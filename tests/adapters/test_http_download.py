"""Tests for the shared hardened HTTP download helper (SSRF + DoS guards + DNS-rebinding
IP pin) used by Image and Audio. Regression coverage for upstream #9993."""

import ipaddress
import socket
from unittest import mock

import pytest
import requests as real_requests
import urllib3.util.connection

from dspy.adapters.types import _http_download
from dspy.adapters.types._http_download import (
    UnsafeURLError,
    _pinned_get,
    _pinned_netloc,
    _PinnedHTTPSAdapter,
    assert_public_url,
    download_bytes,
)


def _fake_getaddrinfo(*ips):
    """Return a getaddrinfo stub that resolves any host to the given IP string(s)."""
    def _resolver(host, port, *args, **kwargs):
        out = []
        for ip in ips:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port or 0, 0, 0) if family == socket.AF_INET6 else (ip, port or 0)
            out.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return out
    return _resolver


class _FakeResponse:
    def __init__(self, *, status_code=200, headers=None, chunks=(b"ok",), location=None):
        self.status_code = status_code
        self.headers = dict(headers or {})
        if location is not None:
            self.headers["Location"] = location
        self.is_redirect = status_code in (301, 302, 303, 307, 308)
        self._chunks = list(chunks)
        self.content = b"".join(self._chunks)

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def close(self):
        pass


# ---- assert_public_url: SSRF guard --------------------------------------------------------

@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",          # loopback
        "10.0.0.5",           # private
        "192.168.1.10",       # private
        "172.16.0.1",         # private
        "169.254.169.254",    # link-local / cloud metadata
        "100.64.0.1",         # RFC 6598 CGNAT / shared address space (not is_private)
        "::1",                # IPv6 loopback
        "fc00::1",            # IPv6 unique-local (private)
        "fe80::1",            # IPv6 link-local
        "::ffff:169.254.169.254",  # IPv4-mapped IPv6 of the metadata endpoint
        "2002:a9fe:a9fe::",   # 6to4 embedding of 169.254.169.254 (metadata)
        "2002:0a00:0001::",   # 6to4 embedding of 10.0.0.1 (private)
        "64:ff9b::a00:1",     # NAT64 embedding of 10.0.0.1
        "2001:0:0:0:0:0:f5ff:fffe",  # Teredo embedding of 10.0.0.1
        "0.0.0.0",            # unspecified
    ],
)
def test_assert_public_url_rejects_internal_addresses(ip):
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo(ip)):
        with pytest.raises(UnsafeURLError):
            assert_public_url("http://malicious.example.com/x")


def test_assert_public_url_allows_public_address_and_returns_it():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")):
        assert assert_public_url("http://example.com/x") == ["93.184.216.34"]


def test_assert_public_url_returns_ordered_deduped_ips():
    with mock.patch.object(
        _http_download.socket,
        "getaddrinfo",
        _fake_getaddrinfo("93.184.216.34", "93.184.216.34", "1.1.1.1"),
    ):
        assert assert_public_url("http://example.com/x") == ["93.184.216.34", "1.1.1.1"]


def test_assert_public_url_rejects_when_any_resolved_ip_is_internal():
    # A host resolving to both a public and a private IP must be refused (weakest link).
    with mock.patch.object(
        _http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34", "10.0.0.1")
    ):
        with pytest.raises(UnsafeURLError):
            assert_public_url("http://mixed.example.com/x")


def test_assert_public_url_allows_public_6to4():
    # A 6to4 address wrapping a PUBLIC v4 must be allowed (only the embedded-internal ones block).
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("2002:5db8:d822::")):
        assert_public_url("http://public6to4.example.com/x")  # 93.184.216.34 embedded -> public


def test_assert_public_url_rejects_non_http_scheme():
    with pytest.raises(UnsafeURLError):
        assert_public_url("file:///etc/passwd")
    with pytest.raises(UnsafeURLError):
        assert_public_url("gopher://example.com/")


def test_assert_public_url_rejects_backslash_authority():
    # http://169.254.169.254\@public.com/ : urlparse sees host=public.com but urllib3 would
    # connect to 169.254.169.254. The backslash makes the URL ambiguous -> refuse outright.
    with pytest.raises(UnsafeURLError):
        assert_public_url("http://169.254.169.254\\@public.example.com/latest/meta-data/")


def test_assert_public_url_rejects_userinfo():
    with pytest.raises(UnsafeURLError):
        assert_public_url("http://public.example.com@evil.internal/x")


def test_assert_public_url_bounds_dns_resolution():
    import time as _time

    def _slow(host, port, *a, **k):
        _time.sleep(0.5)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))]

    with mock.patch.object(_http_download, "_DNS_TIMEOUT", 0.05), \
         mock.patch.object(_http_download.socket, "getaddrinfo", _slow):
        with pytest.raises(UnsafeURLError, match="DNS resolution"):
            assert_public_url("http://slow-dns.example.com/x")


def test_assert_public_url_rejects_unresolvable_host():
    def _boom(*a, **k):
        raise socket.gaierror("nope")
    with mock.patch.object(_http_download.socket, "getaddrinfo", _boom):
        with pytest.raises(UnsafeURLError):
            assert_public_url("http://does-not-resolve.invalid/x")


def test_assert_public_url_rejects_decimal_ip_that_resolves_internal():
    # http://2130706433/ is 127.0.0.1 in decimal; getaddrinfo normalizes it, and the
    # normalized address is what gets range-checked.
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("127.0.0.1")):
        with pytest.raises(UnsafeURLError):
            assert_public_url("http://2130706433/")


# ---- IP pinning: DNS-rebinding TOCTOU defense ---------------------------------------------

def test_pin_holds_under_dns_rebinding():
    """A resolver that answers the validation query with a public IP and every later query
    with an internal one must NOT redirect the socket: the connect dials the validated IP.
    (Pre-pin, requests re-resolved the hostname at connect time and dialed 127.0.0.1.)"""
    hostname_lookups = {"n": 0}

    def _rebinding_resolver(host, port, *a, **k):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            hostname_lookups["n"] += 1
            ip = "93.184.216.34" if hostname_lookups["n"] == 1 else "127.0.0.1"
        else:
            ip = host  # IP literal: real getaddrinfo echoes it back, no DNS involved
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    dialed = []

    def _recording_connect(address, *a, **k):
        dialed.append(address)
        raise OSError("connection blocked by test")

    with mock.patch.object(_http_download.socket, "getaddrinfo", _rebinding_resolver), \
         mock.patch.object(real_requests.utils, "get_environ_proxies", lambda url, no_proxy=None: {}), \
         mock.patch.object(urllib3.util.connection, "create_connection", _recording_connect):
        with pytest.raises(real_requests.exceptions.ConnectionError):
            download_bytes("http://rebind.example.com/payload")

    assert dialed, "no connection was attempted"
    dialed_hosts = {addr[0] for addr in dialed}
    assert dialed_hosts == {"93.184.216.34"}, f"socket dialed {dialed_hosts}, pin did not hold"
    # The hostname was resolved exactly once (validation); the connect never asked DNS again.
    assert hostname_lookups["n"] == 1


def test_pinned_netloc_formats():
    assert _pinned_netloc("93.184.216.34", None) == "93.184.216.34"
    assert _pinned_netloc("93.184.216.34", 8080) == "93.184.216.34:8080"
    assert _pinned_netloc("2606:2800:220:1::1", None) == "[2606:2800:220:1::1]"
    assert _pinned_netloc("2606:2800:220:1::1", 8443) == "[2606:2800:220:1::1]:8443"


def test_pinned_https_adapter_preserves_sni_and_cert_hostname():
    """The adapter's pool kwargs must survive the requests >= 2.32 per-request pool_kwargs
    path (get_connection_with_tls_context merges with connection_pool_kw, not replaces)."""
    adapter = _PinnedHTTPSAdapter("example.com")
    assert adapter.poolmanager.connection_pool_kw["server_hostname"] == "example.com"
    assert adapter.poolmanager.connection_pool_kw["assert_hostname"] == "example.com"

    req = real_requests.PreparedRequest()
    req.prepare(method="GET", url="https://93.184.216.34/x", headers={"Host": "example.com"})
    conn = adapter.get_connection_with_tls_context(req, verify=True)
    # server_hostname is not an explicit HTTPSConnectionPool param -> lands in conn_kw;
    # assert_hostname is explicit -> stored on the pool.
    assert conn.conn_kw.get("server_hostname") == "example.com"
    assert getattr(conn, "assert_hostname", None) == "example.com"


def test_pinned_get_rewrites_url_and_sets_host_header():
    session = mock.Mock()
    session.get.return_value = _FakeResponse()
    with mock.patch.object(_http_download, "requests") as mreq:
        mreq.utils.get_environ_proxies.return_value = {}
        mreq.Session.return_value = session
        resp, close_transport = _pinned_get(
            "http://example.com:8080/dog.png",
            ["93.184.216.34"],
            verify=True,
            timeout=(5, 5),
        )
    close_transport()
    url = session.get.call_args.args[0]
    kwargs = session.get.call_args.kwargs
    assert url == "http://93.184.216.34:8080/dog.png"
    assert kwargs["headers"]["Host"] == "example.com:8080"
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (5, 5)
    session.close.assert_called_once()


def test_pinned_get_mounts_tls_adapter_for_https_only():
    session = mock.Mock()
    session.get.return_value = _FakeResponse()
    with mock.patch.object(_http_download, "requests") as mreq:
        mreq.utils.get_environ_proxies.return_value = {}
        mreq.Session.return_value = session
        _pinned_get("https://example.com/x", ["93.184.216.34"], verify=True, timeout=(5, 5))
        assert session.mount.call_count == 1
        prefix, adapter = session.mount.call_args.args
        assert prefix == "https://"
        assert isinstance(adapter, _PinnedHTTPSAdapter)
        assert adapter.poolmanager.connection_pool_kw["server_hostname"] == "example.com"

        session.mount.reset_mock()
        _pinned_get("http://example.com/x", ["93.184.216.34"], verify=True, timeout=(5, 5))
        session.mount.assert_not_called()


def test_pinned_get_falls_back_to_next_validated_ip():
    session = mock.Mock()
    ok = _FakeResponse()
    session.get.side_effect = [real_requests.exceptions.ConnectionError("refused"), ok]
    with mock.patch.object(_http_download, "requests") as mreq:
        mreq.utils.get_environ_proxies.return_value = {}
        mreq.Session.return_value = session
        resp, _ = _pinned_get(
            "http://example.com/x", ["93.184.216.34", "1.1.1.1"], verify=True, timeout=(5, 5)
        )
    assert resp is ok
    dialed = [c.args[0] for c in session.get.call_args_list]
    assert dialed == ["http://93.184.216.34/x", "http://1.1.1.1/x"]


def test_pinned_get_raises_when_all_validated_ips_fail():
    session = mock.Mock()
    session.get.side_effect = real_requests.exceptions.ConnectionError("refused")
    with mock.patch.object(_http_download, "requests") as mreq:
        mreq.utils.get_environ_proxies.return_value = {}
        mreq.Session.return_value = session
        with pytest.raises(real_requests.exceptions.ConnectionError):
            _pinned_get("http://example.com/x", ["93.184.216.34", "1.1.1.1"], verify=True, timeout=(5, 5))
    assert session.close.call_count == 2


def test_pinned_get_uses_unpinned_fetch_through_proxy():
    # A proxy performs the connect itself: the local resolve-vs-connect TOCTOU does not
    # exist on that path, and an IP-rewritten URL would break proxy ACLs -> fetch unpinned.
    with mock.patch.object(_http_download, "requests") as mreq:
        mreq.utils.get_environ_proxies.return_value = {"http": "http://proxy.local:3128"}
        mreq.get.return_value = _FakeResponse()
        resp, close_transport = _pinned_get(
            "http://example.com/x", ["93.184.216.34"], verify=True, timeout=(5, 5)
        )
        close_transport()
        assert mreq.get.call_args.args[0] == "http://example.com/x"  # original URL, not the IP
        mreq.Session.assert_not_called()


# ---- download_bytes: guarded fetch --------------------------------------------------------

def _fake_pinned_get(*responses):
    """Return a _pinned_get stub yielding the given responses in order."""
    seq = list(responses)

    def _stub(url, validated_ips, *, verify, timeout):
        assert validated_ips, "download_bytes must pass the validated IPs to the transport"
        assert timeout is not None
        return seq.pop(0), (lambda: None)

    return _stub


def test_download_bytes_public_ok():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(
             _http_download, "_pinned_get",
             _fake_pinned_get(_FakeResponse(headers={"Content-Type": "image/png"}, chunks=[b"\x89PNG"])),
         ):
        resp = download_bytes("http://example.com/dog.png")
    assert resp.content == b"\x89PNG"


def test_download_bytes_refuses_internal_host_before_fetch():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")), \
         mock.patch.object(_http_download, "_pinned_get") as mpin:
        with pytest.raises(UnsafeURLError):
            download_bytes("http://metadata.example.com/latest/meta-data/")
        mpin.assert_not_called()  # no connection attempted


def test_download_bytes_revalidates_redirect_to_internal():
    # A public URL that 302-redirects to an internal address must be refused on the second hop.
    resolves = {"example.com": "93.184.216.34", "internal.example.com": "10.0.0.9"}

    def _resolver(host, port, *a, **k):
        ip = resolves[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    with mock.patch.object(_http_download.socket, "getaddrinfo", _resolver), \
         mock.patch.object(
             _http_download, "_pinned_get",
             _fake_pinned_get(_FakeResponse(status_code=302, location="http://internal.example.com/x")),
         ):
        with pytest.raises(UnsafeURLError):
            download_bytes("http://example.com/start")


def test_download_bytes_enforces_size_cap():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(
             _http_download, "_pinned_get",
             _fake_pinned_get(_FakeResponse(chunks=[b"a" * 4, b"b" * 4, b"c" * 4])),
         ), \
         mock.patch.object(_http_download, "_MAX_BYTES", 8):
        with pytest.raises(UnsafeURLError, match="maximum size"):
            download_bytes("http://example.com/huge")


def test_download_bytes_enforces_total_deadline():
    # Force the wall-clock deadline to be already exceeded on the first check.
    times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(_http_download, "_pinned_get", _fake_pinned_get(_FakeResponse(chunks=[b"x"]))), \
         mock.patch.object(_http_download.time, "monotonic", lambda: next(times)), \
         mock.patch.object(_http_download, "_TOTAL_DEADLINE", 10):
        with pytest.raises(UnsafeURLError, match="deadline"):
            download_bytes("http://example.com/slow")
