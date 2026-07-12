"""Tests for the shared hardened HTTP download helper (SSRF + DoS guards) used by
Image and Audio. Regression coverage for upstream #9993."""

import socket
from unittest import mock

import pytest

from dspy.adapters.types import _http_download
from dspy.adapters.types._http_download import (
    UnsafeURLError,
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


def test_assert_public_url_allows_public_address():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")):
        assert_public_url("http://example.com/x")  # must NOT raise


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


# ---- download_bytes: guarded fetch --------------------------------------------------------

def test_download_bytes_public_ok():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(_http_download, "requests") as mreq:
        mreq.get.return_value = _FakeResponse(headers={"Content-Type": "image/png"}, chunks=[b"\x89PNG"])
        mreq.compat.urljoin = lambda a, b: b
        resp = download_bytes("http://example.com/dog.png")
    assert resp.content == b"\x89PNG"
    # requests.get must be called with a positive timeout and redirects disabled.
    kwargs = mreq.get.call_args.kwargs
    assert kwargs.get("allow_redirects") is False
    assert kwargs.get("timeout") is not None


def test_download_bytes_refuses_internal_host_before_fetch():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")), \
         mock.patch.object(_http_download, "requests") as mreq:
        with pytest.raises(UnsafeURLError):
            download_bytes("http://metadata.example.com/latest/meta-data/")
        mreq.get.assert_not_called()  # no connection attempted


def test_download_bytes_revalidates_redirect_to_internal():
    # A public URL that 302-redirects to an internal address must be refused on the second hop.
    seq = [
        _FakeResponse(status_code=302, location="http://internal.example.com/x"),
    ]
    resolves = {"example.com": "93.184.216.34", "internal.example.com": "10.0.0.9"}

    def _resolver(host, port, *a, **k):
        ip = resolves[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    with mock.patch.object(_http_download.socket, "getaddrinfo", _resolver), \
         mock.patch.object(_http_download, "requests") as mreq:
        mreq.get.side_effect = seq
        mreq.compat.urljoin = lambda a, b: b
        with pytest.raises(UnsafeURLError):
            download_bytes("http://example.com/start")


def test_download_bytes_enforces_size_cap():
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(_http_download, "requests") as mreq, \
         mock.patch.object(_http_download, "_MAX_BYTES", 8):
        mreq.get.return_value = _FakeResponse(chunks=[b"a" * 4, b"b" * 4, b"c" * 4])
        mreq.compat.urljoin = lambda a, b: b
        with pytest.raises(UnsafeURLError, match="maximum size"):
            download_bytes("http://example.com/huge")


def test_download_bytes_enforces_total_deadline():
    # Force the wall-clock deadline to be already exceeded on the first check.
    times = iter([1000.0, 1000.0, 2000.0, 2000.0, 2000.0])
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34")), \
         mock.patch.object(_http_download, "requests") as mreq, \
         mock.patch.object(_http_download.time, "monotonic", lambda: next(times)), \
         mock.patch.object(_http_download, "_TOTAL_DEADLINE", 10):
        mreq.get.return_value = _FakeResponse(chunks=[b"x"])
        mreq.compat.urljoin = lambda a, b: b
        with pytest.raises(UnsafeURLError, match="deadline"):
            download_bytes("http://example.com/slow")
