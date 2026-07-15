"""Shared, hardened HTTP download helper for URL-backed media types (Image, Audio).

Centralizes the SSRF + timeout + size guards so Image and Audio do not each re-implement
(and drift on) the fetch path. What it guarantees over a bare ``requests.get``:

1. SSRF filter: the URL's host is resolved and every resolved IP is checked; a fetch that
   would reach a private / loopback / link-local / reserved / multicast / CGNAT address
   (including the cloud-metadata endpoint 169.254.169.254 and IPv4-mapped IPv6 forms) is
   refused *before* any connection is made. Redirects are followed manually and re-validated
   hop by hop, so a public URL cannot 3xx-bounce to an internal one. Ambiguous authorities
   (backslash / whitespace / userinfo) that make the validating parser and the connecting
   parser disagree on the host are rejected outright.
2. IP pinning (DNS-rebinding TOCTOU defense): the socket connects to one of the exact IPs
   that passed validation. The hostname is NOT re-resolved at connect time, so an
   attacker-controlled nameserver that answers the validation query with a public IP and
   the connect query with an internal one gains nothing: the connect never asks DNS again.
   For HTTPS the pin preserves SNI and certificate verification against the ORIGINAL
   hostname (``server_hostname`` + ``assert_hostname`` on the connection pool), and the
   ``Host`` header always carries the original host. When an environment proxy applies,
   the proxy performs the connect (and the resolution) itself: the local
   resolve-vs-connect TOCTOU this pin closes does not exist on that path, so the fetch
   goes through the proxy unpinned (rewriting the URL to an IP would only break proxy
   ACLs and CONNECT-by-name semantics).
3. DoS guard: a per-read timeout plus a hard *total* wall-clock deadline (enforced by a
   watchdog that closes the socket, so a slow-trickle peer that keeps the per-read timeout
   from ever firing still cannot hold the connection open) plus a maximum-byte cap; the
   response is requested with ``Accept-Encoding: identity`` so a small compressed body cannot
   expand past the cap.
"""

import ipaddress
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Callable
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException

# Per-read (inactivity) timeout handed to requests, and the total wall-clock deadline for the
# whole download. The per-read timeout catches a fully silent peer; the watchdog-enforced
# deadline catches a slow-trickle peer that keeps the read timeout from ever firing.
_READ_TIMEOUT = 30
_TOTAL_DEADLINE = 60
_DNS_TIMEOUT = 10  # bound getaddrinfo so a slow/never-answering resolver cannot stall the fetch
_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB
_MAX_REDIRECTS = 5

# IPv6 transition prefixes that embed an IPv4 address. Python's ipaddress is_private/is_reserved
# does NOT reliably flag these across the whole supported range (e.g. 2002::/16 is absent from
# 3.10/3.11's _private_networks), so we decode the embedded IPv4 ourselves and range-check THAT.
_SIXTOFOUR = ipaddress.ip_network("2002::/16")
_TEREDO = ipaddress.ip_network("2001::/32")
_NAT64 = [ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48")]

# Networks that route to shared/internal infrastructure but that Python's is_private / is_reserved
# family does NOT flag (and, for CGNAT, whose is_global verdict has varied across CPython versions,
# so we enumerate them explicitly rather than rely on is_global for version-stable behavior).
_EXTRA_DISALLOWED_NETWORKS = [
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 CGNAT / shared address space
]

# Characters that must never appear raw in an http(s) URL: a backslash or raw whitespace makes
# stdlib urlparse (RFC 3986) and requests/urllib3 (WHATWG-ish) disagree on where the authority
# ends, so the host we validate would not be the host we connect to.
_FORBIDDEN_URL_CHARS = ("\\", " ", "\t", "\n", "\r", "\x00")


class UnsafeURLError(ValueError):
    """Raised when a URL is refused by the SSRF guard (or is otherwise not fetchable)."""


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> "ipaddress.IPv4Address | None":
    """Return the IPv4 address embedded in an IPv4-mapped / 6to4 / Teredo / NAT64 IPv6 address,
    or None. Traffic to these v6 forms is delivered to the embedded v4, so that v4 is what the
    SSRF range check must judge."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return mapped
    packed = ip.packed
    if ip in _SIXTOFOUR:  # 2002:AABB:CCDD:: -> A.B.C.D in bytes 2..5
        return ipaddress.IPv4Address(packed[2:6])
    if ip in _TEREDO:  # client IPv4 is the last 32 bits, bitwise-complemented
        return ipaddress.IPv4Address(int.from_bytes(packed[12:16], "big") ^ 0xFFFFFFFF)
    if any(ip in net for net in _NAT64):  # embedded IPv4 in the low 32 bits
        return ipaddress.IPv4Address(packed[12:16])
    return None


def _classify_ip(addr: str) -> ipaddress._BaseAddress:
    """Parse ``addr`` into an ip address, unwrapping an IPv4-in-IPv6 embedding so the embedded v4
    address is what gets range-checked (``::ffff:169.254.169.254`` and ``2002:a9fe:a9fe::`` must
    both be judged as the link-local v4 they really carry)."""
    ip = ipaddress.ip_address(addr)
    if ip.version == 6:
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            return embedded
    return ip


def _is_disallowed(ip: ipaddress._BaseAddress) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    return any(ip in net for net in _EXTRA_DISALLOWED_NETWORKS)


def assert_public_url(url: str) -> list[str]:
    """Refuse ``url`` unless it is an unambiguous http(s) URL whose host resolves *only* to
    public IPs; return the validated addresses.

    Returns the resolved IP strings, deduplicated, in ``getaddrinfo`` preference order. The
    caller pins the connection to one of these exact addresses (see :func:`_pinned_get`), so
    a nameserver that answers differently on a second lookup cannot redirect the socket.

    Raises :class:`UnsafeURLError` for a non-http(s) scheme, an ambiguous authority (backslash /
    whitespace / userinfo, which can desync the validating and connecting parsers), a missing or
    unresolvable host, or any resolved address in a private / loopback / link-local / reserved /
    multicast / CGNAT range. Resolving through ``getaddrinfo`` also normalizes decimal/octal/hex
    IP encodings (e.g. ``http://2130706433/``) to the real address before the range check.
    """
    if any(c in url for c in _FORBIDDEN_URL_CHARS):
        raise UnsafeURLError(f"Refusing to download from a URL with a backslash or whitespace: {url!r}")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Refusing to download from non-http(s) URL scheme: {parsed.scheme!r}")
    if "@" in (parsed.netloc or ""):
        # A media URL never needs userinfo; refusing it removes the "http://good@evil/" and
        # "http://evil\\@good/" authority-splitting divergence between parsers.
        raise UnsafeURLError(f"Refusing to download from a URL with embedded credentials: {url!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError(f"Refusing to download from URL with no host: {url!r}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Bound the resolution: a malicious authoritative nameserver that never answers would
    # otherwise stall getaddrinfo (and the whole download) past the total deadline.
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(socket.getaddrinfo, host, port)
        try:
            infos = future.result(timeout=_DNS_TIMEOUT)
        except FuturesTimeoutError as e:
            raise UnsafeURLError(f"DNS resolution of {host!r} exceeded {_DNS_TIMEOUT}s") from e
        except socket.gaierror as e:
            raise UnsafeURLError(f"Could not resolve host {host!r}: {e}") from e

    # Ordered dedup: keep getaddrinfo's preference order so the pin tries the OS-preferred
    # address first (a plain set would randomize which validated IP gets dialed).
    resolved: list[str] = []
    seen: set[str] = set()
    for info in infos:
        addr = info[4][0]
        if addr not in seen:
            seen.add(addr)
            resolved.append(addr)
    if not resolved:
        raise UnsafeURLError(f"Host {host!r} resolved to no addresses")

    for addr in resolved:
        try:
            ip = _classify_ip(addr)
        except ValueError as e:
            raise UnsafeURLError(f"Host {host!r} resolved to an unparseable address {addr!r}: {e}") from e
        if _is_disallowed(ip):
            raise UnsafeURLError(
                f"Refusing to download from {host!r}: it resolves to non-public address {ip} "
                f"(private/loopback/link-local/reserved/CGNAT). This blocks SSRF to internal "
                f"services and cloud metadata endpoints."
            )
    return resolved


class _PinnedHTTPSAdapter(HTTPAdapter):
    """Transport adapter for a URL rewritten to a validated IP literal: TLS still handshakes
    (SNI) and verifies the certificate against the ORIGINAL hostname.

    ``server_hostname`` / ``assert_hostname`` are set as pool kwargs; urllib3's
    ``_merge_pool_kwargs`` folds them into every pool this adapter creates, including the
    per-request ``pool_kwargs`` path requests >= 2.32 uses (``get_connection_with_tls_context``
    merges, it does not replace)."""

    def __init__(self, tls_hostname: str) -> None:
        self._tls_hostname = tls_hostname
        super().__init__()

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["server_hostname"] = self._tls_hostname
        pool_kwargs["assert_hostname"] = self._tls_hostname
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


def _pinned_netloc(ip: str, port: "int | None") -> str:
    """Authority for a URL that dials ``ip`` directly (IPv6 literals get bracketed)."""
    host = f"[{ip}]" if ":" in ip else ip
    return host if port is None else f"{host}:{port}"


def _pinned_get(
    url: str,
    validated_ips: "list[str]",
    *,
    verify,
    timeout,
) -> "tuple[requests.Response, Callable[[], None]]":
    """One redirect-free GET of ``url`` that connects to a *validated* IP instead of
    re-resolving the hostname (the DNS-rebinding pin). Returns ``(response, close_transport)``;
    the caller MUST invoke ``close_transport()`` once the (streamed) response is consumed.

    Tries each validated IP in order and falls back to the next on a connection error, so a
    multi-homed host keeps the availability it had when requests iterated ``getaddrinfo``
    results itself. When an environment proxy applies to ``url`` the fetch is deliberately
    unpinned (see module docstring, point 2).
    """
    common = {
        "stream": True,
        "allow_redirects": False,
        "verify": verify,
        "timeout": timeout,
    }
    # Accept-Encoding: identity so a compressed body cannot decompress past the byte cap.
    if requests.utils.get_environ_proxies(url):
        resp = requests.get(url, headers={"Accept-Encoding": "identity"}, **common)
        return resp, (lambda: None)

    parsed = urlparse(url)
    host = parsed.hostname
    host_header = host if parsed.port is None else f"{host}:{parsed.port}"

    last_exc: RequestsConnectionError | None = None
    for ip in validated_ips:
        pinned_url = urlunparse(parsed._replace(netloc=_pinned_netloc(ip, parsed.port)))
        session = requests.Session()
        try:
            if parsed.scheme == "https":
                session.mount("https://", _PinnedHTTPSAdapter(host))
            resp = session.get(
                pinned_url,
                headers={"Accept-Encoding": "identity", "Host": host_header},
                **common,
            )
            return resp, session.close
        except RequestsConnectionError as e:
            session.close()
            last_exc = e
        except BaseException:
            session.close()
            raise
    raise last_exc  # every validated IP refused the connection


def download_bytes(url: str, *, verify: bool = True) -> requests.Response:
    """Fetch ``url`` and return the completed :class:`requests.Response` (with ``.content`` and
    ``.headers`` populated), enforcing the SSRF guard + IP pin on every hop and the total
    deadline + size cap on the body.

    ``verify`` is forwarded to requests for TLS certificate verification.
    """
    deadline = time.monotonic() + _TOTAL_DEADLINE
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validated_ips = assert_public_url(current)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnsafeURLError(f"Download exceeded the total deadline of {_TOTAL_DEADLINE}s")

        read_timeout = max(1.0, min(_READ_TIMEOUT, remaining))
        resp, close_transport = _pinned_get(
            current, validated_ips, verify=verify, timeout=(read_timeout, read_timeout)
        )
        try:
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                resp.close()
                if not location:
                    raise UnsafeURLError("Redirect response without a Location header")
                current = requests.compat.urljoin(current, location)
                continue

            resp.raise_for_status()
            return _read_capped(resp, deadline)
        finally:
            close_transport()

    raise UnsafeURLError(f"Too many redirects (> {_MAX_REDIRECTS}) while downloading {url!r}")


def _read_capped(resp: requests.Response, deadline: float) -> requests.Response:
    """Buffer the body under a hard total deadline and a byte cap, then close the connection.

    A watchdog timer closes the raw socket at the deadline so a slow-trickle peer (which keeps
    each individual read under the per-read timeout) cannot outlast the total budget: the blocked
    read is interrupted and iter_content raises.
    """
    deadline_hit = threading.Event()

    def _on_deadline():
        deadline_hit.set()
        raw = getattr(resp, "raw", None)
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), _on_deadline)
    timer.start()
    chunks = []
    total = 0
    try:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if deadline_hit.is_set() or time.monotonic() > deadline:
                raise UnsafeURLError(f"Download exceeded the total deadline of {_TOTAL_DEADLINE}s")
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_BYTES:
                raise UnsafeURLError(f"Download exceeded the maximum size of {_MAX_BYTES} bytes")
            chunks.append(chunk)
    except (RequestException, OSError) as e:
        if deadline_hit.is_set():
            raise UnsafeURLError(f"Download exceeded the total deadline of {_TOTAL_DEADLINE}s") from e
        raise
    finally:
        timer.cancel()
        resp.close()

    resp._content = b"".join(chunks)
    return resp
