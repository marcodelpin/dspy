import socket
from unittest import mock

import pytest

from dspy.adapters.types import _http_download
from dspy.adapters.types._http_download import UnsafeURLError
from dspy.adapters.types.audio import Audio, _normalize_audio_format, encode_audio


def _fake_getaddrinfo(ip):
    def _resolver(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    return _resolver


@pytest.mark.parametrize(
    "input_format, expected_format",
    [
        # Case 1: Standard format (no change)
        ("wav", "wav"),
        ("mp3", "mp3"),

        # Case 2: The 'x-' prefix
        ("x-wav", "wav"),
        ("x-mp3", "mp3"),
        ("x-flac", "flac"),

        # Case 3: The edge case
        ("my-x-format", "my-x-format"),
        ("x-my-format", "my-format"),

        # Case 4: Empty string and edge cases
        ("", ""),
        ("x-", ""),
    ],
)
def test_normalize_audio_format(input_format, expected_format):
    """
    Tests that the _normalize_audio_format helper correctly removes 'x-' prefixes.
    This single test covers the logic for from_url, from_file, and encode_audio.
    """
    assert _normalize_audio_format(input_format) == expected_format


def test_encode_audio_does_not_auto_download_from_url():
    """Regression (#9993): implicit coercion of a URL string into an Audio field must NOT
    auto-download. Auto-fetching untrusted strings (tool output, retrieved docs) is an
    SSRF/DoS surface. Downloading is opt-in via the explicit, guarded Audio.from_url().
    """
    with pytest.raises(ValueError, match=r"Audio\.from_url"):
        encode_audio("http://example.com/sound.wav")


def test_from_url_goes_through_ssrf_guard():
    """Regression (#9993): Audio.from_url must refuse a host that resolves to an internal
    address (here the cloud-metadata endpoint) before any data is returned.
    """
    with mock.patch.object(_http_download.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")):
        with pytest.raises(UnsafeURLError):
            Audio.from_url("http://metadata.example.com/latest/meta-data/")


def test_from_url_downloads_from_public_host():
    """Audio.from_url succeeds for a public host and encodes the fetched bytes."""
    fake = mock.Mock()
    fake.headers = {"Content-Type": "audio/wav"}
    fake.content = b"RIFFfake"
    with mock.patch("dspy.adapters.types.audio.download_bytes", return_value=fake) as mock_dl:
        audio = Audio.from_url("http://example.com/sound.wav")
    mock_dl.assert_called_once()
    assert audio.audio_format == "wav"
