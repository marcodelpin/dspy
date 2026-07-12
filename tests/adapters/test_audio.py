from unittest import mock

import pytest

from dspy.adapters.types.audio import Audio, _normalize_audio_format


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


def test_from_url_passes_timeout():
    """Regression (#9993): Audio.from_url must bound the download with a timeout.

    Without a timeout, a slow/hanging endpoint blocks the request indefinitely.
    """
    fake_response = mock.Mock()
    fake_response.headers = {"Content-Type": "audio/wav"}
    fake_response.content = b"RIFFfake"
    fake_response.raise_for_status = mock.Mock()

    with mock.patch("dspy.adapters.types.audio.requests.get", return_value=fake_response) as mock_get:
        Audio.from_url("http://example.com/sound.wav")

    mock_get.assert_called_once()
    timeout = mock_get.call_args.kwargs.get("timeout")
    assert timeout is not None and timeout > 0, "Audio.from_url must pass a positive timeout to requests.get"
