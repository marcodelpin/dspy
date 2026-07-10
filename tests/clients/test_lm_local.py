from unittest import mock
from unittest.mock import patch

from dspy.clients.lm_local import LocalProvider


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_with_spaces_in_path(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "/path/to/my models/llama"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert "--model-path" in command
        model_index = command.index("--model-path")
        assert command[model_index + 1] == "/path/to/my models/llama"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_construction_prevents_injection(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "model --trust-remote-code"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert "--model-path" in command
        model_index = command.index("--model-path")
        assert command[model_index + 1] == "model --trust-remote-code"


@patch("dspy.clients.lm_local.threading.Thread")
@patch("dspy.clients.lm_local.subprocess.Popen")
@patch("dspy.clients.lm_local.get_free_port")
@patch("dspy.clients.lm_local.wait_for_server")
def test_command_is_list_not_string(mock_wait, mock_port, mock_popen, mock_thread):
    mock_port.return_value = 8000
    mock_process = mock.Mock()
    mock_process.pid = 12345
    mock_process.stdout.readline.return_value = ""
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process

    lm = mock.Mock(spec=[])
    lm.model = "meta-llama/Llama-2-7b"
    lm.launch_kwargs = {}
    lm.kwargs = {}

    with mock.patch.dict("sys.modules", {"sglang": mock.Mock(), "sglang.utils": mock.Mock()}):
        LocalProvider.launch(lm, launch_kwargs={})

        assert mock_popen.called
        call_args = mock_popen.call_args
        command = call_args[0][0]

        assert isinstance(command, list)
        assert command[0] == "python"
        assert command[1] == "-m"
        assert command[2] == "sglang.launch_server"
        assert "--model-path" in command
        assert "--port" in command
        assert "--host" in command


def test_sft_max_length_kwarg_maps_to_installed_trl():
    """trl >=0.16 renamed the SFTConfig `max_seq_length` constructor arg to `max_length`. The SFT
    config builder must map the sequence-length value onto whichever kwarg the installed trl exposes,
    so training works across trl versions and no longer raises TypeError on trl >=0.16 (#8762)."""
    from dspy.clients.lm_local import _sft_max_length_kwarg

    # trl >=0.16: SFTConfig exposes `max_length` (no `max_seq_length`)
    class NewSFTConfig:
        def __init__(self, output_dir=None, max_length=None, packing=None):
            pass

    assert _sft_max_length_kwarg(NewSFTConfig, 1024) == {"max_length": 1024}

    # trl <0.16: SFTConfig exposes `max_seq_length`
    class OldSFTConfig:
        def __init__(self, output_dir=None, max_seq_length=None, packing=None):
            pass

    assert _sft_max_length_kwarg(OldSFTConfig, 1024) == {"max_seq_length": 1024}
