from unittest.mock import Mock, patch

import dspy
from dspy.teleprompt.utils import create_n_fewshot_demo_sets, eval_candidate_program
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass


def test_eval_candidate_program_full_trainset():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 10

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs["devset"]) == len(trainset)
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_full"}
    assert result == 0


def test_eval_candidate_program_minibatch():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(return_value=0)
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    evaluate.assert_called_once()
    _, called_kwargs = evaluate.call_args
    assert len(called_kwargs["devset"]) == batch_size
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_minibatch"}
    assert result == 0

def test_eval_candidate_program_failure():
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = Mock(side_effect=ValueError("Error"))
    batch_size = 3

    result = eval_candidate_program(batch_size, trainset, candidate_program, evaluate)

    assert result.score == 0.0


def test_create_n_fewshot_demo_sets_passes_metric_threshold_for_unshuffled():
    """Verify that metric_threshold is passed to BootstrapFewShot for the unshuffled (seed=-1) case.

    Regression test for https://github.com/stanfordnlp/dspy/issues/9308
    """
    student = DummyModule()
    student.predictor = dspy.Predict("input -> output")
    trainset = [dspy.Example(input="test", output="test").with_inputs("input")]

    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    with patch("dspy.teleprompt.utils.BootstrapFewShot") as MockBootstrap:
        mock_instance = Mock()
        mock_instance.compile.return_value = student
        MockBootstrap.return_value = mock_instance

        create_n_fewshot_demo_sets(
            student=student,
            num_candidate_sets=4,  # -3, -2, -1, 0 → hits seed=-1
            trainset=trainset,
            max_labeled_demos=1,
            max_bootstrapped_demos=1,
            metric=lambda ex, pred, trace=None: 1.0,
            teacher_settings={},
            metric_threshold=0.9,
        )

        # Find the call where seed == -1 (unshuffled few-shot)
        # BootstrapFewShot should be called at least twice: once for seed=-1, once for seed>=0
        calls = MockBootstrap.call_args_list
        assert len(calls) >= 1, "BootstrapFewShot was never called"

        # Every BootstrapFewShot call should include metric_threshold
        for call in calls:
            _, kwargs = call
            assert "metric_threshold" in kwargs, (
                f"metric_threshold missing from BootstrapFewShot call: {kwargs}"
            )
            assert kwargs["metric_threshold"] == 0.9, (
                f"metric_threshold={kwargs['metric_threshold']}, expected 0.9"
            )


def test_create_n_fewshot_demo_sets_labeled_only_does_not_crash():
    """Labeled-only optimization (max_bootstrapped_demos=0, max_labeled_demos>0) must not crash in the
    shuffled few-shot branch. Previously `rng.randint(min_num_samples, max_bootstrapped_demos)` became
    `randint(1, 0)` and raised ValueError('empty range for randrange() (1, 0)'). Regression test for
    https://github.com/stanfordnlp/dspy/issues/9938.
    """
    student = DummyModule()
    student.predictor = dspy.Predict("input -> output")
    trainset = [dspy.Example(input="test", output="test").with_inputs("input")]

    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    with patch("dspy.teleprompt.utils.BootstrapFewShot") as mock_bootstrap:
        mock_instance = Mock()
        mock_instance.compile.return_value = student
        mock_bootstrap.return_value = mock_instance

        # num_candidate_sets=4 -> seeds -3,-2,-1,0 -> seed=0 hits the shuffled few-shot branch that
        # previously crashed on randint(1, 0) with max_bootstrapped_demos=0.
        create_n_fewshot_demo_sets(
            student=student,
            num_candidate_sets=4,
            trainset=trainset,
            max_labeled_demos=2,
            max_bootstrapped_demos=0,
            metric=lambda ex, pred, trace=None: 1.0,
            teacher_settings={},
        )

        # Completing without ValueError is the core assertion; the branches must request 0 bootstrapped
        # demos (the guarded size) rather than sampling from an empty range.
        assert mock_bootstrap.call_args_list, "BootstrapFewShot was never called"
        for _, kwargs in mock_bootstrap.call_args_list:
            assert kwargs.get("max_bootstrapped_demos") == 0
