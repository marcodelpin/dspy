import pytest

import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2


def test_estimate_lm_calls_returns_two_estimate_lines():
    """Regression guard for the resurrected _estimate_lm_calls (#9849): it must compute the two
    estimate strings without error."""
    dspy.settings.configure(lm=dspy.utils.DummyLM([{"answer": "x"}]))
    opt = MIPROv2(metric=lambda *a, **k: 1.0, auto=None, num_candidates=2)
    prompt_line, task_line = opt._estimate_lm_calls(
        program=dspy.Predict("question -> answer"),
        num_trials=10,
        minibatch=False,
        minibatch_size=35,
        minibatch_full_eval_steps=5,
        valset=[1, 2, 3],
        program_aware_proposer=True,
        num_instruct_candidates=2,
    )
    assert "prompt model calls" in prompt_line
    assert "Program Evaluation" in task_line


def test_compile_invokes_estimate_lm_calls(monkeypatch):
    """#9849: _estimate_lm_calls was fully implemented but never called from compile(). compile() must
    invoke it so the per-run LM-call estimate is emitted."""
    dspy.settings.configure(lm=dspy.utils.DummyLM([{"answer": "x"}]))
    opt = MIPROv2(metric=lambda *a, **k: 1.0, auto=None, num_candidates=2, num_threads=1)

    called = {}
    original = MIPROv2._estimate_lm_calls

    def spy(self, *args, **kwargs):
        called["yes"] = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MIPROv2, "_estimate_lm_calls", spy)

    class _StopError(Exception):
        pass

    def _stop(*args, **kwargs):
        raise _StopError()

    # Short-circuit compile right after the estimate call so the test stays fast and deterministic.
    monkeypatch.setattr(MIPROv2, "_bootstrap_fewshot_examples", _stop)

    program = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="q", answer="a").with_inputs("question")] * 3

    with pytest.raises(_StopError):
        opt.compile(program, trainset=trainset, num_trials=2, valset=trainset, minibatch=False)

    assert called.get("yes") is True


def test_mipro_rejects_non_lm_prompt_model():
    """MIPROv2 must reject a prompt_model/task_model that is not an LM instance (e.g. a bare
    model-name string) at construction time with a clear error, instead of letting it sail through
    and crash deep inside proposal with an opaque AttributeError. Regression test for #1930."""
    with pytest.raises(ValueError, match="prompt_model must be a"):
        MIPROv2(
            metric=lambda *a, **k: 1.0,
            auto=None,
            num_candidates=2,
            prompt_model="openai/gpt-4o",
            task_model=dspy.utils.DummyLM([{"answer": "x"}]),
        )
