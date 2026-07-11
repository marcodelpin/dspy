import pytest

import dspy
from dspy.teleprompt import Ensemble
from dspy.utils import DummyLM


class MockProgram(dspy.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args, **kwargs):
        return self.output


# Simple reduction function to test with
def mock_reduce_fn(outputs):
    return sum(outputs) / len(outputs)


def test_ensemble_without_reduction():
    """Test that Ensemble correctly combines outputs without applying a reduce_fn."""
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble()
    ensembled_program = ensemble.compile(programs)

    outputs = ensembled_program()
    assert len(outputs) == 5, "Ensemble did not combine the correct number of outputs"


def test_ensemble_with_reduction():
    """Test that Ensemble correctly applies a reduce_fn to combine outputs."""
    programs = [MockProgram(i) for i in range(5)]
    ensemble = Ensemble(reduce_fn=mock_reduce_fn)
    ensembled_program = ensemble.compile(programs)

    output = ensembled_program()
    expected_output = sum(range(5)) / 5
    assert output == expected_output, "Ensemble did not correctly apply the reduce_fn"


def test_ensemble_with_size_limitation():
    """Test that specifying a size limits the number of programs used in the ensemble."""
    programs = [MockProgram(i) for i in range(10)]
    ensemble_size = 3
    ensemble = Ensemble(size=ensemble_size)
    ensembled_program = ensemble.compile(programs)

    outputs = ensembled_program()
    assert len(outputs) == ensemble_size, "Ensemble did not respect the specified size limitation"


def test_ensemble_deterministic_behavior():
    """Verify that the Ensemble class raises an assertion for deterministic behavior."""
    with pytest.raises(
        AssertionError,
        match="TODO: Implement example hashing for deterministic ensemble.",
    ):
        Ensemble(deterministic=True)


def test_ensemble_state_round_trips_compiled_candidates():
    """An ensemble of already-compiled (_compiled=True) candidates must serialize and reload each
    candidate's state. named_parameters() skips _compiled sub-modules, so the default dump_state
    returned {} (losing every candidate's demos) and load_state raised KeyError 'programs[0]...'.
    Regression test for #775."""

    class SimpleProg(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict_func = dspy.Predict("question -> answer")

    dspy.configure(lm=DummyLM([{"answer": "x"}]))

    prog0 = SimpleProg()
    prog0.predict_func.demos = [dspy.Example(question="q0", answer="a0")]
    prog0._compiled = True
    prog1 = SimpleProg()
    prog1.predict_func.demos = [dspy.Example(question="q1", answer="a1")]
    prog1._compiled = True

    ensembled = Ensemble(reduce_fn=None).compile([prog0, prog1])

    state = ensembled.dump_state()
    assert state != {}, "ensemble of compiled candidates serialized to an empty state"

    # Reload into a fresh ensemble of uncompiled candidates.
    fresh = Ensemble(reduce_fn=None).compile([SimpleProg(), SimpleProg()])
    fresh.load_state(state)

    assert fresh.programs[0].predict_func.demos[0]["question"] == "q0"
    assert fresh.programs[1].predict_func.demos[0]["question"] == "q1"
