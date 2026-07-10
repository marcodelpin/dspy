import pydantic

import dspy
from dspy.propose.utils import get_dspy_source_code


class _NarrativeTimePeriod(pydantic.BaseModel):
    """MARKER_NESTED_MODEL_DOCSTRING: a period of narrative time."""

    start_year: int
    end_year: int


def test_get_dspy_source_code_natural_dynamic_signature():
    """Regression guard: get_dspy_source_code must not crash on a dynamically-created signature.
    On dspy 3.3.0b1 dynamic signatures populate __pydantic_parent_namespace__, so this passes; the
    guard keeps it passing if that population behavior ever regresses."""

    class M(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(dspy.Signature("question -> answer"))

    assert isinstance(get_dspy_source_code(M()), str)


def test_get_dspy_source_code_handles_none_pydantic_parent_namespace():
    """#9937: get_dspy_source_code subscripted item.signature.__pydantic_parent_namespace__ without a
    None-guard. On dspy 3.2.1 dynamic signatures had __pydantic_parent_namespace__ == None, raising
    'TypeError: NoneType object is not subscriptable'. The guard must tolerate the None case
    regardless of the pydantic version's population behavior."""

    class M(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(dspy.Signature("question -> answer"))

    m = M()
    # Reproduce the historical 3.2.1 condition the issue reported.
    m.predict.signature.__pydantic_parent_namespace__ = None
    assert isinstance(get_dspy_source_code(m), str)


def test_get_dspy_source_code_emits_nested_pydantic_model_source():
    """#7934: get_dspy_source_code emitted the enclosing module + the signature repr but never the
    SOURCE of custom pydantic models referenced in field annotations (e.g. list[NestedModel]), so a
    MIPROv2 proposer saw only the type name with no field definitions."""

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        periods: list[_NarrativeTimePeriod] = dspy.OutputField()

    class M(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(Sig)

    code = get_dspy_source_code(M())
    # The model's docstring only appears if its class SOURCE was emitted (the signature repr shows
    # only the annotation name list[_NarrativeTimePeriod]).
    assert "MARKER_NESTED_MODEL_DOCSTRING" in code
