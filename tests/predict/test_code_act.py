import pytest

import dspy
from dspy import Signature
from dspy.predict import CodeAct
from dspy.utils import DummyLM

pytestmark = pytest.mark.deno


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

def add(a: float, b: float) -> float:
    "add two numbers"
    return a + b

def test_codeact_code_generation():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "code_output_0": '"2\\n"',
        "generated_code_0": "result = add(1,1)\nprint(result)",
    }
    assert program.interpreter is None  # no user-provided interpreter: forward() used a per-call one


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")

def extract_maximum_minimum(input_list: str) -> dict[str, float]:
    numbers = list(map(float, input_list.split(",")))
    return {"maximum": max(numbers), "minimum": min(numbers)}

def test_codeact_support_multiple_fields():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(ExtremumFinder, tools=[extract_maximum_minimum])
    res = program(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert res.trajectory == {
        "code_output_0": '"{\'maximum\': 6.0, \'minimum\': 2.0}\\n"',
        "generated_code_0": "result = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)",
    }
    assert program.interpreter is None  # no user-provided interpreter: forward() used a per-call one


def test_codeact_code_parse_failure():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nparse(error\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "parse(error",
        "observation_0": "Failed to execute the generated code: Invalid Python syntax. message: ",
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }
    assert program.interpreter is None  # no user-provided interpreter: forward() used a per-call one


def test_codeact_code_execution_failure():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nunknown+1\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "unknown+1",
        "observation_0": 'Failed to execute the generated code: NameError: ["name \'unknown\' is not defined"]',
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }
    assert program.interpreter is None  # no user-provided interpreter: forward() used a per-call one


class CustomTool:
    def __call__(self, a: float, b: float) -> float:
        return a + b

def test_codeact_tool_validation():
    with pytest.raises(ValueError, match="CodeAct only accepts functions and not callable objects."):
        CodeAct(BasicQA, tools=[CustomTool()])


def _simple_tool(x: int) -> int:
    return x


def test_codeact_truncate_trajectory_drops_oldest_iteration_boundary_aware():
    """CodeAct steps have a variable key count (1 on failure: observation_i; 2 on success:
    generated_code_i + code_output_i|observation_i), unlike ReAct's fixed 4 keys/step. Popping
    ReAct's fixed keys[:4] slice cuts across iteration boundaries and desynchronizes the
    trajectory. CodeAct must drop exactly the earliest iteration's keys."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    trajectory = {
        "generated_code_0": "print(1)",   # iter 0: success -> 2 keys
        "code_output_0": "1",
        "observation_1": "parse failure",  # iter 1: failure -> 1 key
        "generated_code_2": "print(2)",    # iter 2: execution error -> 2 keys
        "observation_2": "exec error",
    }
    out = program.truncate_trajectory(trajectory)

    # Oldest iteration (0) fully removed; every later iteration is preserved intact.
    assert "generated_code_0" not in out
    assert "code_output_0" not in out
    assert out["observation_1"] == "parse failure"
    assert out["generated_code_2"] == "print(2)"
    assert out["observation_2"] == "exec error"
    assert out is trajectory


def test_codeact_truncate_trajectory_single_iteration_raises():
    """A trajectory with only one iteration cannot be truncated (dropping it leaves no context),
    mirroring ReAct's single-tool-call guard."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    with pytest.raises(ValueError):
        program.truncate_trajectory({"generated_code_0": "print(1)", "code_output_0": "1"})


@pytest.mark.asyncio
async def test_codeact_async_code_generation():
    """dspy-374: CodeAct had no aforward, so acall() resolved via MRO to ReAct.aforward, which
    references self.react (never set by CodeAct.__init__) -> AttributeError. CodeAct now has its own
    async path mirroring forward()."""
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = await program.acall(question="What is 1+1?")
    assert res.answer == "2"
    # The async path must build the same trajectory shape as forward(): one iteration with the
    # generated code and its captured output (exact stdout serialization is an interpreter detail).
    assert set(res.trajectory.keys()) == {"generated_code_0", "code_output_0"}
    assert res.trajectory["generated_code_0"] == "result = add(1,1)\nprint(result)"
    assert "2" in res.trajectory["code_output_0"]
    assert program.interpreter is None  # no user-provided interpreter: aforward() used a per-call one
