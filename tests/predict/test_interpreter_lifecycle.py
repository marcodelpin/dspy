"""Interpreter lifecycle tests for ProgramOfThought and CodeAct.

These tests use a fake interpreter (no Deno required) to verify that each
forward() call runs against its own isolated interpreter instance, so that
concurrent execution (e.g. dspy.Evaluate with num_threads > 1) does not
share or prematurely shut down an interpreter across threads.

Regression tests for https://github.com/stanfordnlp/dspy/issues/9082
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.predict import CodeAct


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


def add(a: float, b: float) -> float:
    "add two numbers"
    return a + b


def make_fake_interpreter_class(execute_error: Exception | None = None):
    """Build a fresh fake interpreter class mimicking PythonInterpreter semantics.

    Like the real PythonInterpreter, an instance is owned by the first thread
    that uses it (cross-thread use raises RuntimeError). Unlike the real class
    (which lazily respawns its subprocess after shutdown), executing after
    shutdown always fails here, so these tests can detect a shutdown() racing
    an in-flight call - the failure mode reported in issue #9082, where the
    race lands mid-execute on a live pipe ('I/O operation on closed file').
    """

    class FakeInterpreter:
        instances: ClassVar[list] = []

        def __init__(self, *args, **kwargs):
            self._owner_thread = None
            self.was_shut_down = False
            self.executed = []
            FakeInterpreter.instances.append(self)

        def execute(self, code, variables=None):
            current_thread = threading.current_thread().ident
            if self._owner_thread is None:
                self._owner_thread = current_thread
            elif self._owner_thread != current_thread:
                raise RuntimeError(
                    "PythonInterpreter is not thread-safe and cannot be shared across threads."
                )
            # Widen the race window so a concurrent shutdown() from another
            # thread lands while this call is still "executing".
            time.sleep(0.01)
            if self.was_shut_down:
                raise ValueError("I/O operation on closed file")
            if execute_error is not None:
                raise execute_error
            self.executed.append(code)
            return "42"

        __call__ = execute

        def shutdown(self):
            self.was_shut_down = True
            self._owner_thread = None

    return FakeInterpreter


def make_pot(fake_cls, monkeypatch, **kwargs):
    monkeypatch.setattr("dspy.predict.program_of_thought.PythonInterpreter", fake_cls)
    pot = ProgramOfThought(BasicQA, **kwargs)
    pot.code_generate = lambda **kw: dspy.Prediction(generated_code="```python\nresult = 42\n```")
    pot.generate_output = lambda **kw: dspy.Prediction(answer="42")
    return pot

def make_codeact(fake_cls, monkeypatch, **kwargs):
    monkeypatch.setattr("dspy.predict.code_act.PythonInterpreter", fake_cls)
    monkeypatch.setattr("dspy.predict.program_of_thought.PythonInterpreter", fake_cls)
    program = CodeAct(BasicQA, tools=[add], **kwargs)
    program.codeact = lambda **kw: dspy.Prediction(
        generated_code="```python\nprint(add(1, 1))\n```", finished=True
    )
    program.extractor = lambda **kw: dspy.Prediction(answer="42")
    return program


def test_pot_concurrent_calls_use_isolated_interpreters(monkeypatch):
    fake_cls = make_fake_interpreter_class()
    pot = make_pot(fake_cls, monkeypatch)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pot, question="What is 6*7?") for _ in range(16)]
        results = [future.result() for future in futures]

    assert all(result.answer == "42" for result in results)
    # Every interpreter that ran code was shut down afterwards (no leaks).
    used = [inst for inst in fake_cls.instances if inst.executed]
    assert len(used) == 16
    assert all(inst.was_shut_down for inst in used)


def test_pot_fresh_interpreter_per_call(monkeypatch):
    fake_cls = make_fake_interpreter_class()
    pot = make_pot(fake_cls, monkeypatch)

    for _ in range(3):
        assert pot(question="What is 6*7?").answer == "42"

    used = [inst for inst in fake_cls.instances if inst.executed]
    assert len(used) == 3
    assert len({id(inst) for inst in used}) == 3
    assert all(inst.was_shut_down for inst in used)
    # No user-provided interpreter: the attribute stays None.
    assert pot.interpreter is None


def test_pot_user_provided_interpreter_still_used(monkeypatch):
    fake_cls = make_fake_interpreter_class()
    user_interpreter = fake_cls()
    pot = make_pot(fake_cls, monkeypatch, interpreter=user_interpreter)

    assert pot(question="What is 6*7?").answer == "42"

    assert user_interpreter.executed, "user-provided interpreter must be the one executing code"
    # The caller owns the lifecycle of a user-provided interpreter (mirrors dspy.RLM).
    assert not user_interpreter.was_shut_down


def test_pot_interpreter_assigned_after_construction_is_used(monkeypatch):
    # Covers both post-construction customization and programs saved before
    # this attribute semantics change (a restored instance is non-None).
    fake_cls = make_fake_interpreter_class()
    pot = make_pot(fake_cls, monkeypatch)
    custom = fake_cls()
    pot.interpreter = custom

    assert pot(question="What is 6*7?").answer == "42"

    assert custom.executed
    assert not custom.was_shut_down


def test_pot_interpreter_shut_down_on_max_iters_error(monkeypatch):
    fake_cls = make_fake_interpreter_class(execute_error=Exception("boom"))
    pot = make_pot(fake_cls, monkeypatch)
    pot.code_regenerate = lambda **kw: dspy.Prediction(generated_code="```python\nresult = 42\n```")

    with pytest.raises(RuntimeError, match="Max hops reached"):
        pot(question="What is 6*7?")

    assert fake_cls.instances and all(inst.was_shut_down for inst in fake_cls.instances)


def test_codeact_concurrent_calls_use_isolated_interpreters(monkeypatch):
    fake_cls = make_fake_interpreter_class()
    program = make_codeact(fake_cls, monkeypatch)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(program, question="What is 1+1?") for _ in range(16)]
        results = [future.result() for future in futures]

    assert all(result.answer == "42" for result in results)
    used = [inst for inst in fake_cls.instances if inst.executed]
    assert len(used) == 16
    assert all(inst.was_shut_down for inst in used)


def test_codeact_user_provided_interpreter_still_used(monkeypatch):
    fake_cls = make_fake_interpreter_class()
    user_interpreter = fake_cls()
    program = make_codeact(fake_cls, monkeypatch, interpreter=user_interpreter)

    assert program(question="What is 1+1?").answer == "42"

    assert user_interpreter.executed
    # The caller owns the lifecycle of a user-provided interpreter (mirrors dspy.RLM).
    assert not user_interpreter.was_shut_down


@pytest.mark.deno
def test_pot_concurrent_calls_with_real_interpreter():
    pot = ProgramOfThought(BasicQA)
    pot.code_generate = lambda **kw: dspy.Prediction(
        generated_code="```python\nresult = sum(range(100))\nSUBMIT({'answer': result})\n```"
    )
    pot.generate_output = lambda **kw: dspy.Prediction(answer="4950")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(pot, question=f"q{i}") for i in range(8)]
        results = [future.result() for future in futures]

    assert all(result.answer == "4950" for result in results)
