import random

from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn

        import dspy

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

            def dump_state(self, json_mode=True):
                # named_parameters() skips sub-modules flagged _compiled=True (kept frozen elsewhere),
                # so the default state dump of an ensemble of already-compiled candidates is empty and
                # save/load loses every candidate. Serialize each candidate's own state directly so the
                # round-trip works regardless of the _compiled flag (#775).
                return {"programs": [program.dump_state(json_mode=json_mode) for program in self.programs]}

            def load_state(self, state, **kwargs):
                for program, program_state in zip(self.programs, state["programs"], strict=False):
                    program.load_state(program_state, **kwargs)

        return EnsembledProgram()
