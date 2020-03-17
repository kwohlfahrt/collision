from numpy import dtype
from .reduce import ReductionProgram, Reducer

class SumProgram(ReductionProgram):
    accumulator = [("0", "ADD")]

class Summer(Reducer):
    program_type = SumProgram
