from numpy import dtype
from .reduce import ReductionProgram, Reducer

class BoundsProgram(ReductionProgram):
    accumulator = [("INFINITY", "min"), ("-INFINITY", "max")]

    def __init__(self, ctx, coord_dtype=dtype(('float32', 3))):
        super().__init__(ctx, coord_dtype)

class Bounds(Reducer):
    program_type = BoundsProgram

    def __init__(self, ctx, ngroups, group_size, coord_dtype=dtype(('float32', 3)),
                 program=None):
        super().__init__(ctx, ngroups, group_size, coord_dtype, program)
