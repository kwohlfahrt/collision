from numpy import dtype
from pathlib import Path
import pyopencl as cl
from .misc import Program, np_unsigned_dtypes, dtype_decl

np_unsigned_dtypes = set(map(dtype, np_unsigned_dtypes))

class OffsetProgram(Program):
    src = Path(__file__).parent / "offset.cl"

    def __init__(self, ctx, value_dtype=dtype('uint32'), offset_dtype=dtype('uint32')):
        self.value_dtype = dtype(value_dtype)
        self.offset_dtype = dtype(offset_dtype)
        self.kernel_args = {'find_offsets': [None, None, value_dtype]}

        if self.value_dtype not in np_unsigned_dtypes:
            raise ValueError("Invalid value dtype: {}".format(self.value_dtype))
        if self.offset_dtype not in np_unsigned_dtypes:
            raise ValueError("Invalid offset dtype: {}".format(self.offset_dtype))

        super().__init__(ctx, [
            "-DVALUE_TYPE='{}'".format(dtype_decl(self.value_dtype)),
            "-DOFFSET_TYPE='{}'".format(dtype_decl(self.offset_dtype)),
        ])

class OffsetFinder:
    def __init__(self, ctx, value_dtype=dtype('uint32'), offset_dtype=dtype('uint32'),
                 program=None):
        if program is None:
            program = OffsetProgram(ctx, value_dtype, offset_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Sorter and program contexts must match")
            if program.value_dtype != value_dtype:
                raise ValueError("Sorter and program value dtypes must match")
            if program.offset_dtype != offset_dtype:
                raise ValueError("Sorter and program offset dtypes must match")
        self.program = program

    def find_offsets(self, cq, values_buf, n_values, offsets_buf, n_offsets,
                     wait_for=None):
        wait_for = wait_for or []
        return self.program.kernels['find_offsets'](
            cq, (n_values - 1,), None, values_buf, offsets_buf, n_offsets
        )
