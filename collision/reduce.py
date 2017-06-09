from numpy import dtype
from pathlib import Path
import pyopencl as cl
from .misc import Program, dtype_decl, dtype_sizeof, np_float_dtypes

class ReductionProgram(Program):
    src = Path(__file__).parent / "reduce.cl"
    kernel_args = {'bounds1': [None, dtype('uint64'), None, None],
                   'bounds2': [None, None]}

    def __init__(self, ctx, value_dtype):
        self.value_dtype = dtype(value_dtype)
        self.acc_dtype = dtype((value_dtype, len(self.acc_inits)))
        super().__init__(ctx, [
            "-DVALDTYPE={}".format(dtype_decl(self.value_dtype)),
            "-DACC_SIZE={}".format(len(self.acc_inits)),
            "-DACC_INIT='{{{}}}'".format(
                ', '.join(map("(VALDTYPE)({})".format, self.acc_inits))
            ),
        ])

class Reducer:
    program_type = ReductionProgram

    def __init__(self, ctx, ngroups, group_size, value_dtype, program=None):
        if program is None:
            program = self.program_type(ctx, value_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Collider and program context must match")
            if program.value_dtype != value_dtype:
                raise ValueError("Reducer and program value dtypes must match")
        self.program = program

        self.ngroups = ngroups
        self.group_size = group_size

        self._group_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.ngroups * dtype_sizeof(self.program.acc_dtype)
        )

    def resize(self, ngroups=None, group_size=None):
        ctx = self.program.context
        if ngroups is None:
            ngroups = self.ngroups
        if group_size is None:
            group_size = self.group_size

        old_group_size = self.group_size
        self.ngroups = ngroups
        self.group_size = group_size

        if self.group_size != old_group_size:
            self._group_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.ngroups * dtype_sizeof(self.program.acc_dtype)
            )

    def reduce(self, cq, size, values_buf, output_buf, wait_for=None):
        e = self.program.kernels['bounds1'](
            cq, (self.ngroups,), (self.group_size,),
            values_buf, size, self._group_buf,
            cl.LocalMemory(self.group_size * dtype_sizeof(self.program.acc_dtype)),
            g_times_l=True, wait_for=wait_for
        )

        e = self.program.kernels['bounds2'](
            cq, (1,), (self.ngroups,),
            self._group_buf, output_buf,
            g_times_l=True, wait_for=[e]
        )

        return e
