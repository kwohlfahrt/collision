from numpy import dtype, zeros
from pathlib import Path
import pyopencl as cl
from misc import Program

class ReductionProgram(Program):
    src = Path(__file__).parent / "reduce.cl"
    kernel_args = {'bounds1': [None, dtype('uint64'), None, None],
                   'bounds2': [None, None]}

class Reducer:
    value_dtype = dtype('float32')

    def __init__(self, ctx, ngroups, group_size, program=None):
        if program is None:
            program = ReductionProgram(ctx)
        elif program.context != ctx:
            raise ValueError("Collider and program context must match")
        self.program = program

        self.ngroups = ngroups
        self.group_size = group_size

        self._group_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.group_size * 2 * 3 * self.value_dtype.itemsize
        )

    def reduce(self, cq, size, values_buf, output_buf, wait_for=None):
        e = self.program.kernels['bounds1'](
            cq, (self.ngroups,), (self.group_size,),
            values_buf, size, self._group_buf,
            cl.LocalMemory(2 * 3 * self.group_size * self.value_dtype.itemsize),
            g_times_l=True, wait_for=wait_for
        )

        e = self.program.kernels['bounds2'](
            cq, (1,), (self.ngroups,),
            self._group_buf, output_buf,
            g_times_l=True, wait_for=[e]
        )

        return e
