from numpy import dtype, zeros
from pathlib import Path
import pyopencl as cl
from .misc import Program, dtype_decl, np_float_dtypes

class ReductionProgram(Program):
    src = Path(__file__).parent / "reduce.cl"
    kernel_args = {'bounds1': [None, dtype('uint64'), None, None],
                   'bounds2': [None, None]}

    def __init__(self, ctx, coord_dtype=dtype('float32')):
        coord_dtype = dtype(coord_dtype)
        if coord_dtype not in np_float_dtypes:
            raise ValueError("Invalid dtype: {}".format(coord_dtype))
        self.coord_dtype = coord_dtype

        super().__init__(ctx, ["-DDTYPE={}".format(dtype_decl(dtype((coord_dtype, 3))))])

class Reducer:
    def __init__(self, ctx, ngroups, group_size, coord_dtype=dtype('float32'), program=None):
        if program is None:
            program = ReductionProgram(ctx, coord_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Collider and program context must match")
            if program.coord_dtype != coord_dtype:
                raise ValueError("Reducer and program coord dtypes must match")
        self.program = program

        self.ngroups = ngroups
        self.group_size = group_size

        self._group_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.ngroups * 2 * 4 * self.program.coord_dtype.itemsize
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
                self.ngroups * 2 * 4 * self.program.coord_dtype.itemsize
            )

    def reduce(self, cq, size, values_buf, output_buf, wait_for=None):
        e = self.program.kernels['bounds1'](
            cq, (self.ngroups,), (self.group_size,),
            values_buf, size, self._group_buf,
            cl.LocalMemory(2 * 4 * self.group_size * self.program.coord_dtype.itemsize),
            g_times_l=True, wait_for=wait_for
        )

        e = self.program.kernels['bounds2'](
            cq, (1,), (self.ngroups,),
            self._group_buf, output_buf,
            g_times_l=True, wait_for=[e]
        )

        return e
