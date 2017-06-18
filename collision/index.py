from numpy import dtype
from pathlib import Path
import pyopencl as cl
from .misc import SimpleProgram, np_unsigned_dtypes, dtype_decl

np_unsigned_dtypes = set(map(dtype, np_unsigned_dtypes))

class IndexProgram(SimpleProgram):
    src = Path(__file__).parent / "index.cl"
    kernel_args = {'gather': [None, None, None], 'scatter': [None, None, None]}

    def __init__(self, ctx, value_dtype=dtype('uint32'), index_dtype=dtype('uint32')):
        self.value_dtype = dtype(value_dtype)
        self.index_dtype = dtype(index_dtype)
        if self.index_dtype not in np_unsigned_dtypes:
            raise ValueError("Invalid index dtype: {}".format(self.index_dtype))

        super().__init__(ctx, [
            "-DINDEX_TYPE='{}'".format(dtype_decl(self.index_dtype)),
            "-DVALUE_TYPE='{}'".format(dtype_decl(self.value_dtype))
        ])

class Indexer:
    def __init__(self, ctx, value_dtype=dtype('uint32'), index_dtype=dtype('uint32'),
                 program=None):
        if program is None:
            program = IndexProgram(ctx, value_dtype, index_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Sorter and program contexts must match")
            if program.index_dtype != index_dtype:
                raise ValueError("Sorter and program index dtypes must match")
            if program.value_dtype != value_dtype:
                raise ValueError("Sorter and program value dtypes must match")
        self.program = program

    def gather(self, cq, size, in_values_buf, indices_buf, out_values_buf, wait_for=None):
        wait_for = wait_for or []

        index = self.program.kernels['gather'](
            cq, (size,), None,
            in_values_buf, indices_buf, out_values_buf,
            wait_for=wait_for
        )
        return index

    def scatter(self, cq, size, in_values_buf, indices_buf, out_values_buf, wait_for=None):
        wait_for = wait_for or []

        index = self.program.kernels['scatter'](
            cq, (size,), None,
            in_values_buf, indices_buf, out_values_buf,
            wait_for=wait_for
        )
        return index
