import pyopencl as cl
from numpy import dtype
from functools import reduce
import operator as op

class Program:
    def __init__(self, ctx, src, options=None, includes=None):
        options = options or []
        if includes is not None:
            options.append(' '.join(map("-I {}".format, includes)))

        self.program = cl.Program(ctx, src).build(' '.join(options))
        self.kernels = {name: getattr(self.program, name) for name in self.kernel_args}
        for name, kernel in self.kernels.items():
            kernel.set_scalar_arg_dtypes(self.kernel_args[name])

    @property
    def context(self):
        return self.program.get_info(cl.program_info.CONTEXT)


class SimpleProgram(Program):
    def __init__(self, ctx, options=None):
        with self.src.open("r") as f:
            super().__init__(ctx, f.read(), options, [self.src.parent])


def roundUp(x, base=1):
  return (x // base + bool(x % base)) * base

def nextPowerOf2(x):
    return 2 ** (x - 1).bit_length()

def product(xs):
    return reduce(op.mul, xs, 1)

np_integer_dtypes = list(map('int{}'.format, [8, 16, 32, 64]))
np_unsigned_dtypes = list(map('u{}'.format, np_integer_dtypes))
np_float_dtypes = list(map('float{}'.format, [16, 32, 64]))
np_dtypes = np_integer_dtypes + np_unsigned_dtypes + np_float_dtypes

c_integer_dtypes = ['char', 'short', 'int', 'long']
c_unsigned_dtypes = list(map('u{}'.format, c_integer_dtypes))
c_float_dtypes = ['half', 'float', 'double']
c_dtypes = c_integer_dtypes + c_unsigned_dtypes + c_float_dtypes

cl_vector_sizes = {2 ** (n + 1) for n in range(4)} | {3}

np_c_dtypes = dict(zip(map(dtype, np_dtypes), c_dtypes))

def dtype_decl(dt):
    if not dt.shape:
        return np_c_dtypes[dt]
    try:
        n, = dt.shape
    except TypeError:
        raise ValueError("Too many vector dimensions: {}".format(dt.shape))
    if n not in cl_vector_sizes:
        raise ValueError("Invalid vector size: {}".format(n))
    return "{}{}".format(np_c_dtypes[dt.base], n)

def dtype_sizeof(dt):
    if dt.base in np_dtypes:
        *shape, n = dt.shape or (1,)
        if n != 1 and n not in cl_vector_sizes:
            raise ValueError("Invalid vector size: {}".format(n))
        n = 4 if n == 3 else n
        return dt.base.itemsize * product(shape) * n
    else:
        subtype, shape = dt.subdtype
        return product(shape) * dtype_sizeof(subtype)
