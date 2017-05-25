import numpy as np
import pyopencl as cl
import pytest
from pathlib import Path

from .common import cl_env
from collision.misc import dtype_decl

def pytest_generate_tests(metafunc):
    if 'offset_dtype' in metafunc.fixturenames:
        metafunc.parametrize("offset_dtype", map(np.dtype, ['uint32']), scope='module')
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", map(np.dtype, ['uint32']), scope='module')

@pytest.fixture(scope='module')
def offset_kernels(cl_env, offset_dtype, value_dtype):
    kernel_args = {'find_offsets': [None, None]}
    ctx, cq = cl_env

    src = Path(__file__).parent / ".." / "collision" / "offset.cl"
    buildopts = ["-DOFFSET_TYPE='{}'".format(dtype_decl(offset_dtype)),
                 "-DVALUE_TYPE='{}'".format(dtype_decl(value_dtype)),
                 "-I {}".format(src.parent)]

    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build(' '.join(buildopts))
    kernels = {name: getattr(program, name) for name in kernel_args}
    for name, kernel in kernels.items():
        kernel.set_scalar_arg_dtypes(kernel_args[name])
    return kernels

def test_offset(cl_env, offset_kernels, offset_dtype, value_dtype):
    ctx, cq = cl_env

    values = np.array([0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 4, 5, 5], dtype=value_dtype)
    expected = np.array([0, 2, 7, 7, 10, 11,], dtype=offset_dtype)
    values_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
    offset_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY,
                           expected.nbytes)
    e = offset_kernels['find_offsets'](
        cq, (len(values) - 1,), None, values_buf, offset_buf,
    )
    (offset_map, _) = cl.enqueue_map_buffer(
        cq, offset_buf, cl.map_flags.READ, 0,
        len(expected), expected.dtype, wait_for=[e], is_blocking=True
    )
    np.testing.assert_equal(offset_map, expected)
