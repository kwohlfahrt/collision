import numpy as np
import pyopencl as cl
import pytest

from collision.summer import *
from collision.misc import dtype_sizeof

def pytest_generate_tests(metafunc):
    if 'coord_dtype' in metafunc.fixturenames:
        # float32 has precision issues
        metafunc.parametrize("coord_dtype", list(map(dtype, [
            ('float64', 3), ('uint32', 4), 'int32'
        ])), scope='module')

@pytest.fixture(scope='module')
def program(cl_env, coord_dtype):
    ctx, cq = cl_env
    return SumProgram(ctx, coord_dtype)

def test_sum(cl_env, program, coord_dtype):
    ctx, cq = cl_env

    size, ngroups, group_size = 100, 4, 8
    reducer = Summer(ctx, ngroups, group_size, coord_dtype, program)

    if coord_dtype.shape == (3,):
        value_dtype = dtype((coord_dtype.base, 4))
    else:
        value_dtype = coord_dtype
    values = (np.random.uniform(0, 200, size=(size,) + value_dtype.shape)
              .astype(value_dtype.base))

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        dtype_sizeof(coord_dtype)
    )

    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, value_dtype.shape, value_dtype.base,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = values.sum(axis=0)
    if coord_dtype.shape == (3,):
        out_buf = out_buf[..., :3]
        expected = expected[..., :3]
    np.testing.assert_almost_equal(out_buf, expected)
