import numpy as np
import pyopencl as cl
import pytest
from collision.offset import *

from .common import cl_env

np.random.seed(4)

def pytest_generate_tests(metafunc):
    if 'offset_dtype' in metafunc.fixturenames:
        metafunc.parametrize(
            "offset_dtype", map(np.dtype, ['uint32', 'uint64']), scope='module'
        )
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize(
            "value_dtype", map(np.dtype, ['uint32', 'uint64']), scope='module'
        )

@pytest.fixture(scope='module')
def offset_program(cl_env, offset_dtype, value_dtype):
    ctx, cq = cl_env
    return OffsetProgram(ctx, value_dtype, offset_dtype)

def test_offset(cl_env, offset_dtype, value_dtype, offset_program):
    ctx, cq = cl_env
    finder = OffsetFinder(ctx, value_dtype, offset_dtype, offset_program)

    values = np.array([0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 4, 5, 5], dtype=value_dtype)
    expected = np.array([0, 2, 7, 7, 10, 11, 13], dtype=offset_dtype)
    values_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
    offset_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY,
                           len(expected) * offset_dtype.itemsize)

    e = finder.find_offsets(cq, values_buf, len(values), offset_buf, int(values.max())+2)
    (offset_map, _) = cl.enqueue_map_buffer(
        cq, offset_buf, cl.map_flags.READ,
        0, len(expected), offset_dtype,
        wait_for=[e], is_blocking=True
    )

    np.testing.assert_equal(offset_map, expected)


def test_offset_missing(cl_env, offset_dtype, value_dtype, offset_program):
    ctx, cq = cl_env
    finder = OffsetFinder(ctx, value_dtype, offset_dtype, offset_program)

    values = np.array([1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3], dtype=value_dtype)
    expected = np.array([0, 0, 7, 7, 13, 13, 13], dtype=offset_dtype)
    values_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=values)
    offset_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY,
                           len(expected) * offset_dtype.itemsize)

    e = finder.find_offsets(cq, values_buf, len(values), offset_buf, 7)
    (offset_map, _) = cl.enqueue_map_buffer(
        cq, offset_buf, cl.map_flags.READ,
        0, len(expected), offset_dtype,
        wait_for=[e], is_blocking=True
    )

    np.testing.assert_equal(offset_map, expected)
