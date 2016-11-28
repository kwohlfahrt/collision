import numpy as np
import pyopencl as cl
import pytest
from collision.reduce import *
from .common import cl_env

def pytest_generate_tests(metafunc):
    if 'coord_dtype' in metafunc.fixturenames:
        metafunc.parametrize("coord_dtype", ['float32', 'float64'], scope='module')

@pytest.fixture(scope='module')
def program(cl_env, coord_dtype):
    ctx, cq = cl_env
    return ReductionProgram(ctx, coord_dtype)

@pytest.mark.parametrize("size,ngroups,group_size", [(24,2,4), (100, 4, 8)])
def test_bounds(cl_env, program, coord_dtype, size, ngroups, group_size):
    ctx, cq = cl_env

    reducer = Reducer(ctx, ngroups, group_size, coord_dtype, program)
    values = np.random.normal(size=(size, 3)).astype(coord_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        2 * 3 * values.dtype.itemsize
    )

    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, (2, 3), values.dtype,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = np.stack([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_equal(out_buf, expected)

@pytest.mark.parametrize("size,old_shape,new_shape", [(100,(2,4),(4, 8))])
def test_bounds_resized(cl_env, program, coord_dtype, size, old_shape, new_shape):
    ctx, cq = cl_env

    reducer = Reducer(ctx, *old_shape, coord_dtype, program=program)
    reducer.resize(*new_shape)
    values = np.random.normal(size=(size, 3)).astype(coord_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        2 * 3 * values.dtype.itemsize
    )

    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, (2, 3), values.dtype,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = np.stack([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_equal(out_buf, expected)

# Need > 1 parameter due to pytest #2067
@pytest.mark.parametrize("size,ngroups,group_size", [(24,2,4), (100, 4, 8)])
def test_auto_program(cl_env, size, ngroups, group_size, coord_dtype):
    ctx, cq = cl_env

    reducer = Reducer(ctx, ngroups, group_size, coord_dtype)
    values = np.random.normal(scale=1e8, size=(size, 3)).astype(coord_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        2 * 3 * values.dtype.itemsize
    )
    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, (2, 3), values.dtype,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = np.stack([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_equal(out_buf, expected)
