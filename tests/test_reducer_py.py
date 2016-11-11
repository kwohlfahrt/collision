import numpy as np
import pyopencl as cl
import pytest
from reduce import *

@pytest.fixture(scope='module')
def cl_reduce():
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    program = ReductionProgram(ctx)
    return ctx, cq, program

@pytest.mark.parametrize("size,ngroups,group_size", [(24,2,4), (100, 4, 8)])
def test_bounds(cl_reduce, size, ngroups, group_size):
    ctx, cq, program = cl_reduce

    reducer = Reducer(ctx, ngroups, group_size, program)
    values = np.random.normal(size=(size, 3)).astype(reducer.value_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        2 * 3 * reducer.value_dtype.itemsize
    )

    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, (2, 3), reducer.value_dtype,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = np.stack([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_equal(out_buf, expected)

@pytest.mark.parametrize("size,old_shape,new_shape", [(100,(2,4),(4, 8))])
def test_bounds_resized(cl_reduce, size, old_shape, new_shape):
    ctx, cq, program = cl_reduce

    reducer = Reducer(ctx, *old_shape, program=program)
    reducer.resize(*new_shape)
    values = np.random.normal(size=(size, 3)).astype(reducer.value_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_ONLY |
        cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.WRITE_ONLY,
        2 * 3 * reducer.value_dtype.itemsize
    )

    calc_reduce = reducer.reduce(cq, len(values), values_buf, out_buf)
    (out_buf, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, (2, 3), reducer.value_dtype,
        wait_for=[calc_reduce], is_blocking=True
    )

    expected = np.stack([values.min(axis=0), values.max(axis=0)])
    np.testing.assert_equal(out_buf, expected)
