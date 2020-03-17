import numpy as np
import pyopencl as cl
import pytest

from collision.index import *

from .common import cl_env

def pytest_generate_tests(metafunc):
    if 'index_dtype' in metafunc.fixturenames:
        metafunc.parametrize(
            "index_dtype", map(np.dtype, ['uint32', 'uint64']), scope='module'
        )
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize(
            "value_dtype", map(np.dtype, ['uint32', ('float64', 2)]), scope='module'
        )

def test_gather(cl_env, value_dtype, index_dtype):
    ctx, cq = cl_env

    size = 240
    nindices = 30
    indexer = Indexer(ctx, value_dtype, index_dtype)
    values = (np.random.uniform(0, 1000, (size,) + value_dtype.shape)
              .astype(value_dtype.base))
    indices = np.random.choice(size, size=nindices, replace=False).astype(index_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    values_out_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, nindices * value_dtype.itemsize
    )
    index_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=indices
    )

    e = indexer.gather(cq, nindices, values_buf, index_buf, values_out_buf)
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_out_buf, cl.map_flags.READ,
        0, (nindices,) + value_dtype.shape, value_dtype.base,
        wait_for=[e], is_blocking=True
    )
    np.testing.assert_equal(values_map, values[indices])

def test_scatter(cl_env, value_dtype, index_dtype):
    ctx, cq = cl_env

    size = 240
    nindices = 30
    indexer = Indexer(ctx, value_dtype, index_dtype)
    values = (np.random.uniform(0, 1000, (nindices,) + value_dtype.shape)
              .astype(value_dtype.base))
    indices = np.random.choice(size, size=nindices, replace=False).astype(index_dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    values_out_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, size * value_dtype.itemsize
    )
    index_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=indices
    )

    e = cl.enqueue_fill_buffer(
        cq, values_out_buf, np.full(1, 1.0, value_dtype), 0, size * value_dtype.itemsize
    )
    e = indexer.scatter(cq, nindices, values_buf, index_buf, values_out_buf, wait_for=[e])
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_out_buf, cl.map_flags.READ,
        0, (size,) + value_dtype.shape, value_dtype.base,
        wait_for=[e], is_blocking=True
    )

    selection = np.zeros(size, dtype='bool')
    selection[indices] = True
    np.testing.assert_equal(values_map[indices], values)
    np.testing.assert_equal(values_map[~selection], 1.0)
