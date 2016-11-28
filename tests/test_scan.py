import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian
from .common import cl_env

def pytest_generate_tests(metafunc):
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32', 'uint64'], scope='module')


@pytest.fixture(scope='module')
def scan_kernels(cl_env):
    kernel_args = {'local_scan': [None, None], 'block_scan': [None, None]}
    ctx, cq = cl_env

    src = Path(__file__).parent / ".." / "collision" / "scan.cl"
    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build()
    kernels = {name: getattr(program, name) for name in kernel_args}
    for name, kernel in kernels.items():
        kernel.set_scalar_arg_dtypes(kernel_args[name])
    return kernels


def test_scan(cl_env, scan_kernels):
    ctx, cq = cl_env

    values = np.array([17,  6, 24, 28, 18, 22,  2,  1,
                       25, 17,  7, 17,  3, 19,  8, 23], dtype='uint32')

    block_size = 4
    nblocks = len(values) // 2 // block_size

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    block_sums_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE,
        nblocks * values.dtype.itemsize
    )
    calc_scan = scan_kernels['local_scan'](
        cq, (len(values) // 2,), (block_size,),
        values_buf, block_sums_buf,
    )

    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[calc_scan], is_blocking=True,
    )
    (block_sums_map, _) = cl.enqueue_map_buffer(
        cq, block_sums_buf, cl.map_flags.READ,
        0, (nblocks,), values.dtype,
        wait_for=[calc_scan], is_blocking=True,
    )

    expected = np.array([  0, 17,  23,  47,  75,  93, 115, 117,
                           0, 25,  42,  49,  66,  69,  88,  96], dtype=values.dtype)
    np.testing.assert_equal(values_map, expected)
    expected = np.array([ 118, 119], dtype=values.dtype)
    np.testing.assert_equal(block_sums_map, expected)


def test_block_scan(cl_env, scan_kernels):
    ctx, cq = cl_env

    values = np.array([  0, 17,  23,  47,  75,  93, 115, 117,
                         0, 25,  42,  49,  66,  69,  88,  96], dtype='uint32')
    block_sums = np.array([ 118, 119], dtype=values.dtype)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    block_sums_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=block_sums
    )

    calc_block_scan = scan_kernels['local_scan'](
        cq, (1,), (len(block_sums),),
        block_sums_buf, None,
        g_times_l=True
    )

    (block_sums_map, _) = cl.enqueue_map_buffer(
        cq, block_sums_buf, cl.map_flags.READ,
        0, block_sums.shape, block_sums.dtype,
        wait_for=[calc_block_scan], is_blocking=True
    )
    expected = np.array([   0, 118], dtype=values.dtype)
    np.testing.assert_equal(block_sums_map, expected)

    calc_scan = scan_kernels['block_scan'](
        cq, (len(values) // 2,), (4,),
        values_buf, block_sums_buf,
        wait_for=[calc_block_scan]
    )
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[calc_scan], is_blocking=True
    )
    expected = np.array([   0, 17,  23,  47,  75,  93, 115, 117,
                          118, 143, 160, 167, 184, 187, 206, 214], dtype=values.dtype)
    np.testing.assert_equal(values_map, expected)

