import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian
from .common import cl_env
from .test_scan import scan_kernels

np.random.seed(4)

def pytest_generate_tests(metafunc):
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32', 'uint64'], scope='module')


@pytest.fixture(scope='module')
def radix_kernels(cl_env, request, value_dtype):
    kernel_args = {'block_sort': [None, None, None, None, None, None,
                                  np.dtype('int8'), np.dtype('int8')],}
    c_dtypes = {'uint32': 'int', 'uint64': 'long'}
    ctx, cq = cl_env

    src = Path(__file__).parent / ".." / "collision" / "radix.cl"
    buildopts = ["-D DTYPE={}".format(c_dtypes[value_dtype]),
                 "-I {}".format(src.parent)]

    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build(' '.join(buildopts))
    kernels = {name: getattr(program, name) for name in kernel_args}
    for name, kernel in kernels.items():
        kernel.set_scalar_arg_dtypes(kernel_args[name])
    return kernels


def radix_key(values, radix_bits, radix_pass):
    return (values >> (radix_pass * radix_bits)) & ((2 ** radix_bits) - 1)


# group_size must be power of 2 (for scan to work)
@pytest.mark.parametrize("ngroups,group_size", [(1, 8), (4, 8), (8, 32)])
def test_block_sort_random(cl_env, radix_kernels, value_dtype, ngroups, group_size):
    ctx, cq = cl_env

    radix_bits = 4
    histogram_len = 2 ** radix_bits

    keys = np.random.randint(0, 64, size=(ngroups, group_size * 2), dtype=value_dtype)

    keys_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, keys.nbytes)
    histogram_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                              ngroups * histogram_len * np.dtype('uint32').itemsize)

    local_keys = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    count = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    local_histogram = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)

    for radix_pass in range(keys.dtype.itemsize // radix_bits):
        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.WRITE_INVALIDATE_REGION, 0,
            (ngroups, group_size * 2), keys.dtype, wait_for=[], is_blocking=True
        )
        keys_map[...] = keys
        del keys_map

        e = radix_kernels['block_sort'](
            cq, (ngroups,), (group_size,),
            keys_buf, histogram_buf, local_keys, local_keys, local_histogram, count,
            radix_bits, radix_pass, g_times_l=True,
        )

        keys = keys.reshape(ngroups, group_size * 2)
        order = np.argsort(radix_key(keys, radix_bits, radix_pass), kind='mergesort')
        grid = np.ogrid[tuple(slice(0, s) for s in keys.shape)]

        expected = keys[grid[:-1] + [order]]

        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.READ, 0,
            (ngroups, group_size * 2), keys.dtype, wait_for=[e], is_blocking=True
        )
        np.testing.assert_equal(keys_map, expected)

        (histogram_map, _) = cl.enqueue_map_buffer(
            cq, histogram_buf, cl.map_flags.READ, 0,
            (histogram_len, ngroups), np.dtype('uint32'), wait_for=[e], is_blocking=True
        )

        for group_keys, histogram in zip(keys, histogram_map.T):
            group_keys = radix_key(group_keys, radix_bits, radix_pass).astype('uint16')
            expected = np.bincount(group_keys, minlength=16)
            np.testing.assert_equal(histogram, expected)
