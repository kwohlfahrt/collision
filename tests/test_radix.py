import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from inspect import signature
from itertools import product as cartesian

from collision.misc import dtype_decl

np.random.seed(4)

def pytest_generate_tests(metafunc):
    params = signature(metafunc.function).parameters
    if 'key_dtype' in params:
        metafunc.parametrize(
            "key_dtype", map(np.dtype, ['uint32', 'uint64']), scope='module'
        )
    elif 'key_dtype' in metafunc.fixturenames:
        metafunc.parametrize("key_dtype", map(np.dtype, ['uint32']), scope='module')
    if 'value_dtype' in params:
        metafunc.parametrize(
            "value_dtype", map(np.dtype, ['uint32', 'float64']), scope='module'
        )
    elif 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", map(np.dtype, ['uint32']), scope='module')

def prefix_sum(x, axis=None):
    r = np.zeros_like(x)
    r[1:] = np.cumsum(x, axis)[:-1]
    return r

sizes = [(1, 8), (3, 8), (4, 8), (8, 32), (16, 128)]

@pytest.fixture(scope='module')
def radix_kernels(cl_env, request, value_dtype, key_dtype):
    kernel_args = {'block_sort': [None, None, None, None, None, None, None, None, None,
                                  np.dtype('uint8'), np.dtype('uint8')],
                   'scatter': [None, None, None, None, None, None, None, None,
                               np.dtype('uint8'), np.dtype('uint8')]}
    c_dtypes = {'uint32': 'int', 'uint64': 'long'}
    ctx, cq = cl_env

    src = Path(__file__).parent / ".." / "collision" / "radix.cl"
    buildopts = ["-DKEY_TYPE='{}'".format(dtype_decl(key_dtype)),
                 "-DVALUE_TYPE='{}'".format(dtype_decl(value_dtype)),
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
@pytest.mark.parametrize("ngroups,group_size", sizes)
def test_block_sort_random(cl_env, radix_kernels, key_dtype, ngroups, group_size):
    ctx, cq = cl_env

    radix_bits = 4
    histogram_len = 2 ** radix_bits

    keys = np.random.randint(0, 64, size=(ngroups, group_size * 2), dtype=key_dtype)

    keys_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, keys.nbytes)
    histogram_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                              ngroups * histogram_len * np.dtype('uint32').itemsize)

    local_keys = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    local_values = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    count = cl.LocalMemory(group_size * 2 * np.dtype('uint32').itemsize)
    local_histogram = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)

    for radix_pass in range(keys.dtype.itemsize * 8 // radix_bits):
        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.WRITE_INVALIDATE_REGION, 0,
            (ngroups, group_size * 2), keys.dtype, wait_for=[], is_blocking=True
        )
        keys_map[...] = keys
        del keys_map

        e = radix_kernels['block_sort'](
            cq, (ngroups,), (group_size,),
            keys_buf, local_keys, local_keys, None, local_values, local_values,
            histogram_buf, local_histogram, count,
            radix_bits, radix_pass, g_times_l=True,
        )

        keys = keys.reshape(ngroups, group_size * 2)
        order = np.argsort(radix_key(keys, radix_bits, radix_pass), kind='mergesort')
        grid = np.ogrid[tuple(slice(0, s) for s in keys.shape)]

        (histogram_map, _) = cl.enqueue_map_buffer(
            cq, histogram_buf, cl.map_flags.READ, 0,
            (histogram_len, ngroups), np.dtype('uint32'), wait_for=[e], is_blocking=True
        )
        i = 0
        for group_keys, histogram in zip(keys, histogram_map.T):
            group_keys = radix_key(group_keys, radix_bits, radix_pass).astype('uint16')
            expected = np.bincount(group_keys, minlength=16)
            try:
                np.testing.assert_equal(histogram, expected)
            except AssertionError:
                print((radix_pass, i))
                raise
            i += 1

        expected = keys[grid[:-1] + [order]]
        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.READ, 0,
            (ngroups, group_size * 2), keys.dtype, wait_for=[e], is_blocking=True
        )
        np.testing.assert_equal(keys_map, expected)


# group_size must be power of 2 (for scan to work)
@pytest.mark.parametrize("ngroups,group_size", sizes)
def test_scatter(cl_env, radix_kernels, key_dtype, ngroups, group_size):
    ctx, cq = cl_env

    radix_bits = 4
    histogram_len = 2 ** radix_bits
    keys = np.random.randint(0, 64, size=(ngroups, group_size * 2), dtype=key_dtype)
    keys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, keys.nbytes)
    out_keys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)
    histogram_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, histogram_len * ngroups * np.dtype('uint32').itemsize
    )
    offset_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, histogram_len * ngroups * np.dtype('uint32').itemsize
    )

    for radix_pass in range(keys.dtype.itemsize * 8 // radix_bits):
        radix_keys = radix_key(keys, radix_bits, radix_pass).astype('uint16')
        order = np.argsort(radix_keys, kind='mergesort')
        grid = np.ogrid[tuple(slice(0, s) for s in keys.shape)]
        block_keys = keys[grid[:-1] + [order]] # Partially sort

        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.WRITE_INVALIDATE_REGION, 0,
            keys.shape, keys.dtype, wait_for=[], is_blocking=True
        )
        keys_map[...] = block_keys
        del keys_map

        radix_keys = radix_key(block_keys, radix_bits, radix_pass).astype('uint16')

        (histogram_map, _) = cl.enqueue_map_buffer(
            cq, histogram_buf, cl.map_flags.WRITE_INVALIDATE_REGION, 0,
            (histogram_len, ngroups), np.dtype('uint32'), wait_for=[], is_blocking=True
        )
        (offset_map, _) = cl.enqueue_map_buffer(
            cq, offset_buf, cl.map_flags.WRITE_INVALIDATE_REGION, 0,
            (histogram_len, ngroups), np.dtype('uint32'), wait_for=[], is_blocking=True
        )
        histogram_map[...] = np.array([np.bincount(group_keys, minlength=16)
                                       for group_keys in radix_keys], dtype='uint32').T
        offset_map[...] = prefix_sum(histogram_map.flat).reshape(histogram_len, ngroups)
        del histogram_map, offset_map

        local_offset = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)
        local_histogram = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)

        e = radix_kernels['scatter'](
            cq, (ngroups,), (group_size,),
            keys_buf, out_keys_buf, None, None,
            offset_buf, local_offset, histogram_buf, local_histogram,
            radix_bits, radix_pass, g_times_l=True,
        )

        (keys_map, _) = cl.enqueue_map_buffer(
            cq, out_keys_buf, cl.map_flags.READ, 0,
            (ngroups, group_size * 2), keys.dtype, wait_for=[e], is_blocking=True
        )

        expected = block_keys.flat[np.argsort(radix_keys, axis=None, kind='mergesort')]
        np.testing.assert_equal(keys_map, expected.reshape(ngroups, 2 * group_size))


@pytest.mark.parametrize("ngroups,group_size", sizes)
def test_sort(cl_env, radix_kernels, key_dtype, ngroups, group_size):
    ctx, cq = cl_env

    radix_bits = 4
    histogram_len = 2 ** radix_bits

    keys = np.random.randint(0, 64, size=ngroups * group_size * 2, dtype=key_dtype)
    sort_keys = keys

    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )
    out_keys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)
    histogram_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                              ngroups * histogram_len * np.dtype('uint32').itemsize)
    offset_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                           ngroups * histogram_len * np.dtype('uint32').itemsize)

    local_keys = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    local_values = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    count = cl.LocalMemory(group_size * 2 * np.dtype('uint32').itemsize)
    local_offset = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)
    local_histogram = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)

    for radix_pass in range(keys.dtype.itemsize * 8 // radix_bits):
        e = radix_kernels['block_sort'](
            cq, (ngroups,), (group_size,),
            keys_buf, local_keys, local_keys, None, local_values, local_values,
            histogram_buf, local_histogram, count,
            radix_bits, radix_pass, g_times_l=True,
        )
        e = cl.enqueue_copy(
            cq, offset_buf, histogram_buf, wait_for=[e],
            byte_count=ngroups * histogram_len * np.dtype('uint32').itemsize,
        )
        (offset_map, _) = cl.enqueue_map_buffer(
            cq, offset_buf, cl.map_flags.READ | cl.map_flags.WRITE, 0,
            (histogram_len, ngroups),  np.dtype('uint32'),
            wait_for=[e], is_blocking=True,
        )
        offset_map[...] = prefix_sum(offset_map.flat).reshape(histogram_len, ngroups)
        del offset_map

        e = radix_kernels['scatter'](
            cq, (ngroups,), (group_size,),
            keys_buf, out_keys_buf, None, None,
            offset_buf, local_offset, histogram_buf, local_histogram,
            radix_bits, radix_pass, g_times_l=True,
        )
        e = cl.enqueue_copy(
            cq, keys_buf, out_keys_buf, byte_count=keys.nbytes, wait_for=[e]
        )

        radix_keys = radix_key(sort_keys, radix_bits, radix_pass)
        sort_keys = sort_keys[np.argsort(radix_keys, kind='mergesort')]
        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.READ, 0,
            len(keys), keys.dtype, wait_for=[e], is_blocking=True,
        )
        np.testing.assert_equal(keys_map, sort_keys)
        del keys_map

    (keys_map, _) = cl.enqueue_map_buffer(
        cq, keys_buf, cl.map_flags.READ, 0,
        len(keys), keys.dtype, wait_for=[e], is_blocking=True,
    )
    np.testing.assert_equal(keys_map, keys[np.argsort(keys, kind='mergesort')])
    del keys_map


@pytest.mark.parametrize("ngroups,group_size", sizes)
def test_argsort(cl_env, radix_kernels, key_dtype, value_dtype, ngroups, group_size):
    ctx, cq = cl_env

    radix_bits = 4
    histogram_len = 2 ** radix_bits

    sort_keys = keys = np.random.randint(0, 64, size=ngroups * group_size * 2, dtype=key_dtype)
    sort_values = values = np.random.uniform(
        -1000, 1000, size=ngroups * group_size * 2
    ).astype(value_dtype)

    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )
    out_keys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_values_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, values.nbytes)
    histogram_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                              ngroups * histogram_len * np.dtype('uint32').itemsize)
    offset_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                           ngroups * histogram_len * np.dtype('uint32').itemsize)

    local_keys = cl.LocalMemory(group_size * 2 * keys.dtype.itemsize)
    local_values = cl.LocalMemory(group_size * 2 * values.dtype.itemsize)
    count = cl.LocalMemory(group_size * 2 * np.dtype('uint32').itemsize)
    local_offset = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)
    local_histogram = cl.LocalMemory(histogram_len * np.dtype('uint32').itemsize)

    for radix_pass in range(keys.dtype.itemsize * 8 // radix_bits):
        e = radix_kernels['block_sort'](
            cq, (ngroups,), (group_size,),
            keys_buf, local_keys, local_keys, values_buf, local_values, local_values,
            histogram_buf, local_histogram, count,
            radix_bits, radix_pass, g_times_l=True,
        )
        e = cl.enqueue_copy(
            cq, offset_buf, histogram_buf, wait_for=[e],
            byte_count=ngroups * histogram_len * np.dtype('uint32').itemsize,
        )
        (offset_map, _) = cl.enqueue_map_buffer(
            cq, offset_buf, cl.map_flags.READ | cl.map_flags.WRITE, 0,
            (histogram_len, ngroups),  np.dtype('uint32'),
            wait_for=[e], is_blocking=True,
        )
        offset_map[...] = prefix_sum(offset_map.flat).reshape(histogram_len, ngroups)
        del offset_map

        e = radix_kernels['scatter'](
            cq, (ngroups,), (group_size,),
            keys_buf, out_keys_buf, values_buf, out_values_buf,
            offset_buf, local_offset, histogram_buf, local_histogram,
            radix_bits, radix_pass, g_times_l=True,
        )
        e = cl.enqueue_copy(
            cq, keys_buf, out_keys_buf, byte_count=keys.nbytes, wait_for=[e]
        )
        e = cl.enqueue_copy(
            cq, values_buf, out_values_buf, byte_count=values.nbytes, wait_for=[e]
        )

        radix_keys = radix_key(sort_keys, radix_bits, radix_pass)
        order = np.argsort(radix_keys, kind='mergesort')
        sort_keys = sort_keys[order]
        (keys_map, _) = cl.enqueue_map_buffer(
            cq, keys_buf, cl.map_flags.READ, 0,
            len(keys), keys.dtype, wait_for=[e], is_blocking=True,
        )
        np.testing.assert_equal(keys_map, sort_keys)
        del keys_map

        sort_values = sort_values[order]
        (values_map, _) = cl.enqueue_map_buffer(
            cq, values_buf, cl.map_flags.READ, 0,
            len(values), values.dtype, wait_for=[e], is_blocking=True,
        )
        np.testing.assert_equal(values_map, sort_values)
        del values_map

    order = np.argsort(keys, kind='mergesort')

    (keys_map, _) = cl.enqueue_map_buffer(
        cq, keys_buf, cl.map_flags.READ, 0,
        len(keys), keys.dtype, wait_for=[e], is_blocking=True,
    )
    np.testing.assert_equal(keys_map, keys[order])
    del keys_map

    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ, 0,
        len(values), values.dtype, wait_for=[e], is_blocking=True,
    )
    np.testing.assert_equal(values_map, values[order])
    del values_map
