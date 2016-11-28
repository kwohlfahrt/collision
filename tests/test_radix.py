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
    kernel_args = {'histogram': [None, None, np.dtype('int32'),
                                 np.dtype('int8'), np.dtype('int8')],
                   'scatter': [None, None, None, None, np.dtype('int32'),
                               None, np.dtype('int8'), np.dtype('int8')]}
    c_dtypes = {'uint32': 'int', 'uint64': 'long'}
    ctx, cq = cl_env

    buildopts = ["-D DTYPE={}".format(c_dtypes[value_dtype])]

    src = Path(__file__).parent / ".." / "collision" / "radix.cl"
    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build(buildopts)
    kernels = {name: getattr(program, name) for name in kernel_args}
    for name, kernel in kernels.items():
        kernel.set_scalar_arg_dtypes(kernel_args[name])
    return kernels


def radix_key(values, radix_bits, radix_pass):
    return (values >> (radix_pass * radix_bits)) & ((2 ** radix_bits) - 1)


def histogram(values, radix_bits, radix_pass):
    values_radix = radix_key(values, radix_bits, radix_pass)
    radixes = np.arange(2 ** radix_bits)
    return (values_radix.reshape(-1, 1) == radixes.reshape(1, -1)).sum(axis=0)


def test_histogram(cl_env, radix_kernels, value_dtype):
    ctx, cq = cl_env

    radix_bits = 4
    ngroups = 2
    group_items = 4

    values = np.array([ 69,  130, 206, 111,  38,   6, 100, 230,
                        143, 100, 161,  95, 165, 113, 169, 222,
                        116,  24, 213, 111,  76, 155, 128, 116,
                        109, 127, 160, 113, 199, 221, 236,  50], dtype=value_dtype)
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    histogram_shape = (2 ** radix_bits), ngroups, group_items
    histogram_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE,
        (2 ** radix_bits) * ngroups * group_items * np.dtype('uint32').itemsize
    )

    item_size = len(values) // ngroups // group_items
    for radix_pass in [0, 1, 2]:
        clear_histogram = cl.enqueue_fill_buffer(
            cq, histogram_buf, np.zeros(1, dtype='uint32'),
            0, (2 ** radix_bits) * ngroups * group_items * np.dtype('uint32').itemsize
        )
        calc_hist = radix_kernels['histogram'](
            cq, (ngroups,), (group_items,),
            histogram_buf, values_buf, len(values), radix_pass, radix_bits,
            g_times_l=True, wait_for=[clear_histogram]
        )

        (histogram_map, _) = cl.enqueue_map_buffer(
            cq, histogram_buf, cl.map_flags.READ,
            0, ((2 ** radix_bits), ngroups, group_items), np.dtype('uint32'),
            wait_for=[calc_hist], is_blocking=True,
        )

        for i, (group, item) in enumerate(cartesian(range(ngroups), range(group_items))):
            values_section = values[item_size * i:item_size * (i + 1)]
            expected = histogram(values_section, radix_bits, radix_pass)

            np.testing.assert_equal(histogram_map[:, group, item], expected)
        del histogram_map


def test_scatter(cl_env, radix_kernels, value_dtype):
    ctx, cq = cl_env

    radix_bits = 4
    radix_pass = 0

    in_keys = np.array([1, 4, 6, 2, 8, 8, 3, 1], dtype=value_dtype)

    buckets = np.arange(2 ** radix_bits)
    histogram = (in_keys.reshape(-1, 1) < buckets.reshape(1, -1)).sum(axis=0, dtype='uint32')

    in_keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=in_keys
    )
    histogram_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=histogram
    )
    out_keys_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, in_keys.nbytes
    )

    calc_scatter = radix_kernels['scatter'](
        cq, (1,), (1,),
        in_keys_buf, out_keys_buf, None, None, len(in_keys),
        histogram_buf, radix_pass, radix_bits,
        g_times_l=True
    )

    (out_keys_map, _) = cl.enqueue_map_buffer(
        cq, out_keys_buf, cl.map_flags.READ,
        0, in_keys.shape, in_keys.dtype,
        wait_for=[calc_scatter], is_blocking=True,
    )

    expected = np.sort(in_keys)
    np.testing.assert_equal(expected, out_keys_map)


def test_sort_pass(cl_env, radix_kernels, scan_kernels, value_dtype):
    ctx, cq = cl_env

    radix_bits = 4
    radix_pass = 0
    ngroups = 2
    group_items = 4
    scan_block_len = 16

    values = np.array([ 69,  130, 206, 111,  38,   6, 100, 230,
                        143, 100, 161,  95, 165, 113, 169, 222,
                        116,  24, 213, 111,  76, 155, 128, 116,
                        109, 127, 160, 113, 199, 221, 236,  50], dtype=value_dtype)
    # Filled at beginning of loop
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, values.nbytes
    )
    out_values_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )

    histogram_len = (2 ** radix_bits) * ngroups * group_items
    histogram_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE,
        histogram_len * np.dtype('uint32').itemsize
    )
    block_sums_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE,
        (histogram_len // 2 // scan_block_len) * values.dtype.itemsize
    )

    fill_values = cl.enqueue_copy(
        cq, values_buf, out_values_buf, byte_count=values.nbytes
    )
    for radix_pass in range(0,2):
        clear_histogram = cl.enqueue_fill_buffer(
            cq, histogram_buf, np.zeros(1, dtype='uint32'),
            0, (2 ** radix_bits) * ngroups * group_items * np.dtype('uint32').itemsize
        )
        calc_hist = radix_kernels['histogram'](
            cq, (ngroups,), (group_items,),
            histogram_buf, values_buf, len(values), radix_pass, radix_bits,
            g_times_l=True, wait_for=[fill_values, clear_histogram]
        )
        calc_scan = scan_kernels['local_scan'](
            cq, (histogram_len // 2,), (scan_block_len,),
            histogram_buf, block_sums_buf,
            wait_for=[calc_hist]
        )
        calc_block_scan = scan_kernels['local_scan'](
            cq, (1,), (histogram_len // 2 // scan_block_len,),
            block_sums_buf, None,
            g_times_l=True, wait_for=[calc_scan]
        )
        calc_scan = scan_kernels['block_scan'](
            cq, (histogram_len // 2,), (scan_block_len,),
            histogram_buf, block_sums_buf,
            wait_for=[calc_block_scan]
        )
        calc_scatter = radix_kernels['scatter'](
            cq, (ngroups,), (group_items,),
            values_buf, out_values_buf, None, None, len(values),
            histogram_buf, radix_pass, radix_bits,
            g_times_l=True, wait_for=[calc_scan]
        )
        fill_values = cl.enqueue_copy(
            cq, values_buf, out_values_buf, byte_count=values.nbytes,
            wait_for=[calc_scatter]
        )

        (values_map, _) = cl.enqueue_map_buffer(
            cq, out_values_buf, cl.map_flags.READ,
            0, values.shape, values.dtype,
            wait_for=[calc_scatter], is_blocking=True,
        )

        values_radix = radix_key(values, radix_bits, radix_pass)
        values = values[np.argsort(values_radix, kind='mergesort')]
        np.testing.assert_equal(values_map, values)
        del values_map

    (values_map, _) = cl.enqueue_map_buffer(
        cq, out_values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[calc_scatter], is_blocking=True,
    )
    np.testing.assert_equal(values_map, values)
    del values_map
