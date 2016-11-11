import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian
from radix import *

np.random.seed(4)

@pytest.fixture(scope='module')
def cl_env():
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    return ctx, cq

@pytest.fixture(scope='module')
def cl_scan(cl_env):
    ctx, cq = cl_env
    program = PrefixScanProgram(ctx)
    return ctx, cq, program

@pytest.fixture(scope='module')
def cl_sort(cl_scan):
    ctx, cq, scan_program = cl_scan
    radix_program = RadixProgram(ctx)
    return ctx, cq, radix_program, scan_program

@pytest.mark.parametrize("size,group_size", [(1023, 4), (20, 4), (96, 6)])
def test_scanner_errs(cl_scan, size, group_size):
    ctx, cq, program = cl_scan
    with pytest.raises(ValueError):
        PrefixScanner(ctx, size, group_size, program=program)

@pytest.mark.parametrize("old_shape,new_shape", [
    ((1024, 4), (1023, 4)),
])
def test_scanner_resize_errs(cl_scan, old_shape, new_shape):
    ctx, cq, program = cl_scan
    scanner = PrefixScanner(ctx, *old_shape, program=program)
    with pytest.raises(ValueError):
        scanner.resize(*new_shape)

@pytest.mark.parametrize("size,group_size,expected", [
    (1024, 4, (128, 16, 2)),
    (20, 2, (8, 2)),
    (24, 4, (8,)),
    (1032, 4, (136, 24, 4)),
    (160, 4, (24, 4)),
    (320, 4, (40, 8)),
])
def test_block_levels(cl_scan, size, group_size, expected):
    ctx, cq, program = cl_scan
    scanner = PrefixScanner(ctx, size, group_size, program=program)
    assert scanner.block_lengths == expected

@pytest.mark.parametrize("size,group_size", [(20, 2), (24, 4), (1024, 4), (160, 4), (320, 4)])
def test_prefix_sum(cl_scan, size, group_size):
    ctx, cq, program = cl_scan
    scanner = PrefixScanner(ctx, size, group_size, program=program)

    values = np.random.randint(0, size, size=size, dtype='uint32')
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    calc_scan = scanner.prefix_sum(cq, values_buf)

    expected = np.cumsum(values)
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[calc_scan], is_blocking=True
    )
    assert values_map[0] == 0
    np.testing.assert_equal(values_map[1:], expected[:-1])

@pytest.mark.parametrize("old_shape,new_shape", [
    ((20, 2), (24, 4)),
    ((1024, 4), (160, 4)),
    ((24, 2), (None, 4)),
    ((160, 4), (1024, None)),
])
def test_scanner_resized(cl_scan, old_shape, new_shape):
    ctx, cq, program = cl_scan
    scanner = PrefixScanner(ctx, *old_shape, program=program)
    scanner.resize(*new_shape)

    size = new_shape[0] or old_shape[0]
    values = np.random.randint(0, 100, size=size, dtype='uint32')
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    calc_scan = scanner.prefix_sum(cq, values_buf)

    expected = np.cumsum(values)
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[calc_scan], is_blocking=True
    )
    assert values_map[0] == 0
    np.testing.assert_equal(values_map[1:], expected[:-1])

@pytest.mark.parametrize("size,ngroups,group_size,bits", [(24, 3, 4, 3), (24, 4, 4, 4),])
def test_sorter_errs(cl_sort, size, ngroups, group_size, bits):
    ctx, cq, program, scan_program = cl_sort
    with pytest.raises(ValueError):
        sorter = RadixSorter(ctx, size, ngroups, group_size, bits, program, scan_program)

@pytest.mark.parametrize("old_shape,new_shape", [((24, 3, 4, 4), (24, 4, 4, 4))])
def test_sorter_resize_errs(cl_sort, old_shape, new_shape):
    ctx, cq, program, scan_program = cl_sort
    sorter = RadixSorter(ctx, *old_shape, program=program, scan_program=scan_program)
    with pytest.raises(ValueError):
        sorter.resize(*new_shape)

@pytest.mark.parametrize("bits,expected", [(1, 32), (2, 16), (4, 8), (8, 4)])
def test_num_passes(cl_sort, bits, expected):
    ctx, cq, program, scan_program = cl_sort
    sorter = RadixSorter(ctx, 24, 3, 4, bits, program, scan_program)
    assert sorter.num_passes == expected

@pytest.mark.parametrize("size,ngroups,group_size", [(16000,10,32), (20,5,4)])
def test_sorter(cl_sort, size, ngroups, group_size):
    ctx, cq, program, scan_program = cl_sort
    sorter = RadixSorter(ctx, size, ngroups, group_size, program=program, scan_program=scan_program)
    data = np.random.randint(500, size=size, dtype='uint32')
    data_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY , data.nbytes
    )

    calc_sort = sorter.sort(cq, data_buf, out_buf)

    (out_map, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, data.shape, data.dtype,
        wait_for=[calc_sort], is_blocking=True
    )
    np.testing.assert_equal(out_map, np.sort(data))

@pytest.mark.parametrize("old_shape,new_shape", [
    ((16000,10,32), (20,5,4)),
    ((20,5,4), (16000,10,32)),
])
def test_sorter_resized(cl_sort, old_shape, new_shape):
    ctx, cq, program, scan_program = cl_sort
    sorter = RadixSorter(ctx, *old_shape, program=program, scan_program=scan_program)
    sorter.resize(*new_shape)

    size = new_shape[0] or old_shape[0]
    data = np.random.randint(500, size=size, dtype='uint32')
    data_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data
    )
    out_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY , data.nbytes
    )

    calc_sort = sorter.sort(cq, data_buf, out_buf)

    (out_map, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, data.shape, data.dtype,
        wait_for=[calc_sort], is_blocking=True
    )
    np.testing.assert_equal(out_map, np.sort(data))

def test_arg_sorter(cl_sort):
    ctx, cq, program, scan_program = cl_sort
    group_size = 32
    ngroups = 10
    size = group_size * ngroups * 50
    sorter = RadixSorter(ctx, size, ngroups, group_size, program=program, scan_program=scan_program)
    keys = np.random.randint(500, size=size, dtype='uint32')
    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )

    values = np.random.randint(500, size=size, dtype='uint32')
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )

    out_keys_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY , keys.nbytes
    )
    out_values_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY , values.nbytes
    )

    calc_sort = sorter.sort(cq, keys_buf, out_keys_buf, values_buf, out_values_buf)

    (out_keys_map, _) = cl.enqueue_map_buffer(
        cq, out_keys_buf, cl.map_flags.READ,
        0, keys.shape, keys.dtype,
        wait_for=[calc_sort], is_blocking=True
    )
    np.testing.assert_equal(out_keys_map, np.sort(keys))

    (out_values_map, _) = cl.enqueue_map_buffer(
        cq, out_values_buf, cl.map_flags.READ,
        0, keys.shape, keys.dtype,
        wait_for=[calc_sort], is_blocking=True
    )
    np.testing.assert_equal(out_values_map, values[np.argsort(keys, kind='mergesort')])

def test_auto_program(cl_env):
    ctx, cq = cl_env
    group_size = 32
    ngroups = 10
    size = group_size * ngroups * 50
    sorter = RadixSorter(ctx, size, ngroups, group_size)
