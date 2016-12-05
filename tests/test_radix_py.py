import numpy as np
import pyopencl as cl
import pytest
from collision.radix import *
from .common import cl_env
from .test_scan_py import scan_program

np.random.seed(4)

def pytest_generate_tests(metafunc):
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32', 'uint64'], scope='module')


@pytest.fixture(scope='module')
def sort_program(cl_env, scan_program, value_dtype):
    ctx, cq = cl_env
    return RadixProgram(ctx, value_dtype)


@pytest.mark.parametrize("size,group_size,bits", [
    (128, 8, 3), (128, 9, 4), (122, 8, 4), (128, 4, 4)
])
def test_sorter_errs(cl_env, sort_program, scan_program, value_dtype,
                     size, group_size, bits):
    ctx, cq = cl_env
    with pytest.raises(ValueError):
        sorter = RadixSorter(ctx, size, group_size, bits,
                            value_dtype, sort_program, scan_program)


def test_dtype_errs(cl_env, scan_program, sort_program, value_dtype):
    ctx, cq = cl_env
    sorter_dtype = {'uint32': 'uint64', 'uint64': 'uint32'}[value_dtype]
    with pytest.raises(ValueError):
        sorter = RadixSorter(ctx, 128, 8, 4, value_dtype=sorter_dtype,
                             program=sort_program, scan_program=scan_program)


@pytest.mark.parametrize("old_shape,new_shape", [((64, 8, 4), (64, 5, 4))])
def test_sorter_resize_errs(cl_env, sort_program, scan_program, value_dtype, old_shape, new_shape):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, *old_shape, value_dtype=value_dtype,
                         program=sort_program, scan_program=scan_program)
    with pytest.raises(ValueError):
        sorter.resize(*new_shape)


@pytest.mark.parametrize("bits,group_size,expected", [
    (1, 4, 32), (2, 4, 16), (4, 8, 8), (8, 128, 4)
])
def test_num_passes(cl_env, sort_program, scan_program, value_dtype,
                    bits, group_size, expected):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, 512, group_size, bits, value_dtype, sort_program, scan_program)
    if value_dtype == np.dtype('uint64'):
        expected *= 2
    assert sorter.num_passes == expected


@pytest.mark.parametrize("size,group_size", [(32, 8), (15360,32)])
def test_sorter(cl_env, sort_program, scan_program, value_dtype, size, group_size):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, size, group_size, value_dtype=value_dtype,
                         program=sort_program, scan_program=scan_program)
    data = np.random.randint(500, size=size, dtype=value_dtype)
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
    ((15360,32), (32,8)), ((32,8), (15360,32)),
])
def test_sorter_resized(cl_env, sort_program, scan_program, value_dtype, old_shape, new_shape):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, *old_shape, value_dtype=value_dtype,
                         program=sort_program, scan_program=scan_program)
    sorter.resize(*new_shape)

    size = new_shape[0] or old_shape[0]
    data = np.random.randint(500, size=size, dtype=value_dtype)
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


@pytest.mark.parametrize("size,group_size", [(32, 8), (15360,32)])
def test_arg_sorter(cl_env, sort_program, scan_program, value_dtype, size, group_size):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, size, group_size, value_dtype=value_dtype,
                         program=sort_program, scan_program=scan_program)
    keys = np.random.randint(500, size=size, dtype=value_dtype)
    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )

    values = np.random.randint(500, size=size, dtype=value_dtype)
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
    size = group_size * 64
    sorter = RadixSorter(ctx, size, group_size)
