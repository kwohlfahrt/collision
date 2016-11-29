import numpy as np
import pyopencl as cl
import pytest
from functools import partial
from collision.radix import RadixProgram, RadixSorter
from ..common import cl_env
from .test_scan import scan_program

def pytest_generate_tests(metafunc):
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32', 'uint64'], scope='module')


@pytest.fixture(scope='module')
def radix_program(cl_env, value_dtype):
    ctx, cq = cl_env
    return RadixProgram(ctx, value_dtype)


def sort_keys(cq, sorter, *args):
    cl.wait_for_events([sorter.sort(cq, *args)])


@pytest.mark.parametrize("size,gen,ngroups,group_size", [
    (307200, partial(np.random.randint, 0, 1000), 16, 128),
    (307200, partial(np.random.randint, 0, 307200), 16, 128),
    (307200, np.arange, 16, 128),
])
def test_sort_keys(cl_env, radix_program, scan_program, value_dtype,
                   size, gen, ngroups, group_size, benchmark):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, size, ngroups, group_size, value_dtype=value_dtype,
                         program=radix_program, scan_program=scan_program)

    keys = gen(size, dtype=value_dtype)
    expected = np.sort(keys)

    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )
    out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)

    benchmark(sort_keys, cq, sorter, keys_buf, out_buf)

    (out_map, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, keys.shape, keys.dtype,
        wait_for=[], is_blocking=True
    )
    np.testing.assert_equal(out_map, expected)


@pytest.mark.parametrize("size,gen,ngroups,group_size", [
    (307200, partial(np.random.randint, 0, 1000), 16, 128),
    (307200, partial(np.random.randint, 0, 307200), 16, 128),
    (307200, np.arange, 16, 128),
])
def test_sort_values(cl_env, radix_program, scan_program, value_dtype,
                     size, gen, ngroups, group_size, benchmark):
    ctx, cq = cl_env
    sorter = RadixSorter(ctx, size, ngroups, group_size, value_dtype=value_dtype,
                         program=radix_program, scan_program=scan_program)

    keys = gen(size, dtype=value_dtype)
    values = np.arange(size, dtype=value_dtype)
    expected_keys = np.sort(keys)
    expected_values = values[np.argsort(keys, kind='mergesort')]

    keys_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys
    )
    out_keys_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)

    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    out_values_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, values.nbytes)

    benchmark(sort_keys, cq, sorter, keys_buf, out_keys_buf, values_buf, out_values_buf)

    for out_buf, expected in [(out_keys_buf, expected_keys),
                              (out_values_buf, expected_values)]:
        (out_map, _) = cl.enqueue_map_buffer(
            cq, out_buf, cl.map_flags.READ,
            0, expected.shape, expected.dtype,
            wait_for=[], is_blocking=True
        )
        np.testing.assert_equal(out_map, expected)
