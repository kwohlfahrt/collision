import numpy as np
import pyopencl as cl
import pytest
from inspect import signature
from functools import partial
from collision.radix import RadixProgram, RadixSorter
from ..common import cl_env
from .test_scan import scan_program

def pytest_generate_tests(metafunc):
    params = signature(metafunc.function).parameters
    if 'key_dtype' in params:
        metafunc.parametrize("key_dtype", ['uint32', 'uint64'], scope='module')
    elif 'key_dtype' in metafunc.fixturenames:
        metafunc.parametrize("key_dtype", ['uint32'], scope='module')
    if 'value_dtype' in params:
        metafunc.parametrize("value_dtype", ['uint32', 'float64'], scope='module')
    elif 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32'], scope='module')


@pytest.fixture(scope='module')
def radix_program(cl_env, key_dtype, value_dtype):
    ctx, cq = cl_env
    return RadixProgram(ctx, key_dtype, value_dtype)


def radix_sort_setup(cq, bufs, values):
    for buf, value in zip(bufs, values):
        (buf_map, _) = cl.enqueue_map_buffer(
            cq, buf, cl.map_flags.WRITE_INVALIDATE_REGION,
            0, value.shape, value.dtype,
            wait_for=[], is_blocking=True
        )
        buf_map[...] = value
        del buf_map


def radix_sort(cq, sorter, *args):
    cl.wait_for_events([sorter.sort(cq, *args)])


@pytest.mark.parametrize("size,gen,group_size,rounds", [
    (307200, partial(np.random.randint, 0, 1000), 128, 100),
    (307200, partial(np.random.randint, 0, 307200), 128, 100),
    (307200, np.arange, 128, 100),
])
def test_radix_sort(cl_env, radix_program, scan_program, key_dtype,
                    size, gen, group_size, rounds, benchmark):
    ctx, cq = cl_env
    sorter = RadixSorter(
        ctx, size, group_size, key_dtype=key_dtype,
        program=radix_program, scan_program=scan_program
    )

    keys = gen(size, dtype=key_dtype)
    expected = np.sort(keys)

    keys_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, keys.nbytes)
    out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, keys.nbytes)

    if key_dtype == np.dtype('uint64'):
        rounds //= 2
    benchmark.pedantic(radix_sort, (cq, sorter, keys_buf, out_buf),
                       setup=partial(radix_sort_setup, cq, [keys_buf], [keys]),
                       rounds=rounds, warmup_rounds=10)

    (out_map, _) = cl.enqueue_map_buffer(
        cq, out_buf, cl.map_flags.READ,
        0, keys.shape, keys.dtype,
        wait_for=[], is_blocking=True
    )
    np.testing.assert_equal(out_map, expected)


@pytest.mark.parametrize("size,gen,group_size,rounds", [
    (307200, partial(np.random.randint, 0, 1000), 128, 100),
    (307200, partial(np.random.randint, 0, 307200), 128, 100),
    (307200, np.arange, 128, 100),
])
def test_sort_values(cl_env, radix_program, scan_program, key_dtype, value_dtype,
                     size, gen, group_size, rounds, benchmark):
    ctx, cq = cl_env
    sorter = RadixSorter(
        ctx, size, group_size, key_dtype=key_dtype, value_dtype=value_dtype,
        program=radix_program, scan_program=scan_program
    )

    keys = gen(size, dtype=key_dtype)
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

    if key_dtype == np.dtype('uint64'):
        rounds //= 2
    benchmark.pedantic(radix_sort, (cq, sorter, keys_buf, out_keys_buf, values_buf, out_values_buf),
                       setup=partial(radix_sort_setup, cq, [keys_buf, values_buf], [keys, values]),
                       rounds=rounds, warmup_rounds=10)

    for out_buf, expected in [(out_keys_buf, expected_keys),
                              (out_values_buf, expected_values)]:
        (out_map, _) = cl.enqueue_map_buffer(
            cq, out_buf, cl.map_flags.READ,
            0, expected.shape, expected.dtype,
            wait_for=[], is_blocking=True
        )
        np.testing.assert_equal(out_map, expected)
