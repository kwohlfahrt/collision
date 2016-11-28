import numpy as np
import pyopencl as cl
import pytest
from collision.radix import *
from .common import cl_env

np.random.seed(4)

def pytest_generate_tests(metafunc):
    if 'value_dtype' in metafunc.fixturenames:
        metafunc.parametrize("value_dtype", ['uint32', 'uint64'], scope='module')


@pytest.fixture(scope='module')
def scan_program(cl_env):
    ctx, cq = cl_env
    return PrefixScanProgram(ctx)


@pytest.mark.parametrize("size,group_size", [(1023, 4), (20, 4), (96, 6)])
def test_scanner_errs(cl_env, scan_program, size, group_size):
    ctx, cq = cl_env
    with pytest.raises(ValueError):
        PrefixScanner(ctx, size, group_size, program=scan_program)


@pytest.mark.parametrize("old_shape,new_shape", [
    ((1024, 4), (1023, 4)),
])
def test_scanner_resize_errs(cl_env, scan_program, old_shape, new_shape):
    ctx, cq = cl_env
    scanner = PrefixScanner(ctx, *old_shape, program=scan_program)
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
def test_block_levels(cl_env, scan_program, size, group_size, expected):
    ctx, cq = cl_env
    scanner = PrefixScanner(ctx, size, group_size, program=scan_program)
    assert scanner.block_lengths == expected


@pytest.mark.parametrize("size,group_size", [(20, 2), (24, 4), (1024, 4), (160, 4), (320, 4)])
def test_prefix_sum(cl_env, scan_program, size, group_size):
    ctx, cq = cl_env
    scanner = PrefixScanner(ctx, size, group_size, program=scan_program)

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
def test_scanner_resized(cl_env, scan_program, old_shape, new_shape):
    ctx, cq = cl_env
    scanner = PrefixScanner(ctx, *old_shape, program=scan_program)
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
