import numpy as np
import pyopencl as cl
import pytest
from functools import partial
from collision.radix import PrefixScanProgram, PrefixScanner
from ..common import cl_env

@pytest.fixture(scope='module')
def scan_program(cl_env):
    ctx, cq = cl_env
    return PrefixScanProgram(ctx)

def prefix_sum_setup(cq, values_buf, values):
    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, values.shape, values.dtype,
        wait_for=[], is_blocking=True
    )
    values_map[...] = values
    del values_map

def prefix_sum(cq, scanner, values_buf):
    cl.wait_for_events([scanner.prefix_sum(cq, values_buf)])

# Use size large enough that t > 100*μs
@pytest.mark.parametrize("size,group_size,rounds", [
    (307200, 128, 4000),
    (1536000, 128, 800),
    (3072000, 128, 400),
])
def test_scanner(cl_env, scan_program, size, group_size, rounds, benchmark):
    ctx, cq = cl_env
    scanner = PrefixScanner(ctx, size, group_size, program=scan_program)

    values = np.random.randint(0, 128, size=size, dtype='uint32')
    expected = np.cumsum(values)

    values_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, values.nbytes)

    calc_scan = benchmark.pedantic(prefix_sum, (cq, scanner, values_buf),
                                   setup=partial(prefix_sum_setup, cq, values_buf, values),
                                   rounds=rounds, warmup_rounds=10)

    (values_map, _) = cl.enqueue_map_buffer(
        cq, values_buf, cl.map_flags.READ,
        0, values.shape, values.dtype,
        wait_for=[], is_blocking=True
    )
    assert values_map[0] == 0
    np.testing.assert_equal(values_map[1:], expected[:-1])
