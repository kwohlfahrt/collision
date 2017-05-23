import numpy as np
import pyopencl as cl
import pytest
from collision.offset import OffsetProgram, OffsetFinder
from ..common import cl_env

@pytest.fixture(scope='module')
def scan_program(cl_env):
    ctx, cq = cl_env
    return PrefixScanProgram(ctx)


@pytest.fixture(scope='module')
def offset_program(cl_env):
    ctx, cq = cl_env
    return OffsetProgram(ctx)


def find_offsets(cq, finder, *args):
    cl.wait_for_events([finder.find_offsets(cq, *args)])


# Use size large enough that t > 100*μs
@pytest.mark.parametrize("size,maxval,rounds", [
    (2**21, 2*10**3, 800),
    (2**21, 2*10**6, 800),
])
def test_find_offsets(cl_env, offset_program, size, maxval, rounds, benchmark):
    ctx, cq = cl_env
    finder = OffsetFinder(ctx, program=offset_program)

    values = np.sort(np.random.randint(0, maxval, size, dtype='uint32'))
    values_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=values
    )
    output_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, maxval * np.dtype('uint32').itemsize
    )
    benchmark(find_offsets, cq, finder, values_buf, len(values), output_buf, maxval)
