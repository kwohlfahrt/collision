import numpy as np
import pyopencl as cl
import pytest
from collision.reduce import ReductionProgram, Reducer
from ..common import cl_env

@pytest.fixture(scope='module')
def reduce_program(cl_env):
    ctx, cq = cl_env
    return ReductionProgram(ctx)


def reduce(cq, reducer, *args):
    cl.wait_for_events([reducer.reduce(cq, *args)])


# Use size large enough that t > 100*μs
@pytest.mark.parametrize("size, ngroups,group_size,rounds", [
    (1536000, 64, 128, 800),
    (3072000, 64, 128, 400),
])
def test_reducer(cl_env, reduce_program, size, ngroups, group_size, rounds, benchmark):
    ctx, cq = cl_env
    reducer = Reducer(ctx, ngroups, group_size, program=reduce_program)

    values = np.random.uniform(0.0, 1.0, size=(size, 3)).astype('float32')
    expected = np.array([np.min(values, axis=0), np.max(values, axis=0)])

    values_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=values)
    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, expected.nbytes)

    calc_scan = benchmark.pedantic(reduce, (cq, reducer, size, values_buf, output_buf),
                                   rounds=rounds, warmup_rounds=10)

    (output_map, _) = cl.enqueue_map_buffer(
        cq, output_buf, cl.map_flags.READ,
        0, expected.shape, expected.dtype,
        wait_for=[], is_blocking=True
    )
    np.testing.assert_equal(output_map, expected)
