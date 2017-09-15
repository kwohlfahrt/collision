import numpy as np
import pyopencl as cl
import pytest
from collision.collision import CollisionProgram, Collider
from ..common import cl_env

@pytest.fixture(scope='module')
def collision_programs(cl_env):
    from collision.radix import RadixProgram, PrefixScanProgram
    from collision.bounds import BoundsProgram

    ctx, cq = cl_env
    program = CollisionProgram(ctx)
    radix_program = RadixProgram(ctx)
    scan_program = PrefixScanProgram(ctx)
    reducer_program = BoundsProgram(ctx)
    return program, (radix_program, scan_program), reducer_program

def collide(cq, collider, *args):
    cl.wait_for_events([collider.get_collisions(cq, *args)])


# Use size large enough that t > 100*μs
@pytest.mark.parametrize("npoints,rmax,ngroups,group_size,rounds", [
    (307200, 0.06, 8, 128, 10),
    (307201, 0.06, 8, 128, 10), # Uneven npoints
])
def test_collide(cl_env, collision_programs, npoints, rmax,
                 ngroups, group_size, rounds, benchmark):
    ctx, cq = cl_env

    coords = np.random.uniform(-1.0, 1.0, (npoints, 3)).astype(dtype='float32')
    radii = np.random.uniform(0.1*rmax, rmax, (len(coords), 1)).astype(coords.dtype)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coords.dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coords.dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=radii)
    n_collisions_buf = cl.Buffer(ctx, cl.mem_flags.HOST_READ_ONLY | cl.mem_flags.READ_WRITE,
                                 np.dtype('int32').itemsize)


    collider = Collider(ctx, len(coords), ngroups, group_size, coord_dtype=coords.dtype)
    benchmark.pedantic(collide, (cq, collider, coords_buf, radii_buf,
                                 n_collisions_buf, None, 0),
                       rounds=rounds, warmup_rounds=10)

    # No expected, as full set is too large
