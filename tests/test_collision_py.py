import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian
from collision import *

@pytest.fixture(scope='module')
def cl_env():
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    return ctx, cq

@pytest.fixture(scope='module')
def cl_collision(cl_env):
    ctx, cq = cl_env
    program = CollisionProgram(ctx)
    return ctx, cq, program

def find_collisions(coords, radii):
    min_bounds = coords - radii.reshape(-1, 1)
    max_bounds = coords + radii.reshape(-1, 1)
    collisions = ((max_bounds.reshape(-1, 1, 3) > min_bounds.reshape(1, -1, 3)) &
                  (min_bounds.reshape(-1, 1, 3) < max_bounds.reshape(1, -1, 3)))
    collisions = collisions.all(axis=-1)
    collisions = np.triu(collisions, 1)
    return set(zip(*reversed(np.nonzero(collisions))))

def test_collision(cl_collision):
    ctx, cq, program = cl_collision

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype='float32')
    radii = np.ones(len(coords), dtype='float32')
    expected = {(0, 1), (4, 5)}

    collider = Collider(program, len(coords))

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    range_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array([coords.min(axis=0), coords.max(axis=0)], dtype=coords.dtype)
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf, range_buf,
                                collisions_buf, len(expected))
    assert n == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n, 2), collider.id_dtype,
        is_blocking=True
    )
    assert set(map(tuple, collisions_map)) == expected

@pytest.mark.parametrize("size", [5, 20, 100, 256])
def test_random_collision(cl_collision, size):
    ctx, cq, program = cl_collision

    collider = Collider(program, size)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(collider.coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(collider.coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    range_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array([coords.min(axis=0), coords.max(axis=0)], dtype=coords.dtype)
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf, range_buf,
                                collisions_buf, len(expected))
    assert n == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n, 2), collider.id_dtype,
        is_blocking=True
    )
    collisions = set(map(tuple, collisions_map))

    # Need to test both directions, order is not clear from tree
    assert (collisions | set(map(tuple, map(reversed, collisions))) ==
            expected | set(map(tuple, map(reversed, expected))))
