import numpy as np
import pyopencl as cl
import pytest
from radix import RadixProgram, PrefixScanProgram
from reduce import ReductionProgram
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
    radix_program = RadixProgram(ctx)
    scan_program = PrefixScanProgram(ctx)
    reducer_program = ReductionProgram(ctx)
    return ctx, cq, program, (radix_program, scan_program), reducer_program

def find_collisions(coords, radii):
    min_bounds = coords - radii.reshape(-1, 1)
    max_bounds = coords + radii.reshape(-1, 1)
    collisions = ((max_bounds.reshape(-1, 1, 3) > min_bounds.reshape(1, -1, 3)) &
                  (min_bounds.reshape(-1, 1, 3) < max_bounds.reshape(1, -1, 3)))
    collisions = collisions.all(axis=-1)
    collisions = np.triu(collisions, 1)
    return set(zip(*reversed(np.nonzero(collisions))))

@pytest.mark.parametrize("size,sorter_shape,expected", [
    (24, (3,8), 24), (23, (3,8), 24), (25, (3,8), 48)
])
def test_padded_size(cl_collision, size, sorter_shape, expected):
    ctx, cq, program, sorter_programs, reducer_program = cl_collision

    collider = Collider(ctx, size, sorter_shape, program,
                        sorter_programs, reducer_program)

    assert collider.padded_size == expected

def test_collision(cl_collision):
    ctx, cq, program, sorter_programs, reducer_program = cl_collision

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype='float32')
    radii = np.ones(len(coords), dtype='float32')
    expected = {(0, 1), (4, 5)}

    collider = Collider(ctx, len(coords), (3, 2), program,
                        sorter_programs, reducer_program)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf, collisions_buf, len(expected))
    assert n == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n, 2), collider.id_dtype,
        is_blocking=True
    )
    assert set(map(tuple, collisions_map)) == expected

@pytest.mark.parametrize("size,sorter_shape", [(5,(5,1)), (20,(5,4)),
                                               (100,(5,4)), (256,(4,32)),
                                               (317, (4, 16))])
def test_random_collision(cl_collision, size, sorter_shape):
    ctx, cq, program, sorter_programs, reducer_program = cl_collision
    collider = Collider(ctx, size, sorter_shape, program,
                        sorter_programs, reducer_program)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(collider.coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(collider.coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf, collisions_buf, len(expected))
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

@pytest.mark.parametrize("old_shape,new_shape", [((5,(5,1)), (20,(5,4)))])
def test_random_collision_resized(cl_collision, old_shape, new_shape):
    ctx, cq, program, sorter_programs, reducer_program = cl_collision

    collider = Collider(ctx, *old_shape, program=program,
                        sorter_programs=sorter_programs,
                        reducer_program=reducer_program)
    collider.resize(*new_shape)

    np.random.seed(4)
    size = new_shape[0] or old_shape[0]
    coords = np.random.random((size, 3)).astype(collider.coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(collider.coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf, collisions_buf, len(expected))
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


@pytest.mark.parametrize("size,sorter_shape", [(5,(5,1))])
def test_auto_program(cl_env, size, sorter_shape):
    ctx, cq = cl_env
    collider = Collider(ctx, size, sorter_shape)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(collider.coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(collider.coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )

    n = collider.get_collisions(cq, coords_buf, radii_buf,
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
