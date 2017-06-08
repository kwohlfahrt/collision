import numpy as np
import pyopencl as cl
import pytest
from inspect import signature
from collision.collision import *
from .common import cl_env

def pytest_generate_tests(metafunc):
    params = signature(metafunc.function).parameters
    if 'coord_dtype' in params:
        metafunc.parametrize(
            "coord_dtype", map(dtype, ['float32', 'float64']), scope='module'
        )
    elif 'coord_dtype' in metafunc.fixturenames:
        metafunc.parametrize("coord_dtype", [dtype('float32')], scope='module')


@pytest.fixture(scope='module')
def collision_programs(cl_env, coord_dtype):
    from collision.radix import RadixProgram, PrefixScanProgram
    from collision.reduce import ReductionProgram

    ctx, cq = cl_env
    program = CollisionProgram(ctx, coord_dtype)
    radix_program = RadixProgram(ctx)
    scan_program = PrefixScanProgram(ctx)
    reducer_program = ReductionProgram(ctx, coord_dtype)
    return program, (radix_program, scan_program), reducer_program


def find_collisions(coords, radii):
    min_bounds = coords - radii.reshape(-1, 1)
    max_bounds = coords + radii.reshape(-1, 1)
    collisions = ((max_bounds.reshape(-1, 1, 3) > min_bounds.reshape(1, -1, 3)) &
                  (min_bounds.reshape(-1, 1, 3) < max_bounds.reshape(1, -1, 3)))
    collisions = collisions.all(axis=-1)
    collisions = np.tril(collisions, -1)
    return set(zip(*reversed(np.nonzero(collisions))))


@pytest.mark.parametrize("size,ngroups,group_size,expected", [
    (48, 3, 8, 48), (47, 3, 8, 48), (49, 3, 8, 64),
])
def test_padded_size(cl_env, collision_programs, size, ngroups, group_size, expected):
    ctx, cq = cl_env
    collider = Collider(ctx, size, ngroups, group_size, 'float32', *collision_programs)
    assert collider.padded_size == expected


def test_collision(cl_env, coord_dtype, collision_programs):
    ctx, cq = cl_env

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype=coord_dtype)
    radii = np.ones(len(coords), dtype=coord_dtype)
    expected = {(0, 1), (4, 5)}

    collider = Collider(ctx, len(coords), 3, 8, coord_dtype, *collision_programs)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, collisions_buf, len(expected))

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, collider.counter_dtype,
        wait_for=[e], is_blocking=True
    )
    assert n_collisions_map[0] == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions_map[0], 2), collider.id_dtype,
        wait_for=[e], is_blocking=True
    )
    assert set(map(tuple, collisions_map)) == expected


@pytest.mark.parametrize("size,ngroups,group_size", [
    (120, 5, 8), (256, 4, 32), (317, 4, 16), (341, 4, 64)
])
def test_random_collision(cl_env, coord_dtype, collision_programs, size, ngroups, group_size):
    ctx, cq = cl_env
    collider = Collider(ctx, size, ngroups, group_size, coord_dtype, *collision_programs)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, collisions_buf, len(expected))

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, collider.counter_dtype,
        wait_for=[e], is_blocking=True
    )
    assert n_collisions_map[0] == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions_map[0], 2), collider.id_dtype,
        wait_for=[e], is_blocking=True
    )

    # Need to sort, order is undefined
    collisions = set(map(tuple, np.sort(collisions_map, axis=1)))
    assert collisions == expected


@pytest.mark.parametrize("old_shape,new_shape", [
    ((350, 8, 64), (351, 8, 64)),
    ((350, 8, 64), (351, None, None))
])
def test_random_collision_resized(cl_env, coord_dtype, collision_programs, old_shape, new_shape):
    ctx, cq = cl_env

    collider = Collider(ctx, *old_shape, coord_dtype, *collision_programs)
    collider.resize(*new_shape)

    np.random.seed(4)
    size = new_shape[0] or old_shape[0]
    coords = np.random.random((size, 3)).astype(coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, collisions_buf, len(expected))

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, collider.counter_dtype,
        wait_for=[e], is_blocking=True
    )
    assert n_collisions_map[0] == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions_map[0], 2), collider.id_dtype,
        wait_for=[e], is_blocking=True
    )

    # Need to sort, order is undefined
    collisions = set(map(tuple, np.sort(collisions_map, axis=1)))
    assert collisions == expected


@pytest.mark.parametrize("size, ngroups, group_size", [(8, 1, 8)])
def test_auto_program(cl_env, coord_dtype, size, ngroups, group_size):
    ctx, cq = cl_env
    collider = Collider(ctx, size, ngroups, group_size)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, len(expected) * 2 * collider.id_dtype.itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, collisions_buf, len(expected))

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, collider.counter_dtype,
        wait_for=[e], is_blocking=True
    )
    assert n_collisions_map[0] == len(expected)

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions_map[0], 2), collider.id_dtype,
        wait_for=[e], is_blocking=True
    )

    # Need to sort, order is undefined
    collisions = set(map(tuple, np.sort(collisions_map, axis=1)))
    assert collisions == expected


@pytest.mark.parametrize("size, ngroups, group_size", [(100, 10, 8)])
def test_count_only(cl_env, coord_dtype, collision_programs, size, ngroups, group_size):
    ctx, cq = cl_env
    collider = Collider(ctx, size, ngroups, group_size, coord_dtype, *collision_programs)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, None, 0)

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, collider.counter_dtype,
        wait_for=[e], is_blocking=True
    )
    assert n_collisions_map[0] == len(expected)

@pytest.mark.parametrize("size, ngroups, group_size", [(100, 5, 8)])
def test_count_err(cl_env, coord_dtype, collision_programs, size,  ngroups, group_size):
    ctx, cq = cl_env
    collider = Collider(ctx, size, ngroups, group_size, coord_dtype, *collision_programs)

    np.random.seed(4)
    coords = np.random.random((size, 3)).astype(coord_dtype)
    radius = 1 / (size ** 0.5) # Keep number of collisions under control
    radii = np.random.uniform(0, radius, len(coords)).astype(coord_dtype)
    expected = find_collisions(coords, radii)

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, len(coords) * 4 * coord_dtype.itemsize
    )
    (coords_map, _) = cl.enqueue_map_buffer(
        cq, coords_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coords), 4), coord_dtype,
        is_blocking=True
    )
    coords_map[..., :3] = coords
    del coords_map
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, collider.counter_dtype.itemsize
    )

    with pytest.raises(ValueError):
        e = collider.get_collisions(cq, coords_buf, radii_buf, n_collisions_buf, None, len(expected))


@pytest.mark.parametrize("dt", ['float32', np.dtype('float32'),
                                'float64', np.dtype('float64')])
def test_collider_dtype(cl_env, dt):
    ctx, cq = cl_env
    collider = Collider(ctx, 100, 5, 8, coord_dtype=dt)
    assert collider.program.coord_dtype == np.dtype(dt)
    assert collider.reducer.program.coord_dtype == np.dtype(dt)
