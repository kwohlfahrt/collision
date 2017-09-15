import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from inspect import signature
from itertools import product as cartesian
from collision.collision import Node
from .common import cl_env
from collision.misc import dtype_decl, roundUp

np.random.seed(4)

kernel_args = {'generateBVH': [None, None, np.dtype('uint32')],
               'fillInternal': [None, None, np.dtype('uint32')],
               'leafBounds': [None, None, None, None, np.dtype('uint32')],
               'internalBounds': [None, None, None, np.dtype('uint32')],
               'calculateCodes': [None, None, None, np.dtype('uint32')],
               'traverse': [None, None, np.dtype('uint32'), None, None],}


def pytest_generate_tests(metafunc):
    params = signature(metafunc.function).parameters
    if 'coord_dtype' in params:
        metafunc.parametrize(
            "coord_dtype", map(np.dtype, ['float32', 'float64']), scope='module'
        )
    elif 'coord_dtype' in metafunc.fixturenames:
        metafunc.parametrize(
            "coord_dtype", map(np.dtype, ['float32']), scope='module'
        )


@pytest.fixture(scope='module')
def kernels(cl_env, coord_dtype):
    ctx, cq = cl_env

    buildopts = ["-DDTYPE={}".format(dtype_decl(coord_dtype))]

    src = Path(__file__).parent / ".." / "collision"/ "collision.cl"
    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build(buildopts)
        kernels = {name: getattr(program, name) for name in kernel_args}
        for name, kernel in kernels.items():
            arg_types = kernel_args[name]
            if arg_types is not None:
                kernel.set_scalar_arg_dtypes(arg_types)

    return kernels


def test_fill_internal(cl_env, kernels):
    ctx, cq = cl_env

    n = 8
    ids = np.random.permutation(n).astype('uint32')
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, (2 * n - 1) * Node.itemsize
    )
    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ids
    )

    fill_internal = kernels['fillInternal'](
        cq, (roundUp(n, 32),), None,
        nodes_buf, ids_buf, n,
    )

    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        (n - 1) * Node.itemsize, n, Node,
        wait_for=[fill_internal], is_blocking=True
    )
    nodes_map.dtype = Node

    np.testing.assert_equal(nodes_map['data'][:, 0], ids)
    np.testing.assert_equal(nodes_map['right_edge'], np.arange(n))


def test_generate_bvh(cl_env, kernels):
    ctx, cq = cl_env

    # From Figure 3
    codes = np.array([0b00001, 0b00010, 0b00100, 0b00101,
                      0b10011, 0b11000, 0b11001, 0b11110], dtype='uint32')
    ids = np.arange(len(codes), dtype='uint32')
    n_nodes = len(codes) * 2 - 1

    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=codes
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )
    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ids
    )

    fill_internal = kernels['fillInternal'](
        cq, (roundUp(len(codes), 32),), None,
        nodes_buf, ids_buf, len(codes),
    )
    generate_bvh = kernels['generateBVH'](
        cq, (roundUp(len(codes) - 1, 32),), None,
        codes_buf, nodes_buf, len(codes),
        wait_for=[fill_internal]
    )
    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        0, n_nodes, Node,
        wait_for=[generate_bvh], is_blocking=True
    )
    nodes_map.dtype = Node

    leaf = len(codes) - 1
    expected = np.array([(-1, 7, [3, 4]),
                         (3, 1, [leaf+0, leaf+1]),
                         (3, 3, [leaf+2, leaf+3]),
                         (0, 3, [1, 2]),
                         (0, 7, [leaf+4, 5]),
                         (4, 7, [6, leaf+7]),
                         (5, 6, [leaf+5, leaf+6])], dtype=Node)
    np.testing.assert_equal(nodes_map[1:leaf], expected[1:])
    np.testing.assert_equal(nodes_map[['data', 'right_edge']][0],
                            expected[['data', 'right_edge']][0])

    expected_parents = np.array([1, 1, 2, 2, 4, 6, 6, 5])
    np.testing.assert_equal(nodes_map['parent'][leaf:], expected_parents)
    np.testing.assert_equal(nodes_map['right_edge'][leaf:], np.arange(len(codes)))
    np.testing.assert_equal(nodes_map['data'][leaf:, 0], np.arange(len(codes)))


def test_generate_odd_bvh(cl_env, kernels):
    ctx, cq = cl_env

    # From Figure 3
    codes = np.array([0b00001, 0b00010, 0b00100, 0b00101,
                      0b10011, 0b11000, 0b11001], dtype='uint32')
    ids = np.arange(len(codes), dtype='uint32')
    n_nodes = len(codes) * 2 - 1

    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=codes
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )
    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ids
    )

    fill_internal = kernels['fillInternal'](
        cq, (roundUp(len(codes), 32),), None,
        nodes_buf, ids_buf, len(codes),
    )
    generate_bvh = kernels['generateBVH'](
        cq, (roundUp(len(codes) - 1, 32),), None,
        codes_buf, nodes_buf, len(codes),
        wait_for=[fill_internal]
    )
    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        0, n_nodes, Node,
        wait_for=[generate_bvh], is_blocking=True
    )
    nodes_map.dtype = Node

    leaf = len(codes) - 1
    expected = np.array([(-1, 6, [3, 4]),
                         (3, 1, [leaf+0, leaf+1]),
                         (3, 3, [leaf+2, leaf+3]),
                         (0, 3, [1, 2]),
                         (0, 6, [leaf+4, 5]),
                         (4, 6, [leaf+5, leaf+6])], dtype=Node)
    np.testing.assert_equal(nodes_map[1:leaf], expected[1:])
    np.testing.assert_equal(nodes_map[['right_edge', 'data']][0],
                            expected[['right_edge', 'data']][0])

    expected_parents = np.array([1, 1, 2, 2, 4, 5, 5])
    np.testing.assert_equal(nodes_map['parent'][leaf:], expected_parents)
    np.testing.assert_equal(nodes_map['data'][leaf:, 0], np.arange(len(codes)))


def test_compute_bounds(cl_env, kernels, coord_dtype):
    ctx, cq = cl_env

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0]], dtype=coord_dtype)
    radii = np.ones(len(coords), dtype=coord_dtype)
    leaf = len(coords) - 1
    nodes = np.array([(-1, 3, [leaf+0, 1]),
                      ( 0, 3, [leaf+3, 2]),
                      ( 1, 2, [leaf+1, leaf+2]),
                      ( 0, 0, [2, -1]),
                      ( 2, 1, [0, -1]),
                      ( 2, 2, [1, -1]),
                      ( 1, 3, [3, -1])], dtype=Node)

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
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=nodes
    )
    bounds_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * 4 * 2 * coords.dtype.itemsize
    )
    flags_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * np.dtype('uint32').itemsize
    )

    clear_flags = cl.enqueue_fill_buffer(
        cq, flags_buf, np.zeros(1, dtype='uint32'),
        0, len(nodes) * np.dtype('uint32').itemsize
    )
    calc_leaf_bounds = kernels['leafBounds'](
        cq, (roundUp(len(coords), 32),), None,
        bounds_buf, coords_buf, radii_buf, nodes_buf, len(coords),
    )
    calc_bounds = kernels['internalBounds'](
        cq, (roundUp(len(coords), 32),), None,
        bounds_buf, flags_buf, nodes_buf, len(coords),
        wait_for=[calc_leaf_bounds, clear_flags]
    )
    (bounds_map, _) = cl.enqueue_map_buffer(
        cq, bounds_buf, cl.map_flags.READ,
        0, (len(nodes), 2, 4), coord_dtype,
        wait_for=[calc_bounds], is_blocking=True
    )

    expected = np.array([[[-6.0,-7.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-6.0,-1.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-1.0, 0.0, 2.0], [ 5.0, 2.0, 9.0]],
                         [[-5.0,-7.0, 2.0], [-3.0,-5.0, 4.0]],
                         [[-1.0, 0.0, 2.0], [ 1.0, 2.0, 4.0]],
                         [[ 3.0, 0.0, 7.0], [ 5.0, 2.0, 9.0]],
                         [[-6.0,-1.0,-2.0], [-4.0, 1.0, 0.0]]], dtype=coord_dtype)
    np.testing.assert_equal(bounds_map[:, :, :3], expected)


def test_codes(cl_env, kernels, coord_dtype):
    ctx, cq = cl_env

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype=coord_dtype)
    coord_range = np.array([coords.min(axis=0),
                            coords.max(axis=0)], dtype=coords.dtype)
    expected = np.array([862940378, 862940378, 1073741823,
                         20332620, 302580864, 306295426], dtype='int32')

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
    range_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY, 2 * 4 * coord_dtype.itemsize
    )
    (range_map, _) = cl.enqueue_map_buffer(
        cq, range_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
        0, (len(coord_range), 4), coord_dtype,
        is_blocking=True
    )
    range_map[..., :3] = coord_range
    del range_map
    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(coords) * np.dtype('uint32').itemsize
    )
    calc_codes = kernels['calculateCodes'](
        cq, (roundUp(len(coords), 32),), None,
        codes_buf, coords_buf, range_buf, len(coords),
    )

    (codes_map, _) = cl.enqueue_map_buffer(
        cq, codes_buf, cl.map_flags.READ | cl.map_flags.WRITE,
        0, (len(coords),), np.dtype('uint32'),
        wait_for=[calc_codes], is_blocking=True
    )
    np.testing.assert_equal(codes_map, expected)
    del codes_map


def test_traverse(cl_env, kernels, coord_dtype):
    ctx, cq = cl_env

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype=coord_dtype)
    radii = np.ones(len(coords), dtype=coord_dtype)
    n_nodes = len(coords) * 2 - 1

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
    range_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array([coords.min(axis=0), coords.max(axis=0)], dtype=coords.dtype)
    )
    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(coords) * np.dtype('uint32').itemsize
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )
    bounds_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * 4 * 2 * coords.dtype.itemsize
    )
    flags_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * np.dtype('uint32').itemsize
    )
    n_collisions = 2
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, n_collisions * 2 * np.dtype('uint32').itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, np.dtype('uint32').itemsize
    )

    calc_codes = kernels['calculateCodes'](
        cq, (roundUp(len(coords), 32),), None,
        codes_buf, coords_buf, range_buf, len(coords),
    )

    # Would use radix sort
    (codes_map, _) = cl.enqueue_map_buffer(
        cq, codes_buf, cl.map_flags.READ | cl.map_flags.WRITE,
        0, (len(coords),), np.dtype('uint32'),
        wait_for=[calc_codes], is_blocking=True
    )
    order = np.argsort(codes_map, kind='mergesort').astype('uint32')
    codes_map[...] = codes_map[order]
    del codes_map

    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=order
    )

    fill_internal = kernels['fillInternal'](
        cq, (roundUp(len(coords), 32),), None,
        nodes_buf, ids_buf, len(coords),
    )
    generate_bvh = kernels['generateBVH'](
        cq, (roundUp(len(coords)-1, 32),), None,
        codes_buf, nodes_buf, len(coords),
        wait_for=[calc_codes, fill_internal]
    )
    clear_flags = cl.enqueue_fill_buffer(
        cq, flags_buf, np.zeros(1, dtype='uint32'),
        0, n_nodes * np.dtype('uint32').itemsize
    )
    calc_bounds = kernels['leafBounds'](
        cq, (roundUp(len(coords), 32),), None,
        bounds_buf, coords_buf, radii_buf, nodes_buf, len(coords),
        wait_for=[generate_bvh]
    )
    calc_bounds = kernels['internalBounds'](
        cq, (roundUp(len(coords), 32),), None,
        bounds_buf, flags_buf, nodes_buf, len(coords),
        wait_for=[clear_flags, calc_bounds]
    )
    clear_collisions = cl.enqueue_fill_buffer(
        cq, collisions_buf, np.array([-1], dtype='uint32'),
        0, n_collisions * 2 * np.dtype('uint32').itemsize
    )
    clear_n_collisions = cl.enqueue_fill_buffer(
        cq, n_collisions_buf, np.zeros(1, dtype='uint32'),
        0, np.dtype('uint32').itemsize
    )
    find_collisions = kernels['traverse'](
        cq, (len(coords),), None,
        collisions_buf, n_collisions_buf, n_collisions, nodes_buf, bounds_buf,
        wait_for=[clear_collisions, clear_n_collisions, calc_bounds],
    )

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, np.dtype('uint32'),
        wait_for=[find_collisions], is_blocking=True
    )

    assert n_collisions_map[0] == n_collisions

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions, 2), np.dtype('uint32'),
        wait_for=[find_collisions], is_blocking=True
    )
    expected = {(0, 1), (4, 5)}
    assert set(map(tuple, collisions_map)) == expected


def test_problem_codes(cl_env, kernels, coord_dtype):
    from .test_collision_py import find_collisions
    ctx, cq = cl_env

    codes = np.array([0b00000000000000000000000000000000,
                      0b00000000000000000000000000000000,
                      0b00000110110000110100000100000010,
                      0b00001001001001001001001001001001,
                      0b00001001001001001001001001001001,
                      0b00010010010010010010010010010010,
                      0b00010010010010010010010010010010,
                      0b00010010011010010010011011011010,
                      0b00011001001011001001011001001011,
                      0b00011011011011011011011011011011,
                      0b00100100010000100010110100010110,
                      0b00100100100100100100100100100100,
                      0b00100100100101101101100101100100,
                      0b00101001101001101101101101101001,
                      0b00101101101101101101101101101101,
                      0b00110110110110110110110110110110, # This node had no parent
                      0b00110110110110110110110110110110,
                      0b00110110110110110110110110110110,
                      0b00111111111111111111111111111111,
                      0b00111111111111111111111111111111,
                      0b00111111111111111111111111111111], dtype='uint32')
    ids = np.arange(len(codes), dtype='uint32')
    n_nodes = 2 * len(codes) - 1

    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=codes
    )
    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ids
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )

    fill_internal = kernels['fillInternal'](
        cq, (roundUp(len(codes), 32),), None,
        nodes_buf, ids_buf, len(codes),
    )

    generate_bvh = kernels['generateBVH'](
        cq, (roundUp(len(codes) - 1, 32),), None,
        codes_buf, nodes_buf, len(codes),
        wait_for=[fill_internal]
    )

    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        0, n_nodes, Node,
        wait_for=[generate_bvh], is_blocking=True
    )
    nodes_map.dtype = Node
    assert set(nodes_map['parent'][1:]) == set(range(len(codes) - 1))
