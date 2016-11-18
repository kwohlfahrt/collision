import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian
from collision import Node

np.random.seed(4)

kernel_args = {'generateBVH': None,
               'fillInternal': None,
               'leafBounds': None,
               'internalBounds': None,
               'calculateCodes': None,
               'traverse': [None, None, np.dtype('uint32'), None, None],}


def pytest_generate_tests(metafunc):
    if 'coord_dtype' in metafunc.fixturenames:
        metafunc.parametrize("coord_dtype", ['float32', 'float64'], scope='module')


@pytest.fixture(scope='module')
def cl_kernels(coord_dtype):
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)

    c_dtypes = {'float32': 'float', 'float64': 'double'}
    buildopts = ["-D DTYPE={}".format(c_dtypes[coord_dtype])]

    src = Path(__file__).parent / ".." / "collision.cl"
    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build(buildopts)
        kernels = {name: getattr(program, name) for name in kernel_args}
        for name, kernel in kernels.items():
            arg_types = kernel_args[name]
            if arg_types is not None:
                kernel.set_scalar_arg_dtypes(arg_types)

    return ctx, cq, kernels


def test_fill_internal(cl_kernels):
    ctx, cq, kernels = cl_kernels

    n = 8
    ids = np.random.permutation(n).astype('uint32')
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, (2 * n - 1) * Node.itemsize
    )
    ids_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ids
    )

    fill_internal = kernels['fillInternal'](
        cq, (n,), None,
        nodes_buf, ids_buf
    )

    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        (n - 1) * Node.itemsize, n, Node,
        wait_for=[fill_internal], is_blocking=True
    )
    nodes_map.dtype = Node

    np.testing.assert_equal(nodes_map['data'][:, 0], ids)
    np.testing.assert_equal(nodes_map['right_edge'], np.arange(n))


def test_generate_bvh(cl_kernels):
    ctx, cq, kernels = cl_kernels

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
        cq, (len(codes),), None,
        nodes_buf, ids_buf
    )
    generate_bvh = kernels['generateBVH'](
        cq, (len(codes) - 1,), None,
        codes_buf, nodes_buf,
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


def test_generate_odd_bvh(cl_kernels):
    ctx, cq, kernels = cl_kernels

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
        cq, (len(codes),), None,
        nodes_buf, ids_buf
    )
    generate_bvh = kernels['generateBVH'](
        cq, (len(codes) - 1,), None,
        codes_buf, nodes_buf,
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


def test_compute_bounds(cl_kernels, coord_dtype):
    ctx, cq, kernels = cl_kernels

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
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
    radii_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=nodes
    )
    bounds_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * 3 * 2 * coords.dtype.itemsize
    )
    flags_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * np.dtype('uint32').itemsize
    )

    clear_flags = cl.enqueue_fill_buffer(
        cq, flags_buf, np.zeros(1, dtype='uint32'),
        0, len(nodes) * np.dtype('uint32').itemsize
    )
    calc_leaf_bounds = kernels['leafBounds'](
        cq, (len(coords),), None,
        bounds_buf, coords_buf, radii_buf, nodes_buf,
    )
    calc_bounds = kernels['internalBounds'](
        cq, (len(coords),), None,
        bounds_buf, flags_buf, nodes_buf,
        wait_for=[calc_leaf_bounds, clear_flags]
    )
    (bounds_map, _) = cl.enqueue_map_buffer(
        cq, bounds_buf, cl.map_flags.READ,
        0, (len(nodes), 2, 3), coord_dtype,
        wait_for=[calc_bounds], is_blocking=True
    )

    expected = np.array([[[-6.0,-7.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-6.0,-1.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-1.0, 0.0, 2.0], [ 5.0, 2.0, 9.0]],
                         [[-5.0,-7.0, 2.0], [-3.0,-5.0, 4.0]],
                         [[-1.0, 0.0, 2.0], [ 1.0, 2.0, 4.0]],
                         [[ 3.0, 0.0, 7.0], [ 5.0, 2.0, 9.0]],
                         [[-6.0,-1.0,-2.0], [-4.0, 1.0, 0.0]]], dtype=coord_dtype)
    np.testing.assert_equal(bounds_map, expected)


def test_traverse(cl_kernels, coord_dtype):
    ctx, cq, kernels = cl_kernels

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype=coord_dtype)
    radii = np.ones(len(coords), dtype=coord_dtype)
    n_nodes = len(coords) * 2 - 1

    coords_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords
    )
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
        ctx, cl.mem_flags.READ_WRITE, n_nodes * 3 * 2 * coords.dtype.itemsize
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
        cq, (len(coords),), None,
        codes_buf, coords_buf, range_buf,
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
        cq, (len(coords),), None,
        nodes_buf, ids_buf
    )
    generate_bvh = kernels['generateBVH'](
        cq, (len(coords)-1,), None,
        codes_buf, nodes_buf,
        wait_for=[calc_codes, fill_internal]
    )
    clear_flags = cl.enqueue_fill_buffer(
        cq, flags_buf, np.zeros(1, dtype='uint32'),
        0, n_nodes * np.dtype('uint32').itemsize
    )
    calc_bounds = kernels['leafBounds'](
        cq, (len(coords),), None,
        bounds_buf, coords_buf, radii_buf, nodes_buf,
        wait_for=[generate_bvh]
    )
    calc_bounds = kernels['internalBounds'](
        cq, (len(coords),), None,
        bounds_buf, flags_buf, nodes_buf,
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


def test_problem_codes(cl_kernels, coord_dtype):
    from test_collision_py import find_collisions
    ctx, cq, kernels = cl_kernels

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
        cq, (len(codes),), None,
        nodes_buf, ids_buf
    )

    generate_bvh = kernels['generateBVH'](
        cq, (len(codes) - 1,), None,
        codes_buf, nodes_buf,
        wait_for=[fill_internal]
    )

    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        0, n_nodes, Node,
        wait_for=[generate_bvh], is_blocking=True
    )
    nodes_map.dtype = Node
    assert set(nodes_map['parent'][1:]) == set(range(len(codes) - 1))
