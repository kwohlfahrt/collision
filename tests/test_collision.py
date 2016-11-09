import numpy as np
import pyopencl as cl
from pathlib import Path
import pytest
from itertools import product as cartesian

np.random.seed(4)

kernel_args = {'generateBVH': None,
               'fillInternal': None,
               'generateBounds': None,
               'calculateCodes': None,
               'traverse': [None, None, np.dtype('uint64'), None, None],}
Node = np.dtype([('parent', 'uint32'), ('data', 'uint32', 2)])

@pytest.fixture(scope='module')

def cl_kernels():
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)

    src = Path(__file__).parent / ".." / "collision.cl"
    with src.open("r") as f:
        program = cl.Program(ctx, f.read()).build()
        kernels = {name: getattr(program, name) for name in kernel_args}
        for name, kernel in kernels.items():
            arg_types = kernel_args[name]
            if arg_types is not None:
                kernel.set_scalar_arg_dtypes(arg_types)

    return ctx, cq, kernels


def test_fill_internal(cl_kernels):
    ctx, cq, kernels = cl_kernels

    n = 8
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, (2 * n - 1) * Node.itemsize
    )

    fill_internal = kernels['fillInternal'](
        cq, (n,), None,
        nodes_buf
    )

    (nodes_map, _) = cl.enqueue_map_buffer(
        cq, nodes_buf, cl.map_flags.READ,
        (n - 1) * Node.itemsize, n, Node,
        wait_for=[fill_internal], is_blocking=True
    )
    nodes_map.dtype = Node

    np.testing.assert_equal(nodes_map['data'][:, 0], np.arange(n, dtype='uint32'))


def test_generate_bvh(cl_kernels):
    ctx, cq, kernels = cl_kernels

    # From Figure 3
    codes = np.array([0b00001, 0b00010, 0b00100, 0b00101,
                      0b10011, 0b11000, 0b11001, 0b11110], dtype='uint32')
    n_nodes = len(codes) * 2 - 1

    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=codes
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )

    fill_internal = kernels['fillInternal'](
        cq, (len(codes),), None,
        nodes_buf
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
    expected = np.array([(-1, [3, 4]),
                         (3, [leaf+0, leaf+1]),
                         (3, [leaf+2, leaf+3]),
                         (0, [1, 2]),
                         (0, [leaf+4, 5]),
                         (4, [6, leaf+7]),
                         (5, [leaf+5, leaf+6])], dtype=Node)
    np.testing.assert_equal(nodes_map[1:leaf], expected[1:])
    np.testing.assert_equal(nodes_map['data'][0], expected['data'][0])

    expected_parents = np.array([1, 1, 2, 2, 4, 6, 6, 5], dtype='uint32')
    np.testing.assert_equal(nodes_map['parent'][leaf:], expected_parents)
    expected_ids = np.arange(len(codes), dtype='uint32')
    np.testing.assert_equal(nodes_map['data'][leaf:, 0], np.arange(len(codes), dtype='uint32'))


def test_generate_odd_bvh(cl_kernels):
    ctx, cq, kernels = cl_kernels

    # From Figure 3
    codes = np.array([0b00001, 0b00010, 0b00100, 0b00101,
                      0b10011, 0b11000, 0b11001], dtype='uint32')
    n_nodes = len(codes) * 2 - 1

    codes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=codes
    )
    nodes_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * Node.itemsize
    )

    fill_internal = kernels['fillInternal'](
        cq, (len(codes),), None,
        nodes_buf
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
    expected = np.array([(-1, [3, 4]),
                         (3, [leaf+0, leaf+1]),
                         (3, [leaf+2, leaf+3]),
                         (0, [1, 2]),
                         (0, [leaf+4, 5]),
                         (4, [leaf+5, leaf+6])], dtype=Node)
    np.testing.assert_equal(nodes_map[1:leaf], expected[1:])
    np.testing.assert_equal(nodes_map['data'][0], expected['data'][0])

    expected_parents = np.array([1, 1, 2, 2, 4, 5, 5], dtype='uint32')
    np.testing.assert_equal(nodes_map['parent'][leaf:], expected_parents)
    expected_ids = np.arange(len(codes), dtype='uint32')
    np.testing.assert_equal(nodes_map['data'][leaf:, 0], np.arange(len(codes), dtype='uint32'))


def test_compute_bounds(cl_kernels):
    ctx, cq, kernels = cl_kernels

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0]], dtype='float32')
    radii = np.ones(len(coords), dtype='float32')
    leaf = len(coords) - 1
    nodes = np.array([(-1, (leaf+0, 1)),
                      ( 0, (leaf+3, 2)),
                      ( 1, (leaf+1, leaf+2)),
                      ( 0, (0, -1)),
                      ( 2, (1, -1)),
                      ( 2, (2, -1)),
                      ( 1, (3, -1))], dtype=Node)

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
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * 3 * 2 * np.dtype('float32').itemsize
    )
    flags_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, len(nodes) * np.dtype('uint32').itemsize
    )

    clear_flags = cl.enqueue_fill_buffer(
        cq, flags_buf, np.zeros(1, dtype='uint32'),
        0, len(nodes) * np.dtype('uint32').itemsize
    )
    calc_bounds = kernels['generateBounds'](
        cq, (len(coords),), None,
        bounds_buf, flags_buf, coords_buf, radii_buf, nodes_buf,
        wait_for=[clear_flags]
    )
    (bounds_map, _) = cl.enqueue_map_buffer(
        cq, bounds_buf, cl.map_flags.READ,
        0, (len(nodes), 2, 3), np.dtype('float32'),
        wait_for=[calc_bounds], is_blocking=True
    )

    expected = np.array([[[-6.0,-7.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-6.0,-7.0,-2.0], [ 5.0, 2.0, 9.0]],
                         [[-5.0,-7.0, 2.0], [ 5.0, 2.0, 9.0]],
                         [[-1.0, 0.0, 2.0], [ 1.0, 2.0, 4.0]],
                         [[ 3.0, 0.0, 7.0], [ 5.0, 2.0, 9.0]],
                         [[-5.0,-7.0, 2.0], [-3.0,-5.0, 4.0]],
                         [[-6.0,-1.0,-2.0], [-4.0, 1.0, 0.0]]], dtype='float32')
    np.testing.assert_equal(bounds_map, expected)


def test_traverse(cl_kernels):
    ctx, cq, kernels = cl_kernels

    coords = np.array([[ 0.0, 1.0, 3.0],
                       [ 4.0, 1.0, 8.0],
                       [-4.0,-6.0, 3.0],
                       [-5.0, 0.0,-1.0],
                       [-5.0, 0.5,-0.5]], dtype='float32')
    radii = np.ones(len(coords), dtype='float32')
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
        ctx, cl.mem_flags.READ_WRITE, n_nodes * 3 * 2 * np.dtype('float32').itemsize
    )
    flags_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, n_nodes * np.dtype('float32').itemsize
    )
    n_collisions = 2 + 5 # Reports self-collisions
    collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, n_collisions * 2 * np.dtype('uint32').itemsize
    )
    n_collisions_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_WRITE, np.dtype('uint64').itemsize
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
    codes_map[...] = np.sort(codes_map, kind='mergesort')
    del codes_map
    fill_internal = kernels['fillInternal'](
        cq, (len(coords),), None,
        nodes_buf
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
    calc_bounds = kernels['generateBounds'](
        cq, (len(coords),), None,
        bounds_buf, flags_buf, coords_buf, radii_buf, nodes_buf,
        wait_for=[clear_flags, generate_bvh]
    )
    clear_collisions = cl.enqueue_fill_buffer(
        cq, collisions_buf, np.array([-1], dtype='uint32'),
        0, n_collisions * 2 * np.dtype('uint32').itemsize
    )
    clear_n_collisions = cl.enqueue_fill_buffer(
        cq, n_collisions_buf, np.zeros(1, dtype='uint64'),
        0, np.dtype('uint64').itemsize
    )
    find_collisions = kernels['traverse'](
        cq, (len(coords),), None,
        collisions_buf, n_collisions_buf, n_collisions, nodes_buf, bounds_buf,
        wait_for=[clear_collisions, clear_n_collisions, calc_bounds],
        global_offset=(0,)
    )

    (n_collisions_map, _) = cl.enqueue_map_buffer(
        cq, n_collisions_buf, cl.map_flags.READ,
        0, 1, np.dtype('uint64'),
        wait_for=[find_collisions], is_blocking=True
    )

    assert n_collisions_map[0] == n_collisions

    (collisions_map, _) = cl.enqueue_map_buffer(
        cq, collisions_buf, cl.map_flags.READ,
        0, (n_collisions, 2), np.dtype('uint32'),
        wait_for=[find_collisions], is_blocking=True
    )
    expected = set(zip(range(5), (range(5)))) | {(3, 4), (4, 3)}
    assert set(map(tuple, collisions_map)) == expected
