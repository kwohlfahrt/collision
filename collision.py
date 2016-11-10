from numpy import dtype, zeros
from pathlib import Path
from itertools import accumulate, chain, tee
import pyopencl as cl
from misc import Program

Node = dtype([('parent', 'uint32'), ('right_edge', 'uint32'), ('data', 'uint32', 2)])

class CollisionProgram(Program):
    src = Path(__file__).parent / "collision.cl"
    kernel_args = {'calculateCodes': [None, None, None],
                   'fillInternal': [None, None],
                   'generateBVH': [None, None],
                   'generateBounds': [None, None, None, None, None],
                   'traverse': [None, None, dtype('uint64'), None, None]}

class Collider:
    code_dtype = dtype('uint32')
    coord_dtype = dtype('float32')
    flag_dtype = dtype('uint32') # Smallest atomic
    counter_dtype = dtype('uint64')
    id_dtype = dtype('uint32')

    def __init__(self, program, size):
        ctx = program.context
        self.program = program
        self.size = size
        self.n_nodes = size * 2 - 1

        self._ids_buf = cl.Buffer(
            # TODO: can be host-no-access with radix sort
            ctx, cl.mem_flags.READ_WRITE,
            self.size * self.id_dtype.itemsize
        )
        self._codes_buf = cl.Buffer(
            # TODO: can be host-no-access with radix sort
            ctx, cl.mem_flags.READ_WRITE,
            self.size * self.code_dtype.itemsize
        )
        self._nodes_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.n_nodes * Node.itemsize
        )
        self._bounds_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.n_nodes * 2 * 3 * self.coord_dtype.itemsize
        )
        self._flags_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.n_nodes * self.flag_dtype.itemsize
        )
        self._n_collisions_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_READ_ONLY,
            self.counter_dtype.itemsize
        )

    def get_collisions(self, cq, coords_buf, radii_buf, range_buf,
                       collisions_buf, n_collisions, wait_for=None):
        from numpy import argsort # TODO: radix-sort

        clear_flags = cl.enqueue_fill_buffer(
            cq, self._flags_buf, zeros(1, dtype='uint32'),
            0, self.n_nodes * self.flag_dtype.itemsize
        )
        clear_n_collisions = cl.enqueue_fill_buffer(
            cq, self._n_collisions_buf, zeros(1, dtype='uint64'),
            0, self.counter_dtype.itemsize
        )
        # TODO: Won't be necessary to map once radix sort is used
        (ids_map, map_ids) = cl.enqueue_map_buffer(
            cq, self._ids_buf, cl.map_flags.WRITE_INVALIDATE_REGION,
            0, (self.size,), self.id_dtype,
            is_blocking=False
        )

        calc_codes = self.program.kernels['calculateCodes'](
            cq, (self.size,), None,
            self._codes_buf, coords_buf, range_buf,
        )

        # TODO: Hook up radix sort
        (codes_map, _) = cl.enqueue_map_buffer(
            cq, self._codes_buf, cl.map_flags.READ | cl.map_flags.WRITE,
            0, (self.size,), self.code_dtype,
            wait_for=[calc_codes], is_blocking=True
        )
        order = argsort(codes_map, kind='mergesort').astype(self.id_dtype)
        codes_map[...] = codes_map[order]
        cl.wait_for_events([map_ids])
        ids_map[...] = order
        del codes_map, ids_map

        fill_internal = self.program.kernels['fillInternal'](
            cq, (self.size,), None,
            self._nodes_buf, self._ids_buf
        )
        generate_bvh = self.program.kernels['generateBVH'](
            cq, (self.size-1,), None,
            self._codes_buf, self._nodes_buf,
            wait_for=[calc_codes, fill_internal]
        )
        calc_bounds = self.program.kernels['generateBounds'](
            cq, (self.size,), None,
            self._bounds_buf, self._flags_buf, coords_buf, radii_buf, self._nodes_buf,
            wait_for=[clear_flags, generate_bvh]
        )
        find_collisions = self.program.kernels['traverse'](
            cq, (self.size,), None,
            collisions_buf, self._n_collisions_buf, n_collisions,
            self._nodes_buf, self._bounds_buf,
            wait_for=[clear_n_collisions, calc_bounds],
        )

        (n_collisions_map, _) = cl.enqueue_map_buffer(
            cq, self._n_collisions_buf, cl.map_flags.READ,
            0, 1, self.counter_dtype,
            wait_for=[find_collisions], is_blocking=True
        )
        return int(n_collisions_map[0])
