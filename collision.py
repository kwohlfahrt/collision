from numpy import dtype, zeros
from pathlib import Path
from itertools import accumulate, chain, tee
import pyopencl as cl
from misc import Program
from radix import RadixSorter, RadixProgram

Node = dtype([('parent', 'uint32'), ('right_edge', 'uint32'), ('data', 'uint32', 2)])

class CollisionProgram(Program):
    src = Path(__file__).parent / "collision.cl"
    kernel_args = {'range': [None],
                   'calculateCodes': [None, None, None],
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

    def __init__(self, program, size, sorter):
        ctx = program.context
        self.program = program
        self.size = size
        if sorter.size != self.size:
            raise ValueError("Sorter size ({}) must match collider size ({})"
                             .format(sorter.size, self.size))
        self.sorter = sorter

        # Can't sort in-place
        self._ids_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.size * self.id_dtype.itemsize
        ) for _ in range(2)]
        self._codes_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.size * self.code_dtype.itemsize
        ) for _ in range(2)]

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

    def resize(self, size=None, sorter_shape=(None, None, None)):
        ctx = self.program.context
        self.sorter.resize(size, *sorter_shape)
        old_size = self.size
        self.size = size

        if old_size != size:
            self._ids_bufs = [cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.size * self.id_dtype.itemsize
            ) for _ in range(2)]
            self._codes_bufs = [cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.size * self.code_dtype.itemsize
            ) for _ in range(2)]

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

    @property
    def n_nodes(self):
        return self.size * 2 - 1

    def get_collisions(self, cq, coords_buf, radii_buf, range_buf,
                       collisions_buf, n_collisions, wait_for=None):
        fill_ids = self.program.kernels['range'](
            cq, (self.size,), None,
            self._ids_bufs[0]
        )
        clear_flags = cl.enqueue_fill_buffer(
            cq, self._flags_buf, zeros(1, dtype='uint32'),
            0, self.n_nodes * self.flag_dtype.itemsize
        )
        clear_n_collisions = cl.enqueue_fill_buffer(
            cq, self._n_collisions_buf, zeros(1, dtype='uint64'),
            0, self.counter_dtype.itemsize
        )
        calc_codes = self.program.kernels['calculateCodes'](
            cq, (self.size,), None,
            self._codes_bufs[0], coords_buf, range_buf,
        )

        self.sorter.sort(cq, *self._codes_bufs, *self._ids_bufs)

        fill_internal = self.program.kernels['fillInternal'](
            cq, (self.size,), None,
            self._nodes_buf, self._ids_bufs[1]
        )
        generate_bvh = self.program.kernels['generateBVH'](
            cq, (self.size-1,), None,
            self._codes_bufs[1], self._nodes_buf,
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
