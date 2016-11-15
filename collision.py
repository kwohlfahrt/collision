from numpy import dtype, zeros, array
from pathlib import Path
from itertools import accumulate, chain, tee
import pyopencl as cl
from misc import Program
from radix import RadixSorter, roundUp
from reduce import Reducer

Node = dtype([('parent', 'uint32'), ('right_edge', 'uint32'), ('data', 'uint32', 2)])

class CollisionProgram(Program):
    src = Path(__file__).parent / "collision.cl"
    kernel_args = {'range': [None],
                   'calculateCodes': [None, None, None],
                   'fillInternal': [None, None],
                   'generateBVH': [None, None],
                   'generateBounds': [None, None, None, None, None],
                   'traverse': [None, None, dtype('uint64'), None, None]}

    def __init__(self, ctx, coord_dtype=dtype('float32')):
        coord_dtype = dtype(coord_dtype)
        if coord_dtype == dtype('float32'):
            def_dtype = 'float'
        elif coord_dtype == dtype('float64'):
            def_dtype = 'double'
        else:
            raise ValueError("Invalid dtype: {}".format(coord_dtype))
        self.coord_dtype = coord_dtype

        super().__init__(ctx, ["-D DTYPE={}".format(def_dtype)])


class Collider:
    code_dtype = dtype('uint32')
    flag_dtype = dtype('uint32') # Smallest atomic
    counter_dtype = dtype('uint64')
    id_dtype = dtype('uint32')

    def __init__(self, ctx, size, sorter_shape, coord_dtype=dtype('float32'),
                 program=None, sorter_programs=(None, None), reducer_program=None):
        self.size = size

        # self.padded size not available before sorter creation
        padded_size = roundUp(size, sorter_shape[0] * sorter_shape[1])
        self.sorter = RadixSorter(ctx, padded_size, *sorter_shape,
                                  program=sorter_programs[0],
                                  scan_program=sorter_programs[1])
        self.reducer = Reducer(ctx, *sorter_shape, coord_dtype=coord_dtype,
                               program=reducer_program)
        if program is None:
            program = CollisionProgram(ctx)
        else:
            if program.context != ctx:
                raise ValueError("Collider and program context must match")
            if program.coord_dtype != coord_dtype:
                raise ValueError("Collider and program coord_dtype must match")
        self.program = program

        # Can't sort in-place
        self._ids_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.padded_size * self.id_dtype.itemsize
        ) for _ in range(2)]
        self._codes_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.padded_size * self.code_dtype.itemsize
        ) for _ in range(2)]

        self._nodes_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.n_nodes * Node.itemsize
        )
        # Dual-use: storing per-node and scene bounds
        self._bounds_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.n_nodes * 2 * 3 * self.program.coord_dtype.itemsize
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
        old_padded_size = self.padded_size
        old_n_nodes = self.n_nodes
        self.sorter.resize(size, *sorter_shape)
        self.reducer.resize(*sorter_shape[:2])
        self.size = size

        if old_padded_size != self.padded_size:
            self._ids_bufs = [cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.padded_size * self.id_dtype.itemsize
            ) for _ in range(2)]
            self._codes_bufs = [cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.padded_size * self.code_dtype.itemsize
            ) for _ in range(2)]

        if old_n_nodes != self.n_nodes:
            self._nodes_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.n_nodes * Node.itemsize
            )
            self._bounds_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.n_nodes * 2 * 3 * self.program.coord_dtype.itemsize
            )
            self._flags_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.n_nodes * self.flag_dtype.itemsize
            )

    @property
    def n_nodes(self):
        return self.size * 2 - 1

    @property
    def padded_size(self):
        return roundUp(self.size, self.sorter.ngroups * self.sorter.group_size)

    def get_collisions(self, cq, coords_buf, radii_buf, collisions_buf,
                       n_collisions, wait_for=None):
        if wait_for is None:
            wait_for = []

        fill_codes = []
        if self.padded_size != self.size:
            fill_codes.append(cl.enqueue_fill_buffer(
                cq, self._codes_bufs[0], array([-1], dtype=self.code_dtype),
                0, self.padded_size * self.code_dtype.itemsize
            ))
        fill_ids = self.program.kernels['range'](
            cq, (self.size,), None,
            self._ids_bufs[0]
        )
        clear_flags = cl.enqueue_fill_buffer(
            cq, self._flags_buf, zeros(1, dtype=self.flag_dtype),
            0, self.n_nodes * self.flag_dtype.itemsize
        )
        clear_n_collisions = cl.enqueue_fill_buffer(
            cq, self._n_collisions_buf, zeros(1, dtype=self.counter_dtype),
            0, self.counter_dtype.itemsize
        )

        calc_scene_bounds = self.reducer.reduce(cq, self.size, coords_buf, self._bounds_buf)

        calc_codes = self.program.kernels['calculateCodes'](
            cq, (self.size,), None,
            self._codes_bufs[0], coords_buf, self._bounds_buf,
            wait_for=[calc_scene_bounds] + fill_codes
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
            wait_for=[clear_n_collisions, calc_bounds] + wait_for,
        )

        (n_collisions_map, _) = cl.enqueue_map_buffer(
            cq, self._n_collisions_buf, cl.map_flags.READ,
            0, 1, self.counter_dtype,
            wait_for=[find_collisions], is_blocking=True
        )
        return int(n_collisions_map[0])
