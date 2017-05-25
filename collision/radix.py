from numpy import dtype
from pathlib import Path
import pyopencl as cl
from .misc import Program, nextPowerOf2, roundUp, np_unsigned_dtypes, dtype_decl
from .scan import PrefixScanProgram, PrefixScanner

np_unsigned_dtypes = set(map(dtype, np_unsigned_dtypes))

class RadixProgram(Program):
    src = Path(__file__).parent / "radix.cl"
    kernel_args = {'block_sort': [None, None, None, None, None, None, None, None, None,
                                  dtype('uint8'), dtype('uint8')],
                   'scatter': [None, None, None, None, None, None, None, None,
                               dtype('uint8'), dtype('uint8')]}

    def __init__(self, ctx, key_dtype=dtype('uint32'), value_dtype=dtype('uint32')):
        self.key_dtype = dtype(key_dtype)
        self.value_dtype = dtype(value_dtype)
        if self.key_dtype not in np_unsigned_dtypes:
            raise ValueError("Invalid key dtype: {}".format(self.key_dtype))

        super().__init__(ctx, [
            "-D KEY_TYPE='{}'".format(dtype_decl(self.key_dtype)),
            "-D VALUE_TYPE='{}'".format(dtype_decl(self.value_dtype))
        ])

class RadixSorter:
    histogram_dtype = dtype('uint32')

    def __init__(self, ctx, size, group_size, radix_bits=4,
                 key_dtype=dtype('uint32'), value_dtype=dtype('uint32'),
                 program=None, scan_program=None):
        self.check_size(size, group_size, radix_bits, key_dtype)
        self.size = size
        self.group_size = group_size
        self.radix_bits = radix_bits

        if program is None:
            program = RadixProgram(ctx, key_dtype, value_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Sorter and program contexts must match")
            if program.key_dtype != key_dtype:
                raise ValueError("Sorter and program key dtypes must match")
            if program.value_dtype != value_dtype:
                raise ValueError("Sorter and program value dtypes must match")
        self.program = program
        if scan_program is None:
            scan_program = PrefixScanProgram(ctx)
        self.scanner = PrefixScanner(ctx, self.histogram_len, self.group_size, scan_program)

        self._histogram_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.histogram_len * self.histogram_dtype.itemsize
        )
        self._offset_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.histogram_len * self.histogram_dtype.itemsize
        )

    @staticmethod
    def check_size(size, group_size, radix_bits, key_dtype):
        key_dtype = dtype(key_dtype)
        if group_size != nextPowerOf2(group_size):
            raise ValueError("Group size ({}) must be a power of two".format(group_size))
        if (size % (group_size * 2)):
            raise ValueError("Size ({}) must be multiple of 2 * group_size ({})"
                             .format(size, group_size))
        if (key_dtype.itemsize * 8) % radix_bits:
            raise ValueError("Radix bits ({}) must evenly divide item-size ({})"
                             .format(radix_bits, key_dtype.itemsize * 8))
        if (2 ** radix_bits) > group_size * 2:
            raise ValueError("2 ^ radix_bits ({}) must be less than 2 * group_size ({})"
                             .format(radix_bits, group_size))

    def resize(self, size=None, group_size=None, radix_bits=None):
        ctx = self.program.context
        if size is None:
            size = self.size
        if group_size is None:
            group_size = self.group_size
        if radix_bits is None:
            radix_bits = self.radix_bits
        old_histogram_len = self.histogram_len
        old_params = (self.size, self.group_size, self.radix_bits)

        self.check_size(size, group_size, radix_bits, self.program.key_dtype)

        self.size = size
        self.group_size = group_size
        self.radix_bits = radix_bits

        try:
            self.scanner.resize(self.histogram_len, self.group_size)
        except:
            self.size, self.group_size, self.radix_bits = old_params
            raise

        if self.histogram_len != old_histogram_len:
            self._histogram_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.histogram_len * self.histogram_dtype.itemsize
            )
            self._offset_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.histogram_len * self.histogram_dtype.itemsize
            )

    @property
    def num_passes(self):
        return (self.program.key_dtype.itemsize * 8) // self.radix_bits

    @property
    def histogram_len(self):
        length = (2 ** self.radix_bits) * self.size // 2 // self.group_size
        return roundUp(length, 2 * self.group_size) # Round up for scanner

    def sort(self, cq, keys_buf, out_keys_buf,
             in_values_buf=None, out_values_buf=None, wait_for=None):
        wait_for = wait_for or []

        if self.program.value_dtype.shape != (3,):
            value_size = self.program.value_dtype.itemsize
        else:
            value_dtype = self.program.value_dtype.base
            value_size = dtype((value_dtype, 4)).itemsize
        local_keys = cl.LocalMemory(
            self.group_size * 2 * self.program.key_dtype.itemsize
        )
        local_values = cl.LocalMemory(
            self.group_size * 2 * value_size
        )
        local_count = cl.LocalMemory(
            self.group_size * 2 * self.histogram_dtype.itemsize
        )
        local_histogram = local_offset = cl.LocalMemory(
            2 ** self.radix_bits * self.histogram_dtype.itemsize
        )

        for radix_pass in range(self.num_passes):
            block_sort = self.program.kernels['block_sort'](
                cq, (self.size // 2,), (self.group_size,),
                keys_buf, local_keys, local_keys, in_values_buf, local_values, local_values,
                self._histogram_buf, local_histogram, local_count,
                self.radix_bits, radix_pass, wait_for=wait_for
            )
            copy_histogram = cl.enqueue_copy(
                cq, self._offset_buf, self._histogram_buf, wait_for=[block_sort],
                byte_count=self.histogram_len * self.histogram_dtype.itemsize
            )
            calc_scan = self.scanner.prefix_sum(cq, self._offset_buf, [copy_histogram])
            calc_scatter = self.program.kernels['scatter'](
                cq, (self.size // 2,), (self.group_size,),
                keys_buf, out_keys_buf, in_values_buf, out_values_buf,
                self._offset_buf, local_offset, self._histogram_buf, local_histogram,
                self.radix_bits, radix_pass, wait_for=[calc_scan]
            )
            fill_keys = cl.enqueue_copy(
                cq, keys_buf, out_keys_buf, wait_for=[calc_scatter],
                byte_count=self.size * self.program.key_dtype.itemsize,
            )
            wait_for = [fill_keys]
            if in_values_buf is not None and out_values_buf is not None:
                # Copying the whole buffer is faster than a 3-vector rect
                fill_values = cl.enqueue_copy(
                    cq, in_values_buf, out_values_buf, wait_for=[calc_scatter],
                    byte_count=self.size * value_size,
                )
                wait_for.append(fill_values)
        return calc_scatter
