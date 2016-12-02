from numpy import dtype, zeros
from pathlib import Path
import pyopencl as cl
from .misc import Program
from .scan import PrefixScanProgram, PrefixScanner

class RadixProgram(Program):
    src = Path(__file__).parent / "radix.cl"
    kernel_args = {'histogram': [None, None, None, dtype('int32'), dtype('int8'), dtype('int8')],
                   'scatter': [None, None, None, None, dtype('int32'),
                               None, None, dtype('int8'), dtype('int8')],}

    def __init__(self, ctx, value_dtype=dtype('uint32')):
        value_dtype = dtype(value_dtype)
        if value_dtype == dtype('uint32'):
            def_dtype = "int"
        elif value_dtype == dtype('uint64'):
            def_dtype = "long"
        else:
            raise ValueError("Invalid dtype: {}".format(value_dtype))
        self.value_dtype = value_dtype

        super().__init__(ctx, ["-D DTYPE={}".format(def_dtype)])

class RadixSorter:
    histogram_dtype = dtype('uint32')

    def __init__(self, ctx, size, ngroups, group_size, radix_bits=4,
                 value_dtype=dtype('uint32'), program=None, scan_program=None):
        value_dtype = dtype(value_dtype)
        self.check_size(size, ngroups, group_size, radix_bits, value_dtype)
        self.size = size
        self.ngroups = ngroups
        self.group_size = group_size
        self.radix_bits = radix_bits

        if program is None:
            program = RadixProgram(ctx, value_dtype)
        else:
            if program.context != ctx:
                raise ValueError("Sorter and program contexts must match")
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

    @staticmethod
    def check_size(size, ngroups, group_size, radix_bits, value_dtype):
        if (size % (group_size * ngroups)):
            raise ValueError("Size ({}) must be multiple of group_size x ngroups ({} x {})"
                             .format(size, group_size, ngroups))
        if (value_dtype.itemsize * 8) % radix_bits:
            raise ValueError("Radix bits ({}) must evenly divide item-size ({})"
                             .format(radix_bits, value_dtype.itemsize * 8))

    def resize(self, size=None, ngroups=None, group_size=None, radix_bits=None):
        ctx = self.program.context
        if size is None:
            size = self.size
        if ngroups is None:
            ngroups = self.ngroups
        if group_size is None:
            group_size = self.group_size
        if radix_bits is None:
            radix_bits = self.radix_bits
        old_histogram_len = self.histogram_len
        old_params = (self.size, self.ngroups, self.group_size, self.radix_bits)

        self.check_size(size, ngroups, group_size, radix_bits, self.program.value_dtype)

        self.size = size
        self.ngroups = ngroups
        self.group_size = group_size
        self.radix_bits = radix_bits

        try:
            self.scanner.resize(self.histogram_len, self.group_size)
        except:
            self.size, self.ngroups, self.group_size, self.radix_bits = old_params
            raise

        if self.histogram_len != old_histogram_len:
            self._histogram_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
                self.histogram_len * self.histogram_dtype.itemsize
            )

    @property
    def num_passes(self):
        return (self.program.value_dtype.itemsize * 8) // self.radix_bits

    @property
    def histogram_len(self):
        return (2 ** self.radix_bits) * self.ngroups * self.group_size

    def sort(self, cq, keys_buf, out_keys_buf,
             in_values_buf=None, out_values_buf=None, wait_for=None):
        wait_for = wait_for or []
        local_histogram = cl.LocalMemory((2 ** self.radix_bits) * self.group_size
                                         * self.histogram_dtype.itemsize)

        for radix_pass in range(self.num_passes):
            clear_histogram = cl.enqueue_fill_buffer(
                cq, self._histogram_buf, zeros(1, dtype='uint32'),
                0, self.histogram_len * self.histogram_dtype.itemsize
            )
            calc_hist = self.program.kernels['histogram'](
                cq, (self.ngroups,), (self.group_size,),
                self._histogram_buf, local_histogram, keys_buf,
                self.size, radix_pass, self.radix_bits,
                g_times_l=True, wait_for=[clear_histogram] + wait_for
            )
            calc_scan = self.scanner.prefix_sum(cq, self._histogram_buf, [calc_hist])
            calc_scatter = self.program.kernels['scatter'](
                cq, (self.ngroups,), (self.group_size,),
                keys_buf, out_keys_buf, in_values_buf, out_values_buf, self.size,
                self._histogram_buf, local_histogram, radix_pass, self.radix_bits,
                g_times_l=True, wait_for=[calc_scan]
            )
            fill_keys = cl.enqueue_copy(
                cq, keys_buf, out_keys_buf,
                byte_count=self.size * self.program.value_dtype.itemsize,
                wait_for=[calc_scatter]
            )
            wait_for = [fill_keys]
            if in_values_buf is not None and out_values_buf is not None:
                fill_values = cl.enqueue_copy(
                    cq, in_values_buf, out_values_buf,
                    byte_count=self.size * self.program.value_dtype.itemsize,
                    wait_for=[calc_scatter]
                )
                wait_for.append(fill_values)
        return calc_scatter
