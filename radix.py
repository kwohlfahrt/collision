from numpy import dtype, zeros
from pathlib import Path
from itertools import accumulate, chain, tee
import pyopencl as cl
from misc import Program

def ceildiv(a, b):
    return (a + b - 1) // b

def roundUp(x, base=1):
  return (x // base + bool(x % base)) * base

class PrefixScanProgram(Program):
    src = Path(__file__).parent / "radix.cl"
    kernel_args = {'local_scan': [None, None],
                   'block_scan': [None, None]}

class PrefixScanner:
    block_sums_dtype = dtype('uint32')

    def __init__(self, program, size, group_size):
        ctx = program.context
        self.program = program
        if size % (group_size * 2):
            raise ValueError("Size must be multiple of 2 * group_size ({})".format(group_size))
        self.size = size
        self.group_size = group_size

        self._block_sums_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            l * self.block_sums_dtype.itemsize
        ) for l in self.block_lengths]

    @property
    def block_lengths(self):
        from math import ceil, log

        block_sizes = []
        size = roundUp(ceildiv(self.size, self.group_size * 2), self.group_size * 2)
        while size > self.group_size * 2:
            size = roundUp(size, self.group_size * 2)
            block_sizes.append(size)
            size = ceildiv(size, self.group_size * 2)
        block_sizes.append(size)
        return tuple(block_sizes)

    def prefix_sum(self, cq, values_buf, wait_for=None):
        e = self.program.kernels['local_scan'](
            cq, (self.size // 2,), (self.group_size,),
            values_buf, self._block_sums_bufs[0],
            wait_for=wait_for
        )

        bufs = tee(self._block_sums_bufs, 2)
        next(bufs[1])
        for size, in_buf, out_buf in zip(self.block_lengths, *bufs):
            e = self.program.kernels['local_scan'](
                cq, (size // 2,), (self.group_size,),
                in_buf, out_buf,
                wait_for=[e]
            )

        e = self.program.kernels['local_scan'](
            cq, (1,), (self.block_lengths[-1] // 2,),
            self._block_sums_bufs[-1], None,
            g_times_l=True, wait_for=[e]
        )

        bufs = tee(reversed(self._block_sums_bufs), 2)
        next(bufs[1])
        for size, in_buf, out_buf in zip(reversed(self.block_lengths[:-1]), *bufs):
            e = self.program.kernels['block_scan'](
                cq, (size // 2,), (self.group_size,),
                out_buf, in_buf,
                wait_for=[e]
            )

        return self.program.kernels['block_scan'](
            cq, (self.size // 2,), (self.group_size,),
            values_buf, self._block_sums_bufs[0],
            wait_for=[e]
        )

class RadixProgram(Program):
    src = Path(__file__).parent / "radix.cl"
    kernel_args = {'histogram': [None, None, dtype('int32'), dtype('int8'), dtype('int8')],
                   'scatter': [None, None, dtype('int32'), None, dtype('int8'), dtype('int8')]}

class RadixSorter:
    value_dtype = dtype('uint32')
    histogram_dtype = dtype('uint32')

    def __init__(self, program, size, ngroups, group_size, radix_bits=4, scan_program=None):
        ctx = program.context
        self.program = program
        if (size % (group_size * ngroups)):
            raise ValueError("Size must be multiple of group_size x ngroups ({} x {})"
                             .format(group_size, ngroups))
        self.size = size
        self.ngroups = ngroups
        self.group_size = group_size
        if (self.value_dtype.itemsize * 8) % radix_bits:
            raise ValueError("Radix bits ({}) must evenly divide item-size ({})"
                             .format(radix_bits, self.value_dtype.itemsize))
        self.radix_bits = radix_bits

        if scan_program is None:
            scan_program = PrefixScanProgram(ctx)
        self.scanner = PrefixScanner(scan_program, self.histogram_len, self.group_size)

        self._histogram_buf = cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            self.histogram_len * self.histogram_dtype.itemsize
        )

    @property
    def num_passes(self):
        return (self.value_dtype.itemsize * 8) // self.radix_bits

    @property
    def histogram_len(self):
        return (2 ** self.radix_bits) * self.ngroups * self.group_size


    def sort(self, cq, values_buf, out_values_buf, wait_for=None):
        wait_for = wait_for or []
        for radix_pass in range(self.num_passes):
            clear_histogram = cl.enqueue_fill_buffer(
                cq, self._histogram_buf, zeros(1, dtype='uint32'),
                0, self.histogram_len * self.histogram_dtype.itemsize
            )
            calc_hist = self.program.kernels['histogram'](
                cq, (self.ngroups,), (self.group_size,),
                self._histogram_buf, values_buf, self.size, radix_pass, self.radix_bits,
                g_times_l=True, wait_for=[clear_histogram] + wait_for
            )
            calc_scan = self.scanner.prefix_sum(cq, self._histogram_buf, [calc_hist])
            calc_scatter = self.program.kernels['scatter'](
                cq, (self.ngroups,), (self.group_size,),
                values_buf, out_values_buf, self.size, self._histogram_buf,
                radix_pass, self.radix_bits,
                g_times_l=True, wait_for=[calc_scan]
            )
            fill_values = cl.enqueue_copy(
                cq, values_buf, out_values_buf, byte_count=self.size * self.value_dtype.itemsize,
                wait_for=[calc_scatter]
            )
            wait_for = [fill_values]
        return calc_scatter
