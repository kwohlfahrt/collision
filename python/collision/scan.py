from numpy import dtype, zeros
from pathlib import Path
import pyopencl as cl
from itertools import tee, zip_longest
from .misc import SimpleProgram, roundUp, nextPowerOf2

def ceildiv(a, b):
    return (a + b - 1) // b

class PrefixScanProgram(SimpleProgram):
    src = Path(__file__).parent / "scan.cl"
    kernel_args = {'local_scan': [None, None, None],
                   'block_scan': [None, None]}

class PrefixScanner:
    block_sums_dtype = dtype('uint32')

    def __init__(self, ctx, size, group_size, program=None):
        self.check_size(size, group_size)
        self.size = size
        self.group_size = group_size

        if program is None:
            program = PrefixScanProgram(ctx)
        elif program.context != ctx:
            raise ValueError("Scanner and program context must match")
        self.program = program

        self._block_sums_bufs = [cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            l * self.block_sums_dtype.itemsize
        ) for l in self.block_lengths]

    @staticmethod
    def check_size(size, group_size):
        if group_size != nextPowerOf2(group_size):
            raise ValueError("Group size ({}) must be a power of two".format(group_size))
        if size % (group_size * 2):
            raise ValueError("Size must be multiple of 2 * group_size ({})".format(group_size))

    def resize(self, size=None, group_size=None):
        ctx = self.program.context
        if size is None:
            size = self.size
        if group_size is None:
            group_size = self.group_size
        old_block_lengths = self.block_lengths

        self.check_size(size, group_size)
        self.size = size
        self.group_size = group_size

        self._block_sums_bufs = [
            old_buf if new_len == old_len else cl.Buffer(
            ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.HOST_NO_ACCESS,
            new_len * self.block_sums_dtype.itemsize
            ) for new_len, old_len, old_buf in
            zip_longest(self.block_lengths, old_block_lengths, self._block_sums_bufs)
            if new_len is not None
        ]

    @property
    def block_lengths(self):
        from math import ceil, log

        block_sizes = []
        size = roundUp(ceildiv(self.size, self.group_size * 2), self.group_size * 2)
        while size > self.group_size * 2:
            size = roundUp(size, self.group_size * 2)
            block_sizes.append(size)
            size = ceildiv(size, self.group_size * 2)
        block_sizes.append(nextPowerOf2(size))
        return tuple(block_sizes)

    def prefix_sum(self, cq, values_buf, wait_for=None):
        local_size = self.group_size * 2 * self.block_sums_dtype.itemsize
        e = self.program.kernels['local_scan'](
            cq, (self.size // 2,), (self.group_size,),
            values_buf, cl.LocalMemory(local_size), self._block_sums_bufs[0],
            wait_for=wait_for
        )

        bufs = tee(self._block_sums_bufs, 2)
        next(bufs[1])
        for size, in_buf, out_buf in zip(self.block_lengths, *bufs):
            e = self.program.kernels['local_scan'](
                cq, (size // 2,), (self.group_size,),
                in_buf, cl.LocalMemory(local_size), out_buf,
                wait_for=[e]
            )

        local_size = self.block_lengths[-1] * self.block_sums_dtype.itemsize
        e = self.program.kernels['local_scan'](
            cq, (1,), (self.block_lengths[-1] // 2,),
            self._block_sums_bufs[-1], cl.LocalMemory(local_size), None,
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
