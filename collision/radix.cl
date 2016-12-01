// Philippe Helluy. A portable implementation of the radix sort algorithm in OpenCL. 2011

#include "local_scan.cl"

unsigned DTYPE radix_key(unsigned DTYPE key, unsigned char radix_bits, unsigned char pass) {
    unsigned DTYPE mask = (1 << radix_bits) - 1;
    return (key >> (pass * radix_bits)) & mask;
}

// Radix-1 sort a region of 2 * local_size
unsigned int local_bin(const local unsigned DTYPE * const keys, local unsigned int * const count,
                       const unsigned char pass) {
    size_t size = get_local_size(0) * 2;

    for (size_t i = get_local_id(0); i < size; i += get_local_size(0))
        count[i] = radix_key(keys[i], 1, pass);

    barrier(CLK_LOCAL_MEM_FENCE);
    up_sweep(count);

    unsigned int sum = count[size - 1];
    count[size - 1] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    down_sweep(count);

    return size - sum;
}

void local_scatter(const local unsigned DTYPE * const keys, local unsigned DTYPE * const out_keys,
                   const local unsigned int * count, const unsigned int offset,
                   const unsigned char pass) {
    size_t size = get_local_size(0) * 2;

    for (size_t i = get_local_id(0); i < size; i += get_local_size(0)) {
        unsigned DTYPE key = radix_key(keys[i], 1, pass);
        unsigned int new_key = key ? offset + count[i] : i - count[i];

        out_keys[new_key] = keys[i];
    }
}

kernel void block_sort(global unsigned DTYPE * const keys,
                       local unsigned DTYPE * in_local_keys,
                       local unsigned DTYPE * out_local_keys,
                       local unsigned int * const count,
                       const unsigned char radix_bits, const unsigned char pass) {
    // # of elements processed by workgroup
    const size_t group_size = get_local_size(0) * 2;
    const size_t group_start = group_size * get_group_id(0);

    event_t copy;
    copy = async_work_group_copy(in_local_keys, keys + group_start, group_size, 0);
    wait_group_events(1, &copy);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned char i = 0; i < radix_bits; i++) {
        const unsigned int offset = local_bin(in_local_keys, count, radix_bits * pass + i);

        local_scatter(in_local_keys, out_local_keys, count, offset, radix_bits * pass + i);
        barrier(CLK_LOCAL_MEM_FENCE);

        local unsigned DTYPE * const tmp = in_local_keys;
        in_local_keys = out_local_keys;
        out_local_keys = tmp;
    }

    copy = async_work_group_copy(keys + group_start, in_local_keys, group_size, 0);
    wait_group_events(1, &copy);
}
