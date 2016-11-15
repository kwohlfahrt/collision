// Philippe Helluy. A portable implementation of the radix sort algorithm in OpenCL. 2011

// Cannot pass same buffer to two pointers (at least on nVidia)
kernel void local_scan(global unsigned int * const data,
                       global unsigned int * const block_sums) {
    // # of elements processed by workgroup
    const size_t group_size = get_local_size(0) * 2;
    const size_t group_start = group_size * get_group_id(0);

    // up-sweep
    for (size_t i = group_size / 2, o = 1; i > 0; i /= 2, o *= 2) {
        if (get_local_id(0) < i) {
            size_t a = o * (2 * get_local_id(0) + 1) - 1;
            size_t b = o * (2 * get_local_id(0) + 2) - 1;
            data[b + group_start] += data[a + group_start];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        if (block_sums != NULL)
            block_sums[get_group_id(0)] = data[group_start + group_size - 1];
        data[group_start + group_size - 1] = 0;
    }

    // down-sweep
    for (size_t i = 1, o = group_size / 2; i < group_size; i *= 2, o /= 2) {
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (get_local_id(0) < i) {
            size_t a = o * (2 * get_local_id(0) + 1) - 1;
            size_t b = o * (2 * get_local_id(0) + 2) - 1;

            unsigned int tmp = data[a + group_start];
            data[a + group_start] = data[b + group_start];
            data[b + group_start] += tmp;
        }
    }
}

kernel void block_scan(global unsigned int * const data,
                       const global unsigned int * const block_sums) {
    data[get_global_id(0) * 2 + 0] += block_sums[get_group_id(0)];
    data[get_global_id(0) * 2 + 1] += block_sums[get_group_id(0)];
}
