// Philippe Helluy. A portable implementation of the radix sort algorithm in OpenCL. 2011

unsigned int radix_key(unsigned int key, unsigned char radix_bits, unsigned char pass) {
    unsigned int mask = (1 << radix_bits) - 1;
    return (key >> (pass * radix_bits)) & mask;
}

kernel void histogram(global unsigned int * histograms,
                      const global unsigned int * const keys, const unsigned int n,
                      const unsigned char pass, const unsigned char radix_bits) {
    // # of keys processed by this item
    const size_t item_size = n / get_num_groups(0) / get_local_size(0);
    const size_t item_start = item_size * get_global_id(0);

    for (size_t i = item_start; i < item_start + item_size; i++){
        unsigned int key = radix_key(keys[i], radix_bits, pass);
        histograms[key * get_num_groups(0) * get_local_size(0)
                   + get_group_id(0) * get_local_size(0)
                   + get_local_id(0)] += 1;
    }
}

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

kernel void scatter(const global unsigned int * in_keys,
                    global unsigned int * out_keys,
                    const unsigned int n,
                    global unsigned int * histograms,
                    const unsigned char pass, const unsigned char radix_bits) {
    const size_t item_size = n / get_num_groups(0) / get_local_size(0);
    const size_t item_start = item_size * get_global_id(0);

    for (size_t i = item_start; i < item_start + item_size; i++) {
        unsigned int key = radix_key(in_keys[i], radix_bits, pass);
        unsigned int new_idx =
            histograms[key * get_num_groups(0) * get_local_size(0)
                       + get_group_id(0) * get_local_size(0)
                       + get_local_id(0)]++;
        out_keys[new_idx] = in_keys[i];
    }
}
