// Philippe Helluy. A portable implementation of the radix sort algorithm in OpenCL. 2011

unsigned DTYPE radix_key(unsigned DTYPE key, unsigned char radix_bits, unsigned char pass) {
    unsigned DTYPE mask = (1 << radix_bits) - 1;
    return (key >> (pass * radix_bits)) & mask;
}

kernel void histogram(global unsigned int * histograms,
                      const global unsigned DTYPE * const keys, const unsigned int n,
                      const unsigned char pass, const unsigned char radix_bits) {
    // # of keys processed by this item
    const size_t item_size = n / get_num_groups(0) / get_local_size(0);
    const size_t item_start = item_size * get_global_id(0);

    for (size_t i = item_start; i < item_start + item_size; i++){
        unsigned DTYPE key = radix_key(keys[i], radix_bits, pass);
        histograms[key * get_num_groups(0) * get_local_size(0)
                   + get_group_id(0) * get_local_size(0)
                   + get_local_id(0)] += 1;
    }
}

kernel void scatter(const global unsigned DTYPE * in_keys,
                    global unsigned DTYPE * out_keys,
                    global unsigned DTYPE * in_values,
                    global unsigned DTYPE * out_values,
                    const unsigned int n,
                    global unsigned int * histograms,
                    const unsigned char pass, const unsigned char radix_bits) {
    const size_t item_size = n / get_num_groups(0) / get_local_size(0);
    const size_t item_start = item_size * get_global_id(0);

    for (size_t i = item_start; i < item_start + item_size; i++) {
        unsigned DTYPE key = radix_key(in_keys[i], radix_bits, pass);
        unsigned int new_idx =
            histograms[key * get_num_groups(0) * get_local_size(0)
                       + get_group_id(0) * get_local_size(0)
                       + get_local_id(0)]++;
        out_keys[new_idx] = in_keys[i];
        if (out_values != NULL && in_values != NULL)
            out_values[new_idx] = in_values[i];
    }
}
