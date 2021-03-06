#include "local_scan.cl"

KEY_TYPE radix_key(KEY_TYPE key, unsigned char radix_bits, unsigned char pass) {
    KEY_TYPE mask = (1 << radix_bits) - 1;
    return (key >> (pass * radix_bits)) & mask;
}

// Radix-1 sort a region of 2 * local_size
unsigned int local_bin(const local KEY_TYPE * const keys,
                       local unsigned int * const count, const unsigned char pass) {
    size_t size = get_local_size(0) * 2;

    for (size_t i = get_local_id(0); i < size; i += get_local_size(0))
        count[i] = radix_key(keys[i], 1, pass);

    barrier(CLK_LOCAL_MEM_FENCE);
    up_sweep(count, size);

    unsigned int sum = count[size - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
        count[size - 1] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    down_sweep(count, size);

    return size - sum;
}

void local_scatter(const local KEY_TYPE * const keys,
                   local KEY_TYPE * const out_keys,
                   const local VALUE_TYPE * const values,
                   local VALUE_TYPE * const out_values,
                   const local unsigned int * count, const unsigned int offset,
                   const unsigned char pass) {
    size_t size = get_local_size(0) * 2;

    for (size_t i = get_local_id(0); i < size; i += get_local_size(0)) {
        KEY_TYPE key = radix_key(keys[i], 1, pass);
        unsigned int new_key = key ? offset + count[i] : i - count[i];

        out_keys[new_key] = keys[i];
        if (values != NULL)
            out_values[new_key] = values[i];
    }
}

kernel void block_sort(global KEY_TYPE * const keys,
                       local KEY_TYPE * in_local_keys,
                       local KEY_TYPE * out_local_keys,
                       global VALUE_TYPE * const values,
                       local VALUE_TYPE * in_local_values,
                       local VALUE_TYPE * out_local_values,
                       global unsigned int * const histogram,
                       local unsigned int * const local_histogram,
                       local unsigned int * const count,
                       const unsigned char radix_bits, const unsigned char pass) {
    // # of elements processed by workgroup
    const size_t group_size = get_local_size(0) * 2;
    const size_t group_start = group_size * get_group_id(0);
    const size_t histogram_len = 1 << radix_bits;
    event_t copy;

    copy = async_work_group_copy(in_local_keys, keys + group_start, group_size, 0);
    if (values != NULL)
        copy = async_work_group_copy(in_local_values, values + group_start, group_size, copy);
    else
        in_local_values = out_local_values = NULL;
    for (size_t i = get_local_id(0); i < histogram_len; i+= get_local_size(0))
        local_histogram[i] = 0;
    wait_group_events(1, &copy);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned char i = 0; i < radix_bits; i++) {
        const unsigned int offset = local_bin(in_local_keys, count, radix_bits * pass + i);

        local_scatter(in_local_keys, out_local_keys, in_local_values, out_local_values,
                      count, offset, radix_bits * pass + i);
        barrier(CLK_LOCAL_MEM_FENCE);

        local KEY_TYPE * const tmp = in_local_keys;
        in_local_keys = out_local_keys;
        out_local_keys = tmp;

        if (values != NULL) {
            local VALUE_TYPE * const tmp = in_local_values;
            in_local_values = out_local_values;
            out_local_values = tmp;
        }
    }

    for (size_t i = get_local_id(0); i < group_size; i += get_local_size(0))
        atomic_inc(&local_histogram[radix_key(in_local_keys[i], radix_bits, pass)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    copy = async_work_group_copy(keys + group_start, in_local_keys, group_size, 0);
    if (values != NULL)
        copy = async_work_group_copy(values + group_start, in_local_values, group_size, copy);
    copy = async_work_group_strided_copy(histogram + get_group_id(0), local_histogram,
                                         histogram_len, get_num_groups(0), copy);
    wait_group_events(1, &copy);
}

kernel void scatter(const global KEY_TYPE * const keys,
                    global KEY_TYPE * const out_keys,
                    const global VALUE_TYPE * const values,
                    global VALUE_TYPE * const out_values,
                    const global unsigned int * const offset,
                    local unsigned int * const local_offset,
                    const global unsigned int * const histogram,
                    local unsigned int * const local_histogram,
                    const unsigned char radix_bits, const unsigned char pass) {
    // # of elements processed by workgroup
    const size_t group_size = get_local_size(0) * 2;
    const size_t group_start = group_size * get_group_id(0);
    const size_t histogram_len = 1 << radix_bits;
    const size_t histogram_start = histogram_len * get_group_id(0);
    event_t copy;

    copy = async_work_group_strided_copy(local_offset, offset + get_group_id(0),
                                         histogram_len, get_num_groups(0), 0);
    copy = async_work_group_strided_copy(local_histogram, histogram + get_group_id(0),
                                         histogram_len, get_num_groups(0), copy);
    wait_group_events(1, &copy);
    barrier(CLK_LOCAL_MEM_FENCE);

    up_sweep(local_histogram, histogram_len);
    local_histogram[histogram_len - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    down_sweep(local_histogram, histogram_len);

    for (size_t i = get_local_id(0); i < group_size; i += get_local_size(0)) {
        const KEY_TYPE key = radix_key(keys[group_start+i], radix_bits, pass);
        const unsigned int new_idx = local_offset[key] + i - local_histogram[key];
        out_keys[new_idx] = keys[group_start+i];
        if (values != NULL && out_values != NULL)
            out_values[new_idx] = values[group_start+i];
    }
}
