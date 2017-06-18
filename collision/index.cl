kernel void gather(global const VALUE_TYPE * const in_values,
                   global const INDEX_TYPE * const indices,
                   global VALUE_TYPE * const out_values) {
    const size_t i = get_global_id(0);
    out_values[i] = in_values[indices[i]];
}

kernel void scatter(global const VALUE_TYPE * const in_values,
                    global const INDEX_TYPE * const indices,
                    global VALUE_TYPE * const out_values) {
    const size_t i = get_global_id(0);
    out_values[indices[i]] = in_values[i];
}
