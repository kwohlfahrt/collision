// Largest possible offset = len(values) + 1

kernel void find_offsets(global const VALUE_TYPE * const values,
                         global OFFSET_TYPE * const offsets) {
    // Some risk of overflow if a == b == VALUE_TYPE_MAX
    const VALUE_TYPE a = values[get_global_id(0)], b = values[get_global_id(0) + 1];
    for (VALUE_TYPE i = a+1; i <= b; i++)
        offsets[i] = get_global_id(0) + 1;

    if (get_global_id(0) == 0)
        offsets[0] = 0;
}
