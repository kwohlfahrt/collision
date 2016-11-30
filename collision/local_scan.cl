void up_sweep(local unsigned int * data) {
    for (size_t i = get_local_size(0), o = 1; i > 0; i /= 2, o *= 2) {
        if (get_local_id(0) < i) {
            size_t a = o * (2 * get_local_id(0) + 1) - 1;
            size_t b = o * (2 * get_local_id(0) + 2) - 1;
            data[b] += data[a];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void down_sweep(local unsigned int * data) {
    for (size_t i = 1, o = get_local_size(0); i < get_local_size(0) * 2; i *= 2, o /= 2) {
        if (get_local_id(0) < i) {
            size_t a = o * (2 * get_local_id(0) + 1) - 1;
            size_t b = o * (2 * get_local_id(0) + 2) - 1;

            unsigned int tmp = data[a];
            data[a] = data[b];
            data[b] += tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}