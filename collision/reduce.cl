// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/

kernel void bounds1(const global VALDTYPE * const values,
                    const unsigned long n,
                    global VALDTYPE (* const group_accs)[ACC_SIZE],
                    local VALDTYPE (* const scratch)[ACC_SIZE]) {
    VALDTYPE accumulator[ACC_SIZE] = ACC_INIT;

    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        accumulator[0] = min(accumulator[0], values[i]);
        accumulator[1] = max(accumulator[1], values[i]);
    }

    for (size_t i = 0; i < ACC_SIZE; i++)
        scratch[get_local_id(0)][i] = accumulator[i];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            size_t idxs[2] = {get_local_id(0), get_local_id(0) + o};
            scratch[idxs[0]][0] = min(scratch[idxs[0]][0], scratch[idxs[1]][0]);
            scratch[idxs[0]][1] = max(scratch[idxs[0]][1], scratch[idxs[1]][1]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0) {
        for (size_t i = 0; i < ACC_SIZE; i++)
            group_accs[get_group_id(0)][i] = scratch[0][i];
    }
}

kernel void bounds2(global VALDTYPE (* const group_accs)[ACC_SIZE],
                    global VALDTYPE (* const output)[ACC_SIZE]) {
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            size_t idxs[2] = {get_local_id(0), get_local_id(0) + o};
            group_accs[idxs[0]][0] = min(group_accs[idxs[0]][0], group_accs[idxs[1]][0]);
            group_accs[idxs[0]][1] = max(group_accs[idxs[0]][1], group_accs[idxs[1]][1]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        for (size_t i = 0; i < ACC_SIZE; i++)
            output[0][i] = group_accs[0][i];
    }
}
