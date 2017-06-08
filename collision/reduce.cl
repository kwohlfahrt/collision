// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/

kernel void bounds1(const global DTYPE * const coords,
                    const unsigned long n,
                    global DTYPE * const group_bounds,
                    local DTYPE * const scratch) {
    DTYPE accumulator[2] = {{INFINITY, INFINITY, INFINITY},
                            {-INFINITY, -INFINITY, -INFINITY}};
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        accumulator[0] = min(accumulator[0], coords[i]);
        accumulator[1] = max(accumulator[1], coords[i]);
    }

    scratch[get_local_id(0) * 2] = accumulator[0];
    scratch[get_local_id(0) * 2 + 1] = accumulator[1];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            size_t idxs[2] = {2 * get_local_id(0), 2 * (get_local_id(0) + o)};
            scratch[idxs[0]] = min(scratch[idxs[0]], scratch[idxs[1]]);
            scratch[idxs[0] + 1] = max(scratch[idxs[0] + 1], scratch[idxs[1] + 1]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0) {
        group_bounds[2*get_group_id(0)] = scratch[0];
        group_bounds[2*get_group_id(0)+1] = scratch[1];
    }
}

kernel void bounds2(global DTYPE * const group_bounds,
                    global DTYPE * const output) {
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            size_t idxs[2] = {2 * get_local_id(0), 2 * (get_local_id(0) + o)};
            group_bounds[idxs[0]] = min(group_bounds[idxs[0]], group_bounds[idxs[1]]);
            group_bounds[idxs[0]+1] = max(group_bounds[idxs[0]+1], group_bounds[idxs[1]+1]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        output[0] = group_bounds[0];
        output[1] = group_bounds[1];
    }
}
