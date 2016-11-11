// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
constant size_t D = 3;

kernel void bounds1(const global float * const coords,
                    const unsigned long n,
                    global float * const group_bounds,
                    local float * const scratch) {
    float accumulator[2][3] = {{INFINITY, INFINITY, INFINITY},
                               {0, 0, 0}};
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        for (size_t d = 0; d < D; d++) {
            accumulator[0][d] = min(accumulator[0][d], coords[i*D+d]);
            accumulator[1][d] = max(accumulator[1][d], coords[i*D+d]);
        }
    }

    for (size_t d = 0; d < D; d++) {
        scratch[d+0+2*D*get_local_id(0)] = accumulator[0][d];
        scratch[d+D+2*D*get_local_id(0)] = accumulator[1][d];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            for (size_t d = 0; d < D; d++) {
                size_t idxs[2] = {d + 2 * D * (get_local_id(0) + 0),
                                  d + 2 * D * (get_local_id(0) + o)};
                scratch[idxs[0]+0] = min(scratch[idxs[0]+0], scratch[idxs[1]+0]);
                scratch[idxs[0]+D] = max(scratch[idxs[0]+D], scratch[idxs[1]+D]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0)
        for (size_t d = 0; d < D; d++) {
            group_bounds[d+0+2*D*get_group_id(0)] = scratch[d+0];
            group_bounds[d+D+2*D*get_group_id(0)] = scratch[d+D];
        }
}

kernel void bounds2(global float * const group_bounds,
                    global float * const output) {
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            for (size_t d = 0; d < D; d++) {
                size_t idxs[2] = {d + 2 * D * (get_local_id(0) + 0),
                                  d + 2 * D * (get_local_id(0) + o)};
                group_bounds[idxs[0]+0] = min(group_bounds[idxs[0]+0],
                                              group_bounds[idxs[1]+0]);
                group_bounds[idxs[0]+D] = max(group_bounds[idxs[0]+D],
                                              group_bounds[idxs[1]+D]);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0)
        for (size_t d = 0; d < D; d++) {
            output[d+0] = group_bounds[d+0];
            output[d+D] = group_bounds[d+D];
        }
}
