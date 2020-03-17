// http://developer.amd.com/resources/articles-whitepapers/opencl-optimization-case-study-simple-reductions/

#define ADD(x, y) ((x) + (y))

kernel void bounds1(const global VALDTYPE * const values,
                    const unsigned long n,
                    global VALDTYPE (* const group_accs)[ACC_SIZE],
                    local VALDTYPE (* const scratch)[ACC_SIZE]) {
    VALDTYPE accumulator[ACC_SIZE] = {
        {%- for initializer in acc_inits -%} (VALDTYPE)({{initializer}}), {% endfor -%}
    };

    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        {%- for fn in acc_funcs %}
        accumulator[{{loop.index0}}] = {{ fn }}(accumulator[{{loop.index0}}], values[i]);
        {%- endfor %}
    }

    for (size_t i = 0; i < ACC_SIZE; i++)
        scratch[get_local_id(0)][i] = accumulator[i];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t o = get_local_size(0) / 2; o > 0; o /= 2) {
        if (get_local_id(0) < o) {
            {%- for fn in acc_funcs %}
            scratch[get_local_id(0)][{{loop.index0}}] = {{ fn }}(
                scratch[get_local_id(0)][{{loop.index0}}],
                scratch[get_local_id(0)+o][{{loop.index0}}]
            );
            {%- endfor %}
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
            {%- for fn in acc_funcs %}
            group_accs[get_local_id(0)][{{loop.index0}}] = {{ fn }}(
                group_accs[get_local_id(0)][{{loop.index0}}],
                group_accs[get_local_id(0)+o][{{loop.index0}}]
            );
            {%- endfor %}
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        for (size_t i = 0; i < ACC_SIZE; i++)
            output[0][i] = group_accs[0][i];
    }
}
