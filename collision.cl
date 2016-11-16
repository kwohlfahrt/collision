// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-i-collision-detection-gpu/

#define CAT_HELPER(X,Y) X##Y
#define CAT(X,Y) CAT_HELPER(X,Y)

#define VTYPE CAT(DTYPE,3)

kernel void range(global unsigned int * const idxs) {
    idxs[get_global_id(0)] = get_global_id(0);
}

// Interleaves bits with 2 zero bits
// TODO: Expand for long
unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

unsigned int morton(VTYPE pos, const VTYPE min, const VTYPE max) {
    VTYPE norm_pos = (pos - min) / (max - min);
    DTYPE scale = (1 << (sizeof(unsigned int) * 8 / 3)) - 1;
    VTYPE pos_scaled = clamp(pos * scale, 0.0f, scale);

    unsigned int xx = expandBits((unsigned int) pos_scaled.x);
    unsigned int yy = expandBits((unsigned int) pos_scaled.y);
    unsigned int zz = expandBits((unsigned int) pos_scaled.z);
    return (xx << 2) + (yy << 1) + zz;
}

kernel void calculateCodes(global unsigned int * const codes,
                           const global DTYPE * const coords,
                           const global DTYPE * const range) {
    const VTYPE min = vload3(0, range);
    const VTYPE max = vload3(1, range);
    codes[get_global_id(0)] = morton(vload3(get_global_id(0), coords), min, max);
}

// TODO: Paper mentions special case of codes[idx] == codes[idx]
// Section 2 (Binary radix trees) & Section 4
char delta(const global unsigned int * const codes, const unsigned int n,
                    const unsigned int idx, const long offset) {
    if (offset < -((long) idx) || offset + idx >= n)
        return -1;
    return clz(codes[idx] ^ codes[idx + offset]);
}

char isign(const int x) {
    return (x >= 0) * 2 - 1;
}

uint2 determineRange(const global unsigned int * const codes,
                     const unsigned int n, const unsigned int idx) {
    const char d = isign(delta(codes, n, idx, 1) - delta(codes, n, idx, -1));
    const char delta_min = delta(codes, n, idx, -d);

    // Find upper bound
    unsigned int len_max = 1;
    while (delta(codes, n, idx, ((long) len_max) * d) > delta_min)
        len_max *= 2;

    unsigned int len = 0;
    unsigned int step = len_max / 2;

    // Binary search
    for (unsigned int step = len_max / 2; step > 0; step /= 2) {
        long new_len = len + step;
        if (delta(codes, n, idx, new_len*d) > delta_min)
            len = new_len;
    }

    if (d > 0)
        return (uint2)(idx, idx+len);
    else
        return (uint2)(idx-len, idx);
}

unsigned int findSplit(const global unsigned int * const codes, const uint2 range) {
    const uint2 range_codes = (uint2)(codes[range.s0], codes[range.s1]);

    if (range_codes.s0 == range_codes.s1)
        // Split range in the middle
        return (range.s0 + range.s1) / 2;

    const unsigned char prefix = clz(range_codes.s0 ^ range_codes.s1);

    // Initial guess
    unsigned int split = range.s0;
    unsigned int step = range.s1 - range.s0;

    // Binary search
    do {
        step = (step + 1) / 2;
        unsigned int new_split = split + step;

        if (new_split < range.s1) {
            unsigned char split_prefix = clz(range_codes.s0 ^ codes[new_split]);
            if (split_prefix > prefix)
                split = new_split;
        }
    } while (step > 1);

    return split;
}

struct Node {
    unsigned int parent;
    unsigned int right_edge;
    union {
        struct {
            unsigned int id;
        } leaf;
        struct {
            unsigned int children[2];
        } internal;
    };
} __attribute__((packed));

kernel void fillInternal(global struct Node * const nodes,
                         const global unsigned int * const ids) {
    const unsigned int n = get_global_size(0);
    const size_t leaf_start = (n - 1);
    nodes[leaf_start + get_global_id(0)].leaf.id = ids[get_global_id(0)];
    nodes[leaf_start + get_global_id(0)].right_edge = get_global_id(0);
}

kernel void generateBVH(const global unsigned int * const codes,
                        global struct Node * const nodes) {
    const unsigned int n = get_global_size(0) + 1;
    const size_t idx = get_global_id(0);
    const uint2 range = determineRange(codes, n, idx);
    // (N-1) internal nodes + N leaf-nodes
    const size_t leaf_start = n - 1;

    unsigned int split = findSplit(codes, range);
    const unsigned int child_a = (split == range.s0) ? leaf_start + split : split;
    split += 1;
    const unsigned int child_b = (split == range.s1) ? leaf_start + split : split;

    nodes[idx].right_edge = range.s1;
    nodes[idx].internal.children[0] = child_a;
    nodes[idx].internal.children[1] = child_b;
    nodes[child_a].parent = idx;
    nodes[child_b].parent = idx;
}

struct Bound {
    DTYPE min[3];
    DTYPE max[3];
} __attribute__((packed));

kernel void leafBounds(global struct Bound * const bounds,
                       const global DTYPE * const coords,
                       const global DTYPE * const radii,
                       const global struct Node * const nodes) {
    const unsigned int n = get_global_size(0);
    const size_t leaf_start = n - 1;
    const size_t D = 3;
    size_t node_idx = leaf_start + get_global_id(0);
    const unsigned int coords_idx = nodes[node_idx].leaf.id;
    for (size_t d = 0; d < D; d++) {
        bounds[node_idx].min[d] = coords[coords_idx*D+d] - radii[coords_idx];
        bounds[node_idx].max[d] = coords[coords_idx*D+d] + radii[coords_idx];
    }
}

// Have to execute kernel for each depth due to lack of synchronization with CL < 2
// For CL2+, remove check for depth <= level, only break on node_idx == 0 & flag < 1
kernel void internalBounds(global struct Bound * const bounds,
                           global unsigned int * const flags,
                           const global struct Node * const nodes,
                           const unsigned char level) {
    const unsigned int n = get_global_size(0);
    const size_t D = 3;
    const size_t leaf_start = n - 1;
    size_t node_idx = leaf_start + get_global_id(0);

    for (unsigned char depth = 0; depth <= level; depth++) {
        if (node_idx == 0) // Root node
            break;
        node_idx = nodes[node_idx].parent;

        // Mark node as visited (only on first visit per starting node)
        unsigned int flag = atomic_add(&flags[node_idx], depth == level);
        if (flag < 1) break; // Children not yet calculated
        else if (flag == 2) continue; // Node already calculated

        const global unsigned int * child_idxs = nodes[node_idx].internal.children;
        for (size_t d = 0; d < D; d++) {
            bounds[node_idx].min[d] = min(bounds[child_idxs[0]].min[d],
                                          bounds[child_idxs[1]].min[d]);
            bounds[node_idx].max[d] = max(bounds[child_idxs[0]].max[d],
                                          bounds[child_idxs[1]].max[d]);
        }
    }
}

bool checkOverlap(const struct Bound a, const struct Bound b) {
    const size_t D = 3;
    bool collision = 1;
    for (size_t d = 0; d < D; d++)
        collision &= a.max[d] > b.min[d] && a.min[d] < b.max[d];
    return collision;
}

bool isLeaf(const unsigned int idx, const unsigned int n) {
    return idx >= (n - 1);
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

kernel void traverse(global unsigned int * const collisions,
                     global unsigned long * const next,
                     const unsigned long n_collisions,
                     const global struct Node * const nodes,
                     const global struct Bound * const bounds) {
    unsigned int n = get_global_size(0);
    size_t leaf_start = n - 1;
    const unsigned int query_idx = get_global_id(0);
    const struct Bound query = bounds[leaf_start + query_idx];

    unsigned int stack[64];
    unsigned char stack_ptr = 0;
    stack[stack_ptr++] = UCHAR_MAX; // push NULL node (i.e. invalid node)

    // Root node
    unsigned int idx = 0;
    do {
        const unsigned int child_a = nodes[idx].internal.children[0];
        const unsigned int child_b = nodes[idx].internal.children[1];
        bool overlap_a = checkOverlap(query, bounds[child_a]);
        bool overlap_b = checkOverlap(query, bounds[child_b]);

        // Don't report self-collisions, and only in one direction
        overlap_a &= !(nodes[child_a].right_edge <= query_idx);
        overlap_b &= !(nodes[child_b].right_edge <= query_idx);

        if (overlap_a && isLeaf(child_a, n)) {
            const unsigned long collision_idx = atom_inc(next);
            if (collision_idx < n_collisions) {
                collisions[collision_idx*2+0] = nodes[leaf_start+query_idx].leaf.id;
                collisions[collision_idx*2+1] = nodes[child_a].leaf.id;
            }
        }
        if (overlap_b && isLeaf(child_b, n)) {
            const unsigned long collision_idx = atom_inc(next);
            if (collision_idx < n_collisions) {
                collisions[collision_idx*2+0] = nodes[leaf_start+query_idx].leaf.id;
                collisions[collision_idx*2+1] = nodes[child_b].leaf.id;
            }
        }
        const bool traverse_a = (overlap_a && !isLeaf(child_a, n));
        const bool traverse_b = (overlap_b && !isLeaf(child_b, n));
        if (!traverse_a && !traverse_b)
            idx = stack[--stack_ptr];
        else {
            idx = (traverse_a) ? child_a : child_b;
            if (traverse_a && traverse_b)
                stack[stack_ptr++] = child_b;
        }
    } while (idx != UCHAR_MAX);
}
