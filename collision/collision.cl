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
    const DTYPE scale = (1 << (sizeof(unsigned int) * 8 / 3)) - 1;
    pos = (pos - min) / (max - min);
    pos = clamp(pos * scale, 0.0f, scale);

    unsigned int xx = expandBits((unsigned int) pos.x);
    unsigned int yy = expandBits((unsigned int) pos.y);
    unsigned int zz = expandBits((unsigned int) pos.z);
    return (xx << 2) + (yy << 1) + zz;
}

kernel void calculateCodes(global unsigned int * const codes,
                           const global VTYPE * const coords,
                           const global VTYPE * const range,
                           const unsigned int n) {
    if (get_global_id(0) >= n)
        return;
    codes[get_global_id(0)] = morton(coords[get_global_id(0)], range[0], range[1]);
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
                         const global unsigned int * const ids,
                         const unsigned int n) {
    if (get_global_id(0) >= n)
        return;
    const size_t leaf_start = (n - 1);
    nodes[leaf_start + get_global_id(0)].leaf.id = ids[get_global_id(0)];
    nodes[leaf_start + get_global_id(0)].right_edge = get_global_id(0);
}

char delta(const global unsigned int * const codes, const unsigned int n,
           const unsigned int i, const bool forward, const unsigned int offset) {
    if (forward && (i + offset) >= n)
        return -1;
    if (!forward && offset > i)
        return -1;

    const unsigned int j = forward ? i + offset : i - offset;
    if (codes[i] != codes[j])
        return clz(codes[i] ^ codes[j]);
    else
        return sizeof(unsigned int) * 8 + clz(i ^ j);
}

// http://dx.doi.org/10.2312/EGGH/HPG12/033-037
// No access at DOI link, but searchable
kernel void generateBVH(const global unsigned int * const codes,
                        global struct Node * const nodes,
                        const unsigned int n) {
    if (get_global_id(0) >= (n - 1))
        return;
    const unsigned int leaf_start = n - 1;
    const unsigned int i = get_global_id(0);

    const bool forward = delta(codes, n, i, true, 1) > delta(codes, n, i, false, 1);
    const char delta_min = delta(codes, n, i, !forward, 1);

    unsigned int len_max = 2;
    while (delta(codes, n, i, forward, len_max) > delta_min)
        len_max *= 2;

    unsigned int len = 0;
    for (unsigned int t = len_max/2; t > 0; t /= 2)
        if (delta(codes, n, i, forward, len + t) > delta_min)
            len += t;

    const unsigned int j = forward ? i + len : i - len;
    const unsigned char delta_node = delta(codes, n, i, forward, len);
    unsigned int s = 0;
    {
        unsigned int t = len;
        do {
            t = (t + 1) / 2;
            if (delta(codes, n, i, forward, s + t) > delta_node)
                s += t;
        } while (t > 1);
    }
    unsigned int gamma = forward ? i + s : i - s - 1;
    unsigned int child_a = (min(i, j) == gamma) ? leaf_start + gamma : gamma;
    unsigned int child_b = (max(i, j) == gamma + 1) ? leaf_start + gamma + 1 : gamma + 1;

    nodes[i].internal.children[0] = child_a;
    nodes[i].internal.children[1] = child_b;
    nodes[i].right_edge = max(i, j);
    nodes[child_a].parent = i;
    nodes[child_b].parent = i;
}

struct Bound {
    VTYPE min;
    VTYPE max;
};

kernel void leafBounds(global struct Bound * const bounds,
                       const global VTYPE * const coords,
                       const global DTYPE * const radii,
                       const global struct Node * const nodes,
                       const unsigned int n) {
    if (get_global_id(0) >= n)
        return;

    const size_t leaf_start = n - 1;
    size_t node_idx = leaf_start + get_global_id(0);
    const unsigned int coords_idx = nodes[node_idx].leaf.id;
    bounds[node_idx].min = coords[coords_idx] - radii[coords_idx];
    bounds[node_idx].max = coords[coords_idx] + radii[coords_idx];
}

kernel void internalBounds(global struct Bound * const bounds,
                           global unsigned int * const flags,
                           const global struct Node * const nodes) {
    const unsigned int n = get_global_size(0);
    const size_t leaf_start = n - 1;
    size_t node_idx = leaf_start + get_global_id(0);

    do {
        node_idx = nodes[node_idx].parent;
        // Mark internal node as visited, and only process after children
        if (atomic_inc(&flags[node_idx]) < 1)
            break;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        const global unsigned int * child_idxs = nodes[node_idx].internal.children;
        bounds[node_idx].min = min(bounds[child_idxs[0]].min, bounds[child_idxs[1]].min);
        bounds[node_idx].max = max(bounds[child_idxs[0]].max, bounds[child_idxs[1]].max);
    } while (node_idx != 0);
}

bool checkOverlap(const struct Bound a, const struct Bound b) {
    return all(a.max > b.min & a.min < b.max);
}

bool isLeaf(const unsigned int idx, const unsigned int n) {
    return idx >= (n - 1);
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

kernel void traverse(global unsigned int * const collisions,
                     global unsigned int * const next,
                     const unsigned int n_collisions,
                     const global struct Node * const nodes,
                     const global struct Bound * const bounds) {
    unsigned int n = get_global_size(0);
    size_t leaf_start = n - 1;
    const unsigned int query_idx = get_global_id(0);
    const struct Bound query = bounds[leaf_start + query_idx];

    unsigned int stack[64];
    unsigned char stack_ptr = 0;
    stack[stack_ptr++] = UINT_MAX; // push NULL node (i.e. invalid node)

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
            const unsigned int collision_idx = atomic_inc(next);
            if (collision_idx < n_collisions) {
                collisions[collision_idx*2+0] = nodes[leaf_start+query_idx].leaf.id;
                collisions[collision_idx*2+1] = nodes[child_a].leaf.id;
            }
        }
        if (overlap_b && isLeaf(child_b, n)) {
            const unsigned int collision_idx = atomic_inc(next);
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
    } while (idx != UINT_MAX);
}
