#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

mod bounds;
mod morton;
use bounds::Bounds;

struct Node<T> {
    // Necessary for parallel implementations TBD
    #[allow(dead_code)]
    parent: usize,
    right_edge: usize,
    bounds: Bounds,
    data: T,
}

pub struct BoundingVolumeHierarchy {
    n_internal_nodes: usize,
    nodes: Vec<Node<(usize, usize)>>,
    leaves: Vec<Node<usize>>,
}

impl BoundingVolumeHierarchy {
    pub fn new() -> Self {
        Self {
            n_internal_nodes: 0,
            nodes: Vec::new(),
            leaves: Vec::new(),
        }
    }

    fn generate_sub_hierarchy(
        &mut self,
        parent: usize,
        codes: &[u32],
        points: &[(usize, Bounds)],
    ) -> usize {
        if points.len() == 1 {
            let idx = self.leaves.len();
            self.leaves.push(Node {
                parent,
                right_edge: idx,
                bounds: points[0].1,
                data: points[0].0,
            });
            idx + self.n_internal_nodes
        } else {
            let split = morton::find_split(codes);
            let child_codes = codes.split_at(split);
            let child_points = points.split_at(split);

            let idx = self.nodes.len();
            self.nodes.push(Node {
                parent,
                right_edge: 0,
                bounds: bounds::NULL_BOUNDS,
                data: (0, 0),
            });

            let children = (
                self.generate_sub_hierarchy(idx, child_codes.0, child_points.0),
                self.generate_sub_hierarchy(idx, child_codes.1, child_points.1),
            );
            self.nodes[idx].data = children;

            let child_bounds = (*self.get_bounds(children.0), *self.get_bounds(children.1));
            let child_right_edges = (
                self.get_right_edge(children.0),
                self.get_right_edge(children.1),
            );

            self.nodes[idx].bounds.update(&child_bounds.0);
            self.nodes[idx].bounds.update(&child_bounds.1);
            // FIXME: is .0 ever greater than .1?
            self.nodes[idx].right_edge = child_right_edges.0.max(child_right_edges.1);

            idx
        }
    }

    fn is_leaf(&self, idx: usize) -> bool {
        idx >= self.n_internal_nodes
    }

    fn get_bounds(&self, idx: usize) -> &Bounds {
        if self.is_leaf(idx) {
            &self.leaves[idx - self.n_internal_nodes].bounds
        } else {
            &self.nodes[idx].bounds
        }
    }

    fn get_right_edge(&self, idx: usize) -> usize {
        if self.is_leaf(idx) {
            self.leaves[idx - self.n_internal_nodes].right_edge
        } else {
            self.nodes[idx].right_edge
        }
    }

    pub fn build(&mut self, points: &[([f32; 3], f32)]) {
        let scene_bounds = points.iter().map(Bounds::from).collect::<Bounds>();
        let mut data = points
            .iter()
            .map(Bounds::from)
            .enumerate()
            .map(|(i, bounds)| {
                let centre = scene_bounds.normalize(bounds.centre());
                ((i, bounds), morton::code(centre))
            })
            .collect::<Vec<_>>();
        data.sort_unstable_by_key(|v| v.1);

        self.n_internal_nodes = data.len() - 1;
        self.nodes.reserve(self.n_internal_nodes);
        self.leaves.reserve(data.len());

        let (points, codes): (Vec<_>, Vec<_>) = data.into_iter().unzip();

        self.generate_sub_hierarchy(0, &codes, &points);
    }

    fn collide_sub_hierarchy(
        &self,
        idx: usize,
        query_idx: usize,
        acc: &mut Vec<(usize, usize)>,
        other: &(usize, Bounds),
    ) {
        if self.get_bounds(idx).intersects(&other.1) {
            if self.is_leaf(idx) {
                let id = self.leaves[idx - self.n_internal_nodes].data;
                acc.push((other.0, id));
            } else {
                let children = self.nodes[idx].data;
                if self.get_right_edge(children.0) > query_idx {
                    self.collide_sub_hierarchy(children.0, query_idx, acc, other);
                }
                if self.get_right_edge(children.1) > query_idx {
                    self.collide_sub_hierarchy(children.1, query_idx, acc, other);
                }
            };
        };
    }

    pub fn collide(&self) -> Vec<(usize, usize)> {
        let mut collisions = Vec::new();
        for (idx, leaf) in self.leaves.iter().enumerate() {
            self.collide_sub_hierarchy(0, idx, &mut collisions, &(leaf.data, leaf.bounds));
        }
        collisions
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn collide(points: &[f32]) -> Vec<usize> {
    let mut bvh = BoundingVolumeHierarchy::new();
    let points = points
        .chunks_exact(4)
        .map(|chunk| ([chunk[0], chunk[1], chunk[2]], chunk[3]))
        .collect::<Vec<_>>();

    bvh.build(&points);
    let result = bvh.collide();
    let mut flat_result = Vec::with_capacity(result.len());
    for idxs in result {
        flat_result.extend(&[idxs.0, idxs.1]);
    }
    flat_result
}

#[cfg(not(target_arch = "wasm32"))]
pub fn collide(points: &[([f32; 3], f32)]) -> Vec<(usize, usize)> {
    let mut bvh = BoundingVolumeHierarchy::new();
    bvh.build(points);
    bvh.collide()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;

    #[cfg(target_arch = "wasm32")]
    extern crate js_sys;
    #[cfg(target_arch = "wasm32")]
    extern crate wasm_bindgen_test;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[test]
    fn test_build() {
        let mut bvh = BoundingVolumeHierarchy::new();
        let points = [
            ([0.0, 1.0, 3.0], 1.0),
            ([0.0, 1.0, 3.0], 1.0),
            ([4.0, 1.0, 8.0], 1.0),
            ([-4.0, -6.0, 3.0], 1.0),
            ([-5.0, 0.0, -1.0], 1.0),
            ([-5.0, 0.5, -0.5], 1.0),
        ];
        bvh.build(&points);

        let expected = [(0, 1), (4, 5)];
        let mut result = bvh.collide();
        result.sort();
        assert_eq!(result, expected)
    }

    fn naive_collisions(points: &[([f32; 3], f32)]) -> Vec<(usize, usize)> {
        let mut collisions = Vec::new();
        for (i, pt1) in points.iter().enumerate() {
            let bounds1 = Bounds::from(pt1);
            for (j, pt2) in points[(i + 1)..].iter().enumerate() {
                let j = j + i + 1;
                let bounds2 = Bounds::from(pt2);
                if bounds1.intersects(&bounds2) {
                    collisions.push((i, j));
                }
            }
        }
        collisions
    }

    fn sort_pair(pair: &(usize, usize)) -> (usize, usize) {
        (usize::min(pair.0, pair.1), usize::max(pair.0, pair.1))
    }

    #[test]
    fn test_naive_collisions() {
        let points = [
            ([0.0, 1.0, 3.0], 1.0),
            ([0.0, 1.0, 3.0], 1.0),
            ([4.0, 1.0, 8.0], 1.0),
            ([-4.0, -6.0, 3.0], 1.0),
            ([-5.0, 0.0, -1.0], 1.0),
            ([-5.0, 0.5, -0.5], 1.0),
        ];
        let expected = [(0, 1), (4, 5)];
        assert_eq!(naive_collisions(&points), expected);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_random_collisions() {
        use rand::{Rng, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let points = (0..100)
            .map(|_| {
                let (centre, radius) = rng.gen::<([f32; 3], f32)>();
                (centre, radius * 0.1)
            })
            .collect::<Vec<_>>();

        let mut reference = naive_collisions(&points);
        reference.sort();

        let mut result = collide(&points).iter().map(sort_pair).collect::<Vec<_>>();
        result.sort();

        assert!(result.len() > 0);
        assert!(result.len() < points.len() * points.len());
        assert_eq!(result, reference);
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    fn test_random_collisions() {
        fn random() -> f32 {
            js_sys::Math::random() as f32
        }

        let points = (0..100)
            .map(|_| ([random(), random(), random()], random() * 0.1))
            .collect::<Vec<_>>();

        let mut flat_points = Vec::new();
        for (centre, radius) in &points {
            flat_points.extend(centre);
            flat_points.push(*radius);
        }

        let mut reference = naive_collisions(&points);
        reference.sort();

        let flat_result = collide(&flat_points);
        let mut result = flat_result
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .map(|pair| sort_pair(&pair))
            .collect::<Vec<_>>();
        result.sort();

        assert!(result.len() > 0);
        assert!(result.len() < points.len() * points.len());
        assert_eq!(result, reference);
    }
}
