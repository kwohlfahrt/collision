#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use std::iter::repeat;

mod bounds;
mod morton;
use bounds::Bounds;

struct Node<T> {
    parent: usize,
    right_edge: usize,
    bounds: Bounds,
    data: T,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct BoundingVolumeHierarchy {
    n_internal_nodes: usize,
    nodes: Vec<Node<(usize, usize)>>,
    leaves: Vec<Node<usize>>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl BoundingVolumeHierarchy {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    extern crate wasm_bindgen_test;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
}
