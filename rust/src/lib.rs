#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use std::mem;

mod bounds;
mod morton;

enum Data {
    Leaf { id: usize },
    Node { children: (usize, usize) },
}

struct Node {
    parent: usize,
    right_edge: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct BoundingVolumeHierarchy {
    nodes: Vec<Node>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl BoundingVolumeHierarchy {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn build(&mut self, points: &[[f32; 3]]) {
        let bounds = bounds::bounds(points);
        let morton_codes = points
            .iter()
            .map(|point| morton::code(bounds::normalize(&bounds, *point)));
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "wasm32")]
    extern crate wasm_bindgen_test;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn it_works() {}
}
