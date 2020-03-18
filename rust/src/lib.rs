#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

mod bounds;
mod morton;
use bounds::Bounds;

enum Data {
    Leaf { id: usize },
    Node { children: (usize, usize) },
}

struct Node {
    parent: usize,
    right_edge: usize,
    children: (usize, usize),
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct BoundingVolumeHierarchy {
    nodes: Vec<Node>,
    points: Vec<[f32; 3]>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl BoundingVolumeHierarchy {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            points: Vec::new(),
        }
    }

    pub fn build(&mut self, points: &[[f32; 3]]) {
        let bounds: Bounds = unimplemented!(); //Bounds::from(points.iter());
        let morton_codes = points
            .iter()
            .map(|point| morton::code(bounds.normalize(*point)))
            .collect::<Vec<_>>();
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
