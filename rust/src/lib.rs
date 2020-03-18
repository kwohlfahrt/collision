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
    bounds: Bounds,
    data: Data,
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

    pub fn build(&mut self, points: &[([f32; 3], f32)]) {
        let bounds = points.iter().collect::<Bounds>();
        let mut data = points
            .iter()
            .enumerate()
            .map(|(i, point)| ((i, point), morton::code(bounds.normalize(point.0))))
            .collect::<Vec<_>>();
        data.sort_unstable_by_key(|v| v.1);
        self.nodes.reserve(2 * data.len() - 1);

        let (points, codes): (Vec<_>, Vec<_>) = data.into_iter().unzip();

        let generateHierarchy = |parent: usize, first: usize, last: usize| {
            //let split = morton::findSplit(codes);
        };

        generateHierarchy(0, 0, codes.len());
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
