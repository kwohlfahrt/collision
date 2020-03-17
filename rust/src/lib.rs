#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

mod morton;
use morton::morton_number;

mod bounds;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct BoundingVolumeHierarchy {
    morton_codes: Vec<u32>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl BoundingVolumeHierarchy {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            morton_codes: Vec::new(),
        }
    }

    pub fn build(&mut self, points: &[[f32; 3]]) {
        let bounds = bounds::bounds(points);
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
