#[cfg(target_arch="wasm32")]
extern crate wasm_bindgen;
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;


#[cfg_attr(target_arch="wasm32", wasm_bindgen)]
pub fn add(x: f64, y: f64) -> f64 {
    x + y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch="wasm32")]
    extern crate wasm_bindgen_test;
    #[cfg(target_arch="wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg_attr(target_arch="wasm32", wasm_bindgen_test)]
    #[test]
    fn it_works() {
        assert_eq!(add(2.0, 2.0), 4.0);
    }
}
