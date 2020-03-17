// Bit hacking!
fn expand_bits(mut x: u32) -> u32 {
    x = (x | (x << 16)) & 0b11111111000000000000000011111111;
    x = (x | (x << 8)) & 0b00001111000000001111000000001111;
    x = (x | (x << 4)) & 0b00110011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
}

pub fn morton_number(values: &[u16; 3]) -> u32 {
    values
        .iter()
        .map(|v| expand_bits(*v as u32))
        .enumerate()
        .fold(0, |acc, (i, val)| acc | (val << i))
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
    fn test_expand_bits() {
        assert_eq!(
            expand_bits(0b1111111111),
            0b00001001001001001001001001001001
        );
        assert_eq!(
            expand_bits(0b1110111101),
            0b00001001001000001001001001000001
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_morton_number() {
        let values = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [2, 0, 0],
        ];
        for (i, value) in values.iter().enumerate() {
            assert_eq!(morton_number(&value), i as u32);
        }
    }
}
