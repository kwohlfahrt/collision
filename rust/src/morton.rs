// Bit hacking!
fn expand_bits(mut x: u32) -> u32 {
    x = (x | (x << 16)) & 0b11111111000000000000000011111111;
    x = (x | (x << 8)) & 0b00001111000000001111000000001111;
    x = (x | (x << 4)) & 0b00110011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
}

fn interleave(values: [u16; 3]) -> u32 {
    values
        .iter()
        .map(|v| expand_bits(*v as u32))
        .enumerate()
        .fold(0, |acc, (i, val)| acc | (val << i))
}

pub fn code(value: [f32; 3]) -> u32 {
    // each component gets 10 bits in 32-bit morton codes
    let scale = ((1 << 10) - 1) as f32;
    let mut fixed_value: [u16; 3] = [0, 0, 0];

    for i in 0..fixed_value.len() {
        fixed_value[i] = (value[i].max(0.0).min(1.0) * scale) as u16;
    }

    interleave(fixed_value)
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
    fn test_interleave() {
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
            assert_eq!(interleave(*value), i as u32);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_code() {
        assert!(code([0.0, 0.0, 0.0]) < code([1.0, 1.0, 1.0]));
        assert!(code([0.0, 0.0, 0.0]) < code([1.0, 0.0, 0.0]));
        assert!(code([0.5, 0.0, 0.0]) < code([0.5, 0.5, 0.5]));
        assert!(code([0.5, 0.5, 0.5]) < code([0.6, 0.0, 0.0]));
    }
}
