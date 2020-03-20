use std::cmp::Ordering;
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
    // assume input is in [0.0, 1.0]
    // each component gets 10 bits in 32-bit morton codes
    let scale = ((1 << 10) - 1) as f32;
    let mut fixed_value: [u16; 3] = [0, 0, 0];

    for i in 0..fixed_value.len() {
        fixed_value[i] = (value[i].max(0.0).min(1.0) * scale) as u16;
    }

    interleave(fixed_value) as u32
}

fn common_prefix_len(x: u32, y: u32) -> u32 {
    (x ^ y).leading_zeros()
}

pub fn find_split(codes: &[u32]) -> usize {
    // Assumes codes are sorted
    if codes[0] == codes[codes.len() - 1] {
        return codes.len() / 2;
    }

    let target_prefix_len = common_prefix_len(codes[0], codes[codes.len() - 1]) + 1;
    let prefix_mask = !(u32::max_value() >> target_prefix_len);
    let prefix = codes[0] & prefix_mask;

    codes
        // binary_search happens to work, but is not specified to because it may
        // return the index of any matching element, not just the greatest.
        .binary_search_by(|v| {
            if common_prefix_len(prefix, *v) >= target_prefix_len {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .unwrap_or_else(|x| x)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_code() {
        assert!(code([0.0, 0.0, 0.0]) < code([1.0, 1.0, 1.0]));
        assert!(code([0.0, 0.0, 0.0]) < code([1.0, 0.0, 0.0]));
        assert!(code([0.5, 0.0, 0.0]) < code([0.5, 0.5, 0.5]));
        assert!(code([0.5, 0.5, 0.5]) < code([0.6, 0.0, 0.0]));
    }

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len(0b00000000, 0b00000000), 32);
        assert_eq!(common_prefix_len(0b11111111, 0b11111111), 32);
        assert_eq!(common_prefix_len(0b01111111, 0b11111111), 24);
        assert_eq!(common_prefix_len(0b01111111, 0b00111111), 25);
        assert_eq!(common_prefix_len(0b10111111, 0b10011111), 26);
    }

    #[test]
    fn test_find_split() {
        let codess: &[&[u32]] = &[
            // All equal
            &[0b10111111, 0b10111111, 0b10111111, 0b10111111],
            // Nice input
            &[0b00011111, 0b00111111, 0b01011111, 0b01111111],
            // Several elements sharing > common-prefix bits (need the last)
            // Input of length such that search tries one of the middle elements first
            &[
                0b10011111, 0b10011111, 0b10011111, 0b10011111, 0b10011111, 0b10011111, 0b10011111,
                0b10011111, 0b10011111, 0b10111111, 0b10111111,
            ],
            // Random trailing bits
            &[0b10000111, 0b10011011, 0b10010111, 0b10111100, 0b10111110],
        ];
        let splits = [2, 2, 9, 3];

        for (codes, &expected_split) in codess.iter().zip(&splits) {
            let split = find_split(codes);
            assert_eq!(split, expected_split);

            if codes.first().unwrap() != codes.last().unwrap() {
                let (first, second) = codes.split_at(split);
                assert!(
                    common_prefix_len(*first.first().unwrap(), *first.last().unwrap())
                        > common_prefix_len(*codes.first().unwrap(), *codes.last().unwrap())
                );
                assert!(
                    common_prefix_len(*second.first().unwrap(), *second.last().unwrap())
                        > common_prefix_len(*codes.first().unwrap(), *codes.last().unwrap())
                );
            }
        }
    }
}
