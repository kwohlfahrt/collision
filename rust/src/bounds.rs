const INITIAL_BOUNDS: ([f32; 3], [f32; 3]) = (
    [std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY],
    [
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
    ],
);

pub fn update_bounds(bounds: &mut ([f32; 3], [f32; 3]), point: &[f32; 3]) {
    let (min, max) = bounds;
    for i in 0..point.len() {
        min[i] = min[i].min(point[i]);
        max[i] = max[i].max(point[i]);
    }
}

pub fn bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    return points.iter().fold(INITIAL_BOUNDS, |mut acc, v| {
        update_bounds(&mut acc, v);
        acc
    });
}

pub fn normalize(bounds: &([f32; 3], [f32; 3]), point: [f32; 3]) -> [f32; 3] {
    let (min, max) = bounds;
    let mut point = point;
    for i in 0..point.len() {
        point[i] = (point[i] - min[i]) / (max[i] - min[i]);
    }
    point
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
    fn test_update_bounds() {
        let mut acc = INITIAL_BOUNDS;
        update_bounds(&mut acc, &[0.0, 0.0, 0.0]);
        assert_eq!(acc, ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
        update_bounds(&mut acc, &[0.0, -2.0, 1.0]);

        assert_eq!(acc, ([0.0, -2.0, 0.0], [0.0, 0.0, 1.0]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_bounds() {
        assert_eq!(bounds(&[]), INITIAL_BOUNDS);
        assert_eq!(
            bounds(&[[0.0, 0.0, 0.0], [0.0, -2.0, 1.0]]),
            ([0.0, -2.0, 0.0], [0.0, 0.0, 1.0])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_normalize() {
        assert_eq!(
            normalize(&([0.0, -2.0, 0.0], [4.0, 0.0, 1.0]), [2.0, -2.0, 1.0]),
            [0.5, 0.0, 1.0]
        );
    }
}
