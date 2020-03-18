#[derive(PartialEq, Debug)]
pub struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

const NULL_BOUNDS: Bounds = Bounds {
    min: [std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY],
    max: [
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
        std::f32::NEG_INFINITY,
    ],
};

impl Bounds {
    pub fn update(&mut self, other: &Self) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    pub fn normalize(&self, point: [f32; 3]) -> [f32; 3] {
	let mut point = point;
	for i in 0..point.len() {
	    point[i] = (point[i] - self.min[i]) / (self.max[i] - self.min[i]);
	}
	point
    }

}

// Could be more generic
impl<'a> std::iter::FromIterator<&'a ([f32; 3], f32)> for Bounds {
    fn from_iter<I: IntoIterator<Item = &'a ([f32; 3], f32)>>(points: I) -> Self {
        points
            .into_iter()
            .map(Self::from)
            .fold(NULL_BOUNDS, |mut acc, v| {
                acc.update(&v);
                acc
            })
    }
}

impl From<&([f32; 3], f32)> for Bounds {
    fn from(point: &([f32; 3], f32)) -> Self {
        let (centre, radius) = point;
        let mut bounds = NULL_BOUNDS;
        for i in 0..3 {
            bounds.min[i] = centre[i] - radius;
            bounds.max[i] = centre[i] + radius;
        }
        bounds
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
    fn test_update_bounds() {
        let mut acc = NULL_BOUNDS;
        acc.update(&Bounds::from(&([0.0, 0.0, 0.0], 0.0)));
        assert_eq!(
            acc,
            Bounds {
                min: [0.0, 0.0, 0.0],
                max: [0.0, 0.0, 0.0]
            }
        );
        acc.update(&Bounds::from(&([0.0, -2.0, 1.0], 1.0)));
        assert_eq!(
            acc,
            Bounds {
                min: [-1.0, -3.0, 0.0],
                max: [1.0, 0.0, 2.0]
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_bounds() {
        assert_eq!([].iter().collect::<Bounds>(), NULL_BOUNDS);
        assert_eq!(
            [([0.0, 0.0, 0.0], 0.0), ([0.0, -2.0, 1.0], 1.0)]
                .iter()
                .collect::<Bounds>(),
            Bounds {
                min: [-1.0, -3.0, 0.0],
                max: [1.0, 0.0, 2.0]
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn test_normalize() {
	let bounds = Bounds {
	    min: [0.0, -2.0, 0.0],
	    max: [4.0, 0.0, 1.0]
	};
        assert_eq!(
            bounds.normalize([2.0, -2.0, 1.0]), [0.5, 0.0, 1.0]
        );
    }
}
