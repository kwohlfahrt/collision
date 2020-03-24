#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

pub const NULL_BOUNDS: Bounds = Bounds {
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
        for (i, coord) in point.iter_mut().enumerate() {
            *coord = (*coord - self.min[i]) / (self.max[i] - self.min[i]);
        }
        point
    }

    pub fn centre(&self) -> [f32; 3] {
        let mut centre = [0.0, 0.0, 0.0];
        for (i, coord) in centre.iter_mut().enumerate() {
            *coord = self.min[i] + (self.max[i] - self.min[i]) / 2.0;
        }
        centre
    }

    pub fn intersects(&self, other: &Self) -> bool {
        (0..3).all(|i| self.max[i] > other.min[i] && self.min[i] < other.max[i])
    }
}

impl<'a> std::iter::FromIterator<Bounds> for Bounds {
    fn from_iter<I: IntoIterator<Item = Bounds>>(points: I) -> Self {
        points.into_iter().fold(NULL_BOUNDS, |mut acc, v| {
            acc.update(&v);
            acc
        })
    }
}

impl From<&([f32; 3], f32)> for Bounds {
    fn from(point: &([f32; 3], f32)) -> Self {
        let (centre, radius) = point;
        let mut bounds = NULL_BOUNDS;
        for (i, coord) in centre.iter().enumerate() {
            bounds.min[i] = coord - radius;
            bounds.max[i] = coord + radius;
        }
        bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_bounds() {
        assert_eq!([].iter().copied().collect::<Bounds>(), NULL_BOUNDS);
        assert_eq!(
            [([0.0, 0.0, 0.0], 0.0), ([0.0, -2.0, 1.0], 1.0)]
                .iter()
                .map(Bounds::from)
                .collect::<Bounds>(),
            Bounds {
                min: [-1.0, -3.0, 0.0],
                max: [1.0, 0.0, 2.0]
            }
        );
    }

    #[test]
    fn test_normalize() {
        let bounds = Bounds {
            min: [0.0, -2.0, 0.0],
            max: [4.0, 0.0, 1.0],
        };
        assert_eq!(bounds.normalize([2.0, -2.0, 1.0]), [0.5, 0.0, 1.0]);
    }

    #[test]
    fn test_intersects() {
        let bounds = Bounds {
            min: [0.0, -2.0, 0.0],
            max: [4.0, 0.0, 1.0],
        };
        assert!(bounds.intersects(&Bounds {
            min: [0.0, -2.0, 0.0],
            max: [4.0, 0.0, 1.0],
        }));
        assert!(bounds.intersects(&Bounds {
            min: [1.0, -1.0, 0.5],
            max: [10.0, 10.0, 10.0],
        }));
        assert!(!bounds.intersects(&Bounds {
            min: [5.0, -2.0, 0.0],
            max: [10.0, 0.0, 1.0],
        }));
    }
}
