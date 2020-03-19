extern crate collision;
extern crate criterion;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn collide(c: &mut Criterion) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);

    let mut group = c.benchmark_group("collide");
    for size in &[10, 100, 1_000, 10_000] {
        let log_size = (*size as f32).log10();
        // Ensure approximately `size` collisions
        // TODO: Find the correct relation
        let scale = 1.0 / 4.0 / (2.1_f32).powf(log_size - 1.0);

        let points = (0..*size)
            .map(|_| {
                let (centre, radius) = rng.gen::<([f32; 3], f32)>();
                (centre, radius * scale)
            })
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(points.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(points.len() as u64),
            &points,
            |b, points| b.iter(|| collision::collide(points)),
        );
    }
}

fn build(c: &mut Criterion) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);

    let mut group = c.benchmark_group("build");
    for size in &[10, 100, 1_000, 10_000] {
        let log_size = (*size as f32).log10();
        // Ensure approximately `size` collisions
        // TODO: Find the correct relation
        let scale = 1.0 / 4.0 / (2.1_f32).powf(log_size - 1.0);

        let points = (0..*size)
            .map(|_| {
                let (centre, radius) = rng.gen::<([f32; 3], f32)>();
                (centre, radius * scale)
            })
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(points.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(points.len() as u64),
            &points,
            |b, points| {
                b.iter(|| {
                    let mut bvh = collision::BoundingVolumeHierarchy::new();
                    bvh.build(&points);
                })
            },
        );
    }
}

fn collide_bvh(c: &mut Criterion) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);

    let mut group = c.benchmark_group("collide_bvh");
    for size in &[10, 100, 1_000, 10_000] {
        let log_size = (*size as f32).log10();
        // Ensure approximately `size` collisions
        // TODO: Find the correct relation
        let scale = 1.0 / 4.0 / (2.1_f32).powf(log_size - 1.0);

        let points = (0..*size)
            .map(|_| {
                let (centre, radius) = rng.gen::<([f32; 3], f32)>();
                (centre, radius * scale)
            })
            .collect::<Vec<_>>();
        let mut bvh = collision::BoundingVolumeHierarchy::new();
        bvh.build(&points);

        group.throughput(Throughput::Elements(points.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(points.len() as u64),
            &bvh,
            |b, bvh| b.iter(|| bvh.collide()),
        );
    }
}

criterion_group!(benches, collide, build, collide_bvh);
criterion_main!(benches);
