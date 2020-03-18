extern crate collision;
extern crate criterion;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn collide(c: &mut Criterion) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);

    let mut group = c.benchmark_group("collide");
    for size in &[10, 100, 1_000, 10_000] {
        let scale = 0.333 * (-0.01 * *size as f32).exp();
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

criterion_group!(benches, collide);
criterion_main!(benches);
