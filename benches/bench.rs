//! Microbenchmarks for `ltbpf`. Run with `cargo bench`.
//!
//! Uses [Divan](https://docs.rs/divan): per-iteration timing with
//! outlier rejection and a tabular summary, no unstable-toolchain
//! requirement (we set `harness = false` in `Cargo.toml` so Divan's
//! `divan::main()` runs in place of the built-in `#[bench]` harness).
//!
//! # Bench groups
//!
//! - **`step_*`** — one full `ParticleFilter::step` call on a
//!   representative model (the 2-D near-constant-velocity vehicle
//!   from `examples/vehicle.rs`). One variant per [`ResamplerKind`],
//!   fenced at the API boundary and unfenced. The fenced number
//!   reflects realistic deployment cost; the unfenced number is the
//!   upper bound LLVM can reach by fusing across iterations of the
//!   bench loop.
//!
//! - **`weighted_mean`**, **`map_particle`** — estimator microbenches
//!   across N. These run *after* a few `step` warm-up calls so the
//!   weight distribution is realistic (skewed, not uniform).
//!
//! On scalar-only targets (Cortex-M4F) none of these numbers transfer
//! directly; bench on the real target.

use divan::{black_box, Bencher};
use ltbpf::{map_particle, weighted_mean, Buffers, Coord, ParticleFilter, ResamplerKind};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn main() {
    divan::main();
}

// ---------------------------------------------------------------------------
// 2-D near-constant-velocity vehicle model. Mirrors examples/vehicle.rs.
// ---------------------------------------------------------------------------

const DT: f32 = 0.1;
const SIGMA_A: f32 = 0.5;
const SIGMA_GPS: f32 = 5.0;
const SIGMA_IMU: f32 = 0.2;

#[derive(Clone, Default)]
struct Vehicle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

struct Obs {
    gps_x: f32,
    gps_y: f32,
    imu_vx: f32,
    imu_vy: f32,
}

fn sample_prior(rng: &mut SmallRng) -> Vehicle {
    let pos = Normal::new(0.0_f32, 5.0).unwrap();
    let vel = Normal::new(0.0_f32, 2.0).unwrap();
    Vehicle {
        x: pos.sample(rng),
        y: pos.sample(rng),
        vx: vel.sample(rng),
        vy: vel.sample(rng),
    }
}

fn propagate(rng: &mut SmallRng, s: &Vehicle) -> Vehicle {
    let an = Normal::new(0.0_f32, SIGMA_A).unwrap();
    let ax = an.sample(rng);
    let ay = an.sample(rng);
    Vehicle {
        x: s.x + s.vx * DT + 0.5 * ax * DT * DT,
        y: s.y + s.vy * DT + 0.5 * ay * DT * DT,
        vx: s.vx + ax * DT,
        vy: s.vy + ay * DT,
    }
}

fn weight_update(p: &Vehicle, obs: &Obs) -> f32 {
    let r1 = (obs.gps_x - p.x) / SIGMA_GPS;
    let r2 = (obs.gps_y - p.y) / SIGMA_GPS;
    let r3 = (obs.imu_vx - p.vx) / SIGMA_IMU;
    let r4 = (obs.imu_vy - p.vy) / SIGMA_IMU;
    (-0.5 * (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4))
        .max(-50.0)
        .exp()
}

fn fixed_obs() -> Obs {
    // A plausible observation near the prior mean. Reused across
    // every step inside the bench loop — we're measuring the
    // per-step cost, not filter convergence.
    Obs {
        gps_x: 1.0,
        gps_y: 0.5,
        imu_vx: 1.0,
        imu_vy: 0.5,
    }
}

/// Allocate caller-owned buffers for an `n`-particle filter and
/// initialize the current population from the prior. Used to set up
/// each bench's local fixture.
fn make_buffers(rng: &mut SmallRng, n: usize) -> (Vec<Vehicle>, Vec<Vehicle>, Vec<f32>, Vec<u32>) {
    let p_curr: Vec<Vehicle> = (0..n).map(|_| sample_prior(rng)).collect();
    let p_next = vec![Vehicle::default(); n];
    let weights = vec![1.0_f32; n];
    let indices = vec![0_u32; n];
    (p_curr, p_next, weights, indices)
}

// ---------------------------------------------------------------------------
// Full `step()` cost, by resampler kind. The four sizes span two
// orders of magnitude — small enough to still fit on a Cortex-M-class
// target, large enough to make the asymptotics visible on a desktop.
// ---------------------------------------------------------------------------

const STEP_SIZES: &[usize] = &[300, 1_000, 3_000, 10_000];

/// Buffered resampler (the default), fenced at the API boundary.
#[divan::bench(args = STEP_SIZES)]
fn step_buffered_fenced(bencher: Bencher, n: usize) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let (mut p_curr, mut p_next, mut weights, mut indices) = make_buffers(&mut rng, n);
    let mut filter = ParticleFilter::new(
        Buffers {
            particles_curr: &mut p_curr,
            particles_next: &mut p_next,
            weights: &mut weights,
            indices: &mut indices,
        },
        propagate,
        weight_update,
    )
    .with_resampler(ResamplerKind::Buffered);
    let obs = fixed_obs();
    bencher.bench_local(|| {
        let _ = filter.step(black_box(&mut rng), black_box(&obs)).unwrap();
    });
}

/// Buffered resampler, no fences. Upper bound on host x86.
#[divan::bench(args = STEP_SIZES)]
fn step_buffered_unfenced(bencher: Bencher, n: usize) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let (mut p_curr, mut p_next, mut weights, mut indices) = make_buffers(&mut rng, n);
    let mut filter = ParticleFilter::new(
        Buffers {
            particles_curr: &mut p_curr,
            particles_next: &mut p_next,
            weights: &mut weights,
            indices: &mut indices,
        },
        propagate,
        weight_update,
    )
    .with_resampler(ResamplerKind::Buffered);
    let obs = fixed_obs();
    bencher.bench_local(|| {
        let _ = filter.step(&mut rng, &obs).unwrap();
    });
}

/// Streaming resampler, fenced at the API boundary.
#[divan::bench(args = STEP_SIZES)]
fn step_streaming_fenced(bencher: Bencher, n: usize) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let (mut p_curr, mut p_next, mut weights, mut indices) = make_buffers(&mut rng, n);
    let mut filter = ParticleFilter::new(
        Buffers {
            particles_curr: &mut p_curr,
            particles_next: &mut p_next,
            weights: &mut weights,
            indices: &mut indices,
        },
        propagate,
        weight_update,
    )
    .with_resampler(ResamplerKind::Streaming);
    let obs = fixed_obs();
    bencher.bench_local(|| {
        let _ = filter.step(black_box(&mut rng), black_box(&obs)).unwrap();
    });
}

/// Streaming resampler, no fences.
#[divan::bench(args = STEP_SIZES)]
fn step_streaming_unfenced(bencher: Bencher, n: usize) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let (mut p_curr, mut p_next, mut weights, mut indices) = make_buffers(&mut rng, n);
    let mut filter = ParticleFilter::new(
        Buffers {
            particles_curr: &mut p_curr,
            particles_next: &mut p_next,
            weights: &mut weights,
            indices: &mut indices,
        },
        propagate,
        weight_update,
    )
    .with_resampler(ResamplerKind::Streaming);
    let obs = fixed_obs();
    bencher.bench_local(|| {
        let _ = filter.step(&mut rng, &obs).unwrap();
    });
}

// ---------------------------------------------------------------------------
// Estimators. We run a handful of `step` calls first so the weights
// look realistic (a skewed posterior, not the flat 1.0 of the prior).
// ---------------------------------------------------------------------------

const EST_SIZES: &[usize] = &[300, 1_000, 3_000, 10_000];

/// Build a filter, run `warmup` steps, and return the populated
/// `(particles, weights)` pair for an estimator bench.
fn warm_population(n: usize, warmup: usize) -> (Vec<Vehicle>, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let (mut p_curr, mut p_next, mut weights, mut indices) = make_buffers(&mut rng, n);
    {
        let mut filter = ParticleFilter::new(
            Buffers {
                particles_curr: &mut p_curr,
                particles_next: &mut p_next,
                weights: &mut weights,
                indices: &mut indices,
            },
            propagate,
            weight_update,
        );
        let obs = fixed_obs();
        for _ in 0..warmup {
            // Some weight-update cycles will be deep enough into the
            // adaptive-resample regime that weights are non-uniform.
            let _ = filter.step(&mut rng, &obs).unwrap();
        }
    }
    (p_curr, weights)
}

#[divan::bench(args = EST_SIZES)]
fn weighted_mean_4d(bencher: Bencher, n: usize) {
    let (particles, weights) = warm_population(n, 5);
    bencher.bench_local(|| {
        let m = weighted_mean(black_box(&particles), black_box(&weights), |v: &Vehicle| {
            [
                Coord::Linear(v.x),
                Coord::Linear(v.y),
                Coord::Linear(v.vx),
                Coord::Linear(v.vy),
            ]
        });
        black_box(m);
    });
}

#[divan::bench(args = EST_SIZES)]
fn map_particle_vehicle(bencher: Bencher, n: usize) {
    let (particles, weights) = warm_population(n, 5);
    bencher.bench_local(|| {
        let p = map_particle(black_box(&particles), black_box(&weights));
        black_box(p);
    });
}
