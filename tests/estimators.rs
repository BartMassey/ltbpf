//! Estimator tests for `weighted_mean` and `map_particle`. Sanity
//! checks on the math: closed-form weighted centroid on a small linear
//! cloud, angular-mean correctness across the wrap, weighted asymmetry,
//! permutation invariance for unimodal angular inputs, and argmax for
//! `map_particle`.

use core::f32::consts::PI;
use ltbpf::{map_particle, weighted_mean, Coord};

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Shortest-arc absolute angular distance, both inputs in any range.
fn ang_dist(a: f32, b: f32) -> f32 {
    let mut d = (a - b).rem_euclid(2.0 * PI);
    if d > PI {
        d = 2.0 * PI - d;
    }
    d.abs()
}

// -------------------------------------------------------------------
// Linear 2D centroid
// -------------------------------------------------------------------

#[test]
fn weighted_mean_linear_2d_matches_analytic() {
    // Particles: (1,2), (3,4), (5,6), (7,8); weights 1, 2, 3, 4.
    // Sum_w = 10; mean_x = 50/10 = 5; mean_y = 60/10 = 6.
    let particles: Vec<(f32, f32)> = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
    let weights = vec![1.0_f32, 2.0, 3.0, 4.0];
    let m = weighted_mean(&particles, &weights, |p| {
        [Coord::Linear(p.0), Coord::Linear(p.1)]
    });
    let Coord::Linear(mx) = m[0] else { panic!() };
    let Coord::Linear(my) = m[1] else { panic!() };
    assert!(approx(mx, 5.0, 1e-5), "mean_x = {mx} expected 5.0");
    assert!(approx(my, 6.0, 1e-5), "mean_y = {my} expected 6.0");
}

// -------------------------------------------------------------------
// Angular: cluster near 0
// -------------------------------------------------------------------

#[test]
fn weighted_mean_angular_cluster_near_zero() {
    let particles = vec![-0.05_f32, -0.02, 0.0, 0.03, 0.07];
    let weights = vec![1.0_f32; 5];
    let m = weighted_mean(&particles, &weights, |p| [Coord::Angular(*p)]);
    let Coord::Angular(theta) = m[0] else {
        panic!()
    };
    // Arithmetic mean = (−0.05 − 0.02 + 0 + 0.03 + 0.07) / 5 = 0.006.
    assert!(approx(theta, 0.006, 1e-5), "got {theta}");
}

// -------------------------------------------------------------------
// Angular: cluster straddling ±π
// -------------------------------------------------------------------

#[test]
fn weighted_mean_angular_straddles_pi() {
    // {+3.0, −3.0} equal weights — both within 0.142 of ±π. Correct
    // mean is ±π (the *other* side of the circle from 0).
    let particles = vec![3.0_f32, -3.0];
    let weights = vec![1.0_f32, 1.0];
    let m = weighted_mean(&particles, &weights, |p| [Coord::Angular(*p)]);
    let Coord::Angular(theta) = m[0] else {
        panic!()
    };
    let dist_from_pi = ang_dist(theta, PI);
    let dist_from_zero = ang_dist(theta, 0.0);
    assert!(
        dist_from_pi < 0.01,
        "expected mean near ±π, got {theta} (dist {dist_from_pi})"
    );
    assert!(
        dist_from_zero > 3.0,
        "mean should NOT be near 0 (the linear centroid), got {theta}"
    );
}

// -------------------------------------------------------------------
// Angular: weighted asymmetric
// -------------------------------------------------------------------

#[test]
fn weighted_mean_angular_weighted_asymmetric() {
    // Particles 0.1 (w=1), 0.5 (w=3) — both within π of each other,
    // no wrap involved. Expected: (1·0.1 + 3·0.5) / 4 = 1.6/4 = 0.4.
    let particles = vec![0.1_f32, 0.5];
    let weights = vec![1.0_f32, 3.0];
    let m = weighted_mean(&particles, &weights, |p| [Coord::Angular(*p)]);
    let Coord::Angular(theta) = m[0] else {
        panic!()
    };
    assert!(approx(theta, 0.4, 1e-5), "got {theta}");
}

// -------------------------------------------------------------------
// Angular: permutation invariance for a unimodal cloud
// -------------------------------------------------------------------

#[test]
fn weighted_mean_angular_permutation_invariant() {
    // Five angles within ±0.6 rad of 2.8 (well within a single
    // π-radius window). Weights distinct. Try several permutations;
    // results must agree to within f32 accumulation noise.
    let base: Vec<(f32, f32)> = vec![
        (2.4, 1.0),
        (2.6, 2.0),
        (2.8, 1.5),
        (3.0, 0.5),
        (-3.05, 3.0), // wraps; same hemisphere as the others
    ];

    let take = |order: &[usize]| -> f32 {
        let particles: Vec<f32> = order.iter().map(|&i| base[i].0).collect();
        let weights: Vec<f32> = order.iter().map(|&i| base[i].1).collect();
        let m = weighted_mean(&particles, &weights, |p| [Coord::Angular(*p)]);
        let Coord::Angular(t) = m[0] else { panic!() };
        t
    };
    let reference = take(&[0, 1, 2, 3, 4]);
    let perms: [&[usize]; 4] = [
        &[4, 3, 2, 1, 0],
        &[2, 0, 4, 1, 3],
        &[1, 4, 0, 3, 2],
        &[3, 2, 1, 0, 4],
    ];
    for p in perms {
        let v = take(p);
        let d = ang_dist(v, reference);
        assert!(d < 1e-4, "permutation {p:?}: {v} vs {reference} (Δ {d})");
    }
}

// -------------------------------------------------------------------
// map_particle
// -------------------------------------------------------------------

#[test]
fn map_particle_returns_max_weight() {
    let particles = vec![10_i32, 20, 30, 40, 50];
    let weights = vec![0.1_f32, 0.4, 0.9, 0.2, 0.3];
    let p = map_particle(&particles, &weights);
    assert_eq!(p, 30, "expected particle 30 (weight 0.9), got {p}");
}

#[test]
fn map_particle_breaks_ties_by_lower_index() {
    let particles = vec!["a", "b", "c"];
    let weights = vec![0.5_f32, 0.5, 0.5];
    assert_eq!(map_particle(&particles, &weights), "a");
}
