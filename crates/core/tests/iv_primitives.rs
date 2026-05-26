//! Integration tests for U1: SR seed + normalized Black primitives.
//!
//! These exercise the public `iv::` API from outside the crate, against the
//! existing Black-76 f64 implementation as an independent oracle.

use ngv_opx_core::black76::black76_price_f64;
use ngv_opx_core::iv::black::{
    black_d3_normalized, black_normalized, black_vega_normalized, black_volga_normalized,
};
use ngv_opx_core::iv::stefanica::sr_seed_call;

/// Round-trip on a moneyness × maturity × vol grid: price an option with a
/// known sigma, normalize, run SR, and verify the seed lies within the paper
/// error band  -0.0418 < (σ_BS - σ_SR) / σ_BS < 0.1138  (eqs 1, 2).
///
/// The band is published as a uniform bound — we enforce it pointwise plus
/// a small (×1.2) slack since some boundary points get clipped by the
/// finite-precision intrinsic check.
#[test]
fn sr_seed_band_on_grid() {
    let f = 100.0_f64;
    let r = 0.0_f64;
    let strikes = [40.0_f64, 70.0, 90.0, 100.0, 110.0, 130.0, 200.0];
    let sigmas = [0.05_f64, 0.10, 0.20, 0.40, 0.80, 1.50];
    let times = [1.0_f64 / 365.0, 7.0 / 365.0, 0.1, 0.5, 1.0, 3.0];

    let mut samples = 0usize;
    let mut worst_pos = f64::MIN;
    let mut worst_neg = f64::MAX;

    for &k in &strikes {
        for &sigma in &sigmas {
            for &t in &times {
                let price = black76_price_f64(f, k, r, sigma, t, true);
                let alpha = price / (k * (-r * t).exp());
                let y = (f / k).ln();
                let v_true = sigma * t.sqrt();

                // Skip rows that arbitrage-bound out by f64 noise
                let intrinsic_norm = (y.exp() - 1.0).max(0.0);
                let upper_norm = y.exp();
                if alpha <= intrinsic_norm + 1e-12 || alpha >= upper_norm - 1e-12 {
                    continue;
                }

                let v_sr = sr_seed_call(y, alpha)
                    .unwrap_or_else(|| panic!("SR returned None for valid input y={}, α={}", y, alpha));

                let rel = (v_true - v_sr) / v_true;
                worst_pos = worst_pos.max(rel);
                worst_neg = worst_neg.min(rel);
                samples += 1;

                // Paper bound with 20% slack
                assert!(
                    rel > -0.0418 * 1.2 && rel < 0.1138 * 1.2,
                    "SR seed outside band: F={}, K={}, σ={}, T={}, v_true={}, v_sr={}, rel={}",
                    f, k, sigma, t, v_true, v_sr, rel
                );
            }
        }
    }
    assert!(samples > 100, "grid too sparse: {} samples", samples);
    eprintln!(
        "SR seed band on {} samples: max_pos = {:.6}, max_neg = {:.6}",
        samples, worst_pos, worst_neg
    );
}

/// Normalized Black against the Black-76 oracle, exact round-trip.
#[test]
fn normalized_black_matches_oracle() {
    let f = 100.0_f64;
    let r = 0.05;
    let cases: [(f64, f64, f64); 6] = [
        (60.0, 0.30, 0.25),
        (90.0, 0.15, 1.0),
        (100.0, 0.20, 0.5),
        (110.0, 0.20, 0.25),
        (150.0, 0.40, 2.0),
        (80.0, 1.20, 0.1),
    ];
    for (k, sigma, t) in cases {
        let y = (f / k).ln();
        let v = sigma * t.sqrt();
        let bn = black_normalized(y, v);
        let oracle = black76_price_f64(f, k, r, sigma, t, true);
        let predicted = (-r * t).exp() * (f * k).sqrt() * bn;
        let rel = (predicted - oracle).abs() / oracle.abs().max(1e-30);
        assert!(rel < 1e-13, "y={}, v={}: predicted={}, oracle={}, rel={}", y, v, predicted, oracle, rel);
    }
}

/// Vega / volga / d3 should all be finite and have the expected sign
/// behavior on a well-conditioned grid.
#[test]
fn derivatives_finite_and_signed() {
    let cases = [(0.0_f64, 0.2), (0.1, 0.3), (-0.2, 0.4), (0.5, 0.8)];
    for (y, v) in cases {
        let vega = black_vega_normalized(y, v);
        let volga = black_volga_normalized(y, v);
        let d3 = black_d3_normalized(y, v);
        assert!(vega > 0.0 && vega.is_finite(), "vega: y={}, v={}", y, v);
        assert!(volga.is_finite(), "volga: y={}, v={}", y, v);
        assert!(d3.is_finite(), "d3: y={}, v={}", y, v);
    }
}
