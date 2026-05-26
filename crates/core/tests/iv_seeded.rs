//! Tests for the seeded-IV entry point (U3).
//!
//! `black76_implied_vol_with_seed_f64` accepts a caller-supplied σ guess and
//! skips the SR seed. PR 2 (GPU) uses it for the f64 CPU fix-up step, where
//! the GPU's f32 answer is passed in as the seed.

use ngv_opx_core::black76::black76_price_f64;
use ngv_opx_core::iv::stefanica::sr_seed_call;
use ngv_opx_core::iv::{
    black76_implied_vol, black76_implied_vol_with_seed_f64, IvError,
};

/// (F, K, r, T, σ_true, is_call) — covers calls, puts, ITM/ATM/OTM,
/// short/long tenors, varying rates. Same shape as the canonical fixtures
/// in `iv_third_party_xcheck.rs`.
const FIXTURES: &[(f64, f64, f64, f64, f64, bool)] = &[
    (100.0, 90.0, 0.05, 0.25, 0.20, true),
    (100.0, 110.0, 0.05, 0.5, 0.25, true),
    (100.0, 100.0, 0.0, 1.0, 0.20, true),
    (75.0, 80.0, 0.04, 0.25, 0.35, false),
    (75.0, 65.0, 0.05, 0.5, 0.45, false),
    (5000.0, 5000.0, 0.045, 30.0 / 365.0, 0.15, true),
    (5000.0, 4250.0, 0.045, 60.0 / 365.0, 0.35, false),
    (1.10, 1.10, 0.02, 90.0 / 365.0, 0.08, true),
];

/// **Scenario 1**: when the caller passes the SR seed itself as `sigma_seed`,
/// both paths run the same refinement from the same starting point and
/// produce the same σ to within f64 noise.
///
/// Strict bit-identical is not achievable here because the seeded path
/// reconstructs `v_seed = sigma_seed · √T` from the caller's σ, while the
/// unseeded path uses `v_seed` directly from `sr_seed_call`. The `÷√T · √T`
/// round-trip drops a few ULPs.
#[test]
fn seeded_with_sr_value_matches_unseeded() {
    let mut worst = 0.0_f64;
    for &(f, k, r, t, sigma_true, is_call) in FIXTURES {
        let price = black76_price_f64(f, k, r, sigma_true, t, is_call);
        let discount = (-r * t).exp();
        let call_price = if is_call { price } else { price + discount * (f - k) };
        let y = (f / k).ln();
        let alpha = call_price / (k * discount);
        let v_sr = sr_seed_call(y, alpha).expect("SR returns Some");
        let sigma_seed = v_sr / t.sqrt();

        let unseeded = black76_implied_vol(f, k, r, t, price, is_call).expect("unseeded ok");
        let seeded =
            black76_implied_vol_with_seed_f64(f, k, r, t, price, is_call, sigma_seed)
                .expect("seeded ok");
        let diff = (seeded - unseeded).abs();
        if diff > worst {
            worst = diff;
        }
        assert!(
            diff < 1e-14,
            "F={}, K={}, σ_true={}, unseeded={}, seeded={}, |diff|={:.3e}",
            f, k, sigma_true, unseeded, seeded, diff
        );
    }
    eprintln!("Worst |seeded - unseeded| over fixtures: {:.2e}", worst);
}

/// **Scenario 2**: a seed within 1e-3 absolute vol-pts of truth must
/// converge to within 1e-12 of σ_true. (Plan said "converges to 1e-12 in 2
/// steps"; we run 3 steps in production but the tolerance is what we promise
/// to callers.)
#[test]
fn close_seed_recovers_sigma_to_1e_12() {
    let mut worst = 0.0_f64;
    for &(f, k, r, t, sigma_true, is_call) in FIXTURES {
        let price = black76_price_f64(f, k, r, sigma_true, t, is_call);
        for &delta in &[-1e-3_f64, -5e-4, 0.0, 5e-4, 1e-3] {
            let sigma_seed = sigma_true + delta;
            let recovered = black76_implied_vol_with_seed_f64(
                f, k, r, t, price, is_call, sigma_seed,
            )
            .expect("seeded ok");
            let err = (recovered - sigma_true).abs();
            if err > worst {
                worst = err;
            }
            assert!(
                err < 1e-12,
                "F={}, K={}, σ_true={}, seed_offset={:+.0e}, recovered={}, err={:.3e}",
                f, k, sigma_true, delta, recovered, err
            );
        }
    }
    eprintln!("Worst σ recovery from close seed: {:.2e}", worst);
}

/// **Scenario 3**: a wildly bad seed (50% off in either direction) must not
/// panic, must return a finite positive σ, and must not return NaN. We don't
/// promise accuracy — 3 Halley steps from a 50%-off seed may or may not
/// converge — but the result must be well-defined and the function must not
/// crash.
#[test]
fn wildly_bad_seed_is_well_defined_no_panic() {
    for &(f, k, r, t, sigma_true, is_call) in FIXTURES {
        let price = black76_price_f64(f, k, r, sigma_true, t, is_call);
        for &factor in &[0.5_f64, 1.5, 0.1, 5.0] {
            let sigma_seed = sigma_true * factor;
            let res = std::panic::catch_unwind(|| {
                black76_implied_vol_with_seed_f64(f, k, r, t, price, is_call, sigma_seed)
            });
            let result = res.expect("solver panicked on bad seed");
            match result {
                Ok(s) => {
                    assert!(
                        s.is_finite() && s > 0.0,
                        "non-finite/non-positive σ on bad seed: F={}, K={}, σ_true={}, factor={}, got={}",
                        f, k, sigma_true, factor, s
                    );
                }
                Err(e) => panic!(
                    "unexpected error on valid market price with bad seed: F={}, K={}, σ_true={}, factor={}: {:?}",
                    f, k, sigma_true, factor, e
                ),
            }
        }
    }
}

/// Same validation surface as the unseeded entry: bad inputs are typed errors.
#[test]
fn seeded_rejects_invalid_inputs() {
    let good_seed = 0.20_f64;
    // Bad seeds themselves
    assert_eq!(
        black76_implied_vol_with_seed_f64(100.0, 100.0, 0.05, 1.0, 10.0, true, -0.5),
        Err(IvError::NonFinite)
    );
    assert_eq!(
        black76_implied_vol_with_seed_f64(100.0, 100.0, 0.05, 1.0, 10.0, true, 0.0),
        Err(IvError::NonFinite)
    );
    assert_eq!(
        black76_implied_vol_with_seed_f64(100.0, 100.0, 0.05, 1.0, 10.0, true, f64::NAN),
        Err(IvError::NonFinite)
    );
    assert_eq!(
        black76_implied_vol_with_seed_f64(100.0, 100.0, 0.05, 1.0, 10.0, true, f64::INFINITY),
        Err(IvError::NonFinite)
    );
    // Bad surrounding inputs — same rejections as the unseeded entry
    assert_eq!(
        black76_implied_vol_with_seed_f64(100.0, 100.0, 0.05, 0.0, 10.0, true, good_seed),
        Err(IvError::NonPositiveTime)
    );
    assert_eq!(
        black76_implied_vol_with_seed_f64(0.0, 100.0, 0.05, 1.0, 10.0, true, good_seed),
        Err(IvError::NonPositiveForward)
    );
    // Below intrinsic
    let f = 100.0_f64;
    let k = 90.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let intrinsic = (-r * t).exp() * (f - k);
    assert_eq!(
        black76_implied_vol_with_seed_f64(f, k, r, t, intrinsic - 1.0, true, good_seed),
        Err(IvError::BelowIntrinsic)
    );
}
