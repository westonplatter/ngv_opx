//! Edge-case panics-don't-happen tests for the IV solver.
//!
//! Per the plan: "No panics. The solver must never panic on f64 inputs —
//! including NaN, inf, negative prices, K=0, T=0, etc. All bad inputs return
//! IvError."

use ngv_opx_core::iv::{black76_implied_vol, IvError};

fn solve(f: f64, k: f64, r: f64, t: f64, p: f64, is_call: bool) -> Result<f64, IvError> {
    black76_implied_vol(f, k, r, t, p, is_call)
}

#[test]
fn time_zero_is_error() {
    let e = solve(100.0, 100.0, 0.05, 0.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveTime);
}

#[test]
fn negative_time_is_error() {
    let e = solve(100.0, 100.0, 0.05, -1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveTime);
}

#[test]
fn nan_time_is_error() {
    let e = solve(100.0, 100.0, 0.05, f64::NAN, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveTime);
}

#[test]
fn strike_zero_is_error() {
    let e = solve(100.0, 0.0, 0.05, 1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveForward);
}

#[test]
fn forward_zero_is_error() {
    let e = solve(0.0, 100.0, 0.05, 1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveForward);
}

#[test]
fn nan_forward_is_error() {
    let e = solve(f64::NAN, 100.0, 0.05, 1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveForward);
}

#[test]
fn nan_strike_is_error() {
    let e = solve(100.0, f64::NAN, 0.05, 1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonPositiveForward);
}

#[test]
fn inf_rate_is_error() {
    let e = solve(100.0, 100.0, f64::INFINITY, 1.0, 5.0, true).unwrap_err();
    assert_eq!(e, IvError::NonFinite);
}

#[test]
fn negative_inf_rate_is_error() {
    let e = solve(100.0, 100.0, f64::NEG_INFINITY, 1.0, 5.0, true);
    assert!(matches!(e, Err(_)), "expected error for r=-inf, got {:?}", e);
}

#[test]
fn nan_price_is_error() {
    let e = solve(100.0, 100.0, 0.05, 1.0, f64::NAN, true).unwrap_err();
    assert_eq!(e, IvError::NonFinite);
}

#[test]
fn inf_price_is_error() {
    let e = solve(100.0, 100.0, 0.05, 1.0, f64::INFINITY, true);
    assert!(matches!(e, Err(_)), "expected error for p=inf, got {:?}", e);
}

#[test]
fn negative_price_is_below_intrinsic() {
    let e = solve(100.0, 100.0, 0.05, 1.0, -5.0, true).unwrap_err();
    assert_eq!(e, IvError::BelowIntrinsic);
}

#[test]
fn price_below_call_intrinsic_is_error() {
    let f = 100.0_f64;
    let k = 80.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let intrinsic = (-r * t).exp() * (f - k);
    let e = solve(f, k, r, t, intrinsic - 1.0, true).unwrap_err();
    assert_eq!(e, IvError::BelowIntrinsic);
}

#[test]
fn price_above_call_upper_bound_is_error() {
    let f = 100.0_f64;
    let k = 80.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let upper = (-r * t).exp() * f;
    let e = solve(f, k, r, t, upper + 1.0, true).unwrap_err();
    assert_eq!(e, IvError::AboveNoArbitrage);
}

#[test]
fn price_below_put_intrinsic_is_error() {
    let f = 100.0_f64;
    let k = 130.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let intrinsic = (-r * t).exp() * (k - f);
    let e = solve(f, k, r, t, intrinsic - 1.0, false).unwrap_err();
    // After parity conversion, a sub-intrinsic put becomes a sub-intrinsic call.
    assert_eq!(e, IvError::BelowIntrinsic);
}

#[test]
fn price_above_put_upper_bound_is_error() {
    let f = 100.0_f64;
    let k = 130.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let upper = (-r * t).exp() * k;
    let e = solve(f, k, r, t, upper + 1.0, false);
    assert!(matches!(e, Err(_)));
}

/// Stress: the solver must not panic on any combination of garbage inputs.
/// We don't check what error variant we get — just that we got an error
/// or a finite numeric answer, never a panic or NaN/Inf output.
#[test]
fn fuzz_garbage_inputs_no_panic() {
    let garbage = [
        f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e308, 1e308, -1.0, 0.0, 1e-300,
        1.0, 1e10, 1e100,
    ];
    for &f in &garbage {
        for &k in &garbage {
            for &r in &garbage {
                for &t in &garbage {
                    for &p in &garbage {
                        for &is_call in &[true, false] {
                            // catch_unwind in case anything still panics
                            let res = std::panic::catch_unwind(|| solve(f, k, r, t, p, is_call));
                            assert!(res.is_ok(), "panic on (f={f}, k={k}, r={r}, t={t}, p={p})");
                            if let Ok(Ok(s)) = res {
                                assert!(s.is_finite() && s > 0.0, "non-finite/non-positive σ: {}", s);
                            }
                        }
                    }
                }
            }
        }
    }
}
