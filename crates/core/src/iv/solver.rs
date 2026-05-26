//! Scalar Black-76 implied-volatility solver: SR seed + 3 fixed Halley steps.
//!
//! (The original plan specified Householder-3 quartic refinement with 2 steps.
//! In implementation the plan's quartic formula was found to overshoot Newton
//! on deep-ITM rows; we switched to Halley cubic, which needs 3 steps from a
//! 5% seed to reach noise floor. See `iv/householder.rs` and `todo.md` Issue 3
//! for the full story. Quartic Householder stays on the table as future work.)
//!
//! Public entry: [`black76_implied_vol`].
//!
//! Input validation, put-call parity conversion, and the no-arbitrage gates
//! all live here. The SR seed and the refinement are pure math living in
//! `stefanica` and `householder` respectively.

use crate::iv::errors::IvError;
use crate::iv::householder::refine;
use crate::iv::stefanica::sr_seed_call;

/// Black-76 implied volatility via SR seed + 3 fixed Halley steps.
///
/// All inputs in f64. Returns σ (annualized, per-year vol — i.e. v/√T).
///
/// # Errors
/// - `NonPositiveTime`     — `time_years <= 0` or non-finite
/// - `NonPositiveForward`  — `forward <= 0` or `strike <= 0`
/// - `NonFinite`           — any other non-finite input, or the seed itself was non-finite
/// - `BelowIntrinsic`      — `market_price` is below the discounted intrinsic
/// - `AboveNoArbitrage`    — `market_price` is at or above the no-arb upper bound
pub fn black76_implied_vol(
    forward: f64,
    strike: f64,
    rate: f64,
    time_years: f64,
    market_price: f64,
    is_call: bool,
) -> Result<f64, IvError> {
    if !time_years.is_finite() || time_years <= 0.0 {
        return Err(IvError::NonPositiveTime);
    }
    if !forward.is_finite() || forward <= 0.0 || !strike.is_finite() || strike <= 0.0 {
        return Err(IvError::NonPositiveForward);
    }
    if !market_price.is_finite() || !rate.is_finite() {
        return Err(IvError::NonFinite);
    }

    let discount = (-rate * time_years).exp();
    if !discount.is_finite() || discount <= 0.0 {
        return Err(IvError::NonFinite);
    }

    // Convert puts → calls in undiscounted-then-discount-applied space via
    // Black-76 parity: C - P = e^{-rT} · (F - K).
    let call_price = if is_call {
        market_price
    } else {
        market_price + discount * (forward - strike)
    };

    // No-arbitrage gates on the CALL price (after parity conversion).
    let intrinsic = (discount * (forward - strike)).max(0.0);
    let upper_bound = discount * forward;
    // Bounds tol scaled to forward — matches existing `black76_implied_vol_f64`.
    let tol = (1e-9 * forward.max(1.0)).max(1e-12);
    if call_price < intrinsic - tol {
        return Err(IvError::BelowIntrinsic);
    }
    if call_price > upper_bound + tol {
        return Err(IvError::AboveNoArbitrage);
    }
    // Strict-inside check (time value below f64 noise floor → IV indeterminate)
    if call_price <= intrinsic + tol || call_price >= upper_bound - tol {
        return Err(IvError::BelowIntrinsic);
    }

    let y = (forward / strike).ln();
    let alpha = call_price / (strike * discount);
    let v_seed = sr_seed_call(y, alpha).ok_or(IvError::NonFinite)?;

    let b_market = call_price / (discount * (forward * strike).sqrt());
    let v_final = refine(y, v_seed, b_market);
    if !v_final.is_finite() || v_final <= 0.0 {
        return Err(IvError::NonFinite);
    }

    Ok(v_final / time_years.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black76::black76_price_f64;

    /// Round-trip on a representative grid: price with σ_true, recover via
    /// the new solver, agree to 1e-10 in the bulk.
    #[test]
    fn roundtrip_bulk_grid() {
        let f = 100.0_f64;
        let r = 0.05;
        let mut worst = 0.0_f64;
        let mut samples = 0usize;
        for &k in &[60.0_f64, 80.0, 95.0, 100.0, 110.0, 130.0, 170.0] {
            for &sigma in &[0.10_f64, 0.20, 0.40, 0.80] {
                for &t in &[7.0_f64 / 365.0, 0.1, 0.5, 1.0, 3.0] {
                    let price = black76_price_f64(f, k, r, sigma, t, true);
                    let recovered = match black76_implied_vol(f, k, r, t, price, true) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let err = (recovered - sigma).abs();
                    if err > worst {
                        worst = err;
                    }
                    samples += 1;
                    assert!(
                        err < 1e-10,
                        "K={}, σ={}, T={}, recovered={}, err={}",
                        k, sigma, t, recovered, err
                    );
                }
            }
        }
        assert!(samples > 50);
        eprintln!("Worst σ recovery over {} samples: {:.2e}", samples, worst);
    }

    /// Put-call parity in IV space: same σ from a call and from the
    /// parity-derived put.
    #[test]
    fn put_call_parity_in_iv_space() {
        let f = 75.0_f64;
        let r = 0.04;
        let k = 80.0;
        let t = 0.25;
        let sigma = 0.35;
        let call_price = black76_price_f64(f, k, r, sigma, t, true);
        let put_price = black76_price_f64(f, k, r, sigma, t, false);
        let iv_call = black76_implied_vol(f, k, r, t, call_price, true).expect("call IV");
        let iv_put = black76_implied_vol(f, k, r, t, put_price, false).expect("put IV");
        assert!(
            (iv_call - iv_put).abs() < 1e-10,
            "Parity IV mismatch: call={}, put={}",
            iv_call,
            iv_put
        );
    }

    /// Bad inputs return typed errors, never panic.
    #[test]
    fn bad_inputs_are_errors_not_panics() {
        assert_eq!(
            black76_implied_vol(100.0, 100.0, 0.05, -1.0, 10.0, true),
            Err(IvError::NonPositiveTime)
        );
        assert_eq!(
            black76_implied_vol(100.0, 100.0, 0.05, 0.0, 10.0, true),
            Err(IvError::NonPositiveTime)
        );
        assert_eq!(
            black76_implied_vol(0.0, 100.0, 0.05, 1.0, 10.0, true),
            Err(IvError::NonPositiveForward)
        );
        assert_eq!(
            black76_implied_vol(100.0, 0.0, 0.05, 1.0, 10.0, true),
            Err(IvError::NonPositiveForward)
        );
        assert_eq!(
            black76_implied_vol(f64::NAN, 100.0, 0.05, 1.0, 10.0, true),
            Err(IvError::NonPositiveForward)
        );
        assert_eq!(
            black76_implied_vol(100.0, 100.0, f64::INFINITY, 1.0, 10.0, true),
            Err(IvError::NonFinite)
        );
        // Below intrinsic for an in-the-money call
        let f = 100.0_f64;
        let k = 90.0_f64;
        let r = 0.05_f64;
        let t = 0.5_f64;
        let intrinsic = (-r * t).exp() * (f - k);
        assert_eq!(
            black76_implied_vol(f, k, r, t, intrinsic - 1.0, true),
            Err(IvError::BelowIntrinsic)
        );
        // Above no-arb upper bound for a call (price > F·e^{-rT})
        let upper = (-r * t).exp() * f;
        assert_eq!(
            black76_implied_vol(f, k, r, t, upper + 1.0, true),
            Err(IvError::AboveNoArbitrage)
        );
    }
}
