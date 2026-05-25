//! Black-76 model: option pricing on a forward / futures price F.
//!
//! Useful for index futures (e.g. SPX/ES), commodity options, and any market
//! where the natural underlying is a forward rather than a spot. Equivalent to
//! Black-Scholes with `S = F * e^(-rT)` substituted in.
//!
//! All functions use f32 to stay consistent with the existing GPU buffer types,
//! and so a future GPU shader can read the same struct layout.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

const MAX_ITERATIONS: u32 = 100;

// IV solver internals use f64. f32 inputs lose ~7 digits relative precision,
// which on a $90 forward erases time value for deep ITM/OTM at short DTE.
const F64_FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const F64_PI: f64 = std::f64::consts::PI;
const TOLERANCE_F64: f64 = 1e-9;
const MIN_VOL_F64: f64 = 1e-4;
const MAX_VOL_F64: f64 = 5.0;

fn norm_cdf_f64(x: f64) -> f64 {
    // N(x) = 0.5 * erfc(-x / sqrt(2)) — uses libm's accurate erfc so the
    // upper tail (1 - N(d) for large d) is precise instead of underflowing to 0.
    0.5 * libm::erfc(-x * F64_FRAC_1_SQRT_2)
}

fn norm_pdf_f64(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * F64_PI).sqrt()
}

fn b76_price_f64(f: f64, k: f64, r: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    // Always compute the OTM side directly (small N values, no cancellation)
    // and derive the ITM side via put-call parity:
    //   call - put = disc * (F - K)
    let sqrt_t = t.sqrt();
    let sst = sigma * sqrt_t;
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / sst;
    let d2 = d1 - sst;
    let disc = (-r * t).exp();
    let intrinsic_disc = disc * (f - k);

    // OTM call uses N(d1), N(d2) which are small when F << K (d2 < 0).
    // OTM put uses N(-d1), N(-d2) which are small when F >> K (d1 > 0).
    // Pick the formulation whose CDFs are < 0.5 to avoid catastrophic cancellation.
    if f >= k {
        // ITM call / OTM put — compute put directly, derive call via parity.
        let put = disc * (k * norm_cdf_f64(-d2) - f * norm_cdf_f64(-d1));
        if is_call { intrinsic_disc + put } else { put }
    } else {
        // OTM call / ITM put — compute call directly, derive put via parity.
        let call = disc * (f * norm_cdf_f64(d1) - k * norm_cdf_f64(d2));
        if is_call { call } else { call - intrinsic_disc }
    }
}

fn b76_vega_f64(f: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let sst = sigma * sqrt_t;
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / sst;
    let disc = (-r * t).exp();
    disc * f * sqrt_t * norm_pdf_f64(d1)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct B76Params {
    pub forward: f32,
    pub strike: f32,
    pub rate: f32,
    pub volatility: f32,
    pub time_to_maturity: f32,
    pub is_call: f32,
    _pad1: f32,
    _pad2: f32,
}

impl B76Params {
    pub fn new(
        forward: f32,
        strike: f32,
        rate: f32,
        volatility: f32,
        time_years: f32,
        is_call: bool,
    ) -> Self {
        Self {
            forward,
            strike,
            rate,
            volatility,
            time_to_maturity: time_years,
            is_call: if is_call { 1.0 } else { 0.0 },
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct B76IVParams {
    pub forward: f32,
    pub strike: f32,
    pub rate: f32,
    pub time_to_maturity: f32,
    pub market_price: f32,
    pub is_call: f32,
    _pad1: f32,
    _pad2: f32,
}

impl B76IVParams {
    pub fn new(
        forward: f32,
        strike: f32,
        rate: f32,
        time_years: f32,
        market_price: f32,
        is_call: bool,
    ) -> Self {
        Self {
            forward,
            strike,
            rate,
            time_to_maturity: time_years,
            market_price,
            is_call: if is_call { 1.0 } else { 0.0 },
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

fn norm_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 — same coefficients used elsewhere in the crate.
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn norm_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Black-76 price for a single European option on a forward (f32, for GPU
/// pricing batches). For IV-recovery accuracy use `black76_price_f64`.
pub fn black76_price_cpu(
    forward: f32,
    strike: f32,
    rate: f32,
    volatility: f32,
    time_years: f32,
    is_call: bool,
) -> f32 {
    let sqrt_t = time_years.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let d1 = ((forward / strike).ln() + 0.5 * volatility * volatility * time_years) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    let discount = (-rate * time_years).exp();
    if is_call {
        discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    } else {
        discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    }
}

/// Black-76 price (f64). Use this from Python for accuracy on deep ITM/OTM
/// short-DTE options where f32 noise on the intrinsic dominates the time value.
pub fn black76_price_f64(
    forward: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_years: f64,
    is_call: bool,
) -> f64 {
    b76_price_f64(forward, strike, rate, volatility, time_years, is_call)
}

/// Vectorized Black-76 price (f64): takes per-leg slices, returns one price
/// per option. The natural Rust array API — what Python's `black76_vectorized`
/// calls internally with the PyO3 layer stripped off.
///
/// # Panics
/// if the input slices are not all the same length.
pub fn black76_price_batch_f64(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    volatilities: &[f64],
    times: &[f64],
    is_calls: &[bool],
) -> Vec<f64> {
    let n = forwards.len();
    assert!(
        strikes.len() == n
            && rates.len() == n
            && volatilities.len() == n
            && times.len() == n
            && is_calls.len() == n,
        "all input slices must have the same length"
    );
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(b76_price_f64(
            forwards[i],
            strikes[i],
            rates[i],
            volatilities[i],
            times[i],
            is_calls[i],
        ));
    }
    out
}

/// Black-76 IV (f64 inputs/outputs). Returns -1.0 for prices outside
/// [intrinsic, disc*upper_bound] or for `time_years <= 0`.
pub fn black76_implied_vol_f64(
    forward: f64,
    strike: f64,
    rate: f64,
    time_years: f64,
    market_price: f64,
    is_call: bool,
) -> f64 {
    if time_years <= 0.0 || !time_years.is_finite() {
        return -1.0;
    }
    let discount = (-rate * time_years).exp();
    let intrinsic = if is_call {
        (discount * (forward - strike)).max(0.0)
    } else {
        (discount * (strike - forward)).max(0.0)
    };
    let upper_bound = if is_call { discount * forward } else { discount * strike };
    let bounds_tol = (TOLERANCE_F64 * forward.max(1.0)).max(1e-12);
    if market_price > upper_bound + bounds_tol {
        return -1.0;
    }
    // Time value below f64 noise floor at this scale → IV is mathematically
    // indeterminate (any σ in a wide range yields the same observable price).
    // py_vollib returns 0.0 here; we return -1.0 to match our sentinel.
    let time_value = market_price - intrinsic;
    if time_value < bounds_tol {
        return -1.0;
    }

    let rel_price_tol = 1e-10;
    let sigma_bracket_tol = 1e-7;
    let mut lo = MIN_VOL_F64;
    let mut hi = MAX_VOL_F64;
    let mut sigma = ((2.0 * F64_PI / time_years).sqrt()
        * (market_price / (discount * forward)))
        .clamp(0.05, 1.5);

    for _ in 0..MAX_ITERATIONS {
        let price = b76_price_f64(forward, strike, rate, sigma, time_years, is_call);
        let diff = price - market_price;
        let scale = market_price.abs().max(intrinsic).max(1e-30);
        if (diff / scale).abs() < rel_price_tol {
            return sigma;
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        if (hi - lo) < sigma_bracket_tol {
            return 0.5 * (hi + lo);
        }
        let vega = b76_vega_f64(forward, strike, rate, sigma, time_years);
        let mut next = if vega > 1e-14 {
            sigma - diff / vega
        } else {
            f64::NAN
        };
        if !next.is_finite() || next <= lo || next >= hi {
            next = 0.5 * (lo + hi);
        }
        if (next - sigma).abs() < 1e-12 {
            return sigma;
        }
        sigma = next;
    }
    sigma
}

/// Black-76 vega (dPrice / dσ) for a single option.
pub fn black76_vega_cpu(
    forward: f32,
    strike: f32,
    rate: f32,
    volatility: f32,
    time_years: f32,
) -> f32 {
    let sqrt_t = time_years.sqrt();
    let sigma_sqrt_t = volatility * sqrt_t;
    let d1 = ((forward / strike).ln() + 0.5 * volatility * volatility * time_years) / sigma_sqrt_t;
    let discount = (-rate * time_years).exp();
    discount * forward * sqrt_t * norm_pdf(d1)
}

pub fn black76_batch_cpu(options: &[B76Params]) -> Vec<f32> {
    options
        .iter()
        .map(|o| {
            black76_price_cpu(
                o.forward,
                o.strike,
                o.rate,
                o.volatility,
                o.time_to_maturity,
                o.is_call > 0.5,
            )
        })
        .collect()
}

/// Solve implied volatility under Black-76 with Newton-Raphson, with a
/// bisection fallback whenever vega collapses (deep OTM, very short-dated).
/// Returns -1.0 if the market price violates intrinsic bounds.
pub fn black76_implied_vol_cpu(
    forward: f32,
    strike: f32,
    rate: f32,
    time_years: f32,
    market_price: f32,
    is_call: bool,
) -> f32 {
    // At or past expiry there is no time value: IV is mathematically undefined
    // (any σ gives the same intrinsic price, vega is zero). Signal, don't guess.
    if time_years <= 0.0 || !time_years.is_finite() {
        return -1.0;
    }

    // Promote to f64 for the solve. f32 noise on a $90 forward swallows the
    // entire time value for deep ITM / OTM options at short DTE, which would
    // otherwise pin the solver to MIN_VOL and silently lie.
    let f = forward as f64;
    let k = strike as f64;
    let r = rate as f64;
    let t = time_years as f64;
    let mp = market_price as f64;

    let discount = (-r * t).exp();
    let intrinsic = if is_call {
        (discount * (f - k)).max(0.0)
    } else {
        (discount * (k - f)).max(0.0)
    };
    let upper_bound = if is_call { discount * f } else { discount * k };
    // Bounds-check uses an absolute tolerance scaled to F so that bid-below-
    // intrinsic rejection still works at SPX scale; doesn't affect the inner
    // solver's convergence test below.
    let bounds_tol = (TOLERANCE_F64 * f.max(1.0)).max(1e-12);
    if mp > upper_bound + bounds_tol {
        return -1.0;
    }
    let time_value = mp - intrinsic;
    if time_value < bounds_tol {
        return -1.0;
    }

    // Bracketed Newton-Raphson + bisection on monotone price(σ).
    //
    // Convergence is the OR of three tests, because no single one works across
    // the full price scale a PM sees on a real book:
    //   1. Relative price tolerance — handles ATM and near-ATM where prices
    //      are O($1)-O($100).
    //   2. Bracket-width on σ — handles far-OTM/short-dated where the price
    //      is sub-femto-dollar; we can't trust |diff| there but we can still
    //      localize σ to ~1e-7 vol-points by bisection.
    //   3. σ-step size — defensive against pathological non-convergence.
    let rel_price_tol = 1e-10;
    let sigma_bracket_tol = 1e-7;

    let mut lo = MIN_VOL_F64;
    let mut hi = MAX_VOL_F64;
    let mut sigma = ((2.0 * F64_PI / t).sqrt() * (mp / (discount * f))).clamp(0.05, 1.5);

    for _ in 0..MAX_ITERATIONS {
        let price = b76_price_f64(f, k, r, sigma, t, is_call);
        let diff = price - mp;
        let scale = mp.abs().max(intrinsic).max(1e-30);
        if (diff / scale).abs() < rel_price_tol {
            return sigma as f32;
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        if (hi - lo) < sigma_bracket_tol {
            return (0.5 * (hi + lo)) as f32;
        }
        let vega = b76_vega_f64(f, k, r, sigma, t);
        let mut next = if vega > 1e-14 {
            sigma - diff / vega
        } else {
            f64::NAN
        };
        if !next.is_finite() || next <= lo || next >= hi {
            next = 0.5 * (lo + hi);
        }
        if (next - sigma).abs() < 1e-12 {
            return sigma as f32;
        }
        sigma = next;
    }
    sigma as f32
}

pub fn black76_implied_vol_batch_cpu(options: &[B76IVParams]) -> Vec<f32> {
    options
        .iter()
        .map(|o| {
            black76_implied_vol_cpu(
                o.forward,
                o.strike,
                o.rate,
                o.time_to_maturity,
                o.market_price,
                o.is_call > 0.5,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Put-call parity for Black-76: C - P = e^(-rT) * (F - K)
    #[test]
    fn test_b76_put_call_parity() {
        let f = 5000.0;
        let k = 5000.0;
        let r = 0.045;
        let t = 30.0 / 365.0;
        let vol = 0.18;
        let c = black76_price_cpu(f, k, r, vol, t, true);
        let p = black76_price_cpu(f, k, r, vol, t, false);
        let lhs = c - p;
        let rhs = (-r * t).exp() * (f - k);
        assert!(
            (lhs - rhs).abs() < 0.01,
            "Parity violated: C-P={}, disc*(F-K)={}",
            lhs,
            rhs
        );
    }

    /// Round-trip across an SPX-style 5-option surface: price with a known IV,
    /// solve back, and verify recovery within tight tolerance.
    ///
    /// Cases span the kinds of strikes that fail naive solvers:
    ///   1. Front-month ATM call — happy path
    ///   2. 1-year ATM put — long-dated happy path
    ///   3. Deep OTM put, short-dated, 35% vol — left-tail / skew
    ///   4. Deep OTM call, 60d, 12% vol — flat right tail
    ///   5. Very short-dated (7d) slightly OTM put — gamma corner
    #[test]
    fn test_spx_style_iv_recovery() {
        let forward = 5000.0_f32;
        let rate = 0.045_f32;

        // (label, strike, dte_days, true_vol, is_call)
        let cases: [(&str, f32, f32, f32, bool); 5] = [
            ("ATM 30d call",         5000.0, 30.0,  0.15, true),
            ("ATM 365d put",         5000.0, 365.0, 0.18, false),
            ("Deep OTM put 60d",     4250.0, 60.0,  0.35, false),
            ("Deep OTM call 60d",    5750.0, 60.0,  0.12, true),
            ("Short 7d 5% OTM put",  4750.0, 7.0,   0.25, false),
        ];

        for (label, k, dte, true_vol, is_call) in cases {
            let t = dte / 365.0;
            let price = black76_price_cpu(forward, k, rate, true_vol, t, is_call);

            // Sanity: price must be positive and respect intrinsic + upper bound
            assert!(price > 0.0, "[{}] non-positive price {}", label, price);
            let disc = (-rate * t).exp();
            let intrinsic = if is_call {
                (disc * (forward - k)).max(0.0)
            } else {
                (disc * (k - forward)).max(0.0)
            };
            assert!(
                price >= intrinsic - 1e-3,
                "[{}] price {} below intrinsic {}",
                label,
                price,
                intrinsic
            );

            let recovered = black76_implied_vol_cpu(forward, k, rate, t, price, is_call);
            assert!(
                recovered > 0.0,
                "[{}] solver returned sentinel {}",
                label,
                recovered
            );

            // Tail/short-dated cases lose precision at f32 + SPX scale; allow
            // 0.5 vol-pt tolerance for the gamma corner, 0.1 vol-pt elsewhere.
            let tol = if dte <= 7.0 { 5e-3 } else { 1e-3 };
            let err = (recovered - true_vol).abs();
            assert!(
                err < tol,
                "[{}] IV recovery off: true={}, recovered={}, err={}, price={}",
                label,
                true_vol,
                recovered,
                err,
                price
            );
        }
    }

    /// IV solver should reject prices below intrinsic.
    #[test]
    fn test_b76_iv_rejects_below_intrinsic() {
        let f = 5000.0_f32;
        let k = 4500.0_f32;
        let r = 0.04_f32;
        let t = 0.25_f32;
        let disc = (-r * t).exp();
        let intrinsic = disc * (f - k);
        let bad_price = intrinsic - 5.0;
        let iv = black76_implied_vol_cpu(f, k, r, t, bad_price, true);
        assert_eq!(iv, -1.0, "Expected sentinel for sub-intrinsic price, got {}", iv);
    }
}
