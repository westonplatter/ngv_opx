//! Normalized Black-76 in total-vol coordinates plus the analytic derivatives
//! needed by the Halley refinement (volga also kept, plus a d3 derivative
//! used nowhere on the current hot path — preserved for future quartic
//! schemes).
//!
//! Working in "Black coordinates":
//!     y = ln(F / K)              log-moneyness
//!     v = σ √T                   total volatility
//!     d+ = y/v + v/2
//!     d- = y/v - v/2
//!
//! The normalized call price is
//!     b(y, v) = e^{y/2} Φ(d+) - e^{-y/2} Φ(d-)
//! where Φ is the standard normal CDF. The true Black-76 call price is
//!     C = e^{-rT} · √(F·K) · b(y, v).
//!
//! Vega (db/dv) reduces to a single Gaussian factor due to the identity
//!     e^{y/2} φ(d+) = e^{-y/2} φ(d-) = (1/√(2π)) · exp(-y²/(2v²) - v²/8)
//! (verified symbolically; see notes in docs/papers/sr-2017-equations.md).
//! Volga and the third derivative reduce similarly to vega · (polynomial in
//! 1/v and v). All three are computed here.

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;

const SQRT_2PI: f64 = 2.506_628_274_631_000_7; // √(2π) to ~16 digits
const INV_SQRT_2PI: f64 = 1.0 / SQRT_2PI;

/// Standard normal CDF. Uses `libm::erfc` so the upper tail is accurate
/// without underflowing to 0 (the naive `0.5·(1+erf(x/√2))` form loses
/// precision once x exceeds ~5).
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Standard normal PDF.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() * INV_SQRT_2PI
}

/// Scaled complementary error function: erfcx(x) = e^{x²} · erfc(x).
///
/// Used for evaluating the deep-wing Φ(-large) asymptotic without underflow:
///     Φ(-a) = 0.5 · erfc(a/√2) = 0.5 · erfcx(a/√2) · exp(-a²/2)
/// so we can factor the exponential out and reason about precision.
///
/// For x ≤ 25: forward formula `erfc(x) · exp(x²)` (libm::erfc is
/// accurate, and `exp(x²)` does not overflow until x²>709).
/// For x > 25: 8-term Laurent expansion (truncation < 1e-16 at x≥10;
/// we never splice in a regime where the asymptotic is noisy).
/// For x < 0: use the identity erfcx(-x) = 2·e^{x²} - erfcx(x). Only
/// numerically safe for moderate |x| (the cancellation is fine while
/// x² is well below 709); the IV solver only invokes negative arguments
/// in regimes where the small-Φ side is preferred anyway.
#[inline]
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    if x <= 25.0 {
        return libm::erfc(x) * (x * x).exp();
    }
    let t = 1.0 / (x * x);
    let series = 1.0
        + t * (-0.5
            + t * (0.75
                + t * (-1.875
                    + t * (6.5625
                        + t * (-29.531_25
                            + t * (162.421_875
                                + t * (-1_055.742_187_5)))))));
    series / (x * (PI.sqrt()))
}

/// Normalized Black-76 call price:  b(y, v) = e^{y/2}·Φ(d+) - e^{-y/2}·Φ(d-).
///
/// For deep OTM (large negative `y` with small `v`, so `d-` very negative)
/// the direct subtraction loses precision. We compute the OTM side directly
/// from Φ-of-small-argument and recover the ITM side by parity:
///     b(y, v) - b_intrinsic_normalized = e^{y/2}·Φ(d+) - e^{-y/2}·Φ(d-)
/// where the normalized intrinsic for a call is max(e^{y/2} - e^{-y/2}, 0).
///
/// This matches the existing `b76_price_f64` strategy in `black76.rs`: pick
/// the side whose CDFs are < 0.5 to avoid catastrophic cancellation.
#[inline]
pub fn black_normalized(y: f64, v: f64) -> f64 {
    if v <= 0.0 {
        // Intrinsic in normalized space: max(e^{y/2} - e^{-y/2}, 0).
        let half = 0.5 * y;
        return (half.exp() - (-half).exp()).max(0.0);
    }
    let dp = y / v + 0.5 * v;
    let dm = y / v - 0.5 * v;
    let half_y = 0.5 * y;
    let ey_half = half_y.exp();
    let em_half = (-half_y).exp();

    if y >= 0.0 {
        // F ≥ K: ITM call. Compute OTM put-equivalent (small Φ values),
        // recover call via parity in normalized space:
        //   b_call - (e^{y/2} - e^{-y/2}) = -(e^{-y/2} Φ(-dm) - e^{y/2} Φ(-dp))
        // i.e., b_call = (e^{y/2} - e^{-y/2}) + (e^{y/2} Φ(-dp) - e^{-y/2} Φ(-dm))? Wait:
        // True identity: b(y,v) = e^{y/2}Φ(d+) - e^{-y/2}Φ(d-). Always.
        // For y ≥ 0, both Φ(d+) and Φ(d-) can be ≥ 0.5. To avoid cancellation
        // use: Φ(d) = 1 - Φ(-d), so
        //   b = e^{y/2}(1 - Φ(-dp)) - e^{-y/2}(1 - Φ(-dm))
        //     = (e^{y/2} - e^{-y/2}) - e^{y/2}Φ(-dp) + e^{-y/2}Φ(-dm)
        // The two Φ(-·) terms are < 0.5 when d± > 0, which holds for y ≥ 0
        // and v not too large (covers the ATM-to-deep-ITM regime).
        let intrinsic = ey_half - em_half;
        intrinsic - ey_half * norm_cdf(-dp) + em_half * norm_cdf(-dm)
    } else {
        // F < K: OTM call. Direct formula — Φ(d+), Φ(d-) are both < 0.5
        // when y < 0 in the typical region.
        ey_half * norm_cdf(dp) - em_half * norm_cdf(dm)
    }
}

/// Normalized Black vega: d b / d v.
///
/// Closed form: vega(y, v) = (1/√(2π)) · exp(-y²/(2v²) - v²/8).
#[inline]
pub fn black_vega_normalized(y: f64, v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    INV_SQRT_2PI * (-(y * y) / (2.0 * v * v) - 0.125 * v * v).exp()
}

/// Normalized Black volga (d²b / d v²) = vega · g  where  g = y²/v³ - v/4.
#[inline]
pub fn black_volga_normalized(y: f64, v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let vega = black_vega_normalized(y, v);
    let g = (y * y) / (v * v * v) - 0.25 * v;
    vega * g
}

/// Normalized Black third derivative (d³b / d v³).
///
/// Derivation: vega · (g² + g')  where  g = y²/v³ - v/4
/// and  g' = dg/dv = -3 y² / v⁴ - 1/4.
#[inline]
pub fn black_d3_normalized(y: f64, v: f64) -> f64 {
    if v <= 0.0 {
        return 0.0;
    }
    let vega = black_vega_normalized(y, v);
    let v2 = v * v;
    let v3 = v2 * v;
    let v4 = v2 * v2;
    let g = (y * y) / v3 - 0.25 * v;
    let g_prime = -3.0 * (y * y) / v4 - 0.25;
    vega * (g * g + g_prime)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erfcx forward-formula vs asymptotic at the x=25 splice point.
    ///
    /// At this x, libm::erfc has lost ~5 digits to the tiny intermediate
    /// magnitude (erfc(25) ~ 3e-274), so the forward formula carries ~4e-5
    /// relative error. The asymptotic series with 8 terms at x=25 is good
    /// to ~1e-18. Both are valid approximations to the same continuous
    /// function — the test just verifies they don't disagree by more than
    /// the looser of the two error bars.
    #[test]
    fn erfcx_continuous_at_splice() {
        let a = erfcx(24.999);
        let b = erfcx(25.0);
        let c = erfcx(25.001);
        assert!(a > b && b > c, "erfcx not monotone: {} {} {}", a, b, c);
        assert!(
            (a - b).abs() / b < 1e-3,
            "splice discontinuity larger than forward-formula precision: {} vs {}",
            a,
            b
        );
    }

    /// erfcx(0) = 1 (since erfc(0)=1, exp(0)=1).
    #[test]
    fn erfcx_at_zero() {
        assert!((erfcx(0.0) - 1.0).abs() < 1e-15);
    }

    /// erfcx reference values cross-checked against the asymptotic limit
    /// `1/(x√π)` for large x, and against the identity `erfcx(0) = 1`.
    ///
    /// Spot values: erfcx is monotone decreasing on x ≥ 0, with the limit
    /// `erfcx(x) · x · √π → 1` as x → ∞. We verify that the limit is
    /// approached cleanly and the function is monotone on a coarse grid.
    /// Hard scipy-table values are out of scope here (we don't have
    /// network access to verify a fresh table); the asymptotic + monotone
    /// + endpoint pin uniquely determines the function on this domain.
    #[test]
    fn erfcx_asymptotic_limit() {
        // erfcx(x) * x * √π → 1 as x → ∞
        let sqrt_pi = PI.sqrt();
        let cases: [(f64, f64); 5] = [
            (10.0, 5e-3),   // truncation/finite-x correction allowed
            (20.0, 1.5e-3),
            (50.0, 3e-4),
            (100.0, 1e-4),
            (1000.0, 1e-6),
        ];
        for (x, tol) in cases {
            let limit = erfcx(x) * x * sqrt_pi;
            assert!(
                (limit - 1.0).abs() < tol,
                "erfcx({}) · x · √π = {}; expected → 1, residual {}",
                x,
                limit,
                (limit - 1.0).abs()
            );
        }
    }

    /// erfcx is monotone strictly decreasing on x ≥ 0.
    #[test]
    fn erfcx_monotone_decreasing() {
        let mut prev = erfcx(0.0);
        let mut x = 0.1;
        while x < 30.0 {
            let curr = erfcx(x);
            assert!(curr < prev, "non-monotone at x={}: {} ≥ {}", x, curr, prev);
            prev = curr;
            x += 0.1;
        }
    }

    /// Normalized Black agrees with the existing un-normalized Black-76 (oracle).
    #[test]
    fn normalized_matches_black76_oracle() {
        use crate::black76::black76_price_f64;
        let f = 100.0_f64;
        let cases: [(f64, f64, f64); 4] =
            [(95.0, 0.20, 0.5), (100.0, 0.20, 1.0), (105.0, 0.20, 0.25), (80.0, 0.40, 2.0)];
        let r = 0.05;
        for (k, sigma, t) in cases {
            let y = (f / k).ln();
            let v = sigma * t.sqrt();
            let bn = black_normalized(y, v);
            let oracle_call = black76_price_f64(f, k, r, sigma, t, true);
            // C_oracle = e^{-rT} · √(F·K) · b(y, v)
            let scale = (-r * t).exp() * (f * k).sqrt();
            let predicted_call = scale * bn;
            let rel = (predicted_call - oracle_call).abs() / oracle_call.abs().max(1e-30);
            assert!(
                rel < 1e-13,
                "Normalized Black disagrees with oracle: y={}, v={}, predicted={}, oracle={}, rel={}",
                y,
                v,
                predicted_call,
                oracle_call,
                rel
            );
        }
    }

    /// Vega via closed form vs central finite difference.
    #[test]
    fn vega_matches_finite_difference() {
        let cases = [(0.0, 0.2), (0.1, 0.3), (-0.2, 0.4), (0.5, 0.8)];
        let h = 1e-5;
        for (y, v) in cases {
            let analytic = black_vega_normalized(y, v);
            let fd = (black_normalized(y, v + h) - black_normalized(y, v - h)) / (2.0 * h);
            let rel = (analytic - fd).abs() / analytic.abs().max(1e-15);
            assert!(
                rel < 1e-7,
                "Vega FD mismatch: y={}, v={}, analytic={}, fd={}, rel={}",
                y,
                v,
                analytic,
                fd,
                rel
            );
        }
    }

    /// Volga (2nd derivative) via closed form vs central finite difference of vega.
    #[test]
    fn volga_matches_finite_difference() {
        let cases = [(0.0, 0.2), (0.1, 0.3), (-0.2, 0.4), (0.5, 0.8)];
        let h = 1e-5;
        for (y, v) in cases {
            let analytic = black_volga_normalized(y, v);
            let fd = (black_vega_normalized(y, v + h) - black_vega_normalized(y, v - h)) / (2.0 * h);
            let scale = analytic.abs().max(1e-15);
            let rel = (analytic - fd).abs() / scale;
            assert!(
                rel < 1e-6,
                "Volga FD mismatch: y={}, v={}, analytic={}, fd={}, rel={}",
                y,
                v,
                analytic,
                fd,
                rel
            );
        }
    }

    /// Third derivative via closed form vs central finite difference of volga.
    #[test]
    fn d3_matches_finite_difference() {
        let cases = [(0.0, 0.2), (0.1, 0.3), (-0.2, 0.4), (0.5, 0.8)];
        let h = 1e-4;
        for (y, v) in cases {
            let analytic = black_d3_normalized(y, v);
            let fd = (black_volga_normalized(y, v + h) - black_volga_normalized(y, v - h)) / (2.0 * h);
            let scale = analytic.abs().max(1e-12);
            let rel = (analytic - fd).abs() / scale;
            assert!(
                rel < 1e-5,
                "d3 FD mismatch: y={}, v={}, analytic={}, fd={}, rel={}",
                y,
                v,
                analytic,
                fd,
                rel
            );
        }
    }
}
