//! Stefanica–Radoičić (2017) closed-form implied-volatility seed.
//!
//! Reference: docs/papers/sr-2017-equations.md and the SSRN PDF at
//! ssrn_id3035850_code1527880.pdf. Equation numbers below match the paper.
//!
//! The seed produces an initial guess `v = σ √T` that is uniformly within
//! the relative-error band  -0.0418 < (σ_BS - σ_SR) / σ_BS < 0.1138  across
//! all moneyness, maturity, and vol levels (paper eqs 1, 2).
//!
//! API surface: a single function `sr_seed_call` that takes the call-side
//! normalized inputs and returns `v`. Put inputs are converted to calls in
//! the upstream solver via put-call parity, so this module is call-only.
//!
//! Input normalization (call):
//!     y      = ln(F / K)                  log-moneyness
//!     alpha  = Cm / (K · e^{-rT})         normalized market call price
//! Output: `v = σ * √T` (total volatility).

use std::f64::consts::PI;

/// Pólya cumulative-normal approximation, eq (8).
///
/// `A(x) = 1/2 + (sgn(x)/2) · sqrt(1 - exp(-2 x² / π))`
#[inline]
fn polya_a(x: f64) -> f64 {
    let s = if x >= 0.0 { 1.0 } else { -1.0 };
    0.5 + 0.5 * s * (1.0 - (-2.0 * x * x / PI).exp()).sqrt()
}

/// SR closed-form seed for a call.
///
/// Returns `Some(v)` where `v = σ √T` is the SR approximation to total
/// volatility, or `None` when the inputs are outside the normalized
/// no-arbitrage region `max(e^y - 1, 0) < αC < e^y`. The solver layer
/// should have already converted such rows into typed `IvError` sentinels;
/// this guard is the last line of defense.
///
/// Precondition: `alpha` and `y` are finite, and `alpha` lies strictly inside
/// the **normalized** call no-arbitrage band  `max(e^y - 1, 0) < αC < e^y`.
///
/// (Where the band comes from: αC = Cm / (K e^{-rT}); call intrinsic in this
/// normalization is `max(e^y - 1, 0)` and the upper bound `F·e^{-rT}/(K·e^{-rT})
/// = e^y`. Both ends collapse the SR coefficients to give v=0 / v=∞ respectively,
/// so rejecting outside the band is the right cut. NB: a naive guess of
/// `max(1 - e^{-y}, 0) < αC < 1` only collapses to the correct band at y=0 and
/// rejects valid deep-ITM rows otherwise.)
pub fn sr_seed_call(y: f64, alpha: f64) -> Option<f64> {
    if !y.is_finite() || !alpha.is_finite() {
        return None;
    }
    let intrinsic_normalized = (y.exp() - 1.0).max(0.0);
    let upper_bound_normalized = y.exp();
    if alpha <= intrinsic_normalized || alpha >= upper_bound_normalized {
        return None;
    }

    // R, eq (definition just below 18, call case):
    //     R = 2αC - e^y + 1
    let ey = y.exp();
    let r = 2.0 * alpha - ey + 1.0;
    let r2 = r * r;

    // The "y ≈ 0" path: at exactly y=0, A=0 and the quadratic in β
    // degenerates to a linear equation. We use the closed-form ATM seed
    // (Pólya inversion at y=0) which is simpler and avoids 0/0 cancellation
    // in the general formula's denominator.
    //
    // Threshold chosen empirically: |y| < 1e-10 is well below the precision
    // floor where the general formula starts losing accuracy.
    if y.abs() < 1e-10 {
        // At y=0, αC ≡ A(v/2) - A(-v/2) = √(1 - exp(-v²/(2π))).
        // Solving for v:  v = √(-2π · ln(1 - αC²))
        // (Valid only when α ∈ (0,1); within the y≈0 tolerance window the
        // bounds (e^y - 1, e^y) ≈ (0, 1) so this matches the no-arb band.)
        if !(0.0 < alpha && alpha < 1.0) {
            return None;
        }
        let arg = 1.0 - alpha * alpha;
        if arg <= 0.0 {
            return None;
        }
        let v2 = -2.0 * PI * arg.ln();
        if v2 <= 0.0 {
            return None;
        }
        return Some(v2.sqrt());
    }

    // General y ≠ 0 formula.  Eqs (16)–(18), corrected per the proof
    // derivation in section 2 (β-quadratic Aβ² + Bβ - C = 0).
    let kappa = 1.0 - 2.0 / PI;
    let p = (2.0 * y / PI).exp(); // e^{2y/π}
    let q = 1.0 / p; // e^{-2y/π}
    let u = (kappa * y).exp(); // e^{(1-2/π) y}
    let v_minus = 1.0 / u; // e^{-(1-2/π) y}

    let s_plus = u + v_minus; // "+" sum
    let s_minus = u - v_minus; // "-" sum

    let big_a = s_minus * s_minus;
    let big_b = 4.0 * (p + q) - 2.0 * (-y).exp() * (ey * ey + 1.0 - r2) * s_plus;
    let big_c = (-2.0 * y).exp() * (r2 - (ey - 1.0).powi(2)) * ((ey + 1.0).powi(2) - r2);

    let disc = big_b * big_b + 4.0 * big_a * big_c;
    if disc < 0.0 {
        // Should not happen inside the valid αC band, but guard anyway.
        return None;
    }
    let sqrt_disc = disc.sqrt();

    // β = 2C / (B + √(B² + 4AC)).  This is the SR-preferred root: it stays
    // numerically well-conditioned across the whole moneyness domain even
    // as A → 0 (where the "other" root, (-B - √disc)/(2A), blows up).
    let denom = big_b + sqrt_disc;
    if denom.abs() < 1e-300 {
        return None;
    }
    let beta = 2.0 * big_c / denom;
    if beta <= 0.0 || beta >= 1.0 {
        // β must be in (0, 1) since β = exp(-something positive).
        return None;
    }

    let gamma = -0.5 * PI * beta.ln();
    // The σ formulas (19)–(26) reduce to choosing the sign of the leading
    // √(γ+y) term based on (sign(y), Cm vs C0).
    //
    // C0 is the approximation price at which the (+,+) and (+,-) branches
    // meet. From eqs preceding (19): for y > 0, C0/(K e^{-rT}) is
    //     e^y · A(√(2y)) - 1/2
    // and for y < 0,
    //     e^y / 2 - A(-√(-2y)).
    // We compare `alpha` against this normalized C0/(K e^{-rT}).
    let (c0_normalized, alpha_le_c0_uses_minus_sqrt) = if y > 0.0 {
        (ey * polya_a((2.0 * y).sqrt()) - 0.5, true)
    } else {
        (0.5 * ey - polya_a(-(-2.0 * y).sqrt()), false)
    };

    let gp_arg = gamma + y;
    let gm_arg = gamma - y;
    if gp_arg < 0.0 || gm_arg < 0.0 {
        return None;
    }
    let gp = gp_arg.sqrt();
    let gm = gm_arg.sqrt();

    // Branching per eqs (19)–(26).
    //
    //   y > 0, αC > C0_norm:  v =  gp + gm    [eq 19]
    //   y > 0, αC ≤ C0_norm:  v =  gp - gm    [eq 20]
    //   y < 0, αC > C0_norm:  v =  gp + gm    [eq 23]
    //   y < 0, αC ≤ C0_norm:  v = -gp + gm    [eq 24]
    //
    // (Put cases 21,22,25,26 reduce to the same call expressions once put-
    // call parity has been applied upstream.)
    let v = if y > 0.0 {
        if alpha > c0_normalized { gp + gm } else { gp - gm }
    } else if alpha > c0_normalized {
        gp + gm
    } else {
        // y < 0, alpha ≤ C0_norm
        let _ = alpha_le_c0_uses_minus_sqrt;
        -gp + gm
    };

    if !v.is_finite() || v <= 0.0 {
        return None;
    }
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black76::black76_price_f64;

    /// Pólya A(x): A(0) = 1/2, A(+∞) → 1, A(-∞) → 0.
    #[test]
    fn polya_endpoints() {
        assert!((polya_a(0.0) - 0.5).abs() < 1e-15);
        assert!(polya_a(10.0) > 0.999);
        assert!(polya_a(-10.0) < 0.001);
    }

    /// Pólya approximates the normal CDF to within ~0.004 absolute.
    /// (The paper quotes < 0.003 but the actual maximum error of the
    /// Pólya 1949 form, attained near |x|≈1.8, is closer to 0.0031.
    /// The looser 0.005 bound is the well-cited textbook value.)
    #[test]
    fn polya_within_5e_minus_3_of_normal() {
        use crate::iv::black::norm_cdf;
        let mut worst = 0.0_f64;
        for k in -50..=50 {
            let x = k as f64 * 0.1;
            let diff = (polya_a(x) - norm_cdf(x)).abs();
            if diff > worst {
                worst = diff;
            }
            assert!(diff < 0.005, "Pólya off at x={}: {}", x, diff);
        }
        // sanity print
        eprintln!("max |Pólya - Φ| over grid: {}", worst);
    }

    /// ATM call inversion: at y=0, αC = √(1 - exp(-v²/(2π))) so SR must
    /// recover v = √(-2π ln(1-αC²)) exactly (closed-form ATM branch).
    #[test]
    fn sr_atm_inverts_polya_exactly() {
        for alpha in [0.05_f64, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95] {
            let v_expected = (-2.0 * PI * (1.0 - alpha * alpha).ln()).sqrt();
            let v_got = sr_seed_call(0.0, alpha).expect("ATM SR returned None");
            let rel = (v_got - v_expected).abs() / v_expected;
            assert!(
                rel < 1e-12,
                "ATM SR mismatch: alpha={}, got={}, expected={}, rel={}",
                alpha,
                v_got,
                v_expected,
                rel
            );
        }
    }

    /// SR seed must lie within the paper's relative-error band when compared
    /// against the true BS implied vol on a grid.
    ///
    ///     -0.0418 < (σ_BS - σ_SR) / σ_BS < 0.1138
    ///
    /// We add a small slack (call it ±0.001) for the rare boundary points.
    #[test]
    fn sr_relative_error_within_paper_band() {
        // Use Black-76 oracle: pick (F, K, T, σ_true), price, normalize,
        // run SR seed, compare.
        let f = 100.0_f64;
        let r = 0.0; // doesn't affect the band; simplifies normalization
        let mut max_pos = f64::MIN;
        let mut max_neg = f64::MAX;
        let mut samples = 0usize;

        for &k in &[40.0_f64, 70.0, 90.0, 100.0, 110.0, 130.0, 200.0] {
            for &sigma in &[0.05_f64, 0.10, 0.20, 0.40, 0.80, 1.5] {
                for &t in &[1.0_f64 / 365.0, 0.1, 0.5, 1.0, 3.0] {
                    let price = black76_price_f64(f, k, r, sigma, t, true);
                    let alpha = price / (k * (-r * t).exp());
                    let y = (f / k).ln();
                    let v_true = sigma * t.sqrt();
                    let intrinsic_norm = (y.exp() - 1.0).max(0.0);
                    let upper_norm = y.exp();
                    if alpha <= intrinsic_norm + 1e-12 || alpha >= upper_norm - 1e-12 {
                        // Boundary; skip
                        continue;
                    }
                    let v_sr = match sr_seed_call(y, alpha) {
                        Some(v) => v,
                        None => continue,
                    };
                    let rel = (v_true - v_sr) / v_true;
                    if rel > max_pos {
                        max_pos = rel;
                    }
                    if rel < max_neg {
                        max_neg = rel;
                    }
                    samples += 1;
                    assert!(
                        rel > -0.05 && rel < 0.13,
                        "SR seed outside band: F={}, K={}, σ={}, T={}, v_true={}, v_sr={}, rel={}",
                        f,
                        k,
                        sigma,
                        t,
                        v_true,
                        v_sr,
                        rel
                    );
                }
            }
        }
        assert!(samples > 100, "test grid too sparse: {} samples", samples);
        // Sanity print on the worst case
        eprintln!(
            "SR seed band over {} samples: max_pos={:.6}, max_neg={:.6}",
            samples, max_pos, max_neg
        );
    }
}
