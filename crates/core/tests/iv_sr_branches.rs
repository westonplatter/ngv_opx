//! Branch-coverage tests for the four σ-formulas in SR eqs (19)–(26).
//!
//! The SR closed form has four σ branches depending on `(sign(y), αC vs C0)`:
//!
//!   y ≥ 0, αC > C0:   v =  √(γ+y) + √(γ-y)          (eq 19)
//!   y ≥ 0, αC ≤ C0:   v =  √(γ+y) − √(γ-y)          (eq 20)
//!   y < 0, αC > C0:   v =  √(γ+y) + √(γ-y)          (eq 23)
//!   y < 0, αC ≤ C0:   v = −√(γ+y) + √(γ-y)          (eq 24)
//!
//! The last branch's NEGATIVE leading sqrt is a transcription hazard and
//! gets the highest-value coverage here. We construct fixtures designed to
//! land each branch and verify SR seed accuracy within the paper's band.

use ngv_opx_core::black76::black76_price_f64;
use ngv_opx_core::iv::black76_implied_vol;

/// Compute the SR C0 threshold (normalized) for diagnostic classification.
///
/// y ≥ 0 :  C0_norm = e^y · A(√(2y)) − 1/2
/// y < 0 :  C0_norm = e^y / 2 − A(−√(−2y))
///
/// where A is the Pólya cumulative-normal approximation, eq (8).
fn polya_a(x: f64) -> f64 {
    use std::f64::consts::PI;
    let s = if x >= 0.0 { 1.0 } else { -1.0 };
    0.5 + 0.5 * s * (1.0 - (-2.0 * x * x / PI).exp()).sqrt()
}

fn c0_normalized(y: f64) -> f64 {
    if y >= 0.0 {
        y.exp() * polya_a((2.0 * y).sqrt()) - 0.5
    } else {
        0.5 * y.exp() - polya_a(-(-2.0 * y).sqrt())
    }
}

#[derive(Debug, Clone, Copy)]
struct BranchHit {
    y_sign_pos: bool,
    alpha_gt_c0: bool,
}

fn classify(y: f64, alpha: f64) -> BranchHit {
    BranchHit {
        y_sign_pos: y > 0.0,
        alpha_gt_c0: alpha > c0_normalized(y),
    }
}

/// Walk a grid of (F, K, T, σ) and count how often each of the four
/// branches fires. Any branch with zero hits fails the test.
#[test]
fn all_four_branches_get_exercised() {
    let f = 100.0_f64;
    let r = 0.0;
    let strikes = [30.0_f64, 50.0, 75.0, 90.0, 100.0, 110.0, 130.0, 170.0, 250.0];
    let sigmas = [0.05_f64, 0.10, 0.20, 0.40, 0.80, 1.50];
    let times = [7.0_f64 / 365.0, 30.0 / 365.0, 0.25, 1.0, 3.0];

    let mut counts = [[0_usize; 2]; 2]; // [y_pos][alpha_gt_c0]
    let mut worst_per_branch = [[0.0_f64; 2]; 2];

    for &k in &strikes {
        for &sigma in &sigmas {
            for &t in &times {
                let price = black76_price_f64(f, k, r, sigma, t, true);
                let alpha = price / (k * (-r * t).exp());
                let y = (f / k).ln();
                let intrinsic_norm = (y.exp() - 1.0).max(0.0);
                let upper_norm = y.exp();
                if alpha <= intrinsic_norm + 1e-12 || alpha >= upper_norm - 1e-12 {
                    continue;
                }
                let recovered = match black76_implied_vol(f, k, r, t, price, true) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let err = (recovered - sigma).abs();
                let b = classify(y, alpha);
                let i = b.y_sign_pos as usize;
                let j = b.alpha_gt_c0 as usize;
                counts[i][j] += 1;
                if err > worst_per_branch[i][j] {
                    worst_per_branch[i][j] = err;
                }
            }
        }
    }

    eprintln!("Branch hit counts (worst σ error):");
    eprintln!(
        "  y > 0, αC > C0:  {:5} (worst err {:.2e}) — eq 19",
        counts[1][1], worst_per_branch[1][1]
    );
    eprintln!(
        "  y > 0, αC ≤ C0:  {:5} (worst err {:.2e}) — eq 20",
        counts[1][0], worst_per_branch[1][0]
    );
    eprintln!(
        "  y < 0, αC > C0:  {:5} (worst err {:.2e}) — eq 23",
        counts[0][1], worst_per_branch[0][1]
    );
    eprintln!(
        "  y < 0, αC ≤ C0:  {:5} (worst err {:.2e}) — eq 24",
        counts[0][0], worst_per_branch[0][0]
    );

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                counts[i][j] > 0,
                "Branch (y_pos={}, αC>C0={}) was never hit by the test grid",
                i == 1,
                j == 1
            );
            // After Halley refinement, every branch should converge.
            // 1e-6 is well above observed in any branch; this is a
            // regression guard, not a tight contract.
            assert!(
                worst_per_branch[i][j] < 1e-6,
                "Branch (y_pos={}, αC>C0={}) worst error {:.3e} too high",
                i == 1, j == 1, worst_per_branch[i][j]
            );
        }
    }
}

/// Hand-constructed case for the y < 0, αC ≤ C0 branch (eq 24's
/// `-√(γ+y) + √(γ-y)` form). This is the highest-risk branch per the
/// `docs/ngv-solver.md` §"SR branch selection" — a sign error here would slip through aggregate
/// round-trip tests for a long time.
#[test]
fn negative_leading_sqrt_branch_recovers_sigma() {
    // y < 0  →  K > F  →  OTM call
    // αC ≤ C0  →  market price is on the "low" side of the threshold;
    // happens for high-vol OTM calls where time value is large but the
    // forward is below the strike.
    //
    // Picking F=100, K=200 gives y = ln(0.5) ≈ -0.693.
    // Sweep a few OTM-call shapes that *should* land in eq-24 territory.
    // Many concrete (σ, T) combos at deep OTM produce sub-noise-floor prices
    // where the solver rightly bails — so we count any row that lands in the
    // branch AND is solvable, and require at least one such row to be
    // accurate. (The branch is also exercised by all_four_branches_get_exercised
    // on a wider grid; this test exists to assert recovery quality on a
    // hand-picked example.)
    let cases = [
        (100.0_f64, 130.0_f64, 0.40, 1.0),   // 30% OTM call, 1y, 40% vol
        (100.0_f64, 130.0_f64, 0.80, 0.5),   // 30% OTM call, 6m, 80% vol
        (100.0_f64, 150.0_f64, 0.50, 1.0),   // 50% OTM call, 1y, 50% vol
        (100.0_f64, 200.0_f64, 0.80, 3.0),   // 100% OTM call, 3y, 80% vol
    ];
    let r = 0.0;
    let mut hit = false;
    for &(f, k, sigma, t) in &cases {
        let price = black76_price_f64(f, k, r, sigma, t, true);
        let alpha = price / (k * (-r * t).exp());
        let y = (f / k).ln();
        if y >= 0.0 || alpha > c0_normalized(y) {
            continue;
        }
        let recovered = match black76_implied_vol(f, k, r, t, price, true) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("(skipped F={}, K={}, σ={}, T={}: {:?})", f, k, sigma, t, e);
                continue;
            }
        };
        hit = true;
        let err = (recovered - sigma).abs();
        assert!(
            err < 1e-7,
            "eq-24 branch miss: F={}, K={}, σ={}, T={}, recovered={}, err={:.3e}",
            f, k, sigma, t, recovered, err
        );
        eprintln!(
            "eq-24 branch hit: F={}, K={}, σ_true={}, σ_rec={}, err={:.2e}",
            f, k, sigma, recovered, err
        );
    }
    assert!(hit, "eq-24 branch (y<0, αC≤C0) had no solvable hit in the fixture");
}
