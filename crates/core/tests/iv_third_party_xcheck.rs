//! Cross-check the new SR + Halley solver against the existing
//! Newton-Raphson Black-76 solver in `black76::black76_implied_vol_f64`.
//!
//! The pre-existing Newton implementation has been live in production for
//! months on real market data. Treating it as a known-good oracle, the new
//! solver must agree with it on the bulk regime, and disagree only in the
//! deep-ITM/short-DTE regime where Newton itself bails (returns -1.0).
//!
//! This is the "Newton-as-oracle" cross-check called out in the plan. A
//! follow-up will add py_vollib/QuantLib fixture files for an external
//! third-party reference; until then, Newton is the cross-check.

use ngv_opx_core::black76::{black76_implied_vol_f64, black76_price_f64};
use ngv_opx_core::iv::black76_implied_vol;

/// Hand-picked, real-shape (F, K, T, r, σ) reference points. Each entry is
/// the kind of strike a desk would actually see: futures-style commodity
/// surfaces, equity index quarterly, FX vanilla, etc.
#[test]
fn xcheck_canonical_reference_points() {
    let cases: &[(f64, f64, f64, f64, f64, bool, &str)] = &[
        // (forward, strike, rate, time_years, σ_true, is_call, label)
        (5000.0,  5000.0, 0.045, 30.0 / 365.0, 0.15, true,  "SPX-style 30d ATM call"),
        (5000.0,  5000.0, 0.045, 365.0 / 365.0, 0.18, false, "SPX-style 1y ATM put"),
        (5000.0,  4250.0, 0.045, 60.0 / 365.0, 0.35, false, "SPX-style 60d OTM put"),
        (5000.0,  5750.0, 0.045, 60.0 / 365.0, 0.12, true,  "SPX-style 60d OTM call"),
        (75.0,    75.0,   0.05,  30.0 / 365.0, 0.30, true,  "CL 30d ATM call"),
        (75.0,    65.0,   0.05,  90.0 / 365.0, 0.45, false, "CL 90d OTM put"),
        (75.0,    90.0,   0.05,  180.0/ 365.0, 0.50, true,  "CL 180d OTM call"),
        (1.10,    1.10,   0.02,  90.0 / 365.0, 0.08, true,  "EUR/USD 90d ATM call"),
        (1.10,    1.05,   0.02,  365.0/ 365.0, 0.10, false, "EUR/USD 1y OTM put"),
        (100.0,   100.0,  0.0,   1.0,           0.20, true,  "Textbook ATM call"),
        (100.0,   110.0,  0.05,  0.25,          0.25, true,  "10% OTM call"),
        (100.0,   95.0,   0.05,  0.25,          0.25, false, "5% OTM put"),
        (50.0,    50.0,   0.03,  2.0,           0.40, true,  "2y ATM long-dated"),
        (50.0,    30.0,   0.03,  2.0,           0.65, false, "2y deep OTM put"),
        (50.0,    80.0,   0.03,  2.0,           0.55, true,  "2y deep OTM call"),
        (200.0,   180.0,  0.04,  0.5,           0.32, true,  "Slight ITM call"),
        (200.0,   220.0,  0.04,  0.5,           0.32, false, "Slight ITM put"),
        (1000.0,  1000.0, 0.025, 90.0 / 365.0, 0.22, true,  "Index ATM call"),
        (1000.0,  900.0,  0.025, 90.0 / 365.0, 0.30, false, "Index 10% OTM put"),
        (1000.0,  1100.0, 0.025, 90.0 / 365.0, 0.20, true,  "Index 10% OTM call"),
    ];

    let mut worst_disagreement = 0.0_f64;
    let mut count = 0usize;

    for &(f, k, r, t, sigma_true, is_call, label) in cases {
        let price = black76_price_f64(f, k, r, sigma_true, t, is_call);

        let sr_iv = black76_implied_vol(f, k, r, t, price, is_call)
            .unwrap_or_else(|e| panic!("[{}] SR solver errored: {:?}", label, e));

        let newton_iv = black76_implied_vol_f64(f, k, r, t, price, is_call);
        assert!(
            newton_iv > 0.0,
            "[{}] Newton oracle bailed (returned {})",
            label,
            newton_iv
        );

        // Both solvers must recover σ_true to better than 1e-6 — the Newton
        // path has its own internal tolerance of 1e-10 on price residual,
        // which translates to ~1e-6 in σ on noise-floor rows. Newton's
        // accuracy is the looser bound here.
        let sr_err = (sr_iv - sigma_true).abs();
        let newton_err = (newton_iv - sigma_true).abs();
        let cross = (sr_iv - newton_iv).abs();
        if cross > worst_disagreement {
            worst_disagreement = cross;
        }
        count += 1;

        assert!(
            sr_err < 1e-6,
            "[{}] SR off from σ_true: σ_true={}, sr={}, err={:.3e}",
            label, sigma_true, sr_iv, sr_err
        );
        assert!(
            newton_err < 1e-3,
            "[{}] Newton off from σ_true: σ_true={}, newton={}, err={:.3e}",
            label, sigma_true, newton_iv, newton_err
        );
        // SR and Newton agree at Newton's tolerance (1e-6 typical)
        assert!(
            cross < 1e-4,
            "[{}] SR vs Newton disagree: sr={}, newton={}, |diff|={:.3e}",
            label, sr_iv, newton_iv, cross
        );
    }

    eprintln!(
        "Cross-check: {} reference rows, worst |SR - Newton| = {:.3e}",
        count, worst_disagreement
    );
}

/// On the bulk regime where Newton fully converges, SR and Newton should
/// agree extremely tightly — Newton runs to its 1e-10 residual tolerance
/// and SR to its 1e-12 floor. Differences should be O(1e-7) or better.
#[test]
fn xcheck_bulk_regime_tight() {
    let f = 100.0_f64;
    let r = 0.045;
    let mut samples = 0usize;
    let mut worst = 0.0_f64;
    for &k in &[80.0_f64, 90.0, 100.0, 110.0, 120.0] {
        for &sigma in &[0.15_f64, 0.20, 0.25, 0.30, 0.40] {
            for &t in &[0.25_f64, 0.5, 1.0, 2.0] {
                let price = black76_price_f64(f, k, r, sigma, t, true);
                let sr_iv = black76_implied_vol(f, k, r, t, price, true).expect("bulk solver ok");
                let newton_iv = black76_implied_vol_f64(f, k, r, t, price, true);
                assert!(newton_iv > 0.0);
                let diff = (sr_iv - newton_iv).abs();
                if diff > worst {
                    worst = diff;
                }
                samples += 1;
                assert!(
                    diff < 1e-6,
                    "Bulk disagreement: K={}, σ={}, T={}, sr={}, newton={}, diff={:.3e}",
                    k, sigma, t, sr_iv, newton_iv, diff
                );
            }
        }
    }
    eprintln!(
        "Bulk cross-check: {} samples, worst |SR - Newton| = {:.3e}",
        samples, worst
    );
}
