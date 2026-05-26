//! Demo of U1: SR closed-form seed for Black-Scholes implied volatility.
//!
//! Prices a small CL-style option strip with a known sigma, normalizes to
//! SR's call coordinates, runs the seed, and prints the round-trip error
//! alongside the Newton-Raphson oracle's recovered IV for comparison.
//!
//! Run:  cargo run --release --example sr_seed_demo -p ngv-opx-core
//!
//! Expected: SR seed is within a few percent of true sigma uniformly across
//! moneyness, with Newton refinement (U2, not yet built) needed to drive
//! the residual to machine precision. Newton baseline column shown for
//! sanity: it's what we're replacing.

use ngv_opx_core::black76::{black76_implied_vol_f64, black76_price_f64};
use ngv_opx_core::iv::stefanica::sr_seed_call;

fn main() {
    let f = 75.0_f64; // CL ~$75/bbl forward
    let r = 0.05;
    let strikes = [55.0_f64, 65.0, 70.0, 75.0, 80.0, 85.0, 100.0];
    let sigmas = [0.20_f64, 0.35, 0.55];
    let times = [7.0_f64 / 365.0, 30.0 / 365.0, 0.25, 1.0];

    println!(
        "{:>6} {:>7} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "K", "T(yrs)", "σ_true", "price", "α (norm)", "y (lnF/K)", "σ_SR", "σ_Newton"
    );
    println!("{}", "-".repeat(90));

    let mut max_rel_sr = 0.0_f64;
    let mut max_rel_newton = 0.0_f64;
    let mut rows = 0u32;

    for &k in &strikes {
        for &sigma in &sigmas {
            for &t in &times {
                let price = black76_price_f64(f, k, r, sigma, t, true);
                let alpha = price / (k * (-r * t).exp());
                let y = (f / k).ln();
                let v_true = sigma * t.sqrt();

                let sigma_sr = sr_seed_call(y, alpha).map(|v| v / t.sqrt());
                let sigma_newton = black76_implied_vol_f64(f, k, r, t, price, true);

                let sr_str = sigma_sr
                    .map(|s| format!("{:>12.6}", s))
                    .unwrap_or_else(|| format!("{:>12}", "none"));
                let newton_str = if sigma_newton > 0.0 {
                    format!("{:>12.6}", sigma_newton)
                } else {
                    format!("{:>12}", "fail")
                };

                println!(
                    "{:>6.1} {:>7.4} {:>6.2} {:>10.4} {:>10.6} {:>10.4} {} {}",
                    k, t, sigma, price, alpha, y, sr_str, newton_str
                );

                // Skip rows where Newton itself bails (no time value at f64
                // precision — typical for deep ITM with sub-month DTE). SR
                // is undefined on those by construction.
                if let (Some(s), true) = (sigma_sr, sigma_newton > 0.0) {
                    let rel_sr = (s - sigma).abs() / sigma;
                    let rel_n = (sigma_newton - sigma).abs() / sigma;
                    if rel_sr > max_rel_sr {
                        max_rel_sr = rel_sr;
                    }
                    if rel_n > max_rel_newton {
                        max_rel_newton = rel_n;
                    }
                    rows += 1;
                }
                let _ = v_true; // implicit via sigma * sqrt(T)
            }
        }
    }
    println!();
    println!("Restricting to {} rows where both solvers produce a result", rows);
    println!("(deep ITM short-DTE rows where time value < f64 noise are skipped):");
    println!(
        "  max |σ_SR - σ_true| / σ_true     = {:.4} ({:.2}%)",
        max_rel_sr,
        max_rel_sr * 100.0
    );
    println!(
        "  max |σ_Newton - σ_true| / σ_true = {:.2e} ({:.4}%)",
        max_rel_newton,
        max_rel_newton * 100.0
    );
    println!(
        "\nSR seed is a closed-form starting point.  U2 (Householder-3 refinement,\n\
         not yet built) will drive the residual to machine precision in 2 fixed steps,\n\
         replacing the per-row Newton loop currently shipping in black76::implied_vol_f64."
    );
}
