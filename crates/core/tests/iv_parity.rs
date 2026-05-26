//! Put-call parity in IV space.
//!
//! For every (F, K, T, r, σ) the call price and put price are related by
//! C - P = e^{-rT}·(F - K). Feeding the call IV vs the put IV through the
//! solver must return the same σ to machine precision — modulo the noise
//! floor in deep-ITM short-DTE rows where time value is below f64 noise.

use ngv_opx_core::black76::black76_price_f64;
use ngv_opx_core::iv::black::black_vega_normalized;
use ngv_opx_core::iv::black76_implied_vol;

#[test]
fn parity_on_dense_grid() {
    let f = 100.0_f64;
    let strikes = [40.0_f64, 60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0, 250.0];
    let rates = [-0.01_f64, 0.0, 0.045, 0.10];
    let sigmas = [0.05_f64, 0.10, 0.20, 0.40, 0.80, 1.50];
    let times = [7.0_f64 / 365.0, 30.0 / 365.0, 0.25, 1.0, 3.0];

    let mut samples = 0usize;
    let mut worst = 0.0_f64;
    let mut skipped = 0usize;

    for &k in &strikes {
        for &r in &rates {
            for &sigma in &sigmas {
                for &t in &times {
                    let call_price = black76_price_f64(f, k, r, sigma, t, true);
                    let put_price = black76_price_f64(f, k, r, sigma, t, false);

                    let iv_call = black76_implied_vol(f, k, r, t, call_price, true);
                    let iv_put = black76_implied_vol(f, k, r, t, put_price, false);

                    match (iv_call, iv_put) {
                        (Ok(c), Ok(p)) => {
                            // Cap tolerance by the noise floor in this regime
                            let y = (f / k).ln();
                            let vega = black_vega_normalized(y, sigma * t.sqrt());
                            let tol = if vega > 1e-3 { 1e-10 } else { 1e-6 };
                            let diff = (c - p).abs();
                            if diff > worst {
                                worst = diff;
                            }
                            samples += 1;
                            assert!(
                                diff < tol,
                                "Parity break: K={}, r={}, σ={}, T={}, call_iv={}, put_iv={}, diff={:.3e}, vega={:.3e}",
                                k, r, sigma, t, c, p, diff, vega
                            );
                        }
                        _ => {
                            skipped += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "Parity: {} samples, worst |call_iv - put_iv| = {:.2e}, {} skipped",
        samples, worst, skipped
    );
    assert!(samples > 200, "too few samples: {}", samples);
}
