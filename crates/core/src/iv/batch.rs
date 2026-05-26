//! Parallel SoA batch IV API. Errors collapse to `NaN` in the output slice;
//! callers needing per-row reasons use the scalar [`super::solver::black76_implied_vol`].

use rayon::prelude::*;

use crate::iv::solver::black76_implied_vol;

/// Batch Black-76 implied volatility over parallel SoA slices.
///
/// All input slices and `out` must be the same length (asserted). Errors
/// for individual rows are written to `out` as `f64::NAN` — same per-row
/// convention as the existing pipeline's `-1.0` sentinel, but in a form
/// that's safe to propagate through downstream nan-aware numerics (e.g.
/// `numpy.nanmean`) without arithmetic surprises.
///
/// # Panics
/// If any pair of input slices has mismatched length.
pub fn black76_implied_vol_batch(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    times: &[f64],
    market_prices: &[f64],
    is_calls: &[bool],
    out: &mut [f64],
) {
    let n = forwards.len();
    assert!(
        strikes.len() == n
            && rates.len() == n
            && times.len() == n
            && market_prices.len() == n
            && is_calls.len() == n
            && out.len() == n,
        "all input slices and out must be the same length"
    );

    // rayon::par_iter_mut + zip on the input slices. We iterate over `out`
    // by index since rayon's multi-zip ergonomics get noisy past 5 slices.
    out.par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = match black76_implied_vol(
            forwards[i],
            strikes[i],
            rates[i],
            times[i],
            market_prices[i],
            is_calls[i],
        ) {
            Ok(s) => s,
            Err(_) => f64::NAN,
        };
    });
}

/// Single-threaded variant — useful when the caller is already parallelizing
/// at a higher level (e.g. per-symbol) and rayon's nested-pool overhead
/// dominates on small inner batches.
pub fn black76_implied_vol_batch_serial(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    times: &[f64],
    market_prices: &[f64],
    is_calls: &[bool],
    out: &mut [f64],
) {
    let n = forwards.len();
    assert!(
        strikes.len() == n
            && rates.len() == n
            && times.len() == n
            && market_prices.len() == n
            && is_calls.len() == n
            && out.len() == n,
        "all input slices and out must be the same length"
    );
    for i in 0..n {
        out[i] = match black76_implied_vol(
            forwards[i],
            strikes[i],
            rates[i],
            times[i],
            market_prices[i],
            is_calls[i],
        ) {
            Ok(s) => s,
            Err(_) => f64::NAN,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black76::black76_price_f64;

    /// Batch and serial paths produce bit-identical results on the same inputs.
    /// This is the determinism check called out in the plan — rayon's
    /// reductions are absent here (we're just mapping), so bit-equality
    /// across thread counts should be automatic.
    #[test]
    fn batch_matches_serial_bit_for_bit() {
        let n = 1024;
        let forwards: Vec<f64> = (0..n).map(|i| 80.0 + (i as f64) * 0.05).collect();
        let strikes: Vec<f64> = (0..n).map(|i| 90.0 + ((i as f64) * 0.07).sin() * 20.0).collect();
        let rates: Vec<f64> = vec![0.045; n];
        let times: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.001).collect();
        let sigmas: Vec<f64> = (0..n).map(|i| 0.15 + ((i as f64) * 0.03).cos().abs() * 0.5).collect();
        let is_calls: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let market_prices: Vec<f64> = (0..n)
            .map(|i| black76_price_f64(forwards[i], strikes[i], rates[i], sigmas[i], times[i], is_calls[i]))
            .collect();

        let mut out_par = vec![0.0_f64; n];
        let mut out_ser = vec![0.0_f64; n];
        black76_implied_vol_batch(&forwards, &strikes, &rates, &times, &market_prices, &is_calls, &mut out_par);
        black76_implied_vol_batch_serial(
            &forwards, &strikes, &rates, &times, &market_prices, &is_calls, &mut out_ser,
        );

        for i in 0..n {
            // Treat NaN==NaN as equal (per-row sentinel rows)
            if out_par[i].is_nan() && out_ser[i].is_nan() {
                continue;
            }
            assert_eq!(
                out_par[i].to_bits(),
                out_ser[i].to_bits(),
                "Row {}: par={}, ser={}",
                i,
                out_par[i],
                out_ser[i]
            );
        }
    }
}
