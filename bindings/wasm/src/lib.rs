//! WebAssembly bindings for ngv-opx.
//!
//! Exposes Black-76 pricing (and, in U5, implied-vol solving) from the
//! pure-Rust core crate to JavaScript/TypeScript callers in browsers and Node.
//! All numerics are f64 to match the Python production binding.

use ngv_opx_core::black76::{black76_implied_vol_f64, black76_price_batch_f64, black76_price_f64};
use wasm_bindgen::prelude::*;

/// Returns the ngv-opx-wasm package version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Black-76 price for a single option on a forward (f64).
///
/// @param forward         Forward price F of the underlying.
/// @param strike          Strike price K.
/// @param rate            Continuously compounded risk-free rate (annualized).
/// @param volatility      Annualized volatility.
/// @param timeYears       Time to expiry in years.
/// @param isCall          true for call, false for put.
#[wasm_bindgen(js_name = black76)]
pub fn black76(
    forward: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_years: f64,
    is_call: bool,
) -> f64 {
    black76_price_f64(forward, strike, rate, volatility, time_years, is_call)
}

/// Vectorized Black-76 price (f64). All input arrays must be equal length.
///
/// `isCalls` is `Uint8Array` (JS has no boolean typed array): 0 = put, non-zero = call.
///
/// Throws if input lengths disagree.
#[wasm_bindgen(js_name = black76Batch)]
pub fn black76_batch(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    volatilities: &[f64],
    times: &[f64],
    is_calls: &[u8],
) -> Result<Vec<f64>, JsError> {
    let n = forwards.len();
    if strikes.len() != n
        || rates.len() != n
        || volatilities.len() != n
        || times.len() != n
        || is_calls.len() != n
    {
        return Err(JsError::new(&format!(
            "black76Batch: all input arrays must have the same length; got forwards={}, strikes={}, rates={}, vols={}, times={}, is_calls={}",
            n,
            strikes.len(),
            rates.len(),
            volatilities.len(),
            times.len(),
            is_calls.len()
        )));
    }
    let cps: Vec<bool> = is_calls.iter().map(|&b| b != 0).collect();
    Ok(black76_price_batch_f64(
        forwards,
        strikes,
        rates,
        volatilities,
        times,
        &cps,
    ))
}

/// Solve Black-76 implied volatility for a single observed price (f64).
///
/// Returns the sentinel `-1.0` when IV is mathematically undefined:
///   - timeYears <= 0 (vega is zero)
///   - marketPrice below intrinsic
///   - marketPrice above the discounted upper bound
///
/// Callers MUST check for `-1.0` before using the result.
#[wasm_bindgen(js_name = impliedVol)]
pub fn implied_vol(
    forward: f64,
    strike: f64,
    rate: f64,
    time_years: f64,
    market_price: f64,
    is_call: bool,
) -> f64 {
    black76_implied_vol_f64(forward, strike, rate, time_years, market_price, is_call)
}

/// Vectorized Black-76 IV solver (f64). All input arrays must be equal length.
///
/// Independent per-row, so an N-row call returns N IVs. Entries are `-1.0`
/// for rows where IV is undefined — see `impliedVol` for the sentinel contract.
///
/// `isCalls` is `Uint8Array`: 0 = put, non-zero = call.
///
/// Throws if input lengths disagree.
#[wasm_bindgen(js_name = impliedVolBatch)]
pub fn implied_vol_batch(
    forwards: &[f64],
    strikes: &[f64],
    rates: &[f64],
    times: &[f64],
    market_prices: &[f64],
    is_calls: &[u8],
) -> Result<Vec<f64>, JsError> {
    let n = forwards.len();
    if strikes.len() != n
        || rates.len() != n
        || times.len() != n
        || market_prices.len() != n
        || is_calls.len() != n
    {
        return Err(JsError::new(&format!(
            "impliedVolBatch: all input arrays must have the same length; got forwards={}, strikes={}, rates={}, times={}, prices={}, is_calls={}",
            n,
            strikes.len(),
            rates.len(),
            times.len(),
            market_prices.len(),
            is_calls.len()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(black76_implied_vol_f64(
            forwards[i],
            strikes[i],
            rates[i],
            times[i],
            market_prices[i],
            is_calls[i] != 0,
        ));
    }
    Ok(out)
}
