//! Python bindings for the Black-Scholes GPU library.
//!
//! This module provides PyO3 bindings to expose the Rust GPU-accelerated
//! Black-Scholes pricing and implied volatility solvers to Python.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::black76::{black76_implied_vol_f64, black76_price_f64};
use crate::implied_vol::{implied_volatility_batch_cpu, implied_volatility_cpu, GpuIVSolver, IVParams};
use crate::{black_scholes_batch_cpu, black_scholes_cpu, GpuPricer, OptionParams};

/// Calculate the Black-Scholes price for a single option.
///
/// Args:
///     spot: Current price of the underlying asset
///     strike: Strike price of the option
///     rate: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
///     volatility: Volatility of the underlying (annualized, e.g., 0.2 for 20%)
///     time_years: Time to maturity in years
///     is_call: True for call option, False for put option
///
/// Returns:
///     Option price as a float
#[pyfunction]
fn black_scholes(
    spot: f32,
    strike: f32,
    rate: f32,
    volatility: f32,
    time_years: f32,
    is_call: bool,
) -> f32 {
    black_scholes_cpu(spot, strike, rate, volatility, time_years, is_call)
}

/// Calculate Black-Scholes prices for a batch of options.
///
/// Args:
///     spots: Array of underlying asset prices
///     strikes: Array of strike prices
///     rates: Array of risk-free interest rates
///     volatilities: Array of volatilities
///     times: Array of times to maturity in years
///     is_calls: Array of booleans (True=call, False=put)
///     use_gpu: Whether to use GPU acceleration (default: True)
///
/// Returns:
///     NumPy array of option prices
#[pyfunction]
#[pyo3(signature = (spots, strikes, rates, volatilities, times, is_calls, use_gpu=true))]
fn black_scholes_batch<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f32>,
    strikes: PyReadonlyArray1<'py, f32>,
    rates: PyReadonlyArray1<'py, f32>,
    volatilities: PyReadonlyArray1<'py, f32>,
    times: PyReadonlyArray1<'py, f32>,
    is_calls: PyReadonlyArray1<'py, bool>,
    use_gpu: bool,
) -> Bound<'py, PyArray1<f32>> {
    let spots = spots.as_slice().unwrap();
    let strikes = strikes.as_slice().unwrap();
    let rates = rates.as_slice().unwrap();
    let volatilities = volatilities.as_slice().unwrap();
    let times = times.as_slice().unwrap();
    let is_calls = is_calls.as_slice().unwrap();

    let n = spots.len();

    // Build OptionParams array
    let options: Vec<OptionParams> = (0..n)
        .map(|i| {
            OptionParams::new(
                spots[i],
                strikes[i],
                rates[i],
                volatilities[i],
                times[i],
                is_calls[i],
            )
        })
        .collect();

    let results = if use_gpu {
        let pricer = GpuPricer::new();
        pricer.price(&options)
    } else {
        black_scholes_batch_cpu(&options)
    };

    PyArray1::from_vec_bound(py, results)
}

/// Calculate the implied volatility for a single option using Newton-Raphson.
///
/// Args:
///     spot: Current price of the underlying asset
///     strike: Strike price of the option
///     rate: Risk-free interest rate (annualized)
///     time_years: Time to maturity in years
///     market_price: Observed market price of the option
///     is_call: True for call option, False for put option
///
/// Returns:
///     Implied volatility as a float, or -1.0 if calculation fails
#[pyfunction]
fn implied_volatility(
    spot: f32,
    strike: f32,
    rate: f32,
    time_years: f32,
    market_price: f32,
    is_call: bool,
) -> f32 {
    implied_volatility_cpu(spot, strike, rate, time_years, market_price, is_call)
}

/// Calculate implied volatilities for a batch of options.
///
/// Args:
///     spots: Array of underlying asset prices
///     strikes: Array of strike prices
///     rates: Array of risk-free interest rates
///     times: Array of times to maturity in years
///     market_prices: Array of observed market prices
///     is_calls: Array of booleans (True=call, False=put)
///     use_gpu: Whether to use GPU acceleration (default: True)
///
/// Returns:
///     NumPy array of implied volatilities (-1.0 for failed calculations)
#[pyfunction]
#[pyo3(signature = (spots, strikes, rates, times, market_prices, is_calls, use_gpu=true))]
fn implied_volatility_batch<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f32>,
    strikes: PyReadonlyArray1<'py, f32>,
    rates: PyReadonlyArray1<'py, f32>,
    times: PyReadonlyArray1<'py, f32>,
    market_prices: PyReadonlyArray1<'py, f32>,
    is_calls: PyReadonlyArray1<'py, bool>,
    use_gpu: bool,
) -> Bound<'py, PyArray1<f32>> {
    let spots = spots.as_slice().unwrap();
    let strikes = strikes.as_slice().unwrap();
    let rates = rates.as_slice().unwrap();
    let times = times.as_slice().unwrap();
    let market_prices = market_prices.as_slice().unwrap();
    let is_calls = is_calls.as_slice().unwrap();

    let n = spots.len();

    // Build IVParams array (note: IVParams::new expects days, but we have years)
    let options: Vec<IVParams> = (0..n)
        .map(|i| {
            // IVParams::new converts days to years internally, but we already have years.
            // We need to pass days = time_years * 365.0
            IVParams::new(
                spots[i],
                strikes[i],
                rates[i],
                times[i] * 365.0, // Convert years back to days for IVParams::new
                market_prices[i],
                is_calls[i],
            )
        })
        .collect();

    let results = if use_gpu {
        let solver = GpuIVSolver::new();
        solver.solve(&options)
    } else {
        implied_volatility_batch_cpu(&options)
    };

    PyArray1::from_vec_bound(py, results)
}

/// Get the name of the GPU being used for computations.
///
/// Returns:
///     GPU name as a string
#[pyfunction]
fn get_gpu_name() -> String {
    let pricer = GpuPricer::new();
    pricer.gpu_name
}

/// Black-76 price for an option on a forward F (f64).
/// f64 throughout because daily-options recovery on deep ITM/OTM strikes
/// requires more than f32's ~7 digits of headroom.
#[pyfunction]
fn black76(
    forward: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time_years: f64,
    is_call: bool,
) -> f64 {
    black76_price_f64(forward, strike, rate, volatility, time_years, is_call)
}

/// Vectorized Black-76 price for an array of (F, K, r, σ, T, cp) tuples.
/// Inputs are float64 numpy arrays of equal length.
#[pyfunction]
fn black76_vectorized<'py>(
    py: Python<'py>,
    forwards: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    rates: PyReadonlyArray1<'py, f64>,
    volatilities: PyReadonlyArray1<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    is_calls: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<f64>> {
    let fs = forwards.as_slice().unwrap();
    let ks = strikes.as_slice().unwrap();
    let rs = rates.as_slice().unwrap();
    let vs = volatilities.as_slice().unwrap();
    let ts = times.as_slice().unwrap();
    let cps = is_calls.as_slice().unwrap();
    let n = fs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(black76_price_f64(fs[i], ks[i], rs[i], vs[i], ts[i], cps[i]));
    }
    PyArray1::from_vec_bound(py, out)
}

/// Solve Black-76 implied vol for a single observed price (f64).
/// Returns -1.0 if the market price is below intrinsic, above the
/// discounted upper bound, or `time_years <= 0`.
#[pyfunction]
fn black76_implied_volatility(
    forward: f64,
    strike: f64,
    rate: f64,
    time_years: f64,
    market_price: f64,
    is_call: bool,
) -> f64 {
    black76_implied_vol_f64(forward, strike, rate, time_years, market_price, is_call)
}

/// Vectorized Black-76 IV solve for an array of (F, K, r, T, mkt_price, cp) rows.
/// Independent per-row, so an N-row call returns N IVs in one round-trip.
#[pyfunction]
fn black76_implied_volatility_vectorized<'py>(
    py: Python<'py>,
    forwards: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    rates: PyReadonlyArray1<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    market_prices: PyReadonlyArray1<'py, f64>,
    is_calls: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<f64>> {
    let fs = forwards.as_slice().unwrap();
    let ks = strikes.as_slice().unwrap();
    let rs = rates.as_slice().unwrap();
    let ts = times.as_slice().unwrap();
    let mps = market_prices.as_slice().unwrap();
    let cps = is_calls.as_slice().unwrap();
    let n = fs.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(black76_implied_vol_f64(
            fs[i], ks[i], rs[i], ts[i], mps[i], cps[i],
        ));
    }
    PyArray1::from_vec_bound(py, out)
}

/// Python module: NGV option pricer (Black-Scholes today, Black-76 added).
#[pymodule]
fn ngv_opx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(implied_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(implied_volatility_batch, m)?)?;
    m.add_function(wrap_pyfunction!(black76, m)?)?;
    m.add_function(wrap_pyfunction!(black76_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(black76_implied_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(black76_implied_volatility_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_name, m)?)?;
    Ok(())
}
