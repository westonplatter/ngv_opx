use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use ngv_opx_core::black76::{black76_implied_vol_f64, black76_price_f64};
use ngv_opx_core::implied_vol::{implied_volatility_batch_cpu, implied_volatility_cpu, IVParams};
use ngv_opx_core::{black_scholes_batch_cpu, black_scholes_cpu, OptionParams};

#[cfg(feature = "gpu")]
use ngv_opx_gpu::{gpu_available as gpu_available_rs, GpuIVSolver, GpuPricer};

/// Calculate the Black-Scholes price for a single option.
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
/// `use_gpu` is accepted for API compatibility but is a no-op when the wheel
/// was built without the `gpu` feature (i.e. all published distribution wheels).
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

    #[cfg(feature = "gpu")]
    let results = if use_gpu {
        match GpuPricer::try_new() {
            Ok(pricer) => pricer.price(&options),
            Err(_) => black_scholes_batch_cpu(&options),
        }
    } else {
        black_scholes_batch_cpu(&options)
    };

    #[cfg(not(feature = "gpu"))]
    let results = {
        let _ = use_gpu;
        black_scholes_batch_cpu(&options)
    };

    PyArray1::from_vec_bound(py, results)
}

/// Calculate the implied volatility for a single option using Newton-Raphson.
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
/// `use_gpu` is accepted for API compatibility but is a no-op when the wheel
/// was built without the `gpu` feature (i.e. all published distribution wheels).
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
    let options: Vec<IVParams> = (0..n)
        .map(|i| {
            IVParams::new(
                spots[i],
                strikes[i],
                rates[i],
                times[i] * 365.0,
                market_prices[i],
                is_calls[i],
            )
        })
        .collect();

    #[cfg(feature = "gpu")]
    let results = if use_gpu {
        match GpuIVSolver::try_new() {
            Ok(solver) => solver.solve(&options),
            Err(_) => implied_volatility_batch_cpu(&options),
        }
    } else {
        implied_volatility_batch_cpu(&options)
    };

    #[cfg(not(feature = "gpu"))]
    let results = {
        let _ = use_gpu;
        implied_volatility_batch_cpu(&options)
    };

    PyArray1::from_vec_bound(py, results)
}

/// Returns true if a usable GPU adapter is available on this system.
/// Always false when the wheel was built without the `gpu` feature.
#[pyfunction]
fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    return gpu_available_rs();
    #[cfg(not(feature = "gpu"))]
    return false;
}

/// Returns the GPU name, or raises RuntimeError if unavailable.
/// Always raises when the wheel was built without the `gpu` feature.
#[pyfunction]
fn get_gpu_name() -> PyResult<String> {
    #[cfg(feature = "gpu")]
    return GpuPricer::try_new()
        .map(|p| p.gpu_name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    #[cfg(not(feature = "gpu"))]
    return Err(PyRuntimeError::new_err(
        "GPU support not compiled in — reinstall from source with --features gpu",
    ));
}

/// Black-76 price for an option on a forward F (f64).
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
/// Returns -1.0 if the market price is outside the no-arbitrage band.
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
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    Ok(())
}
