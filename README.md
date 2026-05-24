# ngv_opx

[![tests](https://github.com/westonplatter/ngv_opx/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/westonplatter/ngv_opx/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)
[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

A Rust + Python option pricing and implied-volatility toolkit.

The Rust core is exposed to Python via PyO3 / maturin, with both scalar and vectorized NumPy entry points.

## Current Scope

So far, this repo only supports **Black-76 only** — the standard model for options on futures. I'll implemenet other pricing models in the future. See tabel for current matrix of coverage and features.

| Pricer / Model              | Pricing | Implied Vol | 1st Order Greeks | 2nd Order Greeks | GPU Path | Status             |
|-----------------------------|:-------:|:-----------:|:----------------:|:----------------:|:--------:|--------------------|
| **Black-76** (futures)      |   ✅    |     ✅      |        ⛔        |        ⛔         |    🚧    | Shipped (CPU, f64) |
| Black-Scholes-Merton (spot) |   🚧    |     🚧      |        ⛔        |        ⛔         |    🚧    | Planned            |

Legend: ✅ available · 🚧 in progress / experimental · ⛔ not started

> Note: an older `black_scholes` / `implied_volatility` (f32) path is
> still exposed in the Python module from the GPU prototype. Treat it
> as experimental — production code should use the Black-76 entry
> points (`black76`, `black76_vectorized`, `black76_implied_volatility`,
> `black76_implied_volatility_vectorized`), all f64.

## Requirements

- **Black-76 CPU path (production):** runs anywhere Rust + Python do — Linux, macOS (Intel or Apple Silicon), Windows.
- **wgpu/Metal GPU path (experimental, legacy f32 BSM only):** requires macOS with Apple Silicon (M1+).
- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.9+ and [uv](https://github.com/astral-sh/uv)

## Quick Start

> **Not yet published to PyPI.** There is no `pip install ngv_opx`
> wheel today — install from source via the steps below. A PyPI
> release is planned once the Black-76 API stabilizes and greeks land.

Common workflows are driven by [Taskfile](https://taskfile.dev/) — install
it via `brew install go-task` (or see the Taskfile docs for other
platforms), then:

```bash
uv sync --extra dev   # one-time: dev deps incl. maturin
task build            # build the Rust extension into the current uv venv
task example          # run the worked CL Black-76 example
```

### Available Tasks

| Task                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `task build`          | Build the Rust extension into the current uv venv (release mode)            |
| `task example`        | Run the Black-76 IV roundtrip example on the crude oil surface              |
| `task test`           | Run Rust unit tests **and** Python pytest suite                             |
| `task test:rust`      | Rust unit tests only                                                        |
| `task test:py`        | Python pytest suite only (cross-validates Black-76 vs py_vollib)            |
| `task bench`          | CPU benchmark — {Python, Rust} × {single, vectorized} + py_vollib           |
| `task bench:all-pythons` | Run bench against Python 3.11, 3.12, 3.13 (writes JSON per version)      |
| `task bench:compare`  | Cross-version comparison tables + Plotly chart (reads `benchmarks/results/`) |
| `task bench:chart`    | Plotly chart of the 5 pricing paths (writes `benchmarks/bench_chart.html`)  |
| `task list` / `task l`| List all available tasks                                                    |

## Python Usage (Black-76)

```python
import numpy as np
import ngv_opx

# Single option: price a 30-day ATM CL call
price = ngv_opx.black76(
    forward=75.0, strike=75.0, rate=0.045,
    volatility=0.32, time_years=30 / 365, is_call=True,
)

# Single option: recover implied vol from a market price
iv = ngv_opx.black76_implied_volatility(
    forward=75.0, strike=75.0, rate=0.045,
    time_years=30 / 365, market_price=price, is_call=True,
)

# Vectorized: whole surface in one call (f64 arrays)
forwards = np.array([75.0, 75.0, 75.0], dtype=np.float64)
strikes  = np.array([75.0, 80.0, 70.0], dtype=np.float64)
rates    = np.full(3, 0.045)
vols     = np.array([0.32, 0.35, 0.34])
times    = np.array([30, 30, 30]) / 365.0
is_calls = np.array([True, True, False])

prices = ngv_opx.black76_vectorized(forwards, strikes, rates, vols, times, is_calls)
ivs    = ngv_opx.black76_implied_volatility_vectorized(
    forwards, strikes, rates, times, prices, is_calls,
)
```

The IV solver returns `-1.0` as a **sentinel** when IV is mathematically undefined (`T = 0`, price below intrinsic, price above the discounted upper bound). Callers must check for this — see `examples/example.py` for handling and `tests/test_black76_vs_vollib.py` for the contract.

## Algorithm Details

### Black-76 Formula

For an option on a forward `F` struck at `K`, with risk-free rate `r`, vol `σ`, and time to expiry `T`:

```
C = e^(-rT) [ F · N(d₁) - K · N(d₂) ]
P = e^(-rT) [ K · N(-d₂) - F · N(-d₁) ]
```

with `d₁ = (ln(F/K) + ½σ²T) / (σ√T)` and `d₂ = d₁ - σ√T`.

### Implied Volatility

Newton-Raphson on vega, with Brenner-Subrahmanyam (ATM) seed. f64 throughout because deep-ITM/OTM short-dated options on CL push f32 past its ~7 digits of headroom (time value can be sub-1e-7 of the forward).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
