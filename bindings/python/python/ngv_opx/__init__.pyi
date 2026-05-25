"""Type stubs for ngv_opx.

The Black-76 entry points (`black76*`) are the production path — f64
throughout, CPU. The legacy `black_scholes*` / `implied_volatility*`
entry points are f32 and route through the experimental wgpu/Metal
GPU prototype; treat them as experimental.
"""
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Black-76 (production, CPU, f64)
# ---------------------------------------------------------------------------

def black76(
    forward: float,
    strike: float,
    rate: float,
    volatility: float,
    time_years: float,
    is_call: bool,
) -> float:
    """Black-76 price for a single option on a forward.

    Args:
        forward: Forward price F of the underlying (e.g. CL front-month future).
        strike: Strike price K.
        rate: Continuously compounded risk-free rate (annualized, e.g. 0.045).
        volatility: Annualized volatility (e.g. 0.32 for 32%).
        time_years: Time to expiry in years (use act/365 to match the example).
        is_call: True for call, False for put.

    Returns:
        Option price as a float (f64).
    """
    ...

def black76_vectorized(
    forwards: NDArray[np.float64],
    strikes: NDArray[np.float64],
    rates: NDArray[np.float64],
    volatilities: NDArray[np.float64],
    times: NDArray[np.float64],
    is_calls: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Vectorized Black-76 pricer.

    All input arrays must be 1-D, equal length, and contiguous. No
    broadcasting — pass `np.full(N, value)` for constant rates/forwards.

    Args:
        forwards: Forward prices, float64.
        strikes: Strike prices, float64.
        rates: Risk-free rates, float64.
        volatilities: Annualized vols, float64.
        times: Times to expiry in years, float64.
        is_calls: Call/put flags, bool.

    Returns:
        Array of option prices, float64, same length as inputs.
    """
    ...

def black76_implied_volatility(
    forward: float,
    strike: float,
    rate: float,
    time_years: float,
    market_price: float,
    is_call: bool,
) -> float:
    """Solve Black-76 implied volatility for a single observed price.

    Uses Newton-Raphson on vega with a Brenner-Subrahmanyam ATM seed.

    Returns:
        Implied volatility as a float (f64), or the sentinel `-1.0` when
        IV is mathematically undefined:
          - `time_years <= 0` (vega is zero)
          - `market_price` below intrinsic
          - `market_price` above the discounted upper bound

        Callers MUST check for `-1.0` before using the result.
    """
    ...

def black76_implied_volatility_vectorized(
    forwards: NDArray[np.float64],
    strikes: NDArray[np.float64],
    rates: NDArray[np.float64],
    times: NDArray[np.float64],
    market_prices: NDArray[np.float64],
    is_calls: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Vectorized Black-76 IV solver.

    Independent per-row, so an N-row call returns N IVs in one round-trip.
    All input arrays must be 1-D, equal length, and contiguous.

    Returns:
        Array of implied vols, float64. Entries are `-1.0` for rows where
        IV is undefined — see `black76_implied_volatility` for the sentinel
        contract.
    """
    ...

# ---------------------------------------------------------------------------
# Legacy / experimental: f32 Black-Scholes (spot) + wgpu/Metal GPU path
# ---------------------------------------------------------------------------

def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_years: float,
    is_call: bool,
) -> float:
    """[Experimental] Black-Scholes price for a single option (f32).

    Production code should use `black76` instead. f32 precision is
    insufficient for deep-ITM/OTM short-dated options.
    """
    ...

def black_scholes_batch(
    spots: NDArray[np.float32],
    strikes: NDArray[np.float32],
    rates: NDArray[np.float32],
    volatilities: NDArray[np.float32],
    times: NDArray[np.float32],
    is_calls: NDArray[np.bool_],
    use_gpu: bool = True,
) -> NDArray[np.float32]:
    """[Experimental] Batched Black-Scholes pricer (f32, optional GPU).

    `use_gpu=True` requires the Apple Silicon wgpu/Metal path.
    """
    ...

def implied_volatility(
    spot: float,
    strike: float,
    rate: float,
    time_years: float,
    market_price: float,
    is_call: bool,
) -> float:
    """[Experimental] Black-Scholes implied vol (f32). Returns -1.0 on failure."""
    ...

def implied_volatility_batch(
    spots: NDArray[np.float32],
    strikes: NDArray[np.float32],
    rates: NDArray[np.float32],
    times: NDArray[np.float32],
    market_prices: NDArray[np.float32],
    is_calls: NDArray[np.bool_],
    use_gpu: bool = True,
) -> NDArray[np.float32]:
    """[Experimental] Batched Black-Scholes IV solver (f32, optional GPU).

    Returns -1.0 for rows where the solver fails.
    """
    ...

def get_gpu_name() -> str:
    """Return the name of the GPU used by the experimental wgpu/Metal path.

    Raises:
        RuntimeError: if no GPU adapter is available on this system.
    """
    ...

def gpu_available() -> bool:
    """Cheap probe: True if a usable GPU adapter is available on this system.

    Does not initialize a device. Safe to call from CPU-only code paths to
    decide whether to opt in to `use_gpu=True` on batch functions.
    """
    ...
