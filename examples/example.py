"""Black-76 implied-vol roundtrip on a CL (WTI crude) 5-option surface,
plus the 0-DTE / sub-day edge cases mirrored from the test suite.

Generates prices from known vols, then asks the Rust solver to recover them.
Run from repo root: `task example`
(or directly: `cd bindings/python && uv run python ../../examples/example.py`)
"""
import math
import numpy as np
import ngv_opx

FORWARD = 75.00  # CL front-month futures, USD/bbl
RATE = 0.045
DAY = 1.0 / 365.0
HOUR = DAY / 24.0
MINUTE = HOUR / 60.0

# ----------------------------------------------------------------------------
# 1. Solvable surface: price -> IV roundtrip
# ----------------------------------------------------------------------------
# (label, strike, dte_days, true_vol, is_call)
CASES = [
    ("ATM 30d call",        75.0,  30.0, 0.32, True),
    ("ATM 365d put",        75.0, 365.0, 0.28, False),
    ("Deep OTM put 60d",    60.0,  60.0, 0.45, False),  # left-tail skew
    ("Deep OTM call 60d",   90.0,  60.0, 0.38, True),   # right-tail (CL keeps smile)
    ("Short 7d 5% OTM put", 71.0,   7.0, 0.40, False),  # gamma corner
]

strikes   = np.array([c[1] for c in CASES], dtype=np.float64)
times     = np.array([c[2] * DAY for c in CASES], dtype=np.float64)
true_vols = np.array([c[3] for c in CASES], dtype=np.float64)
is_calls  = np.array([c[4] for c in CASES], dtype=np.bool_)
forwards  = np.full(len(CASES), FORWARD, dtype=np.float64)
rates     = np.full(len(CASES), RATE, dtype=np.float64)

prices = ngv_opx.black76_vectorized(forwards, strikes, rates, true_vols, times, is_calls)
recovered = ngv_opx.black76_implied_volatility_vectorized(
    forwards, strikes, rates, times, prices, is_calls,
)

print(f"{'case':<22} {'K':>6} {'dte':>5} {'cp':>3} {'price':>10} "
      f"{'true_iv':>8} {'recovered':>10} {'err':>10}")
print("-" * 90)
for (label, k, dte, true_vol, is_call), px, iv in zip(CASES, prices, recovered):
    cp = "C" if is_call else "P"
    err = abs(iv - true_vol)
    print(f"{label:<22} {k:>6.2f} {dte:>5.0f} {cp:>3} {px:>10.4f} "
          f"{true_vol:>8.4f} {iv:>10.4f} {err:>10.2e}")

max_err = float(np.max(np.abs(recovered - true_vols)))
print(f"\nmax IV recovery error: {max_err:.2e}")
assert max_err < 5e-3, "IV recovery exceeded tolerance"

# ----------------------------------------------------------------------------
# 2. 0 DTE / sub-day edge cases (mirrored from tests/test_black76_vs_vollib.py)
#
# At T=0 vega is zero and the solver returns -1.0 sentinel (IV undefined).
# Sub-day expiries with real time value still recover cleanly.
# ----------------------------------------------------------------------------
SENTINEL = -1.0

print("\nedge cases (0 DTE / sub-day):")
print(f"{'case':<32} {'F':>6} {'K':>6} {'t (yr)':>10} {'price':>10} {'iv':>10}")
print("-" * 90)

def show(label: str, F: float, K: float, t: float, price: float, is_call: bool):
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call)
    iv_str = "SENTINEL" if iv == SENTINEL else f"{iv:.4f}"
    print(f"{label:<32} {F:>6.2f} {K:>6.2f} {t:>10.6f} {price:>10.4f} {iv_str:>10}")
    return iv

# 0 DTE deep ITM call: price == intrinsic, vega == 0 -> sentinel.
iv = show("0 DTE deep ITM call (F=90,K=65)", 90.0, 65.0, 0.0, 90.0 - 65.0, True)
assert iv == SENTINEL

# 0 DTE ATM: any vol -> same (zero) price -> sentinel.
iv = show("0 DTE ATM call (F=K=75)", 75.0, 75.0, 0.0, 0.0, True)
assert iv == SENTINEL

# Same-day "0 DTE" as the desk actually sees it: 10 minutes to expiry,
# annualized. Tiny but nonzero T -> solver recovers cleanly.
t_10m = 10.0 * MINUTE
price_10m = ngv_opx.black76(75.0, 75.0, RATE, 0.45, t_10m, True)
iv = show("10min ATM call (vol=0.45)", 75.0, 75.0, t_10m, price_10m, True)
assert abs(iv - 0.45) < 5e-4

# 6 hours to expiry, ATM, vol=0.45 -- morning-of-expiration daily.
t_6h = 6.0 * HOUR
price_6h = ngv_opx.black76(75.0, 75.0, RATE, 0.45, t_6h, True)
iv = show("6hr ATM call (vol=0.45)", 75.0, 75.0, t_6h, price_6h, True)
assert abs(iv - 0.45) < 5e-4

# 1 DTE ATM, vol=0.40 -- the bread-and-butter daily-options case.
t_1d = 1.0 * DAY
price_1d = ngv_opx.black76(75.0, 75.0, RATE, 0.40, t_1d, True)
iv = show("1 DTE ATM call (vol=0.40)", 75.0, 75.0, t_1d, price_1d, True)
assert abs(iv - 0.40) < 5e-4

# High-vol shock: 80% IV, 7 DTE ATM (covid/Ukraine regime).
t_7d = 7.0 * DAY
price_shock = ngv_opx.black76(75.0, 75.0, RATE, 0.80, t_7d, True)
iv = show("7 DTE ATM call shock (vol=0.80)", 75.0, 75.0, t_7d, price_shock, True)
assert abs(iv - 0.80) < 5e-4

# Below-intrinsic quote -> sentinel (stale/cross quote, must not return fake vol).
F, K, t_2d = 90.0, 65.0, 2.0 * DAY
disc = math.exp(-RATE * t_2d)
intrinsic = disc * (F - K)
iv = show("below-intrinsic call quote", F, K, t_2d, intrinsic - 0.50, True)
assert iv == SENTINEL

print("\nOK")
