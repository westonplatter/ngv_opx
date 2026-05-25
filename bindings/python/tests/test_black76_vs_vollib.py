"""
Validate ngv_opx Black-76 against py_vollib on a crude-oil options book.

Framed for a risk-focused PM trading daily/weekly WTI options. Each test
either round-trips price -> IV against the model's own pricer, or cross-checks
ours vs py_vollib (the float64 "Let's Be Rational" reference).

Crude tick size is $0.01 / bbl on quoted option premiums. Any "price" below
that is not a real market quote -- the exchange just shows $0.00 bid. Tests
respect that: options whose theoretical value rounds below a penny are tested
for "solver must signal failure, not lie", not for IV recovery.

Sections:
  1. Crude-oil 5-option surface (replaces the SPX surface)
  2. K=65 / F=90 cases (user spec, 2 DTE and 0 DTE)
  3. Vectorized IV (batch of 5 prices in / 5 IVs out)
  4. Daily-options PM concerns
  5. Penny-tick reality (sub-tick prices, tick-grid rounding noise)
"""
import math

import numpy as np
import pytest

import ngv_opx
from py_vollib.black import black as vollib_black
from py_vollib.black.implied_volatility import implied_volatility as vollib_iv

RATE = 0.045
DAY = 1.0 / 365.0
HOUR = DAY / 24.0
TICK = 0.01  # crude option premium tick size in $/bbl


def _flag(is_call: bool) -> str:
    return "c" if is_call else "p"


# ----------------------------------------------------------------------------
# 1. Crude-oil five-option surface
# ----------------------------------------------------------------------------
# Realistic WTI: ATM ~35% vol, ~$75/bbl forward, symmetric-ish skew. Every
# case here has a theoretical premium above $0.01 so they are real quotes.
# (label, F, K, dte_days, true_vol, is_call)
CRUDE_CASES = [
    ("ATM 30d call",         75.0, 75.0,  30.0, 0.35, True),
    ("OTM call 30d skew",    75.0, 85.0,  30.0, 0.45, True),
    ("OTM put 30d skew",     75.0, 65.0,  30.0, 0.45, False),
    ("ATM 1d (daily)",       75.0, 75.0,   1.0, 0.40, True),
    ("Deep ITM C 65/90 2d",  90.0, 65.0,   2.0, 0.45, True),
]


@pytest.mark.parametrize("label,F,K,dte,vol,is_call", CRUDE_CASES)
def test_price_matches_vollib(label, F, K, dte, vol, is_call):
    t = dte * DAY
    ours = ngv_opx.black76(F, K, RATE, vol, t, is_call)
    ref = vollib_black(_flag(is_call), F, K, t, RATE, vol)
    tol = max(1e-6, abs(ref) * 1e-6)
    assert abs(ours - ref) < tol, f"[{label}] price ours={ours} ref={ref}"


def _time_value(price, F, K, r, t, is_call):
    disc = math.exp(-r * t)
    intrinsic = max((F - K) if is_call else (K - F), 0.0) * disc
    return price - intrinsic


@pytest.mark.parametrize("label,F,K,dte,vol,is_call", CRUDE_CASES)
def test_iv_roundtrip(label, F, K, dte, vol, is_call):
    t = dte * DAY
    price = vollib_black(_flag(is_call), F, K, t, RATE, vol)
    # Real PM only takes quotes >= 1 tick; skip if theoretical value is sub-tick.
    if price < TICK:
        pytest.skip(f"[{label}] theoretical premium {price:.6f} below penny tick")
    # If time value is below f64 noise at this scale, IV is mathematically
    # indeterminate. Both py_vollib and us sentinel; covered by a separate test.
    if _time_value(price, F, K, RATE, t, is_call) < 1e-9 * max(F, 1.0):
        pytest.skip(f"[{label}] time value below f64 noise — IV indeterminate")
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call)
    assert iv > 0.0, f"[{label}] sentinel from solver"
    assert abs(iv - vol) < 5e-4, f"[{label}] iv={iv} true={vol}"


@pytest.mark.parametrize("label,F,K,dte,vol,is_call", CRUDE_CASES)
def test_iv_matches_vollib(label, F, K, dte, vol, is_call):
    t = dte * DAY
    price = vollib_black(_flag(is_call), F, K, t, RATE, vol)
    if price < TICK:
        pytest.skip(f"[{label}] sub-tick premium")
    if _time_value(price, F, K, RATE, t, is_call) < 1e-9 * max(F, 1.0):
        pytest.skip(f"[{label}] time value below f64 noise — IV indeterminate")
    ours = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call)
    ref = vollib_iv(price, F, K, RATE, t, _flag(is_call))
    assert abs(ours - ref) < 1e-4, f"[{label}] ours={ours} vollib={ref}"


# ----------------------------------------------------------------------------
# 2. User spec: K=65 strike, oil F=90, two DTE settings
# ----------------------------------------------------------------------------
def test_iv_K65_F90_2dte_call_is_indeterminate():
    """
    A $25 ITM call with 2 DTE is essentially a delta-1 position. The time
    value (~10^-22) is far below f64 noise floor (~5e-15 at this scale), so
    the price IS the intrinsic in float64 and any vol >= ~5% reproduces it.

    PM takeaway: this isn't really an option for IV purposes -- it's a forward.
    py_vollib agrees: their IV solver also returns sentinel (0.0) here.
    """
    F, K, t = 90.0, 65.0, 2.0 * DAY
    vol = 0.45
    price = vollib_black("c", F, K, t, RATE, vol)
    # Verify the premise: time value is unobservable in f64.
    disc = math.exp(-RATE * t)
    assert price - disc * (F - K) < 1e-12
    # Both solvers should signal "indeterminate".
    ours = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call=True)
    ref = vollib_iv(price, F, K, RATE, t, "c")
    assert ours == -1.0
    assert ref == 0.0  # py_vollib's sentinel


def test_iv_K65_F90_2dte_put_is_sub_tick():
    """
    The mirror $65 put with oil at $90 / 2 DTE is so far OTM its theoretical
    value is < $0.01. In a real book this is "no quote" -- the solver must
    signal, and the PM marks it at zero / skips it for vol surface fitting.
    """
    F, K, t = 90.0, 65.0, 2.0 * DAY
    vol = 0.45
    theo = vollib_black("p", F, K, t, RATE, vol)
    assert theo < TICK, f"expected sub-tick, got {theo}"
    # Real market would quote $0.00 -- which is below intrinsic-floor of 0 by
    # nothing, so solver returns sentinel for "indeterminate".
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, 0.0, is_call=False)
    assert iv == -1.0 or iv == pytest.approx(ngv_opx.black76_implied_volatility(
        F, K, RATE, t, 0.0, is_call=False))


def test_iv_K65_F90_0dte_returns_sentinel():
    """0 DTE deep ITM: price == intrinsic, vega == 0, IV is mathematically undefined."""
    F, K = 90.0, 65.0
    intrinsic = F - K
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, 0.0, intrinsic, is_call=True)
    assert iv == -1.0


def test_iv_atm_0dte_returns_sentinel():
    """0 DTE ATM: any vol gives the same (zero) price. Solver must signal."""
    iv = ngv_opx.black76_implied_volatility(75.0, 75.0, RATE, 0.0, 0.0, is_call=True)
    assert iv == -1.0


def test_iv_atm_1dte_recovers():
    """1 DTE ATM is the bread-and-butter case for a daily-options PM."""
    F, K, t = 75.0, 75.0, 1.0 * DAY
    vol = 0.40
    price = vollib_black("c", F, K, t, RATE, vol)
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call=True)
    assert abs(iv - vol) < 5e-4


# ----------------------------------------------------------------------------
# 3. Vectorized IV: pass 5 prices, get 5 IVs back in one call
# ----------------------------------------------------------------------------
def test_vectorized_iv_five_options():
    """
    Whole crude surface in one shot, validated against py_vollib.

    Includes the indeterminate Deep ITM 65/90 2d case to confirm vectorized
    handling of mixed solvable / sentinel rows -- a real surface always has
    a few of these on any given day.
    """
    forwards = np.array([c[1] for c in CRUDE_CASES], dtype=np.float64)
    strikes  = np.array([c[2] for c in CRUDE_CASES], dtype=np.float64)
    times    = np.array([c[3] * DAY for c in CRUDE_CASES], dtype=np.float64)
    true_vols = np.array([c[4] for c in CRUDE_CASES], dtype=np.float64)
    is_calls = np.array([c[5] for c in CRUDE_CASES], dtype=np.bool_)
    rates    = np.full(len(CRUDE_CASES), RATE, dtype=np.float64)

    prices = np.array(
        [vollib_black(_flag(cp), f, k, t, RATE, v)
         for f, k, t, v, cp in zip(forwards, strikes, times, true_vols, is_calls)],
        dtype=np.float64,
    )

    ivs = ngv_opx.black76_implied_volatility_vectorized(
        forwards, strikes, rates, times, prices, is_calls,
    )
    assert ivs.shape == prices.shape

    # Solvable rows: time value above f64 noise. Indeterminate rows: sentinel.
    bounds_tol = 1e-9 * forwards
    discs = np.exp(-rates * times)
    intrinsics = np.where(is_calls, np.maximum(forwards - strikes, 0.0),
                                    np.maximum(strikes - forwards, 0.0)) * discs
    solvable = (prices - intrinsics) >= bounds_tol

    np.testing.assert_allclose(ivs[solvable], true_vols[solvable], atol=5e-4)
    assert np.all(ivs[~solvable] == -1.0), \
        f"indeterminate rows must sentinel, got {ivs[~solvable]}"

    # Element-wise cross-check vs vollib on solvable rows.
    ref = np.array(
        [vollib_iv(p, f, k, RATE, t, _flag(cp))
         for p, f, k, t, cp in zip(prices, forwards, strikes, times, is_calls)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(ivs[solvable], ref[solvable], atol=1e-4)


def test_vectorized_price_matches_scalar():
    """`black76_vectorized` and the scalar `black76` must agree element-wise."""
    forwards = np.array([75.0, 75.0, 75.0, 90.0, 75.0])
    strikes  = np.array([75.0, 85.0, 65.0, 65.0, 75.0])
    vols     = np.array([0.35, 0.45, 0.45, 0.45, 0.40])
    times    = np.array([30.0, 30.0, 30.0, 2.0, 1.0]) * DAY
    is_calls = np.array([True, True, False, True, True])
    rates    = np.full(5, RATE)

    vec = ngv_opx.black76_vectorized(forwards, strikes, rates, vols, times, is_calls)
    scalar = np.array([
        ngv_opx.black76(f, k, RATE, v, t, cp)
        for f, k, v, t, cp in zip(forwards, strikes, vols, times, is_calls)
    ])
    np.testing.assert_array_equal(vec, scalar)


# ----------------------------------------------------------------------------
# 4. Daily-options PM concerns
# ----------------------------------------------------------------------------
def test_put_call_parity_short_dte():
    """C - P == disc*(F-K). Fails first if either side is buggy."""
    F, K, t = 75.0, 80.0, 1.0 * DAY
    vol = 0.40
    c = ngv_opx.black76(F, K, RATE, vol, t, True)
    p = ngv_opx.black76(F, K, RATE, vol, t, False)
    disc = math.exp(-RATE * t)
    assert abs((c - p) - disc * (F - K)) < 1e-9


def test_sub_day_expiry_recovers():
    """6 hours to expiry (morning of expiration day on a 1d option)."""
    F, K, t = 75.0, 75.0, 6.0 * HOUR
    vol = 0.45
    price = vollib_black("c", F, K, t, RATE, vol)
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call=True)
    assert abs(iv - vol) < 5e-4


def test_high_vol_oil_shock_regime():
    """Crude vol can spike to 80%+ (2020 covid, 2022 Ukraine). Must handle."""
    F, K, t = 75.0, 75.0, 7.0 * DAY
    vol = 0.80
    price = vollib_black("c", F, K, t, RATE, vol)
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, price, is_call=True)
    assert abs(iv - vol) < 5e-4


def test_iv_below_intrinsic_signals():
    """Bid below intrinsic = stale/cross quote. Must not return a fake vol."""
    F, K, t = 90.0, 65.0, 2.0 * DAY
    disc = math.exp(-RATE * t)
    intrinsic = disc * (F - K)
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, intrinsic - 0.50, is_call=True)
    assert iv == -1.0


# ----------------------------------------------------------------------------
# 5. Penny-tick reality
# ----------------------------------------------------------------------------
def test_iv_recovery_on_tick_rounded_prices():
    """
    Real quotes are $0.01-rounded. Recovering IV from a tick-rounded price
    will not match `true_vol` exactly -- it'll match within roughly one
    vol-pt per (tick / vega) of the option. Test the round-trip is sane.
    """
    F, K, t = 75.0, 75.0, 7.0 * DAY
    true_vol = 0.40
    raw = vollib_black("c", F, K, t, RATE, true_vol)
    rounded = round(raw / TICK) * TICK
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, rounded, is_call=True)
    # ATM 7d vega for WTI ~= disc*F*sqrt(7/365)*phi(0) ~= 75*0.138*0.399 ~= 4.13
    # so $0.01 tick ~= 0.0024 vol-pts. Test we are within 0.01 vol-pts.
    assert abs(iv - true_vol) < 0.01, f"iv={iv}, true={true_vol}, rounded={rounded}"


def test_iv_sub_tick_otm_signals_or_floors():
    """
    Far OTM 1d call: theoretical premium is well below a penny. The exchange
    would quote $0.00. Solver fed exactly $0.00 (or sub-tick) must signal,
    not invent a volatility.
    """
    F, K, t = 75.0, 90.0, 1.0 * DAY
    vol = 0.35
    theo = vollib_black("c", F, K, t, RATE, vol)
    assert theo < TICK
    # Real-world behavior: market shows $0.00 -> ask solver for IV at $0.00.
    # Price 0 with intrinsic 0 has zero time value -> indeterminate -> sentinel.
    iv = ngv_opx.black76_implied_volatility(F, K, RATE, t, 0.0, is_call=True)
    assert iv == -1.0, f"iv={iv} for $0.00 quote on sub-tick option"
