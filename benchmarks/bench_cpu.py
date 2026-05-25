"""
CPU benchmark: {library} x {single, vectorized} for Black-76 pricing.

Five paths, all computing the same prices on the same synthetic WTI book:

  language / lib   | single (loop)              | vectorized
  ---------------- | -------------------------- | --------------------------
  Python (stdlib)  | math.erf, per option       | numpy + scipy.special.ndtr
  Rust  (ngv_opx)  | ngv_opx.black76, per opt   | ngv_opx.black76_vectorized
  py_vollib (ref)  | py_vollib.black, per opt   | (n/a — no batch API)

py_vollib is the float64 "Let's Be Rational" reference quants will recognize.
It's scalar-only by design, so it sits in the single-option column.

Run:
  uv run python benchmarks/bench_cpu.py
  bash benchmarks/bench_all_pythons.sh   # cross-version sweep (3.11/3.12/3.13)
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

# py_lets_be_rational needs _testcapi.DBL_MIN/DBL_MAX on Python 3.12+
if "_testcapi" not in sys.modules:
    _shim = types.ModuleType("_testcapi")
    _shim.DBL_MIN = sys.float_info.min
    _shim.DBL_MAX = sys.float_info.max
    sys.modules["_testcapi"] = _shim

import numpy as np
import pandas as pd
from scipy.special import ndtr
from py_vollib.black import black as vollib_black

import ngv_opx

COLS = ["py-single", "py-vec", "rs-single", "rs-vec", "vollib", "js-single", "js-vec"]

RATE = 0.045
SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
SQRT2 = math.sqrt(2.0)

# Filled once at startup by _collect_js_timings(); maps N -> {"js-single": s,
# "js-vec": s} in seconds-per-batch (already converted from ns/option).
_JS_TIMINGS: dict[int, dict[str, float]] = {}


def _collect_js_timings() -> None:
    """Subprocess `node benchmarks/bench_wasm.mjs`, parse its JSON, and
    populate `_JS_TIMINGS`. The JS bench emits ns/option per (path, N); we
    convert to seconds-per-batch so it slots into the same data model as the
    Python/Rust paths timed inline in this module."""
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "bench_wasm.mjs"
    try:
        proc = subprocess.run(
            ["node", str(script)],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"warning: skipping JS columns — {type(e).__name__}: {e}", file=sys.stderr)
        return
    raw = json.loads(proc.stdout.strip().splitlines()[-1])
    for n in SIZES:
        _JS_TIMINGS[n] = {
            "js-single": raw["js-single"][str(n)] * n / 1e9,
            "js-vec":    raw["js-vec"][str(n)] * n / 1e9,
        }


# ----------------------------------------------------------------------------
# Pure-Python Black-76 (scalar + numpy-vectorized)
# ----------------------------------------------------------------------------
def py_b76_scalar(f, k, r, sigma, t, is_call):
    sqrt_t = math.sqrt(t)
    sst = sigma * sqrt_t
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / sst
    d2 = d1 - sst
    disc = math.exp(-r * t)
    n_d1 = 0.5 * (1.0 + math.erf(d1 / SQRT2))
    n_d2 = 0.5 * (1.0 + math.erf(d2 / SQRT2))
    if is_call:
        return disc * (f * n_d1 - k * n_d2)
    return disc * (k * (1.0 - n_d2) - f * (1.0 - n_d1))


def py_b76_vectorized(f, k, r, sigma, t, is_call):
    sqrt_t = np.sqrt(t)
    sst = sigma * sqrt_t
    d1 = (np.log(f / k) + 0.5 * sigma * sigma * t) / sst
    d2 = d1 - sst
    disc = np.exp(-r * t)
    call = disc * (f * ndtr(d1) - k * ndtr(d2))
    put = disc * (k * ndtr(-d2) - f * ndtr(-d1))
    return np.where(is_call, call, put)


# ----------------------------------------------------------------------------
# Synthetic book + timing
# ----------------------------------------------------------------------------
def make_book(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    forwards = np.full(n, 75.0, dtype=np.float64)
    strikes = rng.uniform(50.0, 100.0, n).astype(np.float64)
    vols = rng.uniform(0.25, 0.65, n).astype(np.float64)
    times = (rng.uniform(1.0, 90.0, n) / 365.0).astype(np.float64)
    is_calls = rng.integers(0, 2, n).astype(bool)
    rates = np.full(n, RATE, dtype=np.float64)
    return forwards, strikes, rates, vols, times, is_calls


def best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def _measure_row(n: int) -> dict:
    """Time all 5 paths at batch size N; return seconds (NaN for skipped cells)."""
    f, k, r, v, t, cp = make_book(n)

    def py_single():
        return [py_b76_scalar(f[i], k[i], r[i], v[i], t[i], bool(cp[i])) for i in range(n)]

    def py_vec():
        return py_b76_vectorized(f, k, r, v, t, cp)

    def rs_single():
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = ngv_opx.black76(f[i], k[i], r[i], v[i], t[i], bool(cp[i]))
        return out

    def rs_vec():
        return ngv_opx.black76_vectorized(f, k, r, v, t, cp)

    def vollib_single():
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = vollib_black("c" if cp[i] else "p", f[i], k[i], t[i], r[i], v[i])
        return out

    rep_fast = 5 if n <= 1_000 else 3
    # Run every path at every size — dial reps down at large N so the slow
    # paths (py_vollib, pure Python) still finish in a reasonable wall-clock.
    rep_slow = 3 if n <= 1_000 else (2 if n <= 10_000 else 1)
    rep_rs_single = 3 if n <= 10_000 else 1

    times = {
        "py-single": best_of(py_single, rep_slow),
        "py-vec":    best_of(py_vec, rep_fast),
        "rs-single": best_of(rs_single, rep_rs_single),
        "rs-vec":    best_of(rs_vec, rep_fast),
        "vollib":    best_of(vollib_single, rep_slow),
        "js-single": _JS_TIMINGS.get(n, {}).get("js-single", float("nan")),
        "js-vec":    _JS_TIMINGS.get(n, {}).get("js-vec",    float("nan")),
    }

    # Correctness sanity at the smallest size.
    if n == 10:
        ps = np.array(py_single())
        assert np.allclose(ps, py_vec(), atol=1e-10)
        assert np.allclose(ps, rs_single(), atol=1e-10)
        assert np.allclose(ps, rs_vec(), atol=1e-10)
        assert np.allclose(ps, vollib_single(), atol=1e-10)
    return times


def _fmt_total(seconds: float) -> str:
    if math.isnan(seconds):
        return "skipped"
    return f"{seconds * 1e3:.3f} ms"


def _fmt_per(seconds: float, n: int) -> str:
    """Per-option cost always in ns, with thousands separator. Keeps every
    row on the same scale so the eye doesn't switch between µs and ns."""
    if math.isnan(seconds):
        return "—"
    return f"{seconds / n * 1e9:,.1f} ns"


def run_bench(save_path: Path | None = None):
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\nBlack-76 PRICE — Python {pyver}   (best-of-N wall-clock)")

    # JS columns are produced by a separate Node subprocess. Run it once
    # before the inline Python/Rust grid so every row has the data ready.
    print("collecting JS (wasm) timings via node...", flush=True)
    _collect_js_timings()

    # Collect raw timings (seconds) into a DataFrame indexed by N.
    raw = pd.DataFrame(
        {n: _measure_row(n) for n in SIZES},
        index=COLS,
    ).T
    raw.index.name = "N"

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "python": pyver,
            "sizes": SIZES,
            "paths": {col: [None if math.isnan(v) else v for v in raw[col].tolist()] for col in COLS},
        }
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"saved raw timings → {save_path}")

    # Render two side-by-side views: total wall-clock and per-option cost.
    totals = raw.apply(lambda col: col.map(_fmt_total))
    per_opt = raw.apply(lambda col: [_fmt_per(s, n) for s, n in zip(col, col.index)])

    # Pretty index labels with thousands separators.
    pretty_index = [f"{n:>9,}" for n in raw.index]
    totals.index = pretty_index
    per_opt.index = pretty_index
    totals.index.name = "N"
    per_opt.index.name = "N"

    print("\n— total wall-clock —")
    print(totals.to_string())
    print("\n— per-option cost —")
    print(per_opt.to_string())

    # Speedup table vs py_vollib at the largest size where every path ran.
    n_ref = max(n for n in SIZES if not math.isnan(raw.at[n, "vollib"])
                and not math.isnan(raw.at[n, "py-single"]))
    base = raw.at[n_ref, "vollib"]
    speedups = pd.Series(
        {
            col: base / raw.at[n_ref, col]
            for col in COLS
            if col != "vollib" and not math.isnan(raw.at[n_ref, col])
        },
        name=f"×vs py_vollib @ N={n_ref:,}",
    ).map(lambda x: f"{x:.1f}x")
    print(f"\n— speedups vs py_vollib at N={n_ref:,} (what migrating gets you) —")
    print(speedups.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, default=None,
                    help="write raw timings (seconds) as JSON to this path")
    args = ap.parse_args()
    run_bench(save_path=args.save)
    print()
