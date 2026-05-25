"""
Plotly chart of the 5 Black-76 pricing paths benchmarked side by side:

  1. py_vollib (scalar, in Python for-loop)        — the quants' reference
  2. Python pure (math.erf, in Python for-loop)
  3. Python vectorized (numpy + scipy.special.ndtr)
  4. ngv_opx Rust scalar (PyO3 call per option, in Python for-loop)
  5. ngv_opx Rust vectorized (one PyO3 call, Rust loop)

X axis: batch size N (log).
Y axis: per-option time in nanoseconds (log) — flat lines mean linear scaling;
        gaps between lines are the speedup multipliers.

Writes benchmarks/bench_chart.html and opens it.

Run: uv run python benchmarks/bench_chart.py
"""
import math
import sys
import time
import types
import webbrowser
from pathlib import Path

if "_testcapi" not in sys.modules:
    _shim = types.ModuleType("_testcapi")
    _shim.DBL_MIN = sys.float_info.min
    _shim.DBL_MAX = sys.float_info.max
    sys.modules["_testcapi"] = _shim

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtr
from py_vollib.black import black as vollib_black

import ngv_opx

RATE = 0.045
SQRT2 = math.sqrt(2.0)
SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


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


def make_book(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = np.full(n, 75.0, dtype=np.float64)
    k = rng.uniform(50.0, 100.0, n).astype(np.float64)
    v = rng.uniform(0.25, 0.65, n).astype(np.float64)
    t = (rng.uniform(1.0, 90.0, n) / 365.0).astype(np.float64)
    cp = rng.integers(0, 2, n).astype(bool)
    r = np.full(n, RATE, dtype=np.float64)
    return f, k, r, v, t, cp


def best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def collect():
    """Returns dict: path_name -> {N: ns_per_option}."""
    # Two axes only:
    #   Language: Python | Rust   (where the *caller* lives)
    #   Pricer:   which library / implementation
    # Note: "Python — ngv_opx" rows are calling Rust under the hood via PyO3.
    # The "Rust — ngv_opx (vectorized)" row is timed by a separate cargo binary
    # and shows what the Rust pricer can do with zero FFI overhead.
    paths = {
        "Python — py_vollib":              {},
        "Python — pure (math.erf)":        {},
        "Python — numpy/scipy":            {},
        "Python — ngv_opx (single)":       {},
        "Python — ngv_opx (vectorized)":   {},
        "JavaScript — @ngv/opx (single)":     {},
        "JavaScript — @ngv/opx (vectorized)": {},
        "Rust — ngv_opx (vectorized)":         {},
    }
    for n in SIZES:
        print(f"  N = {n:,} ...", flush=True)
        f, k, r, v, t, cp = make_book(n)

        rep_fast = 5 if n <= 1_000 else 3
        # Slow paths always run -- the user wants the full grid filled in.
        # Just dial reps down so 1M doesn't take forever.
        rep_slow = 3 if n <= 1_000 else (2 if n <= 10_000 else 1)

        t_vol = best_of(
            lambda: [vollib_black("c" if cp[i] else "p", f[i], k[i], t[i], r[i], v[i]) for i in range(n)],
            rep_slow,
        )
        paths["Python — py_vollib"][n] = t_vol / n * 1e9

        t_py_s = best_of(
            lambda: [py_b76_scalar(f[i], k[i], r[i], v[i], t[i], bool(cp[i])) for i in range(n)],
            rep_slow,
        )
        paths["Python — pure (math.erf)"][n] = t_py_s / n * 1e9

        t_py_v = best_of(lambda: py_b76_vectorized(f, k, r, v, t, cp), rep_fast)
        paths["Python — numpy/scipy"][n] = t_py_v / n * 1e9

        rep_rs_s = 3 if n <= 10_000 else 1
        t_rs_s = best_of(
            lambda: [ngv_opx.black76(f[i], k[i], r[i], v[i], t[i], bool(cp[i])) for i in range(n)],
            rep_rs_s,
        )
        paths["Python — ngv_opx (single)"][n] = t_rs_s / n * 1e9

        t_rs_v = best_of(lambda: ngv_opx.black76_vectorized(f, k, r, v, t, cp), rep_fast)
        paths["Python — ngv_opx (vectorized)"][n] = t_rs_v / n * 1e9

    # Native Rust: run the cargo example, parse its JSON line. No PyO3.
    print("  running native Rust benchmark via cargo...", flush=True)
    import json, subprocess
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "bench_native"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    native = json.loads(proc.stdout.strip().splitlines()[-1])
    for n_str, ns in native.items():
        paths["Rust — ngv_opx (vectorized)"][int(n_str)] = float(ns)

    # JavaScript (wasm): same pattern — subprocess Node, parse the last JSON line.
    # Requires `node` on PATH and bindings/wasm/pkg-node/ already built.
    print("  running JavaScript (wasm) benchmark via node...", flush=True)
    try:
        js_proc = subprocess.run(
            ["node", str(repo_root / "benchmarks" / "bench_wasm.mjs")],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
        js = json.loads(js_proc.stdout.strip().splitlines()[-1])
        for n_str, ns in js["js-single"].items():
            paths["JavaScript — @ngv/opx (single)"][int(n_str)] = float(ns)
        for n_str, ns in js["js-vec"].items():
            paths["JavaScript — @ngv/opx (vectorized)"][int(n_str)] = float(ns)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  warning: skipping JS rows — {type(e).__name__}: {e}", flush=True)

    return paths


def _fmt_ns(x):
    """All values rendered in nanoseconds with comma thousands separator,
    so the eye doesn't have to convert between µs and ns mid-row."""
    if x is None:
        return "—"
    return f"{x:,.1f} ns"


def render(paths) -> Path:
    # Three language axes now: Python, JavaScript (wasm), Rust (native).
    PYTHON_GROUP = [
        "Python — pure (math.erf)",
        "Python — numpy/scipy",
        "Python — py_vollib",
        "Python — ngv_opx (single)",
        "Python — ngv_opx (vectorized)",
    ]
    JS_GROUP = [
        "JavaScript — @ngv/opx (single)",
        "JavaScript — @ngv/opx (vectorized)",
    ]
    RUST_GROUP = [
        "Rust — ngv_opx (vectorized)",
    ]
    ORDERED = PYTHON_GROUP + JS_GROUP + RUST_GROUP

    # Visual differentiation: NGV solution stands out, other libs are muted.
    #   Rust ngv_opx       — purple   (the floor / fastest, highlighted)
    #   JavaScript ngv_opx — green    (the new wasm path, highlighted)
    #   Python ngv_opx     — blue     (the shipped Python product)
    #   Other Python libs  — greys    (the comparison/baseline crowd)
    NGV_PYTHON = {"Python — ngv_opx (single)", "Python — ngv_opx (vectorized)"}
    NGV_JS = set(JS_GROUP)
    NGV_RUST = {"Rust — ngv_opx (vectorized)"}

    color = {
        "Python — py_vollib":                  "#999999",   # grey
        "Python — pure (math.erf)":            "#666666",   # darker grey
        "Python — numpy/scipy":                "#333333",   # darkest grey
        "Python — ngv_opx (single)":           "#4a90e2",   # blue
        "Python — ngv_opx (vectorized)":       "#1f5fb5",   # darker blue
        "JavaScript — @ngv/opx (single)":      "#3aa657",   # green
        "JavaScript — @ngv/opx (vectorized)":  "#1f7a3b",   # darker green
        "Rust — ngv_opx (vectorized)":         "#7b3fb5",   # purple
    }
    dash = {n: ("dot" if n in PYTHON_GROUP else "solid") for n in ORDERED}

    # Column tints — reinforce the "NGV stands out" story.
    OTHER_BG   = "#f3f3f3"  # neutral grey for other Python libs
    NGV_PY_BG  = "#d6e6fb"  # blue for ngv_opx Python rows
    NGV_JS_BG  = "#d6f0dc"  # green for ngv_opx wasm rows
    NGV_RS_BG  = "#e7d5f5"  # purple for ngv_opx Rust native row
    # Matching "chip" colors for the H1 label badges (light bg, dark text).
    PY_CHIP_BG,   PY_CHIP_FG = "#dcdcdc", "#222222"   # light grey chip, near-black text
    JS_CHIP_BG,   JS_CHIP_FG = "#c8e6cf", "#1f7a3b"   # light green chip, dark green text
    RS_CHIP_BG,   RS_CHIP_FG = "#d8c4ea", "#4a1f7a"   # lavender chip, dark purple text

    # Stacked layout: table on top, chart below. Chart gets the lion's share
    # of vertical space since the lines are the main read; table is reference.
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.40, 0.60],
        vertical_spacing=0.06,
    )

    for name in ORDERED:
        by_n = paths[name]
        xs = sorted(by_n.keys())
        ys = [by_n[n] for n in xs]
        if name in PYTHON_GROUP:
            lang = "Python"
        elif name in JS_GROUP:
            lang = "JavaScript"
        else:
            lang = "Rust"
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=name,
                line=dict(color=color[name], dash=dash[name], width=2),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{name}</b><br>language: {lang}"
                    "<br>N=%{x:,}<br>%{y:.1f} ns/option<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    # ---- Table: 3-line stacked header per column ----
    # Each header cell renders:
    #   Row 1: LANGUAGE chip (PYTHON / JAVASCRIPT / RUST)
    #   Row 2: library name  (py_vollib, ngv_opx, @ngv/opx, math.erf, numpy/scipy)
    #   Row 3: variant       (single / vectorized / —)
    # Cells stack vertically inside one Plotly header row, so column widths
    # can shrink and the eye groups by language first, then library, then
    # call shape.
    library_label = {
        "Python — py_vollib":                  "py_vollib",
        "Python — pure (math.erf)":            "math.erf",
        "Python — numpy/scipy":                "numpy/scipy",
        "Python — ngv_opx (single)":           "ngv_opx",
        "Python — ngv_opx (vectorized)":       "ngv_opx",
        "JavaScript — @ngv/opx (single)":      "@ngv/opx",
        "JavaScript — @ngv/opx (vectorized)":  "@ngv/opx",
        "Rust — ngv_opx (vectorized)":         "ngv_opx",
    }
    variant_label = {
        "Python — py_vollib":                  "single",
        "Python — pure (math.erf)":            "single",
        "Python — numpy/scipy":                "vectorized",
        "Python — ngv_opx (single)":           "single",
        "Python — ngv_opx (vectorized)":       "vectorized",
        "JavaScript — @ngv/opx (single)":      "single",
        "JavaScript — @ngv/opx (vectorized)":  "vectorized",
        "Rust — ngv_opx (vectorized)":         "vectorized",
    }

    def column_bg(name: str) -> str:
        if name in NGV_RUST:
            return NGV_RS_BG
        if name in NGV_JS:
            return NGV_JS_BG
        if name in NGV_PYTHON:
            return NGV_PY_BG
        return OTHER_BG

    def h1_chip(name: str) -> str:
        """Return an HTML 'chip' for the H1 (language) row."""
        if name in PYTHON_GROUP:
            text, bg, fg = "PYTHON", PY_CHIP_BG, PY_CHIP_FG
        elif name in JS_GROUP:
            text, bg, fg = "JAVASCRIPT", JS_CHIP_BG, JS_CHIP_FG
        else:
            text, bg, fg = "RUST", RS_CHIP_BG, RS_CHIP_FG
        return (
            f"<span style='background:{bg};color:{fg};font-size:10px;"
            f"font-weight:700;padding:2px 8px;border-radius:3px;"
            f"letter-spacing:1px;'>{text}</span>"
        )

    all_ns = sorted({n for by_n in paths.values() for n in by_n.keys()})

    def stacked_header(name: str) -> str:
        """Build the 3-line stacked cell: language chip / library / variant."""
        chip = h1_chip(name)
        lib = library_label[name]
        var = variant_label[name]
        # Tinted strip behind the library name reinforces the column's
        # language group. The variant row is plain so it reads as a sub-axis.
        bg = column_bg(name)
        return (
            f"{chip}<br>"
            f"<span style='background:{bg};display:inline-block;"
            f"padding:1px 8px;border-radius:3px;'>"
            f"<b style='font-size:12px'>{lib}</b></span><br>"
            f"<span style='font-size:10.5px;color:#555;font-style:italic'>{var}</span>"
        )

    header_vals = [
        "<b>N</b><br><span style='font-size:10px;color:#888'>(batch size)</span>"
    ] + [stacked_header(name) for name in ORDERED]
    header_bg = ["#ececec"] + [column_bg(name) for name in ORDERED]

    col_n = [f"{n:,}" for n in all_ns]
    col_values = [col_n] + [
        [_fmt_ns(paths[name].get(n)) for n in all_ns]
        for name in ORDERED
    ]
    cell_fill = [["white"] * len(all_ns)] + [
        [column_bg(name)] * len(all_ns) for name in ORDERED
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=header_vals,
                fill_color=header_bg,
                align="center",
                font=dict(size=12),
                height=82,  # 3 stacked lines + padding
            ),
            cells=dict(
                values=col_values,
                fill_color=cell_fill,
                align="right",
                font=dict(family="monospace", size=11),
                height=22,
            ),
            # Each column can now be narrower since the stacked header
            # carries less text per line.
            columnwidth=[60] + [110] * len(ORDERED),
        ),
        row=1, col=1,
    )

    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    fig.update_xaxes(title_text="Batch size N", type="log", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="ns per option", type="log", showgrid=True, row=2, col=1)
    fig.update_layout(
        title=dict(
            # Subtitle split across two lines so the color legend doesn't
            # overflow the chart width.
            text=(
                f"Black-76 pricing throughput — Python {pyver}<br>"
                f"<sub>"
                f"<span style='color:#7b3fb5'>● Rust ngv_opx (native)</span>"
                f" &nbsp;·&nbsp; <span style='color:#1f7a3b'>● JS @ngv/opx (wasm)</span>"
                f" &nbsp;·&nbsp; <span style='color:#1f5fb5'>● Python ngv_opx</span>"
                f" &nbsp;·&nbsp; <span style='color:#555'>● other Python libs</span>"
                f"<br>Log-log; lower is faster."
                f"</sub>"
            ),
            x=0.5, xanchor="center",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.0),
        template="plotly_white",
        # Taller overall figure + bigger top margin to make room for the
        # 2-line subtitle, and a taller table area (row_heights above) so
        # the last (N=1,000,000) row isn't clipped.
        height=1100, width=1200,
        margin=dict(l=70, r=30, t=130, b=80),
    )
    out = Path(__file__).parent / "bench_chart.html"
    fig.write_html(out, include_plotlyjs="cdn")
    try:
        out_png = out.with_suffix(".png")
        fig.write_image(out_png, width=1600, height=1467, scale=2)
    except Exception as e:
        print(f"  (skipped PNG export: {e})", flush=True)
    return out


if __name__ == "__main__":
    print("collecting benchmarks...")
    paths = collect()
    out = render(paths)
    print(f"\nwrote {out}")
    try:
        webbrowser.open(out.as_uri())
    except Exception:
        pass
