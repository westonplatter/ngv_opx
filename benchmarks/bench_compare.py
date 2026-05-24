"""
Cross-Python-version comparison of bench_cpu.py results.

Reads every benchmarks/results/py*.json (one per Python version, produced by
`task bench:all-pythons`), then prints pandas tables comparing per-option cost
across versions and writes a faceted Plotly chart.

Tables: one per path, rows = N, columns = Python version, values = ns/option.
Chart : 5 subplots (one per path), each plotting ns/option vs N for every
        Python version on the same axes.

Run:
  task bench:all-pythons    # populates benchmarks/results/
  task bench:compare        # this script
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = Path(__file__).parent / "results"
OUT_HTML = Path(__file__).parent / "bench_compare.html"
PATHS = ["py-single", "py-vec", "rs-single", "rs-vec", "vollib"]


def load_runs() -> dict[str, dict]:
    """{python_version_str: {sizes: [...], paths: {path: [seconds_or_None, ...]}}}"""
    files = sorted(RESULTS_DIR.glob("py*.json"))
    if not files:
        sys.exit(f"no results found under {RESULTS_DIR} — run `task bench:all-pythons` first")
    runs = {}
    for f in files:
        payload = json.loads(f.read_text())
        runs[payload["python"]] = payload
    return dict(sorted(runs.items()))  # sort by version string for stable order


def build_per_option_table(runs: dict, path: str) -> pd.DataFrame:
    """Rows = N (size), columns = python version, values = ns/option."""
    rows = {}
    for ver, payload in runs.items():
        col = {}
        for n, secs in zip(payload["sizes"], payload["paths"][path]):
            col[n] = None if secs is None else (secs / n) * 1e9
        rows[ver] = col
    df = pd.DataFrame(rows)
    df.index.name = "N"
    return df


def fmt_ns(x):
    """Always ns, with thousands separator. Same column = same unit, always."""
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.1f} ns"


def print_tables(runs: dict):
    print(f"\nLoaded {len(runs)} Python version(s): {', '.join(runs.keys())}")
    for path in PATHS:
        df = build_per_option_table(runs, path)
        pretty = df.copy()
        pretty.index = [f"{n:>9,}" for n in pretty.index]
        pretty.index.name = "N"
        pretty = pretty.map(fmt_ns)
        print(f"\n— {path} (ns / option) —")
        print(pretty.to_string())


def render_chart(runs: dict) -> Path:
    titles = PATHS
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )
    # Stable color per Python version, same color across subplots.
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    ver_color = {ver: palette[i % len(palette)] for i, ver in enumerate(runs.keys())}

    for idx, path in enumerate(PATHS):
        row = idx // 3 + 1
        col = idx % 3 + 1
        df = build_per_option_table(runs, path)
        for ver in df.columns:
            ys = df[ver].dropna()
            if ys.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=ys.index.tolist(),
                    y=ys.tolist(),
                    mode="lines+markers",
                    name=f"Py {ver}",
                    legendgroup=ver,
                    showlegend=(idx == 0),  # legend only on first subplot
                    line=dict(color=ver_color[ver], width=2),
                    marker=dict(size=7),
                    hovertemplate=f"{path}<br>Py {ver}<br>N=%{{x:,}}<br>%{{y:.1f}} ns/opt<extra></extra>",
                ),
                row=row, col=col,
            )
        fig.update_xaxes(type="log", title_text="N" if row == 2 else None, row=row, col=col)
        fig.update_yaxes(type="log", title_text="ns/opt" if col == 1 else None, row=row, col=col)

    fig.update_layout(
        title=dict(
            text="Black-76 pricing — per-option cost across Python versions"
                 "<br><sub>log-log; one subplot per path. Same color = same Python version.</sub>",
            x=0.5, xanchor="center",
        ),
        template="plotly_white",
        height=640, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.0),
        margin=dict(l=70, r=30, t=90, b=70),
    )
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    return OUT_HTML


if __name__ == "__main__":
    runs = load_runs()
    print_tables(runs)
    out = render_chart(runs)
    print(f"\nwrote {out}")
