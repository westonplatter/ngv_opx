# Vol Surface Demo

A live implied-volatility visualization that simulates an options market: the backend
streams **individual trades**, and the browser solves each traded price back into an
implied vol using the `@westonplatter/ngv-opx` WASM solver, updating only the points that traded.

There are two separate volatility steps, and they happen on opposite sides:

1. **Backend (vols → prices).** The server knows a *fair* synthetic vol surface and uses
   Black-76 to **price** options. It never sends vols — only prices go over the wire.
2. **Frontend (prices → vols).** When a contract trades, the browser **solves** that one
   price back into an implied vol with the `@westonplatter/ngv-opx` WASM solver and moves that single
   point on the smile.

The stream works in two stages:

- **One snapshot on connect** — the full option chain, so the chart starts with a
  complete surface (every strike × expiry).
- **Then a stream of trades** — small bursts of individual contracts that just traded,
  about every 400 ms. Each trade moves exactly one point; untraded points stay where
  they were. This mirrors real markets, where the surface only updates where prints
  happen — and means the browser solves IV per trade, not the whole chain every tick.

```
        vols → prices                          prices → vols  (the IV solve)
┌───────────────────┐                    ┌─────────────────────────┐
│  FastAPI backend  │  snapshot + trades │  React + Vite frontend  │
│  Black-76 pricing │ ── WebSocket ───►  │  @westonplatter/ngv-opx WASM solver   │
│  :8000/ws         │   (JSON messages)  │  Plotly chart  :5173    │
└───────────────────┘                    └─────────────────────────┘
   knows fair vols                          solves IV per trade
```

## Quick start

```bash
bash start.sh
```

This one-shot script:

1. Builds the `@westonplatter/ngv-opx` WASM bundle (`wasm-pack build --target web`).
2. Installs frontend deps (`npm install`).
3. Installs backend deps with **uv** (`uv sync`).
4. Starts the FastAPI backend on **:8000** and the Vite dev server on **:5173**.

Then open **http://localhost:5173**. Press `Ctrl-C` to stop both servers.

> Requirements: [`uv`](https://docs.astral.sh/uv/), Node.js, and the Rust
> `wasm-pack` toolchain.

## Backend (`backend/`)

A FastAPI app that simulates a synthetic options market — a self-contained **uv project**
(`pyproject.toml` + `uv.lock`), independent of the repo's main Python package.

- **`main.py`** — single `/ws` WebSocket endpoint. On connect it sends one full
  `snapshot`, then loops forever sending `trades` messages every **400 ms**.
- **Market model** — spot ≈ $50, strikes $35–$60 (26 strikes), expiries 1–5 weeks. The
  spot drifts slowly while trades stream.
- **Vol surface** — `skewed_iv()` builds each expiry's fair smile from an ATM vol plus a
  quadratic smile and linear skew in log-moneyness, scaled by `1/√T` so short-dated
  smiles are steeper. The **2wk expiry carries an earnings event** (ATM IV ~60% vs.
  ~30–35% elsewhere).
- **Pricing** — a pure-Python **Black-76** pricer. `generate_snapshot()` prices the whole
  chain for the opening surface; `simulate_trade()` picks a random contract and fills it
  at mid ± part of the spread (someone lifting the ask or hitting the bid).

Two message types, both JSON:

```jsonc
// 1) sent once on connect — the full chain
{ "type": "snapshot", "spot": 50.0, "timestamp": 1717000000.0,
  "quotes": [
    { "expiry": "2wk", "t_years": 0.038356, "strike": 50,
      "call_bid": 0.91, "call_ask": 0.94, "put_bid": 0.90, "put_ask": 0.93 }
  ] }

// 2) streamed ~every 400 ms — a small burst of individual trades
{ "type": "trades", "spot": 50.03, "timestamp": 1717000000.4,
  "trades": [
    { "expiry": "2wk", "t_years": 0.038356, "strike": 50,
      "is_call": true, "price": 0.93, "size": 12 }
  ] }
```

Run the backend on its own:

```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Frontend (`frontend/`)

A React + TypeScript app built with Vite that consumes the trade stream and renders the
vol surface as a 2D chart.

- **`src/App.tsx`** — initializes the `@westonplatter/ngv-opx` WASM module once, connects to the
  backend WebSocket (auto-reconnecting on drop), and keeps the surface in memory so it
  can patch individual points as trades arrive.
- **Snapshot** — on the opening `snapshot`, it picks the OTM option at each strike (call
  when `K ≥ spot`, put otherwise) and calls `impliedVolBatch()` once per expiry to solve
  the whole starting surface.
- **Trades** — for each `trades` message it solves **only the traded contracts** with the
  scalar `impliedVol()` and updates just those points. A `-1` sentinel (no solution) is
  skipped. This is the key behavior: one IV solve per trade, not a full-chain recompute.
- **Chart** — [Plotly](https://plotly.com/javascript/) line chart, **Strike ($) on X,
  Implied Vol (%) on Y**, with **one color-coded line series per expiry** (2wk
  highlighted as the earnings event). Each point fades from bright to dim over a few
  seconds since its last trade, so you can see what's freshly traded vs. stale. A dashed
  vertical marker tracks the current spot, and unified hover compares all expiries at a
  given strike.
- A status bar shows connection state, live spot, and a trades/sec counter.

Key dependencies: `react`, `plotly.js-dist-min`, and `@westonplatter/ngv-opx` (linked locally from
`../../bindings/wasm`). Vite's `optimizeDeps.exclude` keeps esbuild from pre-bundling the
WASM package so the `wasm-pack` `init()` URL resolution works.

Run the frontend on its own (backend must be up):

```bash
cd frontend
npm install
npm run dev
```

## Notes

- The market data is **fully synthetic** — there is no live feed or real pricing.
- The point of the demo is the round-trip: prices generated from known vols on the
  backend are independently re-solved to implied vols on the frontend by the Rust/WASM
  core, and the recovered smiles should match the input shape.
- It also shows the **trade-driven update model** — the surface only moves where a trade
  prints, and IV is solved per trade rather than recomputed for the whole chain each tick.
