# ngv-opx web demo

Tiny Vite + TypeScript app that proves the `@ngv/opx` browser entry works end-to-end. Computes Black-76 implied vol live as you type, plus a 10 000-contract batch benchmark.

## Run

The demo consumes `@ngv/opx` via a local `file:` link to `../../bindings/wasm/`, so make sure that package is built first:

```bash
# from repo root
cd bindings/wasm && wasm-pack build --target web --out-dir pkg-web

# then run the demo
cd ../../examples/web-demo
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

## What it shows

- **Live IV solver** — single-contract `impliedVol(F, K, r, T, price, isCall)` recomputed on every input change. Renders the `-1.0` sentinel as a human-readable "IV undefined" message.
- **Batch benchmark** — 10 000 random contracts through `impliedVolBatch`, timed with `performance.now()`. Shows wall-clock and solve rate.

All numerics are f64 and identical to the Python production binding.
