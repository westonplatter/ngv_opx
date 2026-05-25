# ngv-opx web demo

Tiny Vite + TypeScript app that proves the `@ngv/opx` browser entry works end-to-end. Computes Black-76 implied vol live as you type, plus a 10 000-contract batch benchmark.

This demo consumes `@ngv/opx` via a local `file:` link to `../../bindings/wasm/`, so the wasm package must be built first.

## Run

From the repo root:

```bash
# 1. Build the wasm package for both browser and Node targets.
cd bindings/wasm
wasm-pack build --target web --out-dir pkg-web
wasm-pack build --target nodejs --out-dir pkg-node

# 2. Install demo deps (re-links bindings/wasm/ via file:) and start Vite.
cd ../../examples/web-demo
npm install
npm run dev
```

Vite will open `http://localhost:5173` automatically. With the default inputs (F=K=75, r=0.045, T=30d, price=2.99, call), the **Implied volatility** panel should show `35.118%`. The **Run benchmark** button solves 10 000 random IVs and reports wall-clock time.

## Iterating on the wasm package

The demo links `@ngv/opx` from `../../bindings/wasm/` at `npm install` time — npm hardlinks/copies the package's files, so **changes to `bindings/wasm/` do not propagate automatically**. After editing the wasm crate or the JS shims:

```bash
# Rebuild wasm artifacts.
cd bindings/wasm
wasm-pack build --target web --out-dir pkg-web
wasm-pack build --target nodejs --out-dir pkg-node

# Re-link into the demo.
cd ../../examples/web-demo
rm -rf node_modules package-lock.json
npm install
npm run dev
```

If the page still shows stale code after this, the browser may have cached the old wasm — hard-refresh with `Cmd+Shift+R` (macOS) / `Ctrl+Shift+R` (Linux/Windows), or open the page in a private window.

## Troubleshooting

**`Failed to execute 'compile' on 'WebAssembly': HTTP status code is not ok`** — the bundler couldn't resolve the wasm asset URL. This usually means `bindings/wasm/` was edited but not rebuilt + reinstalled. Run the iterate steps above.

## Building for static hosting

```bash
npm run build
```

Produces `dist/` (≈22 KB wasm + 2.4 KB gzipped JS + `index.html`) — deployable to any static host (GitHub Pages, Vercel, Netlify, Cloudflare Pages).

## What it shows

- **Live IV solver** — single-contract `impliedVol(F, K, r, T, price, isCall)` recomputed on every input change. Renders the `-1.0` sentinel as a human-readable "IV undefined" message.
- **Batch benchmark** — 10 000 random contracts through `impliedVolBatch`, timed with `performance.now()`. Shows wall-clock and solve rate.

All numerics are f64 and identical to the Python production binding.
