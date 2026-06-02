// Node-side Black-76 benchmark for the @westonplatter/ngv-opx wasm binding.
//
// Mirrors the Python rs-single / rs-vec rows but with the JS/wasm caller.
// Generates a synthetic WTI book at every N in SIZES, times two paths
// (single-call loop and Float64Array batch), and prints a single JSON
// object on the last stdout line. bench_chart.py / bench_cpu.py subprocess
// this script and parse that JSON, the same way the native Rust example is
// invoked via `cargo run --example bench_native`.

import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const opx = require(resolve(__dirname, "../bindings/wasm/pkg-node"));

const RATE = 0.045;
const SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000];

function makeBook(n) {
  // Same shape as the Python make_book: F=75, K~U(50,100), vol~U(0.25,0.65),
  // T~U(1,90)/365, isCall~Bernoulli(0.5). Pure JS PRNG — timing-only test,
  // numerical parity is covered by u4/u5.
  const f = new Float64Array(n);
  const k = new Float64Array(n);
  const r = new Float64Array(n);
  const v = new Float64Array(n);
  const t = new Float64Array(n);
  const cp = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    f[i] = 75.0;
    k[i] = 50 + Math.random() * 50;
    r[i] = RATE;
    v[i] = 0.25 + Math.random() * 0.4;
    t[i] = (1 + Math.random() * 89) / 365.0;
    cp[i] = Math.random() < 0.5 ? 0 : 1;
  }
  return { f, k, r, v, t, cp };
}

function bestOf(fn, repeats) {
  let best = Infinity;
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    fn();
    const dt = (performance.now() - t0) / 1000.0; // seconds
    if (dt < best) best = dt;
  }
  return best;
}

const results = { "js-single": {}, "js-vec": {} };

for (const n of SIZES) {
  process.stderr.write(`  N = ${n.toLocaleString()} ...\n`);
  const { f, k, r, v, t, cp } = makeBook(n);

  const repFast = n <= 1_000 ? 5 : 3;
  const repSingle = n <= 10_000 ? 3 : 1;

  // Single-call loop: per-option FFI cost.
  const sSeconds = bestOf(() => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = opx.black76(f[i], k[i], r[i], v[i], t[i], cp[i] !== 0);
    }
    return out;
  }, repSingle);

  // Batch: one wasm call, all the work inside Rust.
  const vSeconds = bestOf(() => opx.black76Batch(f, k, r, v, t, cp), repFast);

  results["js-single"][n] = (sSeconds / n) * 1e9; // ns / option
  results["js-vec"][n] = (vSeconds / n) * 1e9;
}

// LAST stdout line must be valid JSON (matches the bench_native.rs contract).
process.stdout.write(JSON.stringify(results) + "\n");
