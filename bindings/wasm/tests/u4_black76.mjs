// U4 parity test: ngv-opx-wasm Black-76 vs. captured Python baseline.
//
// Run after `wasm-pack build --target nodejs --out-dir pkg-node`:
//   node bindings/wasm/tests/u4_black76.mjs
//
// Exit code: 0 on success, 1 on any failure.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const opx = require(resolve(__dirname, "../pkg-node"));

const baseline = JSON.parse(
  readFileSync(
    resolve(__dirname, "../../../bindings/python/tests/baselines/black76_baseline.json"),
    "utf8",
  ),
);

let failures = 0;
const fail = (msg) => {
  console.error("FAIL:", msg);
  failures += 1;
};
const pass = (msg) => console.log("PASS:", msg);

// --- Smoke
if (typeof opx.version() !== "string") fail("version() not a string");
else pass(`version = ${opx.version()}`);

// --- Singletons (byte-exact vs Python f64)
const s = baseline.singletons;
const c1 = opx.black76(75.0, 75.0, 0.045, 0.35, 30 / 365, true);
const p1 = opx.black76(75.0, 65.0, 0.045, 0.45, 30 / 365, false);
if (c1 !== s.b76_call_atm)
  fail(`call ATM mismatch: wasm=${c1} py=${s.b76_call_atm} diff=${Math.abs(c1 - s.b76_call_atm)}`);
else pass(`call ATM byte-exact (${c1})`);
if (p1 !== s.b76_put_otm)
  fail(`put OTM mismatch: wasm=${p1} py=${s.b76_put_otm} diff=${Math.abs(p1 - s.b76_put_otm)}`);
else pass(`put OTM byte-exact (${p1})`);

// --- Batch parity (200 contracts vs Python f64)
const inp = baseline.inputs;
const f = Float64Array.from(inp.forwards);
const k = Float64Array.from(inp.strikes);
const r = Float64Array.from(inp.rates);
const v = Float64Array.from(inp.vols);
const t = Float64Array.from(inp.times);
const c = Uint8Array.from(inp.is_calls.map((b) => (b ? 1 : 0)));

const prices = opx.black76Batch(f, k, r, v, t, c);
const basePrices = baseline.outputs.prices;
if (prices.length !== basePrices.length) fail(`batch length: ${prices.length} vs ${basePrices.length}`);
let maxDiff = 0;
for (let i = 0; i < prices.length; i++) {
  const d = Math.abs(prices[i] - basePrices[i]);
  if (d > maxDiff) maxDiff = d;
}
if (maxDiff > 1e-10) fail(`batch max-diff ${maxDiff} > 1e-10`);
else pass(`batch parity max-diff ${maxDiff} (n=${prices.length})`);

// --- Edge cases
// t=0 returns intrinsic, discounted by zero rate*time = 1.
const intrinsicCall = opx.black76(100, 90, 0.0, 0.2, 0.0, true);
if (Math.abs(intrinsicCall - 10.0) > 1e-12) fail(`t=0 call intrinsic: ${intrinsicCall} expected 10`);
else pass(`t=0 call intrinsic = ${intrinsicCall}`);

// vol=0 returns intrinsic discounted
const v0Call = opx.black76(100, 95, 0.05, 0.0, 0.25, true);
const discIntrinsic = Math.max(100 - 95, 0) * Math.exp(-0.05 * 0.25);
if (Math.abs(v0Call - discIntrinsic) > 1e-10)
  fail(`vol=0 call: ${v0Call} expected ${discIntrinsic}`);
else pass(`vol=0 call intrinsic = ${v0Call}`);

// --- Error path: mismatched lengths
try {
  opx.black76Batch(
    Float64Array.of(1, 2),
    Float64Array.of(1),
    Float64Array.of(1, 2),
    Float64Array.of(1, 2),
    Float64Array.of(1, 2),
    Uint8Array.of(1, 1),
  );
  fail("mismatched lengths should throw");
} catch (e) {
  pass(`mismatched lengths threw: ${e.message.slice(0, 60)}...`);
}

// --- Empty batch
const empty = opx.black76Batch(
  new Float64Array(0),
  new Float64Array(0),
  new Float64Array(0),
  new Float64Array(0),
  new Float64Array(0),
  new Uint8Array(0),
);
if (empty.length !== 0) fail(`empty batch length: ${empty.length}`);
else pass("empty batch returns empty Float64Array");

console.log(failures ? `\n${failures} FAILED` : `\nALL PASSED`);
process.exit(failures ? 1 : 0);
