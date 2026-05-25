// U5 parity test: ngv-opx-wasm impliedVol vs. captured Python baseline.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const opx = require(resolve(__dirname, "../pkg-node"));

const baseline = JSON.parse(
  readFileSync(
    resolve(__dirname, "../../python/tests/baselines/black76_baseline.json"),
    "utf8",
  ),
);

let failures = 0;
const fail = (msg) => {
  console.error("FAIL:", msg);
  failures += 1;
};
const pass = (msg) => console.log("PASS:", msg);

// --- Singleton: ATM call IV from baseline
const s = baseline.singletons;
const iv1 = opx.impliedVol(75.0, 75.0, 0.045, 30 / 365, 3.0, true);
if (Math.abs(iv1 - s.iv_call_atm) > 1e-12)
  fail(`ATM IV mismatch: wasm=${iv1} py=${s.iv_call_atm}`);
else pass(`ATM IV byte-near-exact (${iv1}, diff ${Math.abs(iv1 - s.iv_call_atm)})`);

// --- Round-trip: price -> IV -> re-price within tolerance
{
  const truVol = 0.32;
  const price = opx.black76(75, 75, 0.045, truVol, 30 / 365, true);
  const ivBack = opx.impliedVol(75, 75, 0.045, 30 / 365, price, true);
  if (Math.abs(ivBack - truVol) > 1e-8)
    fail(`round-trip vol drift: in=${truVol} out=${ivBack}`);
  else pass(`round-trip ok (in=${truVol} out=${ivBack})`);
}

// --- Batch parity vs Python baseline (200 contracts)
const inp = baseline.inputs;
const f = Float64Array.from(inp.forwards);
const k = Float64Array.from(inp.strikes);
const r = Float64Array.from(inp.rates);
const t = Float64Array.from(inp.times);
const mp = Float64Array.from(baseline.outputs.prices);
const c = Uint8Array.from(inp.is_calls.map((b) => (b ? 1 : 0)));

const ivs = opx.impliedVolBatch(f, k, r, t, mp, c);
const baseIvs = baseline.outputs.ivs;
let maxDiff = 0;
let validCount = 0;
for (let i = 0; i < ivs.length; i++) {
  if (ivs[i] === -1.0 || baseIvs[i] === -1.0) continue;
  validCount++;
  const d = Math.abs(ivs[i] - baseIvs[i]);
  if (d > maxDiff) maxDiff = d;
}
if (maxDiff > 1e-10) fail(`IV batch max-diff ${maxDiff} > 1e-10 (n=${validCount})`);
else pass(`IV batch parity max-diff ${maxDiff} (valid=${validCount}/${ivs.length})`);

// --- Sentinel: price below intrinsic -> -1.0
{
  const iv = opx.impliedVol(100, 50, 0.05, 0.25, 1.0, true); // call worth ~50, price=1 < intrinsic
  if (iv !== -1.0) fail(`expected -1.0 sentinel for sub-intrinsic price, got ${iv}`);
  else pass(`sub-intrinsic returns -1.0`);
}

// --- Sentinel: time <= 0 -> -1.0
{
  const iv = opx.impliedVol(100, 100, 0.05, 0.0, 5.0, true);
  if (iv !== -1.0) fail(`expected -1.0 for t=0, got ${iv}`);
  else pass(`t=0 returns -1.0`);
}

// --- Error path: mismatched batch lengths
try {
  opx.impliedVolBatch(
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

console.log(failures ? `\n${failures} FAILED` : `\nALL PASSED`);
process.exit(failures ? 1 : 0);
