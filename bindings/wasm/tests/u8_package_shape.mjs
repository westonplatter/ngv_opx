// U8 test: consume the @westonplatter/ngv-opx package via its public entry points
// (index-node.mjs and index-node.cjs). Validates that the Node-facing
// conditional exports resolve and re-export the expected wasm functions.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG = resolve(__dirname, "..");

let failures = 0;
const fail = (msg) => { console.error("FAIL:", msg); failures += 1; };
const pass = (msg) => console.log("PASS:", msg);

// --- ESM entry
const esm = await import(resolve(PKG, "index-node.mjs"));
for (const name of ["version", "black76", "black76Batch", "impliedVol", "impliedVolBatch", "init"]) {
  if (typeof esm[name] !== "function") fail(`ESM missing export: ${name}`);
}
if (!failures) pass("ESM entry exposes the full surface");

const initResult = await esm.init();
if (initResult.environment !== "node" || initResult.gpu !== false)
  fail(`init() in Node returned ${JSON.stringify(initResult)}, expected {gpu:false, environment:'node'}`);
else pass(`init() Node result: ${JSON.stringify(initResult)}`);

// Smoke-call through ESM entry
const p = esm.black76(75, 75, 0.045, 0.35, 30 / 365, true);
if (Math.abs(p - 2.989958146807046) > 1e-12) fail(`ESM black76 wrong: ${p}`);
else pass(`ESM black76 ATM call = ${p}`);

// --- CJS entry
const require = createRequire(import.meta.url);
const cjs = require(resolve(PKG, "index-node.cjs"));
for (const name of ["version", "black76", "black76Batch", "impliedVol", "impliedVolBatch"]) {
  if (typeof cjs[name] !== "function") fail(`CJS missing export: ${name}`);
}
if (!failures) pass("CJS entry exposes the wasm surface");

const p2 = cjs.black76(75, 75, 0.045, 0.35, 30 / 365, true);
if (p2 !== p) fail(`CJS vs ESM disagree: ${p2} vs ${p}`);
else pass(`CJS black76 matches ESM (${p2})`);

console.log(failures ? `\n${failures} FAILED` : `\nALL PASSED`);
process.exit(failures ? 1 : 0);
