// ESM entry for Node consumers using `import` syntax. wasm-pack's nodejs
// target is CJS internally; we wrap it via createRequire so the package
// works regardless of whether the consumer chose ESM or CJS.

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const mod = require("./pkg-node/ngv_opx_wasm.js");

export const version = mod.version;
export const black76 = mod.black76;
export const black76Batch = mod.black76Batch;
export const impliedVol = mod.impliedVol;
export const impliedVolBatch = mod.impliedVolBatch;

// Node has no GPU path in this binding; the browser entry adds one in U6.
// Node consumers can always call the CPU functions above directly without
// any init step.
export async function init() {
  return { gpu: false, environment: "node" };
}

export default mod;
