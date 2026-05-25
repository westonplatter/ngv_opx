// Browser entry. wasm-pack's --target web output requires an explicit
// `await init()` to instantiate the wasm module before any exported function
// can be called.
//
// Usage:
//   import { init, black76, impliedVolBatch } from "@ngv/opx";
//   await init();
//   const price = black76(75, 75, 0.045, 0.35, 30/365, true);
//
// init() returns a status object describing the environment. A future U6
// will add `init({ gpu: true })` to enable an experimental WebGPU IV path.

import wasmInit, {
  version,
  black76,
  black76Batch,
  impliedVol,
  impliedVolBatch,
} from "./pkg-web/ngv_opx_wasm.js";

let _ready = null;

export async function init(opts = {}) {
  if (!_ready) _ready = wasmInit();
  await _ready;
  return {
    gpu: false, // CPU-only in v1; WebGPU path lands in a follow-up unit
    environment: "browser",
  };
}

export { version, black76, black76Batch, impliedVol, impliedVolBatch };
