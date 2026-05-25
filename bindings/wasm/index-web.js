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

/**
 * Initialize the wasm module.
 *
 * @param {object} [opts]
 * @param {string | URL | Request} [opts.wasmUrl] Explicit URL to the
 *   wasm binary. **Required when consuming this package through a bundler
 *   like Vite or webpack** — pass `new URL("@ngv/opx/pkg-web/ngv_opx_wasm_bg.wasm?url", import.meta.url)`
 *   or use the bundler's `?url` import syntax. If omitted, init falls back
 *   to wasm-pack's default URL resolution, which only works when the
 *   package files are served at their published relative paths (no
 *   bundler / no asset hashing).
 */
export async function init(opts = {}) {
  if (!_ready) _ready = wasmInit(opts.wasmUrl);
  await _ready;
  return {
    gpu: false, // CPU-only in v1; WebGPU path lands in a follow-up unit
    environment: "browser",
  };
}

export { version, black76, black76Batch, impliedVol, impliedVolBatch };
