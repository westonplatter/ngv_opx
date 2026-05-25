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
let _readyUrl = undefined;

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
  // We cache the init promise (including rejections) on purpose: a failed
  // wasm load is almost always a deployment/bundler bug, not a transient
  // network blip, so silently retrying on the next call would hide the
  // root cause. Callers that want to retry should reload the page.
  if (!_ready) {
    _ready = wasmInit(opts.wasmUrl);
    _readyUrl = opts.wasmUrl;
  } else if (
    opts.wasmUrl !== undefined &&
    String(opts.wasmUrl) !== String(_readyUrl)
  ) {
    // Second call asked for a different binary than the one already loading
    // — wasm-pack only instantiates once per module, so the new URL would
    // be silently ignored. Warn so the mismatch surfaces during development.
    console.warn(
      "[@ngv/opx] init() called again with a different wasmUrl; " +
        "the originally-loaded module is reused. " +
        `first=${String(_readyUrl)} second=${String(opts.wasmUrl)}`,
    );
  }
  await _ready;
  return {
    gpu: false, // CPU-only in v1; WebGPU path lands in a follow-up unit
    environment: "browser",
  };
}

export { version, black76, black76Batch, impliedVol, impliedVolBatch };
