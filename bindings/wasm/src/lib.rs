//! WebAssembly bindings for ngv-opx.
//!
//! Exposes Black-76 pricing and implied-vol solving from the pure-Rust core
//! crate to JavaScript/TypeScript callers in browsers and Node.

use wasm_bindgen::prelude::*;

/// Returns the ngv-opx-wasm package version (smoke test for U3).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
