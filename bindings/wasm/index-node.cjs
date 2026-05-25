// CommonJS entry for Node consumers. wasm-pack's --target nodejs output is
// CJS with synchronous wasm load — no async init needed. We re-export it
// directly so `const { black76 } = require('@ngv/opx')` works.

module.exports = require("./pkg-node/ngv_opx_wasm.js");
