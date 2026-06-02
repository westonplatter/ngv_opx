# @westonplatter/ngv-opx

Black-76 option pricing and implied-volatility solver for **browsers and Node**, compiled from Rust to WebAssembly. f64 throughout — same numeric contract as the Python binding.

## Install

```bash
npm install @westonplatter/ngv-opx
```

## Quick start

### Node (ESM)

```ts
import { black76, impliedVolBatch } from "@westonplatter/ngv-opx";

// Single option
const price = black76(75, 75, 0.045, 0.35, 30 / 365, /* isCall */ true);

// Batch — all inputs Float64Array, isCalls Uint8Array (0=put, 1=call)
const ivs = impliedVolBatch(
  Float64Array.of(75, 80, 70),       // forwards
  Float64Array.of(75, 75, 75),       // strikes
  Float64Array.of(0.045, 0.045, 0.045), // rates
  Float64Array.of(30 / 365, 30 / 365, 30 / 365), // times
  Float64Array.of(2.99, 5.50, 0.80), // market prices
  Uint8Array.of(1, 1, 1),            // is_calls
);
```

### Node (CommonJS)

```js
const opx = require("@westonplatter/ngv-opx");
const price = opx.black76(75, 75, 0.045, 0.35, 30 / 365, true);
```

### Browser

The browser entry requires `await init()` before first use.

```ts
import { init, black76, impliedVolBatch } from "@westonplatter/ngv-opx";

await init();
const price = black76(75, 75, 0.045, 0.35, 30 / 365, true);
```

## API

| Function | Signature | Notes |
|---|---|---|
| `black76` | `(F, K, r, σ, T, isCall) → number` | Single option price |
| `black76Batch` | `(Float64Array×5, Uint8Array) → Float64Array` | Vectorized price |
| `impliedVol` | `(F, K, r, T, price, isCall) → number` | Returns `-1.0` sentinel when undefined |
| `impliedVolBatch` | `(Float64Array×5, Uint8Array) → Float64Array` | Per-row independent |
| `init` | `(opts?) → Promise<InitResult>` | Required in browser, no-op in Node |
| `version` | `() → string` | Package version string |

### Sentinel contract

`impliedVol*` returns `-1.0` when IV is mathematically undefined:
- `timeYears <= 0` (vega is zero)
- `marketPrice` below intrinsic
- `marketPrice` above the discounted upper bound

**Callers must check for `-1.0` before using the result.**

### Boolean arrays

JavaScript has no boolean typed array, so batch APIs accept `Uint8Array` for `isCalls`: `0` = put, non-zero = call.

## License

BSD-3-Clause. See `LICENSE`.
