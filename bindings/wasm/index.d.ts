// Public TypeScript surface for @westonplatter/ngv-opx.
//
// All numerics are f64. Browser and Node entries expose the same functions;
// the browser entry also requires `await init()` before first use.

export function version(): string;

/**
 * Black-76 price for a single option on a forward (f64).
 *
 * @param forward      Forward price F of the underlying.
 * @param strike       Strike price K.
 * @param rate         Continuously compounded risk-free rate (annualized).
 * @param volatility   Annualized volatility.
 * @param timeYears    Time to expiry in years.
 * @param isCall       true for call, false for put.
 * @returns            Option price.
 */
export function black76(
  forward: number,
  strike: number,
  rate: number,
  volatility: number,
  timeYears: number,
  isCall: boolean,
): number;

/**
 * Vectorized Black-76 price (f64). All input arrays must be equal length.
 *
 * @param isCalls Uint8Array (0 = put, non-zero = call) — JS lacks a boolean
 *                typed array.
 * @throws If input lengths disagree.
 */
export function black76Batch(
  forwards: Float64Array,
  strikes: Float64Array,
  rates: Float64Array,
  volatilities: Float64Array,
  times: Float64Array,
  isCalls: Uint8Array,
): Float64Array;

/**
 * Solve Black-76 implied volatility for a single observed price (f64).
 *
 * @returns The implied volatility, or the sentinel `-1.0` when IV is
 *          mathematically undefined: timeYears <= 0, marketPrice below
 *          intrinsic, or marketPrice above the discounted upper bound.
 *          Callers MUST check for `-1.0` before using the result.
 */
export function impliedVol(
  forward: number,
  strike: number,
  rate: number,
  timeYears: number,
  marketPrice: number,
  isCall: boolean,
): number;

/**
 * Vectorized Black-76 IV solver (f64). Independent per-row; entries are
 * `-1.0` for rows where IV is undefined.
 *
 * @throws If input lengths disagree.
 */
export function impliedVolBatch(
  forwards: Float64Array,
  strikes: Float64Array,
  rates: Float64Array,
  times: Float64Array,
  marketPrices: Float64Array,
  isCalls: Uint8Array,
): Float64Array;

export interface InitResult {
  /** True if a GPU path is active in this session (browser-only, opt-in). */
  gpu: boolean;
  /** "browser" | "node". */
  environment: "browser" | "node";
}

/**
 * Initialize the wasm module.
 *
 * Required in the browser before calling any other function; a no-op in
 * Node (kept for API symmetry).
 *
 * When consumed through a bundler (Vite, webpack, esbuild, etc.), pass
 * `wasmUrl` explicitly using the bundler's URL-import syntax — bundlers
 * rewrite asset paths and the default `import.meta.url` resolution will
 * 404. Example (Vite):
 *
 * ```ts
 * import wasmUrl from "@westonplatter/ngv-opx/pkg-web/ngv_opx_wasm_bg.wasm?url";
 * await init({ wasmUrl });
 * ```
 */
export function init(opts?: {
  gpu?: boolean;
  wasmUrl?: string | URL | Request;
}): Promise<InitResult>;
