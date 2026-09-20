---
title: "feat: Black-76 first-order greeks (delta, vega, theta, rho)"
type: feat
status: active
created: 2026-09-12
origin:
  - README.md coverage matrix — "1st Order Greeks: ⛔ not started"
---

# feat: Black-76 first-order greeks (delta, vega, theta, rho)

## Summary

Add the four **first-order** greeks for Black-76 — **delta**, **vega**,
**theta**, **rho** — as a new capability alongside pricing and implied vol.
Ship Rust-first in `crates/core`, then plumb the same contract downstream
through the Python (PyO3) and JS/WASM (wasm-bindgen) bindings, matching the
f64 production path already established for `black76` and
`black76_implied_volatility`.

This flips the README coverage matrix cell **Black-76 → 1st Order Greeks**
from ⛔ to ✅.

Gamma, vanna, charm, volga/vomma and the rest of the second-order greeks are
**out of scope** here — they are the separate "2nd Order Greeks" column and a
follow-up plan.

## Decisions (locked)

These were settled before writing and drive every signature below:

1. **API shape: individual functions.** One function per greek
   (`black76_delta`, `black76_vega`, `black76_theta`, `black76_rho`), mirroring
   py_vollib's per-greek layout and the existing `black76` / `black76_vectorized`
   surface. Each is standalone; a private helper computes the shared
   `d1 / d2 / disc / φ(d1)` terms so the formula isn't duplicated four times.
2. **Units: market conventions.**
   - **delta** — raw, per **$1** change in forward `F` (dimensionless-ish; no scaling).
   - **vega** — per **1 vol point** (÷100).
   - **theta** — per **calendar day** (÷365).
   - **rho** — per **1% rate** (÷100).

   These conventions are documented at every entry point. The IV solver's
   internal **raw** vega (`b76_vega_f64`, per 1.00 of σ) is **not** touched —
   the public `black76_vega` is a thin `raw / 100.0` wrapper over it.
3. **Sequencing: Rust → Python → JS.** Land and validate the core math first,
   then each binding re-exports it. No greek math lives in a binding layer.

## Open decisions (resolve during U1)

- **Degenerate-input behavior** (`T <= 0`, `σ <= 0`, non-finite inputs). The IV
  path uses a `-1.0` sentinel, but greeks are **signed** (put delta, theta, and
  rho are routinely negative), so `-1.0` collides with legitimate values.
  **Recommendation: return `NaN`** for degenerate inputs, documented per
  function, and assert the contract in tests. Confirm before coding U1.
- **Theta rate-term parity with py_vollib.** The analytic decay term is
  unambiguous; the `+ r·V` rate term's exact form/sign is a convention that
  differs across references. U1 pins ours by symbolic derivation, then U3
  asserts bit-for-convention parity against `py_vollib` (÷365) so we adopt the
  same sign it does.

## Greek definitions (Black-76 reference)

For an option on forward `F`, strike `K`, rate `r`, vol `σ`, expiry `T`:

```
d1 = (ln(F/K) + ½σ²T) / (σ√T)      d2 = d1 - σ√T      disc = e^(-rT)
C  = disc·[F·N(d1) - K·N(d2)]      P  = disc·[K·N(-d2) - F·N(-d1)]
```

with `N` the normal CDF and `φ` the normal PDF. All four greeks reduce to the
same building blocks the pricer already computes.

| Greek | Definition | Call (raw) | Put (raw) | Market scaling |
|-------|------------|------------|-----------|----------------|
| **Delta** | ∂V/∂F | `disc·N(d1)` | `disc·(N(d1) − 1)` = `−disc·N(−d1)` | none (per $1 of F) |
| **Vega**  | ∂V/∂σ | `disc·F·√T·φ(d1)` | same as call | ÷100 (per 1 vol pt) |
| **Theta** | −∂V/∂T | `−disc·F·φ(d1)·σ/(2√T) + r·C` | `−disc·F·φ(d1)·σ/(2√T) + r·P` | ÷365 (per day) |
| **Rho**   | ∂V/∂r | `−T·C` | `−T·P` | ÷100 (per 1% rate) |

**Design note — rho is `−T·V` in Black-76.** Because the underlying is a
*forward* (which does not depend on `r` in this model), `r` enters only through
the discount factor `disc = e^(-rT)`. So `∂V/∂r = −T·V` exactly, for both calls
and puts. This is deliberately different from Black-Scholes rho
(`±K·T·e^(-rT)·N(±d2)`), and it gives us a cheap exact test (see U1).

**Delta is a forward delta** (`∂/∂F`), not a spot delta — the natural
sensitivity for a futures/forward model. Documented as such at every surface.

## Reuse

- **Vega already exists.** `b76_vega_f64(f, k, r, sigma, t)` in
  `crates/core/src/black76.rs` computes raw vega and is used by the IV solver.
  The public `black76_vega` wraps it (`/100`); we do **not** write a second
  vega implementation, and the solver's call site is unchanged.
- **Robust price for theta/rho.** Theta's `r·V` term and rho's `−T·V` reuse the
  existing cancellation-safe `b76_price_f64` (OTM-direct + parity) rather than
  re-expanding `F·N(d1) − K·N(d2)`, keeping deep ITM/OTM accuracy.
- **Batch shape.** Vectorized greeks copy the SoA loop and length-assert already
  in `black76_price_batch_f64` verbatim.

---

## Implementation Units

Grouped into three PRs matching the Rust → Python → JS sequence.

### PR 1 — Core Rust math (`crates/core`)

#### U1. Scalar greeks (f64) + private shared helper

**Goal:** Land the four scalar greeks as pure f64 functions with market-unit
scaling, plus a private `struct`/helper that computes `d1, d2, disc, φ(d1)`
once so no formula is duplicated.

**Files:**
- `crates/core/src/black76.rs` — add public `black76_delta_f64`,
  `black76_vega_f64` (market-scaled; wraps existing raw `b76_vega_f64`),
  `black76_theta_f64`, `black76_rho_f64`, and a private `b76_greek_ctx(...)`
  helper. (Alternatively a new `crates/core/src/greeks.rs` submodule re-exported
  from `lib.rs`; decide in U1 based on file size — `black76.rs` is already
  ~550 lines, so a submodule is the likely call.)

**Approach:**
- Private helper returns `(d1, d2, disc, pdf_d1, sqrt_t)` from `(F,K,r,σ,T)`.
- Each greek transcribes its table row, applies the market scaling factor, and
  branches on `is_call` only where the table differs (delta, theta, rho).
- Degenerate inputs (`T<=0`, `σ<=0`, non-finite) → `NaN` (per open decision),
  guarded up front.
- Cite the greek definition and any reference equation in doc comments, matching
  the citation style already used in `black76.rs` / the SR solver.

**Test scenarios (`crates/core/tests/` + `#[cfg(test)]`):** independent of any
third-party lib —
- **Finite-difference cross-check.** Central difference of `b76_price_f64` w.r.t.
  each input (bump `F`, `σ`, `T`, `r`) matches the analytic greek (converted back
  to raw units) to ~1e-6 across a moneyness/T/vol grid.
- **Rho exact identity:** `rho_raw == −T · price` to ~1e-12 (calls and puts).
- **Delta bounds & parity:** call delta ∈ `(0, disc)`, put delta ∈ `(−disc, 0)`,
  and `delta_call − delta_put == disc`.
- **Vega:** ≥ 0, and call vega == put vega exactly (same function).
- **Theta put-call:** `theta_call_raw − theta_put_raw == r·(C − P) == r·disc·(F−K)`.
- **Degenerate inputs** return `NaN`, never panic (T=0, σ=0, negative T).

#### U2. Vectorized greeks (f64)

**Goal:** SoA batch entry point per greek, mirroring `black76_price_batch_f64`.

**Files:** same module as U1 — add `black76_delta_batch_f64`,
`black76_vega_batch_f64`, `black76_theta_batch_f64`, `black76_rho_batch_f64`.

**Approach:** copy the equal-length assertion and `Vec::with_capacity` loop from
`black76_price_batch_f64`; call the scalar greek per row.

**Test scenarios:** batch-vs-scalar agreement (batch output equals looping the
scalar fn) on the U1 grid; length-mismatch panics like the pricer.

---

### PR 2 — Python binding (`bindings/python`)

#### U3. PyO3 scalar + vectorized greeks

**Goal:** Expose all four greeks (scalar + vectorized) as f64 pyfunctions,
matching the `black76` / `black76_vectorized` signature style.

**Files:**
- `bindings/python/src/lib.rs` — add `#[pyfunction]` wrappers:
  `black76_delta`, `black76_vega`, `black76_theta`, `black76_rho` (scalar) and
  `black76_delta_vectorized`, `black76_vega_vectorized`,
  `black76_theta_vectorized`, `black76_rho_vectorized`; register all in the
  `#[pymodule]`.
- `bindings/python/python/ngv_opx/__init__.py` — add the eight names to the
  import block and `__all__`.

**Signatures** (f64, `numpy` arrays for batch), e.g.:
```
black76_delta(forward, strike, rate, volatility, time_years, is_call) -> float
black76_vega (forward, strike, rate, volatility, time_years)          -> float   # no is_call
black76_delta_vectorized(forwards, strikes, rates, vols, times, is_calls) -> np.ndarray[f64]
black76_vega_vectorized (forwards, strikes, rates, vols, times)          -> np.ndarray[f64]
```
(Vega omits `is_call`; the other three take it, matching the math.)

**Test scenarios (`bindings/python/tests/`):** a new
`test_black76_greeks_vs_vollib.py` on the existing crude-oil surface —
- Parity vs `py_vollib.black.greeks.analytical` (`delta`/`vega`/`theta`/`rho`),
  applying the **same market scaling** py_vollib uses (vega,rho ÷100; theta
  ÷365). This is where the theta rate-term sign is pinned to py_vollib.
- Vectorized == scalar loop.
- Sentinel/degenerate rows (`T=0`, sub-tick) return `NaN` per the U1 contract,
  consistent with the penny-tick philosophy already in
  `test_black76_vs_vollib.py`.

---

### PR 3 — JS/WASM binding + docs (`bindings/wasm`, README)

#### U4. wasm-bindgen scalar + batch greeks

**Goal:** Same contract in JS/TS, f64, browser + Node.

**Files:**
- `bindings/wasm/src/lib.rs` — `#[wasm_bindgen(js_name = black76Delta)]` etc. for
  scalar `black76Delta/Vega/Theta/Rho` and batch `black76DeltaBatch/...`
  (batch takes `&[f64]` slices + `is_calls: &[u8]`, length-checked with a
  `JsError` like `black76Batch`).
- `bindings/wasm/index.d.ts` — typed declarations + JSDoc (state units and the
  forward-delta convention).
- `bindings/wasm/index-web.js` — import from `pkg-web` and add to the
  `export { ... }` list.
- `bindings/wasm/index-node.mjs` — add `export const black76Delta = mod....`
  lines. (`index-node.cjs` re-exports the whole module, no change needed.)

**Test scenarios:** new `bindings/wasm/tests/*_greeks.mjs` — parity vs the Python
baseline (the repo's existing JS-vs-Python parity pattern), scalar and batch,
including a `NaN`/degenerate row.

#### U5. Docs, README matrix, plan close-out

**Files:**
- `README.md` — flip Black-76 **1st Order Greeks** cell ⛔ → ✅; add a short
  greeks usage snippet under both **Python** and **JavaScript**; note the unit
  conventions and forward-delta.
- `bindings/wasm/README.md` — greeks API section.
- This plan → `status: shipped` once PR 3 merges.

---

## Scope boundaries

**In scope:** delta, vega, theta, rho for Black-76; f64 production path; scalar +
vectorized; Rust core + Python + JS/WASM; unit/parity tests; README/docs.

**Deferred / out of scope:**
- **Second-order greeks** (gamma, vanna, charm, volga/vomma) — separate column,
  separate plan. Gamma is the natural next step (`∂²V/∂F² = disc·φ(d1)/(F·σ√T)`).
- **GPU greeks path** — pricing GPU path is still 🚧; greeks stay CPU/f64 here.
- **f32 / legacy BSM greeks** — the `black_scholes*` path stays as-is.
- **Bundled all-greeks call** — explicitly rejected in favor of individual
  functions (locked decision 1); a bundle could be added later without breaking
  these signatures.
- **Greeks benchmarks** — optional follow-up; the existing bench harness
  (`benchmarks/`) can add a greeks row later, not required to ship.

## Risks & mitigations

- **Theta convention drift** — mitigated by symbolic derivation in U1 + explicit
  py_vollib parity in U3 (the ÷365 and rate-term sign are pinned there).
- **Deep ITM/OTM cancellation in theta/rho** — mitigated by reusing
  `b76_price_f64` (parity-based, cancellation-safe) rather than re-expanding.
- **Sentinel collision** — signed greeks make `-1.0` unusable; `NaN` chosen and
  tested (open decision, confirm before U1).
- **Surface sprawl** (4 greeks × scalar+batch × 3 langs) — contained by the
  strict per-greek naming convention and by reusing existing batch/loop scaffolding.

## Definition of done

- All four greeks available, f64, scalar + vectorized, in Rust, Python, and JS.
- `task test` green (Rust unit + FD/identity tests, Python py_vollib parity,
  JS-vs-Python parity).
- README matrix cell flipped to ✅ with usage snippets and documented conventions.
