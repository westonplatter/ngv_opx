---
title: "feat: SR+Householder IV solver and GPU Black-76 path"
type: feat
status: active
created: 2026-05-25
origin:
  - docs/plans/lets-be-rational
  - docs/plans/gpu-black76
---

# feat: SR+Householder IV solver and GPU Black-76 path

## Summary

Two coordinated workstreams that together replace the current Black-76 IV
hot path with something fast enough for tick-by-tick pricing on the full
universe:

1. **CPU `implied_vol` crate** — Stefanica-Radoičić (2017) closed-form seed
   + fixed Householder-3 refinement, f64 throughout, branch-free per-row
   so it auto-vectorizes and parallelizes cleanly under rayon.
2. **GPU Black-76 path** — wgpu/Metal compute pipeline that prices and
   solves IV in one dispatch, f32 on-device with f64 CPU fix-up, reusing
   the SR seed and the Householder iteration logic from workstream 1.

Workstream 2 strictly depends on 1 (it imports the SR formula and the
Householder step), so the plan ships as two PRs.

## Delivery: Two PRs

- **PR 1 (CPU algorithm switch)** — U1, U2, U3. Lands SR+Householder as the
  default CPU IV path and replaces the Newton-Raphson-Vega solver.
- **PR 2 (GPU Black-76)** — U4, U5, U6, U7. Adds `_gpu`-postfixed entry points
  alongside the CPU defaults, plus crossover benchmark.

## Routing Convention: `_gpu` Postfix

The public surface stays CPU-default. GPU acceleration is opted into per call
via an explicitly-named `_gpu` variant — no boolean kwarg, no hidden routing.

| CPU (default) | GPU variant |
|---|---|
| `black76_price_batch` | `black76_price_batch_gpu` |
| `black76_implied_vol_batch` | `black76_implied_vol_batch_gpu` |
| Python: `ngv_opx.black76_implied_vol(...)` | Python: `ngv_opx.black76_implied_vol_gpu(...)` |

Rules:
- Both variants take and return the same shapes/dtypes — only the compute
  backend differs.
- `_gpu` variants return the same f64 output as CPU (the f32 GPU pass + f64
  CPU fix-up is an implementation detail).
- If no GPU adapter is available, `_gpu` variants raise / return an error
  rather than silently falling back to CPU. Callers who want a fallback
  build it explicitly: `try gpu, except: cpu`.
- The legacy Newton-Raphson path is removed from the public CPU API in PR 1.
  Its iteration kernel is moved to `crates/core/tests/` as a `#[cfg(test)]`
  oracle for cross-checking the new solver — not reachable from production code.

## Problem Frame

Current `crates/core/src/implied_vol.rs` Newton-Raphson is too slow for
the universe size we want to price on every tick. The existing GPU
prototype (BSM, f32, with per-thread early termination) was retired when
the desk moved to commodity futures and never came back. Both gaps need
to be closed before the production pipeline can scale.

## Scope Boundaries

In scope:
- New `crates/core` modules for SR seed, Black-76 normalized + derivatives,
  Householder-3 refinement, batch API.
- New `crates/gpu` Black-76 pricer + IV solver structs, WGSL shaders,
  f64 CPU fix-up.
- Python `_gpu`-postfixed entry points (e.g. `black76_implied_vol_gpu`)
  plumbed through `bindings/python`. CPU functions remain unchanged.
- Crossover benchmarks on a representative CL day.

### Deferred to Follow-Up Work
- Hand-rolled SIMD (wide/pulp) — auto-vec + rayon first.
- Double-single f64 emulation in WGSL — only if f32+fix-up is insufficient.
- Routing layer that auto-selects CPU vs GPU based on batch size — gated on
  the crossover benchmark. Until then, callers explicitly pick a variant.
- `is_call` bitmap packing — only if profiling shows boolean reads bottleneck.

### Outside this product's identity
- Porting the legacy BSM GPU path to f64. It stays as-is with a deprecation note.
- Non-wgpu GPU backends (CUDA/ROCm).
- GPU path as correctness oracle — CPU f64 remains the reference.

---

## High-Level Technical Design

```
                    OptionQuote / SoA slices
                              |
                  +-----------+-----------+
                  |                       |
       [CPU path: *_batch]      [GPU path: *_batch_gpu]
                  |                       |
                  v                       v
        normalize -> SR seed       upload SoA f32 buffers
                  |                       |
            2x Householder-3       WGSL: normalize -> SR -> 2x Householder
                  |                       |
          undo normalization       readback f32 IV
                  |                       |
                  |               CPU f64 fix-up (re-run Householder
                  |                with GPU answer as seed)
                  +-----------+-----------+
                              v
                         Vec<f64> / &mut [f64]
```

Directional only. Both paths share normalization (forward, parity, log-moneyness)
and share the SR+Householder math — the GPU path is a transcription of the CPU
formulas into WGSL, not a separate algorithm.

---

## Implementation Units

---

## PR 1 — CPU algorithm switch (Newton → SR + Householder)

### U1. SR closed-form seed + normalized Black-76 in `crates/core`

**Goal:** Land the math primitives needed by both paths:
1. SR closed-form seed (Pólya-based, no erfcx needed here).
2. Normalized Black-76 call price + analytic derivatives (vega, volga, d3)
   used by the Householder step in U2.
3. erfcx-based deep-wing evaluation of `Phi(·)` for the **true** Black-76
   (used by the Householder step), so the f(v) residual stays accurate when
   `|d_-|` is large.

**Equation reference:** `docs/papers/sr-2017-equations.md` — full transcription
of SR eqs 5–26 with the Pólya A(·) definition, no-arbitrage bounds, the
y-sign / C0-threshold branches, and the ATM-forward simplification. Cite
paper equation numbers in source comments.

**Files:**
- `crates/core/src/iv/mod.rs` (new submodule)
- `crates/core/src/iv/black.rs` — normalized Black-76, derivatives, erfcx-based Phi
- `crates/core/src/iv/stefanica.rs` — SR seed (Pólya-based, eqs 16–26)
- `crates/core/src/iv/errors.rs` — `IvError` enum
- `crates/core/tests/iv_primitives.rs`

**Approach:**
- **SR seed:** transcribe eqs 16–26 with paper equation-number comments. The
  formula is **not singular** at y=0 (A→0 collapses β to C/B); use the
  collapsed form directly under `|y| < eps` guard. Brenner-Subrahmanyam is
  **not** needed.
- **Normalized Black:** straight transcription of the Black-76 formula in
  total-vol coordinates `v = σ√T`. Use `libm`/`statrs` for `Phi(·)`; switch
  to erfcx-based asymptotic when `|d_-| > 8` (threshold from Jäckel) to avoid
  underflow.
- **Derivatives:** vega, volga, d3 derived symbolically; verify with SymPy
  before coding; cross-check with finite-difference in tests.

**Test scenarios:**
- Normalized Black price round-trip against `crates/core/src/black76.rs` (oracle) to 1e-13.
- SR seed relative error within paper bounds `-0.0418 < (σ_BS - σ_SR)/σ_BS < 0.1138`
  across a moneyness/T grid covering σ ∈ [0.01, 3.0], T ∈ [1d, 5y], K/S ∈ [0.3, 3.0].
- Vega/volga/d3 verified via central finite-difference at sample points to 1e-7.
- erfcx vs scipy reference table (values copied into test) to 1e-14 relative.
- ATM-forward case (|y| < 1e-12): SR returns finite, sane vol matching paper eq (3) bound.
- No-arbitrage guards: prices below intrinsic or above upper bound return the right `IvError` variant, never panic.

### U2. Householder-3 refinement + scalar/batch solver API

**Goal:** Compose SR seed + 2 fixed Householder steps into the public solver.
SoA batch API with rayon. Sentinel handling (BelowIntrinsic / AboveNoArbitrage /
NonFinite) detected pre-loop.

**Dependencies:** U1.

**Files:**
- `crates/core/src/iv/householder.rs`
- `crates/core/src/iv/solver.rs`
- `crates/core/src/iv/batch.rs`
- `crates/core/src/lib.rs` (re-export public API)
- `crates/core/Cargo.toml` (add `[features]` block with `halley_only`)
- `crates/core/tests/iv_roundtrip.rs`
- `crates/core/tests/iv_parity.rs`
- `crates/core/tests/iv_edge_cases.rs`
- `crates/core/tests/iv_third_party_xcheck.rs` — cross-check against `py_vollib` / QuantLib reference points (paper has no numerical tables, so we substitute trusted third-party values for ~20 sample triples)

**Approach:** Fixed 2 Householder steps, no per-row convergence check.
Put-call parity converts puts to calls in undiscounted space before SR.
Batch errors become NaN in the output slice. Feature flag `halley_only` to
swap in Halley for debugging.

**Test scenarios:**
- 100k random (S,K,T,r,q,sigma) round-trip: |sigma_recovered - sigma_true| < 1e-12 bulk, < 1e-9 wings.
- Put-call parity in IV space: call IV vs put-via-parity IV agree to 1e-13.
- Third-party cross-check (~20 reference triples generated offline from
  `py_vollib_vectorized` or QuantLib, committed as test fixtures): agree to 1e-10.
  *(Note: the SR paper itself contains no numerical tables — only figures —
  so we substitute third-party references for paper-table reproduction.)*
- No panics on NaN, inf, negative price, K=0, T=0 — all return IvError.
- Determinism across rayon thread counts (bit-exact).

**Performance targets** (single-threaded, modern x86-64 core):
- Scalar API: < 100 ns/option, target 50 ns.
- Batch + rayon: > 50M options/sec on 32 cores for a 10M-row batch.

**Verification:** `cargo test -p ngv-opx-core` green. `cargo bench` scalar and
batch numbers reported in README.

### U3. Seeded IV entry point for GPU fix-up

**Goal:** Surface `black76_implied_vol_with_seed_f64(...)` — a Householder-based
IV entry point that accepts an externally-supplied initial guess instead of
running SR. The GPU fix-up step in U5 uses this with the GPU's f32 answer as
the seed.

**Dependencies:** U2 (lands the Householder loop the seeded entry shares).

**Files:**
- `crates/core/src/iv/solver.rs` (add seeded variant alongside the SR-seeded entry)
- `crates/core/src/lib.rs` (re-export)
- `crates/core/tests/iv_seeded.rs`

**Approach:** Small wrapper around the same 2-step Householder loop from U2;
skips the SR call and uses the caller's seed instead. ~5-line addition once
U2's solver is in place.

**Test scenarios:**
- Output matches the unseeded SR+Householder entry when seed = SR guess.
- Converges to 1e-12 in 2 steps when seed is within 1e-3 vol-pts of truth.
- Handles a wildly bad seed (50% off) gracefully — 2 steps may not converge,
  but no panic; result is well-defined (caller's responsibility to detect).

---

## PR 2 — GPU Black-76 path

### U4. GPU Black-76 pricer (WGSL + Rust scaffolding)

**Goal:** `GpuBlack76Pricer` owning persistent `wgpu::Device`/`Queue`/pipeline.
SoA f32 upload, one dispatch, f32 readback. Mirrors `crates/gpu/src/implied_vol.rs`
structure but rewrites the shader for the Black-76 drift (both legs discounted
by e^(-rT)).

**Dependencies:** U1 (uses the normalized Black-76 evaluator as the CPU oracle in tests; does not depend on the IV solver).

**Files:**
- `crates/gpu/src/black76_pricer.rs`
- `crates/gpu/src/black76_shader.wgsl`
- `crates/gpu/src/lib.rs` (export, share adapter init with existing IV solver)
- `crates/gpu/tests/black76_pricer_parity.rs`

**Approach:** Generalize existing `get_gpu_name()` so both Black-76 and the
legacy BSM pricer share adapter init. Don't reuse BSM shaders.

**Test scenarios:**
- Per-row |gpu_price - cpu_price| < 1e-3 on a representative grid.
- Sentinel rows (T=0) return the documented sentinel value.
- Pricer reuse across multiple `.solve()` calls without re-creating the adapter.

### U5. GPU Black-76 IV solver (WGSL + f64 CPU fix-up)

**Goal:** `GpuBlack76IVSolver` running branch-free SR + 2 Householder steps in
WGSL f32, then CPU walks the result and calls `black76_implied_vol_with_seed_f64`
for any row whose residual exceeds tolerance. Sentinel mask pre-filled with
-1.0 before dispatch.

**Dependencies:** U3, U4.

**Files:**
- `crates/gpu/src/black76_iv.rs`
- `crates/gpu/src/black76_iv_shader.wgsl`
- `crates/gpu/tests/black76_iv_parity.rs`

**Approach:** Sentinels (below-intrinsic, above upper bound, T=0) computed on
CPU pre-dispatch and written into the output as -1.0 so the shader skips them
with one bounds check. No `break` in the shader loop.

**Test scenarios:**
- |gpu_iv - cpu_iv| < 1e-3 vol-pts on solvable rows; exact match on sentinel rows.
- Branch-free verified by inspecting WGSL (no per-row early-exit).
- Fix-up rate measured and reported — most rows should not need CPU touch-up.

### U6. Python binding: `_gpu`-postfixed entry points + `gpu_name()` helper

**Goal:** Expose `_gpu`-postfixed Python entry points alongside the existing
CPU functions: `ngv_opx.black76_price_gpu(...)`, `ngv_opx.black76_implied_vol_gpu(...)`.
CPU functions and dtypes are unchanged. Add `ngv_opx.gpu_name() -> str | None`.

**Dependencies:** U5.

**Files:**
- `bindings/python/src/lib.rs` (PyO3 bindings)
- `bindings/python/python/ngv_opx/__init__.py` (re-exports + type stubs)
- `bindings/python/tests/test_gpu_entrypoints.py`

**Approach:** `_gpu` variants share signature with CPU variants and return f64
ndarrays of the same shape. If no GPU adapter is available, the `_gpu` variant
raises `RuntimeError` (Python side) — no silent CPU fallback.

**Test scenarios:**
- Existing CPU functions unchanged: outputs bit-exact with pre-PR1 baseline.
- `black76_implied_vol_gpu(...)` returns f64 ndarray, same shape as CPU variant,
  values within 1e-3 vol-pts on solvable rows, exact match on sentinels.
- `gpu_name()` returns string when adapter available, `None` otherwise.
- `_gpu` call on a machine with no adapter raises a clear `RuntimeError`,
  not a panic.

### U7. Crossover benchmark + README refresh

**Goal:** Measure CPU vs GPU break-even on a representative CL day. Decide
whether to flip `use_gpu` default. Update README perf table.

**Dependencies:** U6.

**Files:**
- `benchmarks/black76_cpu_vs_gpu.rs` (or `.py` if existing benches are Python)
- `README.md`

**Approach:** Pricing and IV both swept across batch size, realistic sentinel
fraction. Output the break-even N for each. Decision rule: a future routing
layer (deferred — see Scope Boundaries) flips to `_gpu` automatically only
when break-even N is below the median production batch size. Until then,
callers pick the variant explicitly.

**Test scenarios:** `Test expectation: none -- benchmark artifact, not behavior.`
The benchmark itself must run cleanly and produce a CSV/markdown table; no unit assertions.

---

## Key Technical Decisions

- **f32 GPU + f64 CPU fix-up over double-single emulation.** Simpler, faster
  in the common case, and the fix-up loop is the same code as the CPU solver.
  Revisit if measured fix-up rate is high enough to dominate.
- **Fixed 2 Householder steps everywhere.** No runtime convergence check.
  Required for clean auto-vectorization and for branch-free WGSL.
- **SoA over AoS for batch.** Matches market-data memory layout; AoS wrapper
  provided for convenience.
- **Share adapter init between legacy BSM GPU and new Black-76 GPU.** One
  `wgpu::Instance`/`Adapter` per process.

## Risks

- **SR formula transcription errors.** The paper contains no numerical tables
  to reproduce, so transcription correctness rests on: (a) round-trip tests on
  a dense (S,K,T,r,q,σ) grid, (b) ~20 third-party reference triples generated
  offline from `py_vollib_vectorized` or QuantLib and committed as fixtures,
  (c) the published relative-error band `(-0.0418, 0.1138)` enforced
  pointwise. All three must pass before U2 is considered done.
- **f32 precision insufficient on deep-OTM CL dailies.** Mitigation: measure
  fix-up rate in U5 tests; if dominant, escalate to double-single (deferred).
- **wgpu/Metal f64 unavailable.** Already assumed; f32+fix-up is the response.
- **GPU pipeline construction overhead.** Mitigation: persistent pricer/solver
  structs (don't re-create per call like the prototype did).
- **Removing legacy Newton path.** PR 1 removes Newton from the public CPU
  hot path. Mitigation: keep the Newton implementation reachable as a
  `#[cfg(test)]`-only oracle in `crates/core/tests/` so cross-checks stay
  available, and run the full round-trip suite under both algorithms one
  last time before deleting the production-path Newton wiring.

## System-Wide Impact

- `crates/core` gains a new `iv` module. The legacy `implied_vol.rs` Newton
  path is removed from the public API in PR 1; its iteration kernel is moved
  into `crates/core/tests/` as a test-only oracle. No `unsafe`, no public
  re-exports, no production callers.
- `crates/gpu` gains two new pipelines parallel to the legacy BSM ones.
- Python surface gains `_gpu`-postfixed variants and a `gpu_name()` helper.
  CPU functions and dtypes are unchanged.
- README perf table updated once U7 completes.

## Dependencies

- `rayon`, `libm` (or `statrs`), `thiserror` for `crates/core`.
- `wgpu`, `bytemuck` already present in `crates/gpu`.
- `criterion` dev-dep for benches.
