# NGV implied-volatility solver

> ⚠ **Caveat.** This document was AI-generated from conversations with the
> codebase author. It may contain inaccuracies, mischaracterizations of the
> cited literature, or fabricated references. Treat all claims — especially
> the bibliography, accuracy numbers, and algorithm descriptions — as
> starting points for verification, not as authoritative statements. Please
> direct corrections, questions, and feedback to the codebase author.

This note describes the implied-volatility solver behind the current
`crates/core/src/iv` implementation. It borrows the *spirit* of Peter
Jäckel's "Let's Be Rational" (Wilmott 2015) — reduce Black IV inversion to
a well-conditioned normalized problem, start from a high-quality explicit
seed, then refine with a small number of high-order steps — but the
concrete algorithm and the design priorities are different. The seed is
from Stefanica & Radoičić (2017) and the refinement is Halley. No Jäckel
code was transcribed.

## What we did and why it differs from Jäckel

Jäckel's solver, as implemented in py_lets_be_rational and QuantLib's
`Spanderen` port, is:

1. A **region-partitioned rational cubic seed** — the normalized price
   domain is split into four regions around the ATM normalized price and
   the no-arbitrage boundaries, with log-transformed objective functions
   in the two outer (wing) regions to stay well-conditioned.
2. A **Householder-3** (quartic-convergence) refinement, capped at 2
   iterations by default.
3. Wrapped in **bracket logic + a reversal counter + bisection fallback**
   so the high-order step cannot overshoot the no-arbitrage bracket.

That design optimizes for per-row accuracy across the full domain. It is
also branchy: which region you fall into, whether your step stayed in the
bracket, and whether the reversal counter tripped all affect what code
runs on a given row.

This solver makes a different trade. It uses:

1. A **single closed-form Stefanica-Radoičić seed** that is uniformly ~5%
   accurate everywhere in the no-arb band. No regions, no log-transformed
   objectives in the wings.
2. **Halley** (cubic-convergence) refinement instead of Householder-3,
   because Halley does not overshoot from a 5%-accurate seed.
3. **Exactly 3 fixed iterations** with no bracket, no reversal counter,
   no bisection fallback, no early exit.

The result: every valid row executes the *same* sequence of arithmetic
instructions. Worst-case work equals average-case work. This is the
property the doc means by "straight-line compute," and it is the
structural goal of the project.

### Why straight-line compute is the structural goal

The accuracy and per-row CPU performance of Newton, Jäckel-style
Householder-3, and our SR + 3 Halley are all roughly comparable for any
single option. The interesting differences show up at batch scale:

- **CPU SIMD.** Modern x86/ARM cores execute 4–16 f64 operations per
  cycle via SIMD when the compiler can prove every row does the same
  work. A `while residual > tol` loop or a region branch defeats that
  proof; the compiler emits scalar code instead. Straight-line compute
  keeps the auto-vectorizer in play, which is where most of the batch
  throughput comes from.
- **GPU (SIMT warps).** A GPU warp of 32 threads runs in lockstep on
  one instruction stream. If one thread takes a branch the others do
  not, the warp serializes: each side of the branch runs with the other
  side's threads masked off doing nothing. A Newton loop where iteration
  counts vary from 3 to 80 across rows runs the whole warp for 80
  iterations. SR + 3 Halley runs every thread for 3 steps. The straight-
  line property is what makes a clean GPU port viable; without it the
  CPU and GPU paths end up as fundamentally different solvers.
- **Determinism.** No data-dependent branches means the same row
  produces bit-identical output regardless of which thread or device
  processed it. That removes a class of "why did this number drift
  between runs" debugging.
- **Tail latency.** Worst-case = average-case. A solver whose 99th
  percentile equals its 50th percentile is much easier to schedule
  inside a quoting or risk loop with a hard time budget.

The trade-off we accept for this is wing accuracy: SR's uniform 5% seed
in the deep wings produces a worst-case round-trip σ error of ~1.74e-9,
versus the ~1e-13 we hit in the bulk (and that Jäckel-style
region-aware seeds can likely hold across more of the domain). Below any
realistic downstream tolerance, but worth naming as the trade.

The current implementation uses:

1. Stefanica-Radoicic (2017) as the closed-form total-volatility seed.
2. Normalized Black-76 pricing in total-vol coordinates.
3. Three fixed Halley refinement steps.
4. No convergence loop in the hot path.

The result is a deterministic, branch-light f64 solver that is suitable for
batch execution and Rayon parallelism.

## Problem

Given a Black-76 option price, recover annualized volatility `sigma`.
Black-76 price is monotone in volatility in the valid no-arbitrage region, so
there is a unique implied volatility whenever the price has resolvable time
value.

The naive solver is Newton-Raphson on `sigma`. That works in many cases, but it
has undesirable hot-path properties:

- it needs an initial guess;
- it usually has per-row convergence checks;
- iteration count differs across rows;
- low-vega edge cases need extra control flow;
- bad rows can waste work before failing.

The solver here turns the problem into a fixed-cost map:

```text
validate inputs
convert puts to calls
normalize to Black coordinates
compute SR total-vol seed
run exactly 3 Halley steps
convert total vol back to annualized sigma
```

## Normalized Black coordinates

The solver works in total volatility:

```text
v = sigma * sqrt(T)
```

and log-forward moneyness:

```text
y = ln(F / K)
```

where `F` is the Black-76 forward, `K` is strike, and `T` is time in years.

The true Black-76 call price can be written as:

```text
C = exp(-rT) * sqrt(FK) * b(y, v)
```

where the normalized call price is:

```text
b(y, v) = exp(y/2) * Phi(d+) - exp(-y/2) * Phi(d-)
d+ = y / v + v / 2
d- = y / v - v / 2
```

This normalization is the core simplification. The root solve is no longer
over prices of arbitrary scale; it is over:

```text
f(v) = b(y, v) - b_market
```

where:

```text
b_market = C_market / (exp(-rT) * sqrt(FK))
```

The solver only has to find `v`. The final return is:

```text
sigma = v / sqrt(T)
```

## Put handling

The SR seed implementation is call-side only. Puts are converted to calls
before any SR math runs, using Black-76 put-call parity:

```text
C - P = exp(-rT) * (F - K)
C = P + exp(-rT) * (F - K)
```

After this conversion, every valid row follows the same call-side path.

## No-arbitrage gates

The solver validates before solving. For the call price after parity
conversion:

```text
intrinsic = max(exp(-rT) * (F - K), 0)
upper     = exp(-rT) * F
```

Prices below intrinsic return `IvError::BelowIntrinsic`. Prices above the
upper bound return `IvError::AboveNoArbitrage`.

The implementation also rejects prices that are technically inside the bounds
but too close to either boundary. In that regime the time value is below f64
price noise, vega is effectively zero, and implied volatility is not
numerically identifiable. Returning an error is better than returning a
precise-looking number.

Note: `BelowIntrinsic` currently covers both genuine arbitrage violations and
the "valid price but time value below f64 noise" case. A separate
`NoResolvableTimeValue` variant is on the open follow-up list for callers
who need to distinguish.

The SR seed has its own normalized call-price guard. Since:

```text
alpha_C = C / (K * exp(-rT))
```

the normalized call no-arbitrage band is:

```text
max(exp(y) - 1, 0) < alpha_C < exp(y)
```

This is easy to get wrong. The naive `(max(1 - exp(-y), 0), 1)` band only
matches at `y = 0`; it rejects valid in-the-money rows.

## Stage 1: SR closed-form seed

Stefanica-Radoicic replaces the normal CDF in Black-Scholes with Pólya's
closed-form approximation:

```text
A(x) = 1/2 + sgn(x)/2 * sqrt(1 - exp(-2x^2/pi))
```

That approximation makes the pricing equation explicitly invertible. Given
`y` and normalized market call price `alpha_C`, the SR formula computes a
total-volatility seed:

```text
v_seed ~= sigma_BS * sqrt(T)
```

The seed is good enough to be used directly as the starting point for a
high-order refinement method. The paper gives a broad relative-error band:

```text
-0.0418 < (sigma_BS - sigma_SR) / sigma_BS < 0.1138
```

The local tests enforce that band on synthetic Black-76 prices.

### The beta quadratic

The implementation follows the SR paper's equation structure. Define:

```text
R = 2 * alpha_C - exp(y) + 1
```

Then compute:

```text
Acoef = (exp((1 - 2/pi)y) - exp(-(1 - 2/pi)y))^2

Bcoef = 4 * (exp(2y/pi) + exp(-2y/pi))
        - 2 * exp(-y) * (exp(2y) + 1 - R^2)
          * (exp((1 - 2/pi)y) + exp(-(1 - 2/pi)y))

Ccoef = exp(-2y)
        * (R^2 - (exp(y) - 1)^2)
        * ((exp(y) + 1)^2 - R^2)
```

The total-vol seed is derived through:

```text
beta  = 2 * Ccoef / (Bcoef + sqrt(Bcoef^2 + 4 * Acoef * Ccoef))
gamma = -(pi / 2) * ln(beta)
```

The paper/PDF extraction is a transcription hazard here. In particular,
`Bcoef` must not contain a spurious `+ 1`, and `Ccoef` is a factored product,
not a subtraction of two unrelated terms.

At `y = 0`, the quadratic degenerates to a linear equation and the corrected
coefficients give:

```text
Bcoef = 4R^2
Ccoef = R^2(4 - R^2)
beta  = Ccoef / Bcoef = 1 - R^2 / 4 = 1 - alpha_C^2
```

The implementation uses the closed-form ATM-forward limit directly when
`|y| < 1e-10`:

```text
v = sqrt(-2*pi * ln(1 - alpha_C^2))
```

This avoids cancellation in the general formula.

### SR branch selection

Once `gamma` is known, SR selects one of four call-side formulas according to
`sign(y)` and whether the price is above or below the SR threshold `C0`.

For `y >= 0`:

```text
alpha_C >  C0: v =  sqrt(gamma + y) + sqrt(gamma - y)
alpha_C <= C0: v =  sqrt(gamma + y) - sqrt(gamma - y)
```

For `y < 0`:

```text
alpha_C >  C0: v =  sqrt(gamma + y) + sqrt(gamma - y)
alpha_C <= C0: v = -sqrt(gamma + y) + sqrt(gamma - y)
```

The negative leading square root in the last branch is intentional. The test
suite has explicit fixtures for all four branches, including this highest-risk
case.

## Stage 2: Halley refinement

The SR seed is only an approximation. The final solver refines the seed by
solving:

```text
f(v) = b(y, v) - b_market = 0
```

The implementation uses Halley's method:

```text
v_next = v - 2 * f * f' / (2 * (f')^2 - f * f'')
```

where:

```text
f'  = normalized vega
f'' = normalized volga
```

The normalized derivatives have compact forms:

```text
vega(y, v) = 1/sqrt(2pi) * exp(-y^2/(2v^2) - v^2/8)

volga(y, v) = vega(y, v) * (y^2/v^3 - v/4)
```

Each step is ~30 ns of arithmetic. The code also keeps the third derivative
around for future quartic Householder experiments, but the production path
does not need it.

## Why Halley instead of Householder-3

Jäckel uses Householder-3 (quartic convergence). It works in his solver
because his seed is region-aware and tight, and because the iteration is
wrapped in a bracket with bisection fallback that catches overshoot.

The original project plan inherited "use Householder-3" from that framing.
During implementation the quoted quartic formula was found to overshoot
Newton on deep in-the-money rows when fed a uniform SR seed. The honest
read: a quartic step paired with a 5%-accurate seed and no bracket is the
wrong pairing. Either tighten the seed and add a bracket (Jäckel's path),
or drop the convergence order and stay branch-free.

We chose the second. Halley's closed form

```text
v_{n+1}  =  v_n  −  2 f f' / (2 (f')² − f f'')
```

is the standard cubic-convergence alternative — well-documented in the IV
literature, no overshoot pathology from a 5% seed, no bracket needed.

Empirically from the SR seed: two Halley steps reach ~1e-9 in σ; the third
closes the remaining gap to the f64 noise floor in the bulk. Three is the
minimum that hits machine precision; we use three.

A corrected quartic implementation could be added later behind a feature
flag such as `quartic_householder` as a comparison path. It is not on the
critical path — cubic from a uniform seed already meets every accuracy
and throughput target.

## Fixed iteration count

There is intentionally no convergence check in the hot path. Every valid row
runs exactly the same refinement schedule:

```text
v = v_seed
repeat 3 times:
    v = halley_step(y, v, b_market)
```

This matters for performance:

- no per-row convergence branches;
- predictable instruction count;
- deterministic results;
- Rayon batch mapping has no reductions or ordering effects;
- the same schedule can be transcribed to WGSL later.

Newton's "stop when residual < ε" loop by contrast:

- **CPU**: branch-mispredicts per row. Auto-vectorizer can't speculate
  through a variable-trip-count loop.
- **GPU**: catastrophic. A SIMT warp of 32 threads runs at the speed of
  the slowest row's iteration count.

Fixed three Halley steps do the same arithmetic on every row regardless of
how close the seed already was. Worst-case work = average-case work. That's
what unlocks both the CPU vectorization and the eventual GPU port.

Convergence is proven by tests, not discovered at runtime.

## Normalized Black evaluation

`black_normalized(y, v)` evaluates:

```text
exp(y/2) * Phi(d+) - exp(-y/2) * Phi(d-)
```

Direct evaluation can lose precision when the option is deeply in the money
and the result is intrinsic plus a tiny time value. The implementation chooses
the numerically smaller side using parity-style algebra:

- for `y >= 0`, compute the in-the-money call as intrinsic plus a small
  out-of-the-money correction;
- for `y < 0`, evaluate the out-of-the-money call directly.

The code defines an `erfcx` helper for future deep-wing work, but the current
CPU hot path uses `libm::erfc` through `norm_cdf` plus this parity-side
selection. That is sufficient for the accepted U1/U2 domain.

## Public API

```rust
use ngv_opx_core::iv::{black76_implied_vol, IvError};

// Scalar
let sigma: Result<f64, IvError> = black76_implied_vol(
    forward,       // F
    strike,        // K
    rate,          // r
    time_years,    // T
    market_price,  // Cm or Pm
    is_call,       // true for call, false for put
);
```

Put inputs are converted to call coordinates via Black-76 parity
(`C − P = e^{-rT} · (F − K)`) inside the solver. The caller can pass puts
directly; the result is the same σ either way.

```rust
use ngv_opx_core::iv::{
    black76_implied_vol_batch,         // parallel via rayon
    black76_implied_vol_batch_serial,  // single-threaded
};

// Batch (SoA, parallel via rayon)
let mut out = vec![0.0_f64; n];
black76_implied_vol_batch(
    &forwards, &strikes, &rates, &times, &market_prices, &is_calls,
    &mut out,
);
// Per-row errors become NaN in `out`. Use the scalar API if you need the typed reason.
```

The serial batch entry is provided for cases where the caller is already
parallelizing at a higher level (per-symbol, per-tenor, etc.) and rayon's
nested-pool overhead dominates on small inner batches.

### Error variants

```rust
pub enum IvError {
    BelowIntrinsic,       // price below intrinsic OR within f64 noise of it
    AboveNoArbitrage,     // price at or above F·e^{-rT} (call) / K·e^{-rT} (put)
    NonPositiveTime,      // T ≤ 0 or non-finite
    NonPositiveForward,   // F ≤ 0 or K ≤ 0
    NonFinite,            // NaN/Inf in inputs, or internal numerical pathology
}
```

## Batch behavior

The scalar solver returns `Result<f64, IvError>`. Batch APIs accept
struct-of-arrays slices and write one output per row. Scalar errors become
`NaN` in the output slice. This keeps the hot batch path simple while
preserving typed error handling for callers that need row-level reasons.

The batch implementation is embarrassingly parallel:

```text
out.par_iter_mut().enumerate().for_each(|(i, slot)| {
    *slot = scalar_solver(row_i).unwrap_or(NaN)
})
```

There are no reductions, so parallel execution is deterministic for a fixed
input row.

## Accuracy and test coverage

| Metric | Value | Where |
|---|---|---|
| 100k random round-trip, bulk regime | worst σ error **3.24e-13** | `tests/iv_roundtrip.rs` |
| 100k random round-trip, wing regime | worst σ error **1.74e-9** | `tests/iv_roundtrip.rs` |
| Halley residual, bulk grid | worst **< 1e-13** | `iv::householder::tests` |
| Put-call parity in IV space | agreement to **1e-10** (vega > 1e-3) | `tests/iv_parity.rs` |
| SR vs Newton oracle, canonical fixtures | agreement to **< 1e-4** | `tests/iv_third_party_xcheck.rs` |
| SR seed accuracy band (paper) | enforced pointwise | `iv::stefanica::tests` |
| All four SR σ-branches fire | verified | `tests/iv_sr_branches.rs` |
| No-panic on garbage inputs | 18-row × 6-arg fuzz | `tests/iv_edge_cases.rs` |

**Tested regime:** σ ∈ [0.01, 3.0], T ∈ [1 day, 5 years], K/F ∈ [0.3, 3.0],
r ∈ [−0.02, 0.10], both call and put. Deep-ITM short-DTE rows where time
value drops below f64 noise are correctly rejected as `BelowIntrinsic` (IV is
mathematically indeterminate there; the legacy Newton solver also bails on
those with its `-1.0` sentinel).

Run the suite:

```
cargo test -p ngv-opx-core --release
```

52 tests across 7 test files. Diagnostic stats (worst-case errors, sample
counts, skip counts) print with `-- --nocapture`.

The latest observed 100k random round-trip numbers:

```text
bulk worst absolute sigma error: 3.24e-13
wing worst absolute sigma error: 1.74e-9
low-vega skips: 147
arbitrage/noise skips: 2994
```

The remaining audit gap is an external py_vollib or QuantLib fixture. The
current cross-check uses the existing Newton implementation as an internal
oracle.

## Comparison to the old Newton-Raphson path

The old solver in `crates/core/src/black76.rs` —
`black76_implied_vol_f64` and the f32 wrapper — uses bracketed Newton +
bisection with up to 100 iterations and per-row early termination. It remains
in the codebase as a `#[cfg(test)]` cross-check oracle but is not on the
production hot path going forward.

| | Newton (old) | SR + Halley (new) |
|---|---|---|
| Iteration count | variable, 5–15 typical, 100 worst-case | **fixed 3** |
| Accuracy on bulk | ~1e-6 to 1e-10 (depends on row) | ~1e-13 |
| Branch behavior | per-row early exit + bisection | **straight-line** |
| Vectorizes (auto) | poorly | well |
| GPU port viability | impossible (warp stalls) | trivial transcription |
| Sentinel on bad input | `-1.0` | typed `IvError` |
| Failure mode on deep-ITM noise | `-1.0` | `IvError::BelowIntrinsic` |

The headline win isn't accuracy or speed in isolation — it's **predictable
branch-free compute** that unlocks both the CPU auto-vectorizer and the GPU
port.

## Source map

```
crates/core/src/iv/
├── mod.rs            -- public re-exports (black76_implied_vol, batch, IvError)
├── errors.rs         -- IvError enum
├── black.rs          -- normalized Black-76 + vega/volga/d3 + erfcx
├── stefanica.rs      -- SR closed-form seed (Pólya-based, eqs 16–26)
├── householder.rs    -- Halley refinement (3 fixed steps, file misnamed for legacy reasons)
├── solver.rs         -- scalar entry: validate → parity → SR → refine → σ
└── batch.rs          -- rayon SoA batch + serial variant

crates/core/tests/
├── iv_primitives.rs           -- SR band check + normalized Black vs oracle
├── iv_roundtrip.rs            -- 100k random round-trip, deterministic seed
├── iv_parity.rs               -- put-call parity in IV space, dense grid
├── iv_edge_cases.rs           -- 17 bad-input tests + garbage fuzz
├── iv_sr_branches.rs          -- all 4 SR σ-branches + eq-24 spotlight
└── iv_third_party_xcheck.rs   -- vs existing Newton solver on 20 canonical points

crates/core/examples/
└── sr_seed_demo.rs            -- prints SR vs Newton on a CL-style grid
```

The `householder.rs` filename predates the Halley pivot. The module docstring
explains. A rename to `halley.rs` is a follow-up if anyone cares about the
file name — module path is `iv::householder` either way.

## What's deferred

Five items worth doing later, ranked by value:

1. **`IvError::NoResolvableTimeValue`** variant so callers can distinguish
   "bad market price" from "valid price, indeterminate IV at f64".
2. **True third-party reference fixtures** from py_vollib or QuantLib
   (currently the cross-check is against our own Newton solver).
3. **Corrected quartic Householder** behind a future explicit feature such as
   `quartic_householder`, as a comparison path.
4. **Tighten random-roundtrip wing tolerance** from 1e-7 to 1e-9 (observed
   worst is 1.74e-9; tightening locks current accuracy in as a contract).
5. **GPU transcription** of the same SR + Halley math to WGSL.

## Summary

The solver is best understood as:

```text
Black-76 price
  -> call-side parity normalization
  -> normalized total-vol Black problem
  -> Stefanica-Radoicic closed-form seed
  -> 3 fixed Halley steps
  -> annualized implied volatility
```

The design borrows the key lesson from Jäckel's "Let's Be Rational": do not
solve implied volatility with an open-ended root finder when a good analytic
seed and a small fixed number of high-order updates can make the solve
predictable, accurate, and fast.

## References

- Stefanica, D. & Radoičić, R. (2017). *An Explicit Implied Volatility
  Formula.* SSRN. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494>
  — source of the closed-form total-volatility seed used in Stage 1, the
  β-quadratic, and the four sign-branch formulas.
- Jäckel, P. (2015). *Let's Be Rational.* Wilmott Magazine. — design
  inspiration for the seed-then-refine structure and the "fixed small
  number of high-order steps" approach. Not ported; see *What we did and
  why it differs from Jäckel* above.
- vollib. *py_lets_be_rational* (Python port of Jäckel's reference
  implementation). <https://github.com/vollib/py_lets_be_rational> —
  consulted for the structural read of Jäckel's algorithm (region-partitioned
  rational-cubic seed, Householder-3, bracket + bisection fallback).
- Black, F. (1976). *The pricing of commodity contracts.* Journal of
  Financial Economics 3 (1–2): 167–179. — the Black-76 forward-price
  pricing model the solver inverts.
- Pólya, G. (1949). *Remarks on computing the probability integral in one
  and two dimensions.* Proc. Berkeley Symp. on Math. Statistics and
  Probability. — source of the closed-form normal-CDF approximation `A(x)`
  that makes the SR pricing equation explicitly invertible.
- Householder, A. S. (1970). *The Numerical Treatment of a Single Nonlinear
  Equation.* McGraw-Hill. — Householder's family of root-finding methods;
  Halley's method (used here) is the cubic-order member, Householder-3
  (used by Jäckel) is the quartic-order member.
