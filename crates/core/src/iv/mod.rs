//! Implied-volatility solver: Stefanica–Radoičić (2017) seed + 3 fixed Halley
//! refinement steps. (Plan called for Householder-3 quartic; the quartic
//! formula as written in the plan overshoots Newton on deep-ITM rows, so we
//! ship the well-known Halley cubic. Three steps from a 5% seed reach noise
//! floor on the bulk grid. See `iv/householder.rs` for the details and
//! `todo.md` Issue 3 for the rationale.)
//!
//! See docs/plans/2026-05-25-001-feat-sr-householder-and-gpu-black76-plan.md.
//!
//! Public surface:
//!   - [`solver::black76_implied_vol`]     — scalar, returns Result
//!   - [`batch::black76_implied_vol_batch`] — parallel SoA over rayon

pub mod batch;
pub mod black;
pub mod errors;
pub mod householder;
pub mod solver;
pub mod stefanica;

pub use batch::{black76_implied_vol_batch, black76_implied_vol_batch_serial};
pub use errors::IvError;
pub use solver::black76_implied_vol;
