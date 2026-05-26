//! Error variants for the SR + Halley IV solver.
//!
//! These are returned by the scalar API and converted to `NaN` in the batch
//! API output slice (with the per-row reason discarded — callers who need
//! the reason use the `Result`-returning scalar entry).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvError {
    /// Market price below the no-arbitrage intrinsic lower bound.
    BelowIntrinsic,
    /// Market price at or above the no-arbitrage upper bound
    /// (call: F·e^{-rT}; put: K·e^{-rT}).
    AboveNoArbitrage,
    /// Non-positive time to expiry, or non-finite time.
    NonPositiveTime,
    /// Non-positive forward, strike, or other input precondition violation.
    NonPositiveForward,
    /// Input contained NaN/Inf, or the solver produced a non-finite result.
    NonFinite,
}
