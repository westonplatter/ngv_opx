pub mod black76;
pub mod implied_vol;
pub mod iv;

use bytemuck::{Pod, Zeroable};
use std::f32::consts::FRAC_1_SQRT_2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OptionParams {
    pub spot: f32,
    pub strike: f32,
    pub rate: f32,
    pub volatility: f32,
    pub time_to_maturity: f32,
    pub is_call: f32,
    _padding1: f32,
    _padding2: f32,
}

impl OptionParams {
    pub fn new(
        spot: f32,
        strike: f32,
        rate: f32,
        volatility: f32,
        time_to_maturity_years: f32,
        is_call: bool,
    ) -> Self {
        Self {
            spot,
            strike,
            rate,
            volatility,
            time_to_maturity: time_to_maturity_years,
            is_call: if is_call { 1.0 } else { 0.0 },
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }

    pub fn new_from_days(
        spot: f32,
        strike: f32,
        rate: f32,
        volatility: f32,
        days_to_maturity: f32,
        is_call: bool,
    ) -> Self {
        Self::new(spot, strike, rate, volatility, days_to_maturity / 365.0, is_call)
    }

    /// Whether this option is a call (`true`) or a put (`false`).
    ///
    /// `is_call` is stored as an `f32` so the struct is `Pod` and can be
    /// uploaded straight to a GPU buffer. This accessor centralizes the
    /// `> 0.5` decode so call sites don't repeat the magic comparison.
    pub fn is_call_option(&self) -> bool {
        self.is_call > 0.5
    }
}

fn norm_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

fn erf(x: f32) -> f32 {
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let p = 0.3275911_f32;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

pub fn black_scholes_cpu(
    spot: f32,
    strike: f32,
    rate: f32,
    volatility: f32,
    time_years: f32,
    is_call: bool,
) -> f32 {
    let sqrt_t = time_years.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * time_years)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;
    let discount = (-rate * time_years).exp();

    if is_call {
        spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    } else {
        strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    }
}

pub fn black_scholes_batch_cpu(options: &[OptionParams]) -> Vec<f32> {
    options
        .iter()
        .map(|o| {
            black_scholes_cpu(
                o.spot,
                o.strike,
                o.rate,
                o.volatility,
                o.time_to_maturity,
                o.is_call > 0.5,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_black_scholes_call() {
        let price = black_scholes_cpu(100.0, 100.0, 0.05, 0.20, 1.0, true);
        assert!((price - 10.45).abs() < 0.1, "Call price was {}", price);
    }

    #[test]
    fn test_cpu_black_scholes_put() {
        let price = black_scholes_cpu(100.0, 100.0, 0.05, 0.20, 1.0, false);
        assert!((price - 5.57).abs() < 0.1, "Put price was {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let t = 1.0;
        let vol = 0.20;

        let call = black_scholes_cpu(s, k, r, vol, t, true);
        let put = black_scholes_cpu(s, k, r, vol, t, false);
        let parity = s - k * (-r * t).exp();

        assert!(
            (call - put - parity).abs() < 0.0001,
            "Put-call parity violated: C-P={}, S-Ke^(-rT)={}",
            call - put,
            parity
        );
    }

    #[test]
    fn test_itm_otm_pricing() {
        let itm_call = black_scholes_cpu(100.0, 90.0, 0.05, 0.20, 0.25, true);
        let otm_call = black_scholes_cpu(100.0, 110.0, 0.05, 0.20, 0.25, true);
        assert!(itm_call > otm_call);

        let itm_put = black_scholes_cpu(100.0, 110.0, 0.05, 0.20, 0.25, false);
        let otm_put = black_scholes_cpu(100.0, 90.0, 0.05, 0.20, 0.25, false);
        assert!(itm_put > otm_put);
    }
}
