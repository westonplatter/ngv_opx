use bytemuck::{Pod, Zeroable};
use std::f32::consts::{FRAC_1_SQRT_2, PI};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct IVParams {
    pub spot: f32,
    pub strike: f32,
    pub rate: f32,
    pub time_to_maturity: f32,
    pub market_price: f32,
    pub is_call: f32,
    _padding1: f32,
    _padding2: f32,
}

impl IVParams {
    pub fn new(
        spot: f32,
        strike: f32,
        rate: f32,
        days_to_maturity: f32,
        market_price: f32,
        is_call: bool,
    ) -> Self {
        Self {
            spot,
            strike,
            rate,
            time_to_maturity: days_to_maturity / 365.0,
            market_price,
            is_call: if is_call { 1.0 } else { 0.0 },
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

const MAX_ITERATIONS: u32 = 100;
const TOLERANCE: f32 = 1e-6;
const MIN_VOL: f32 = 0.0001;
const MAX_VOL: f32 = 5.0;

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

fn norm_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

pub fn bs_price_cpu(s: f32, k: f32, r: f32, t: f32, sigma: f32, is_call: bool) -> f32 {
    let sqrt_t = t.sqrt();
    let sigma_sqrt_t = sigma * sqrt_t;

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;

    let discount = (-r * t).exp();

    if is_call {
        s * norm_cdf(d1) - k * discount * norm_cdf(d2)
    } else {
        k * discount * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}

fn bs_vega_cpu(s: f32, k: f32, r: f32, t: f32, sigma: f32) -> f32 {
    let sqrt_t = t.sqrt();
    let sigma_sqrt_t = sigma * sqrt_t;

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t;

    s * sqrt_t * norm_pdf(d1)
}

/// Calculate implied volatility for a single option using Newton-Raphson
pub fn implied_volatility_cpu(
    spot: f32,
    strike: f32,
    rate: f32,
    time_years: f32,
    market_price: f32,
    is_call: bool,
) -> f32 {
    let mut sigma = ((2.0 * PI / time_years).sqrt() * (market_price / spot)).clamp(0.1, 1.0);

    let discount = (-rate * time_years).exp();
    let intrinsic = if is_call {
        (spot - strike * discount).max(0.0)
    } else {
        (strike * discount - spot).max(0.0)
    };

    if market_price < intrinsic - TOLERANCE {
        return -1.0;
    }

    for _ in 0..MAX_ITERATIONS {
        let price = bs_price_cpu(spot, strike, rate, time_years, sigma, is_call);
        let diff = price - market_price;

        if diff.abs() < TOLERANCE {
            return sigma;
        }

        let vega = bs_vega_cpu(spot, strike, rate, time_years, sigma);

        if vega < 1e-10 {
            if diff > 0.0 {
                sigma *= 0.5;
            } else {
                sigma *= 1.5;
            }
        } else {
            sigma -= diff / vega;
        }

        sigma = sigma.clamp(MIN_VOL, MAX_VOL);
    }

    sigma
}

pub fn implied_volatility_batch_cpu(options: &[IVParams]) -> Vec<f32> {
    options
        .iter()
        .map(|o| {
            implied_volatility_cpu(
                o.spot,
                o.strike,
                o.rate,
                o.time_to_maturity,
                o.market_price,
                o.is_call > 0.5,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_recovery_cpu() {
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let t = 0.25;
        let true_vol = 0.25;

        let price = bs_price_cpu(spot, strike, rate, t, true_vol, true);
        let recovered = implied_volatility_cpu(spot, strike, rate, t, price, true);

        assert!((recovered - true_vol).abs() < 0.0001);
    }
}
