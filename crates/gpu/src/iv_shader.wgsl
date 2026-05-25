// Implied Volatility Solver using Newton-Raphson Method
// Standard academic approach for backing out IV from option prices

struct IVParams {
    spot: f32,              // Current price of underlying
    strike: f32,            // Strike price
    rate: f32,              // Risk-free interest rate
    time_to_maturity: f32,  // Time to expiration in years
    market_price: f32,      // Observed market price of the option
    is_call: f32,           // 1.0 for call, 0.0 for put
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<storage, read> params: array<IVParams>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;

const PI: f32 = 3.14159265358979323846;
const MAX_ITERATIONS: u32 = 100u;
const TOLERANCE: f32 = 1e-6;
const MIN_VOL: f32 = 0.0001;
const MAX_VOL: f32 = 5.0;

// Error function approximation (Abramowitz and Stegun)
fn erf_approx(x: f32) -> f32 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-abs_x * abs_x);

    return sign * y;
}

// Cumulative normal distribution
fn norm_cdf(x: f32) -> f32 {
    let FRAC_1_SQRT_2 = 0.7071067811865476;
    return 0.5 * (1.0 + erf_approx(x * FRAC_1_SQRT_2));
}

// Standard normal probability density function
fn norm_pdf(x: f32) -> f32 {
    return exp(-0.5 * x * x) / sqrt(2.0 * PI);
}

// Black-Scholes price given volatility
fn bs_price(S: f32, K: f32, r: f32, T: f32, sigma: f32, is_call: bool) -> f32 {
    let sqrt_T = sqrt(T);
    let sigma_sqrt_T = sigma * sqrt_T;

    let d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T;
    let d2 = d1 - sigma_sqrt_T;

    let discount = exp(-r * T);

    if is_call {
        return S * norm_cdf(d1) - K * discount * norm_cdf(d2);
    } else {
        return K * discount * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

// Vega: derivative of option price with respect to volatility
// vega = S * sqrt(T) * N'(d1)
fn bs_vega(S: f32, K: f32, r: f32, T: f32, sigma: f32) -> f32 {
    let sqrt_T = sqrt(T);
    let sigma_sqrt_T = sigma * sqrt_T;

    let d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T;

    return S * sqrt_T * norm_pdf(d1);
}

// Newton-Raphson solver for implied volatility
fn solve_iv(S: f32, K: f32, r: f32, T: f32, market_price: f32, is_call: bool) -> f32 {
    // Initial guess using Brenner-Subrahmanyam approximation
    // sigma_0 ≈ sqrt(2*pi/T) * (C/S) for ATM options
    var sigma = sqrt(2.0 * PI / T) * (market_price / S);
    sigma = clamp(sigma, 0.1, 1.0);  // Reasonable starting range

    // Check for intrinsic value violations
    let discount = exp(-r * T);
    let intrinsic: f32 = select(
        max(K * discount - S, 0.0),  // Put intrinsic
        max(S - K * discount, 0.0),  // Call intrinsic
        is_call
    );

    // If market price is below intrinsic, no valid IV exists
    if market_price < intrinsic - TOLERANCE {
        return -1.0;  // Invalid/no solution
    }

    // Newton-Raphson iteration
    for (var i = 0u; i < MAX_ITERATIONS; i = i + 1u) {
        let price = bs_price(S, K, r, T, sigma, is_call);
        let diff = price - market_price;

        // Check convergence
        if abs(diff) < TOLERANCE {
            return sigma;
        }

        let vega = bs_vega(S, K, r, T, sigma);

        // Avoid division by very small vega
        if vega < 1e-10 {
            // Fall back to bisection step
            if diff > 0.0 {
                sigma = sigma * 0.5;
            } else {
                sigma = sigma * 1.5;
            }
        } else {
            // Newton-Raphson update
            sigma = sigma - diff / vega;
        }

        // Keep sigma in valid range
        sigma = clamp(sigma, MIN_VOL, MAX_VOL);
    }

    // Did not converge - return best estimate
    return sigma;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= arrayLength(&params) {
        return;
    }

    let opt = params[idx];

    let iv = solve_iv(
        opt.spot,
        opt.strike,
        opt.rate,
        opt.time_to_maturity,
        opt.market_price,
        opt.is_call > 0.5
    );

    results[idx] = iv;
}
