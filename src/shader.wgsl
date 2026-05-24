// Black-Scholes Option Pricing Compute Shader

struct OptionParams {
    spot: f32,           // Current price of underlying
    strike: f32,         // Strike price
    rate: f32,           // Risk-free interest rate (annualized, e.g., 0.05 for 5%)
    volatility: f32,     // Volatility (annualized, e.g., 0.20 for 20%)
    time_to_maturity: f32, // Time to expiration in years
    is_call: f32,        // 1.0 for call, 0.0 for put
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<storage, read> params: array<OptionParams>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;

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

// Cumulative normal distribution using erf
fn norm_cdf(x: f32) -> f32 {
    let FRAC_1_SQRT_2 = 0.7071067811865476;
    return 0.5 * (1.0 + erf_approx(x * FRAC_1_SQRT_2));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= arrayLength(&params) {
        return;
    }

    let opt = params[idx];

    let S = opt.spot;
    let K = opt.strike;
    let r = opt.rate;
    let sigma = opt.volatility;
    let T = opt.time_to_maturity;

    let sqrt_T = sqrt(T);
    let sigma_sqrt_T = sigma * sqrt_T;

    // Calculate d1 and d2
    let d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T;
    let d2 = d1 - sigma_sqrt_T;

    // Discount factor
    let discount = exp(-r * T);

    var price: f32;

    if opt.is_call > 0.5 {
        // Call option: C = S * N(d1) - K * e^(-rT) * N(d2)
        price = S * norm_cdf(d1) - K * discount * norm_cdf(d2);
    } else {
        // Put option: P = K * e^(-rT) * N(-d2) - S * N(-d1)
        price = K * discount * norm_cdf(-d2) - S * norm_cdf(-d1);
    }

    results[idx] = price;
}
