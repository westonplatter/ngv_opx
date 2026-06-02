//! How to access and use the GPU pricing software.
//!
//! Run it with:
//!
//! ```sh
//! cargo run -p ngv-opx-gpu --example gpu_pricing
//! ```
//!
//! The example is written so it works everywhere: it probes for a GPU first and
//! falls back to the CPU pricer when no adapter is available (headless CI,
//! containers, machines without a usable graphics backend), so it never panics.

use ngv_opx_core::{black_scholes_cpu, OptionParams};
use ngv_opx_gpu::{gpu_available, GpuPricer};

fn main() {
    // A small option chain: same underlying, a spread of strikes and expiries.
    let options = vec![
        OptionParams::new_from_days(100.0, 95.0, 0.05, 0.20, 30.0, true),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 30.0, true),
        OptionParams::new_from_days(100.0, 105.0, 0.05, 0.20, 30.0, false),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 365.0, true),
    ];

    // 1. Cheap probe — does this machine have a usable GPU adapter? This does
    //    not create a device, so it is safe to call from CPU-only code paths.
    if !gpu_available() {
        println!("No GPU adapter found; pricing on the CPU instead.\n");
        price_on_cpu(&options);
        return;
    }

    // 2. Acquire the GPU. `try_new` returns `Err` instead of panicking when the
    //    adapter or device cannot be obtained, so production code can fall back.
    let pricer = match GpuPricer::try_new() {
        Ok(pricer) => pricer,
        Err(err) => {
            println!("GPU init failed ({err}); falling back to CPU.\n");
            price_on_cpu(&options);
            return;
        }
    };

    println!("Using GPU: {}\n", pricer.gpu_name);

    // 3. Price the whole batch in one dispatch. The GPU shines as the batch
    //    grows; for a handful of options the CPU path is already instant.
    let gpu_prices = pricer.price(&options);

    println!("{:>5} {:>6} {:>6} {:>10}", "Type", "Strike", "Days", "GPU Price");
    println!("{}", "-".repeat(32));
    for (opt, price) in options.iter().zip(&gpu_prices) {
        println!(
            "{:>5} {:>6.0} {:>6.0} {:>10.4}",
            if opt.is_call_option() { "Call" } else { "Put" },
            opt.strike,
            opt.time_to_maturity * 365.0,
            price,
        );
    }
}

/// CPU fallback: price each option with the analytic Black-Scholes solver.
fn price_on_cpu(options: &[OptionParams]) {
    println!("{:>5} {:>6} {:>6} {:>10}", "Type", "Strike", "Days", "CPU Price");
    println!("{}", "-".repeat(32));
    for opt in options {
        let price = black_scholes_cpu(
            opt.spot,
            opt.strike,
            opt.rate,
            opt.volatility,
            opt.time_to_maturity,
            opt.is_call_option(),
        );
        println!(
            "{:>5} {:>6.0} {:>6.0} {:>10.4}",
            if opt.is_call_option() { "Call" } else { "Put" },
            opt.strike,
            opt.time_to_maturity * 365.0,
            price,
        );
    }
}
