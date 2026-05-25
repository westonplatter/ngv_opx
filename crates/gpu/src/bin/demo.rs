use ngv_opx_core::{black_scholes_cpu, OptionParams};
use ngv_opx_gpu::{run_iv_demo, GpuPricer};
use std::env;
use std::time::Instant;

fn run_pricing_demo() {
    println!("Black-Scholes GPU Option Pricer (wgpu + Metal)\n");

    let gpu_pricer = GpuPricer::new();
    println!("Using GPU: {}\n", gpu_pricer.gpu_name);

    let options = vec![
        OptionParams::new_from_days(100.0, 95.0, 0.05, 0.20, 30.0, true),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 30.0, true),
        OptionParams::new_from_days(100.0, 105.0, 0.05, 0.20, 30.0, true),
        OptionParams::new_from_days(100.0, 95.0, 0.05, 0.20, 30.0, false),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 30.0, false),
        OptionParams::new_from_days(100.0, 105.0, 0.05, 0.20, 30.0, false),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 90.0, true),
        OptionParams::new_from_days(100.0, 100.0, 0.05, 0.20, 365.0, true),
    ];

    let gpu_results = gpu_pricer.price(&options);
    let cpu_results: Vec<f32> = options
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
        .collect();

    println!(
        "{:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:>10} {:>10} {:>8}",
        "Spot", "Strike", "Rate", "Vol", "Days", "Type", "GPU Price", "CPU Price", "Diff"
    );
    println!("{}", "-".repeat(80));

    for (i, opt) in options.iter().enumerate() {
        let option_type = if opt.is_call > 0.5 { "Call" } else { "Put" };
        let diff = (gpu_results[i] - cpu_results[i]).abs();
        println!(
            "{:<6.0} {:<6.0} {:<6.2} {:<6.2} {:<6.0} {:<6} {:>10.4} {:>10.4} {:>8.6}",
            opt.spot,
            opt.strike,
            opt.rate,
            opt.volatility,
            opt.time_to_maturity * 365.0,
            option_type,
            gpu_results[i],
            cpu_results[i],
            diff
        );
    }

    println!("\n--- Performance Benchmark: GPU vs CPU (resources reused) ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "Batch Size", "GPU Time", "CPU Time", "Speedup"
    );
    println!("{}", "-".repeat(48));

    let batch_sizes = [
        1, 10, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
        4_000_000,
    ];

    for &size in &batch_sizes {
        let batch: Vec<OptionParams> = (0..size)
            .map(|i| {
                OptionParams::new_from_days(
                    100.0 + (i % 20) as f32,
                    100.0 + ((i / 20) % 20) as f32,
                    0.05,
                    0.15 + (i % 10) as f32 * 0.01,
                    30.0 + (i % 90) as f32,
                    i % 2 == 0,
                )
            })
            .collect();

        if size == 1 {
            let _ = gpu_pricer.price(&batch);
        }

        let gpu_start = Instant::now();
        let gpu_results = gpu_pricer.price(&batch);
        let gpu_duration = gpu_start.elapsed();

        let cpu_start = Instant::now();
        let cpu_results: Vec<f32> = batch
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
            .collect();
        let cpu_duration = cpu_start.elapsed();

        let _sum: f32 = cpu_results.iter().sum::<f32>() + gpu_results.iter().sum::<f32>();

        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        let speedup_str = if speedup >= 1.0 {
            format!("{:.2}x faster", speedup)
        } else {
            format!("{:.2}x slower", 1.0 / speedup)
        };

        println!(
            "{:>10} {:>12.3?} {:>12.3?} {:>10}",
            size, gpu_duration, cpu_duration, speedup_str
        );
    }

    println!(
        "\nNote: GPU overhead is now amortized. Speedup depends on batch size and computation complexity."
    );
}

fn print_usage() {
    println!("NGV Option Pricer (ngv_opx)\n");
    println!("Usage: ngv_opx [command]\n");
    println!("Commands:");
    println!("  price    Run Black-Scholes pricing benchmark (default)");
    println!("  iv       Run implied volatility solver benchmark");
    println!("  all      Run both benchmarks");
    println!("\nExamples:");
    println!("  cargo run --release");
    println!("  cargo run --release -- price");
    println!("  cargo run --release -- iv");
    println!("  cargo run --release -- all");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let command = if args.len() > 1 {
        args[1].as_str()
    } else {
        "price"
    };

    match command {
        "price" => run_pricing_demo(),
        "iv" => run_iv_demo(),
        "all" => {
            run_pricing_demo();
            run_iv_demo();
        }
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            println!("Unknown command: {}\n", command);
            print_usage();
        }
    }
}
