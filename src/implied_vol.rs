use bytemuck::{Pod, Zeroable};
use std::f32::consts::{FRAC_1_SQRT_2, PI};
use std::time::Instant;
use wgpu::util::DeviceExt;

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

// ============================================================================
// CPU Implementation - Newton-Raphson for Implied Volatility
// ============================================================================

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

fn bs_price_cpu(s: f32, k: f32, r: f32, t: f32, sigma: f32, is_call: bool) -> f32 {
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
    // Initial guess using Brenner-Subrahmanyam approximation
    let mut sigma = ((2.0 * PI / time_years).sqrt() * (market_price / spot)).clamp(0.1, 1.0);

    // Check for intrinsic value violations
    let discount = (-rate * time_years).exp();
    let intrinsic = if is_call {
        (spot - strike * discount).max(0.0)
    } else {
        (strike * discount - spot).max(0.0)
    };

    if market_price < intrinsic - TOLERANCE {
        return -1.0; // Invalid - price below intrinsic
    }

    // Newton-Raphson iteration
    for _ in 0..MAX_ITERATIONS {
        let price = bs_price_cpu(spot, strike, rate, time_years, sigma, is_call);
        let diff = price - market_price;

        if diff.abs() < TOLERANCE {
            return sigma;
        }

        let vega = bs_vega_cpu(spot, strike, rate, time_years, sigma);

        if vega < 1e-10 {
            // Fall back to bisection step
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

    sigma // Return best estimate if not converged
}

/// Calculate implied volatility for a batch of options on CPU
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

// ============================================================================
// GPU Implementation
// ============================================================================

pub struct GpuIVSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub gpu_name: String,
}

impl GpuIVSolver {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find GPU adapter");

        let gpu_name = adapter.get_info().name;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .expect("Failed to create device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IV Solver Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("iv_shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IV Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IV Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IV Solver Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            gpu_name,
        }
    }

    /// Calculate implied volatility for a batch of options on GPU
    pub fn solve(&self, options: &[IVParams]) -> Vec<f32> {
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IV Input Buffer"),
                contents: bytemuck::cast_slice(options),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (options.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IV Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IV Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IV Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IV Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IV Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (options.len() as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        results
    }

    /// Solve implied volatility for a single option (convenience method)
    pub fn solve_single(&self, params: IVParams) -> f32 {
        self.solve(&[params])[0]
    }
}

// ============================================================================
// Demo and Benchmarks
// ============================================================================

pub fn run_iv_demo() {
    println!("\n{}", "=".repeat(80));
    println!("Implied Volatility Solver (Newton-Raphson)");
    println!("{}\n", "=".repeat(80));

    // Initialize GPU solver
    let gpu_solver = GpuIVSolver::new();
    println!("Using GPU: {}\n", gpu_solver.gpu_name);

    // Test cases: we know the volatility used to generate these prices
    // So we can verify the IV solver recovers it correctly
    let test_cases = vec![
        // (spot, strike, rate, days, known_vol, is_call)
        (100.0, 100.0, 0.05, 30.0, 0.20, true),   // ATM call, 20% vol
        (100.0, 100.0, 0.05, 30.0, 0.30, true),   // ATM call, 30% vol
        (100.0, 100.0, 0.05, 30.0, 0.40, true),   // ATM call, 40% vol
        (100.0, 95.0, 0.05, 30.0, 0.25, true),    // ITM call
        (100.0, 105.0, 0.05, 30.0, 0.25, true),   // OTM call
        (100.0, 100.0, 0.05, 30.0, 0.20, false),  // ATM put
        (100.0, 95.0, 0.05, 30.0, 0.25, false),   // OTM put
        (100.0, 105.0, 0.05, 30.0, 0.25, false),  // ITM put
        (100.0, 100.0, 0.05, 365.0, 0.20, true),  // 1-year ATM call
    ];

    // Generate market prices using known volatilities
    let options: Vec<IVParams> = test_cases
        .iter()
        .map(|&(spot, strike, rate, days, vol, is_call)| {
            let t = days / 365.0;
            let market_price = bs_price_cpu(spot, strike, rate, t, vol, is_call);
            IVParams::new(spot, strike, rate, days, market_price, is_call)
        })
        .collect();

    // Solve for IV on GPU
    let gpu_ivs = gpu_solver.solve(&options);

    // Solve for IV on CPU
    let cpu_ivs = implied_volatility_batch_cpu(&options);

    // Print results
    println!(
        "{:<6} {:<6} {:<5} {:<8} {:<6} {:>8} {:>8} {:>8} {:>8}",
        "Spot", "Strike", "Days", "MktPrice", "Type", "TrueVol", "GPU IV", "CPU IV", "Error"
    );
    println!("{}", "-".repeat(80));

    for (i, (opt, &(_, _, _, _, true_vol, _))) in options.iter().zip(test_cases.iter()).enumerate()
    {
        let option_type = if opt.is_call > 0.5 { "Call" } else { "Put" };
        let error = (gpu_ivs[i] - true_vol).abs();
        println!(
            "{:<6.0} {:<6.0} {:<5.0} {:<8.4} {:<6} {:>8.4} {:>8.4} {:>8.4} {:>8.6}",
            opt.spot,
            opt.strike,
            opt.time_to_maturity * 365.0,
            opt.market_price,
            option_type,
            true_vol,
            gpu_ivs[i],
            cpu_ivs[i],
            error
        );
    }

    // Performance benchmark
    println!("\n--- Performance Benchmark: IV Solver GPU vs CPU ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "Batch Size", "GPU Time", "CPU Time", "Speedup"
    );
    println!("{}", "-".repeat(48));

    let batch_sizes = [1, 10, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

    for &size in &batch_sizes {
        // Generate random-ish option data
        let batch: Vec<IVParams> = (0..size)
            .map(|i| {
                let spot = 100.0;
                let strike = 90.0 + (i % 20) as f32;
                let rate = 0.05;
                let days = 30.0 + (i % 335) as f32;
                let vol = 0.15 + (i % 30) as f32 * 0.01;
                let is_call = i % 2 == 0;
                let t = days / 365.0;
                let market_price = bs_price_cpu(spot, strike, rate, t, vol, is_call);
                IVParams::new(spot, strike, rate, days, market_price, is_call)
            })
            .collect();

        // Warm up
        if size == 1 {
            let _ = gpu_solver.solve(&batch);
        }

        // GPU benchmark
        let gpu_start = Instant::now();
        let gpu_results = gpu_solver.solve(&batch);
        let gpu_duration = gpu_start.elapsed();

        // CPU benchmark
        let cpu_start = Instant::now();
        let cpu_results = implied_volatility_batch_cpu(&batch);
        let cpu_duration = cpu_start.elapsed();

        // Prevent optimization
        let _sum: f32 = gpu_results.iter().sum::<f32>() + cpu_results.iter().sum::<f32>();

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

    println!("\nNote: IV calculation involves iterative Newton-Raphson (up to 100 iterations per option).");
    println!("This is more compute-intensive than simple Black-Scholes pricing.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_recovery_cpu() {
        // Generate price with known vol, then recover it
        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let t = 0.25; // 3 months
        let true_vol = 0.25;

        let price = bs_price_cpu(spot, strike, rate, t, true_vol, true);
        let recovered_vol = implied_volatility_cpu(spot, strike, rate, t, price, true);

        assert!(
            (recovered_vol - true_vol).abs() < 0.0001,
            "IV recovery failed: true={}, recovered={}",
            true_vol,
            recovered_vol
        );
    }

    #[test]
    #[ignore = "requires GPU adapter (Apple Silicon); run with --ignored"]
    fn test_iv_recovery_gpu() {
        let gpu_solver = GpuIVSolver::new();

        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let days = 90.0;
        let true_vol = 0.30;
        let t = days / 365.0;

        let price = bs_price_cpu(spot, strike, rate, t, true_vol, true);
        let params = IVParams::new(spot, strike, rate, days, price, true);

        let recovered_vol = gpu_solver.solve_single(params);

        assert!(
            (recovered_vol - true_vol).abs() < 0.001,
            "GPU IV recovery failed: true={}, recovered={}",
            true_vol,
            recovered_vol
        );
    }

    #[test]
    #[ignore = "requires GPU adapter (Apple Silicon); run with --ignored"]
    fn test_iv_gpu_matches_cpu() {
        let gpu_solver = GpuIVSolver::new();

        let test_cases = vec![
            (100.0, 95.0, 0.05, 30.0, 0.20, true),
            (100.0, 100.0, 0.05, 60.0, 0.25, true),
            (100.0, 105.0, 0.05, 90.0, 0.30, false),
        ];

        let options: Vec<IVParams> = test_cases
            .iter()
            .map(|&(spot, strike, rate, days, vol, is_call)| {
                let t = days / 365.0;
                let price = bs_price_cpu(spot, strike, rate, t, vol, is_call);
                IVParams::new(spot, strike, rate, days, price, is_call)
            })
            .collect();

        let gpu_ivs = gpu_solver.solve(&options);
        let cpu_ivs = implied_volatility_batch_cpu(&options);

        for i in 0..options.len() {
            let diff = (gpu_ivs[i] - cpu_ivs[i]).abs();
            assert!(
                diff < 0.001,
                "GPU/CPU IV mismatch at {}: GPU={}, CPU={}",
                i,
                gpu_ivs[i],
                cpu_ivs[i]
            );
        }
    }
}
