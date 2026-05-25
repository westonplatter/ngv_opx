use bytemuck;
use ngv_opx_core::implied_vol::{bs_price_cpu, implied_volatility_batch_cpu, IVParams};
use ngv_opx_core::OptionParams;
use std::fmt;
use std::time::Instant;
use wgpu::util::DeviceExt;

// ============================================================================
// Graceful no-GPU handling
// ============================================================================

#[derive(Debug, Clone)]
pub enum GpuInitError {
    NoAdapter,
    DeviceRequestFailed(String),
}

impl fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "no GPU adapter available on this system"),
            GpuInitError::DeviceRequestFailed(msg) => write!(f, "failed to request GPU device: {}", msg),
        }
    }
}

impl std::error::Error for GpuInitError {}

/// Cheap probe: does this system have a usable GPU adapter? Does not create a
/// device, just queries the adapter list. Safe to call from CPU-only code paths.
pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_some()
}

// ============================================================================
// GPU Black-Scholes pricer
// ============================================================================

pub struct GpuPricer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub gpu_name: String,
}

impl GpuPricer {
    /// Construct a `GpuPricer`, returning `Err` on systems without a GPU adapter
    /// or where device acquisition fails. Use this in production code paths so
    /// callers can fall back to CPU instead of panicking.
    pub fn try_new() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuInitError::NoAdapter)?;

        let gpu_name = adapter.get_info().name;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .map_err(|e| GpuInitError::DeviceRequestFailed(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Black-Scholes Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
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
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Black-Scholes Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            gpu_name,
        })
    }

    /// Panicking shorthand for `try_new`. Use in demos/tests; production code
    /// should prefer `try_new` so missing GPU is recoverable.
    pub fn new() -> Self {
        Self::try_new().expect("GPU init failed (no adapter or device error)")
    }

    pub fn price(&self, options: &[OptionParams]) -> Vec<f32> {
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(options),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (options.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
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

    pub fn price_single(&self, params: OptionParams) -> f32 {
        self.price(&[params])[0]
    }
}

impl Default for GpuPricer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GPU IV solver
// ============================================================================

pub struct GpuIVSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub gpu_name: String,
}

impl GpuIVSolver {
    /// Construct a `GpuIVSolver`, returning `Err` on systems without a GPU
    /// adapter or where device acquisition fails.
    pub fn try_new() -> Result<Self, GpuInitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuInitError::NoAdapter)?;

        let gpu_name = adapter.get_info().name;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .map_err(|e| GpuInitError::DeviceRequestFailed(e.to_string()))?;

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

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            gpu_name,
        })
    }

    /// Panicking shorthand for `try_new`. Use in demos/tests; production code
    /// should prefer `try_new`.
    pub fn new() -> Self {
        Self::try_new().expect("GPU init failed (no adapter or device error)")
    }

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

    pub fn solve_single(&self, params: IVParams) -> f32 {
        self.solve(&[params])[0]
    }
}

impl Default for GpuIVSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Demo
// ============================================================================

pub fn run_iv_demo() {
    println!("\n{}", "=".repeat(80));
    println!("Implied Volatility Solver (Newton-Raphson)");
    println!("{}\n", "=".repeat(80));

    let gpu_solver = GpuIVSolver::new();
    println!("Using GPU: {}\n", gpu_solver.gpu_name);

    let test_cases = vec![
        (100.0, 100.0, 0.05, 30.0, 0.20, true),
        (100.0, 100.0, 0.05, 30.0, 0.30, true),
        (100.0, 100.0, 0.05, 30.0, 0.40, true),
        (100.0, 95.0, 0.05, 30.0, 0.25, true),
        (100.0, 105.0, 0.05, 30.0, 0.25, true),
        (100.0, 100.0, 0.05, 30.0, 0.20, false),
        (100.0, 95.0, 0.05, 30.0, 0.25, false),
        (100.0, 105.0, 0.05, 30.0, 0.25, false),
        (100.0, 100.0, 0.05, 365.0, 0.20, true),
    ];

    let options: Vec<IVParams> = test_cases
        .iter()
        .map(|&(spot, strike, rate, days, vol, is_call)| {
            let t = days / 365.0;
            let market_price = bs_price_cpu(spot, strike, rate, t, vol, is_call);
            IVParams::new(spot, strike, rate, days, market_price, is_call)
        })
        .collect();

    let gpu_ivs = gpu_solver.solve(&options);
    let cpu_ivs = implied_volatility_batch_cpu(&options);

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

    println!("\n--- Performance Benchmark: IV Solver GPU vs CPU ---");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "Batch Size", "GPU Time", "CPU Time", "Speedup"
    );
    println!("{}", "-".repeat(48));

    let batch_sizes = [1, 10, 1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000];

    for &size in &batch_sizes {
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

        if size == 1 {
            let _ = gpu_solver.solve(&batch);
        }

        let gpu_start = Instant::now();
        let gpu_results = gpu_solver.solve(&batch);
        let gpu_duration = gpu_start.elapsed();

        let cpu_start = Instant::now();
        let cpu_results = implied_volatility_batch_cpu(&batch);
        let cpu_duration = cpu_start.elapsed();

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU adapter; run with --ignored"]
    fn test_gpu_matches_cpu() {
        let gpu_pricer = GpuPricer::new();
        let options = vec![
            OptionParams::new(100.0, 95.0, 0.05, 0.20, 30.0 / 365.0, true),
            OptionParams::new(100.0, 100.0, 0.05, 0.20, 30.0 / 365.0, true),
            OptionParams::new(100.0, 105.0, 0.05, 0.20, 30.0 / 365.0, false),
        ];
        let gpu_results = gpu_pricer.price(&options);
        for (i, opt) in options.iter().enumerate() {
            let cpu_price = ngv_opx_core::black_scholes_cpu(
                opt.spot,
                opt.strike,
                opt.rate,
                opt.volatility,
                opt.time_to_maturity,
                opt.is_call > 0.5,
            );
            let diff = (gpu_results[i] - cpu_price).abs();
            assert!(diff < 0.001, "GPU/CPU mismatch at {}: GPU={}, CPU={}", i, gpu_results[i], cpu_price);
        }
    }

    #[test]
    #[ignore = "requires GPU adapter; run with --ignored"]
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
            assert!(diff < 0.001);
        }
    }
}
