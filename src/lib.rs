pub mod implied_vol;
pub mod black76;

#[cfg(feature = "python")]
mod python;

use bytemuck::{Pod, Zeroable};
use std::f32::consts::FRAC_1_SQRT_2;
use wgpu::util::DeviceExt;

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
}

pub struct GpuPricer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub gpu_name: String,
}

impl GpuPricer {
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

        Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            gpu_name,
        }
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
    #[ignore = "requires GPU adapter (Apple Silicon); run with --ignored"]
    fn test_gpu_matches_cpu() {
        let gpu_pricer = GpuPricer::new();

        let options = vec![
            OptionParams::new(100.0, 95.0, 0.05, 0.20, 30.0 / 365.0, true),
            OptionParams::new(100.0, 100.0, 0.05, 0.20, 30.0 / 365.0, true),
            OptionParams::new(100.0, 105.0, 0.05, 0.20, 30.0 / 365.0, false),
        ];

        let gpu_results = gpu_pricer.price(&options);

        for (i, opt) in options.iter().enumerate() {
            let cpu_price = black_scholes_cpu(
                opt.spot,
                opt.strike,
                opt.rate,
                opt.volatility,
                opt.time_to_maturity,
                opt.is_call > 0.5,
            );
            let diff = (gpu_results[i] - cpu_price).abs();
            assert!(
                diff < 0.001,
                "GPU/CPU mismatch at index {}: GPU={}, CPU={}, diff={}",
                i,
                gpu_results[i],
                cpu_price,
                diff
            );
        }
    }

    #[test]
    fn test_itm_otm_pricing() {
        let itm_call = black_scholes_cpu(100.0, 90.0, 0.05, 0.20, 0.25, true);
        let otm_call = black_scholes_cpu(100.0, 110.0, 0.05, 0.20, 0.25, true);
        assert!(
            itm_call > otm_call,
            "ITM call ({}) should be > OTM call ({})",
            itm_call,
            otm_call
        );

        let itm_put = black_scholes_cpu(100.0, 110.0, 0.05, 0.20, 0.25, false);
        let otm_put = black_scholes_cpu(100.0, 90.0, 0.05, 0.20, 0.25, false);
        assert!(
            itm_put > otm_put,
            "ITM put ({}) should be > OTM put ({})",
            itm_put,
            otm_put
        );
    }
}
