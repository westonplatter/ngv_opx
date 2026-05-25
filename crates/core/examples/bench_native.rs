//! Native Rust benchmark for ngv_opx Black-76 pricing — no Python, no PyO3.
//!
//! This is the "Rust floor": what you'd get if your whole strategy ran in
//! Rust and never crossed an FFI boundary. Emits JSON on stdout so the
//! Python chart script can splice the numbers in alongside the PyO3-called
//! variants.
//!
//! Run: cargo run --release --example bench_native

use ngv_opx_core::black76::black76_price_batch_f64;
use std::hint::black_box;
use std::time::Instant;

const SIZES: &[usize] = &[10, 100, 1_000, 10_000, 100_000, 1_000_000];
const RATE: f64 = 0.045;

/// Deterministic LCG so the workload doesn't depend on rand_* deps.
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as f64 / (u32::MAX as f64 + 1.0)
    }
}

fn make_book(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut rng = Lcg::new(42);
    let forwards = vec![75.0_f64; n];
    let strikes: Vec<f64> = (0..n).map(|_| 50.0 + rng.next_f64() * 50.0).collect();
    let vols: Vec<f64> = (0..n).map(|_| 0.25 + rng.next_f64() * 0.40).collect();
    let times: Vec<f64> = (0..n).map(|_| (1.0 + rng.next_f64() * 89.0) / 365.0).collect();
    let is_calls: Vec<bool> = (0..n).map(|_| rng.next_f64() > 0.5).collect();
    let rates = vec![RATE; n];
    (forwards, strikes, rates, vols, times, is_calls)
}

fn best_of<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        f();
        let dur = t0.elapsed().as_secs_f64();
        if dur < best {
            best = dur;
        }
    }
    best
}

fn main() {
    let mut batch_ns: Vec<f64> = Vec::new();

    for &n in SIZES {
        let (f, k, r, v, t, cp) = make_book(n);
        let reps = if n <= 1_000 {
            10
        } else if n <= 100_000 {
            5
        } else {
            3
        };

        // Vectorized Rust API: arrays in, array out. Same kernel that powers
        // `black76_vectorized` on the Python side, called natively with no PyO3.
        let secs = best_of(
            || {
                let out = black76_price_batch_f64(&f, &k, &r, &v, &t, &cp);
                black_box(out);
            },
            reps,
        );
        batch_ns.push(secs / n as f64 * 1e9);
    }

    // Emit a single JSON line for easy parsing from the Python chart script.
    let pairs: Vec<String> = SIZES
        .iter()
        .zip(batch_ns.iter())
        .map(|(n, ns)| format!("\"{}\": {:.4}", n, ns))
        .collect();
    println!("{{{}}}", pairs.join(", "));
}
