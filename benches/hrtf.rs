//! Benchmarks for the time-domain `HrtfProcessor`. Each iteration processes one block (128 frames,
//! or 4410 for the `block4410_*` benches), so ns/iter is the per-block cost.
//!
//! To run (nightly only; the cfg keeps `cargo build --all-targets` working on stable):
//! `RUSTFLAGS="--cfg nightly_bench" cargo +nightly bench --bench hrtf`
//! Add `-C target-cpu=x86-64-v3` (or `native`) to RUSTFLAGS to enable AVX2/FMA, and
//! `--cfg hrtf_simd` to benchmark the `std::simd` kernel instead of the scalar loop.
//!
//! To compare with the old FFT implementation, run the same workload on the base commit: see the
//! instructions at the top of `baseline/hrtf_fft.rs`. Keep the two workloads in sync.
#![cfg(nightly_bench)]
#![feature(test)]
extern crate test;

use hrtf::{HrirSphere, HrtfProcessor, HrtfSphere, Vec3};
use test::{black_box, Bencher};

/// Block lengths: a Web Audio render quantum, and 0.1 s at 44.1 kHz.
const QUANTUM: usize = 128;
const LONG: usize = 4410;
const HRIR_BYTES: &[u8] = include_bytes!("IRC_1003_C.bin");

/// Fraction of blocks that carry sound in the "sparse" benches: 3 of every 8 blocks = 37.5%.
const SPARSE_PATTERN: [bool; 8] = [true, false, false, true, false, false, true, false];

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn noise_block(rng: &mut Lcg, len: usize) -> Vec<f32> {
    (0..len).map(|_| 0.5 * rng.next()).collect()
}

/// A source direction that sweeps around the head, changing every block.
fn direction(block: usize) -> Vec3 {
    let a = block as f32 * 0.07;
    Vec3::new(a.sin(), 0.4 * (a * 0.31).sin(), a.cos())
}

fn make_processor(rate: u32) -> HrtfProcessor {
    let sphere = HrirSphere::new(HRIR_BYTES, rate).unwrap();
    HrtfProcessor::new(HrtfSphere::new(&sphere))
}

/// `active(block_index)` says whether that block carries sound or silence.
fn run(b: &mut Bencher, rate: u32, len: usize, active: impl Fn(usize) -> bool) {
    let mut processor = make_processor(rate);
    let mut rng = Lcg(1);
    let loud: Vec<Vec<f32>> = (0..8).map(|_| noise_block(&mut rng, len)).collect();
    let silent = vec![0.0f32; len];
    let mut output = vec![(0.0f32, 0.0f32); len];
    let mut block = 0usize;

    // Warm up so the first (reset) block and the lazily-grown buffers are out of the picture.
    for _ in 0..16 {
        processor.process(&loud[0], &mut output, direction(block), 1.0);
        block += 1;
    }

    b.iter(|| {
        let source = if active(block) {
            &loud[block % 8]
        } else {
            &silent
        };
        output.fill((0.0, 0.0));
        processor.process(
            black_box(source),
            black_box(&mut output),
            direction(block),
            1.0,
        );
        block += 1;
        black_box(&output);
    });
}

fn continuous(_: usize) -> bool {
    true
}
fn sparse(block: usize) -> bool {
    SPARSE_PATTERN[block % 8]
}
fn silence(_: usize) -> bool {
    false
}

macro_rules! benches {
    ($($name:ident: $rate:expr, $len:expr, $active:expr;)*) => {
        $(
            #[bench]
            fn $name(b: &mut Bencher) {
                run(b, $rate, $len, $active);
            }
        )*
    };
}

benches! {
    block128_continuous_44100: 44100, QUANTUM, continuous;
    block128_continuous_48000: 48000, QUANTUM, continuous;
    block128_sparse_44100: 44100, QUANTUM, sparse;
    block128_sparse_48000: 48000, QUANTUM, sparse;
    block128_silence_44100: 44100, QUANTUM, silence;
    block128_silence_48000: 48000, QUANTUM, silence;
    block4410_continuous_44100: 44100, LONG, continuous;
    block4410_continuous_48000: 48000, LONG, continuous;
    block4410_sparse_44100: 44100, LONG, sparse;
    block4410_sparse_48000: 48000, LONG, sparse;
    block4410_silence_44100: 44100, LONG, silence;
    block4410_silence_48000: 48000, LONG, silence;
}
