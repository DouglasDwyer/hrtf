//! Baseline: the same workload as `../hrtf.rs`, written against the old FFT-based `HrtfProcessor`
//! API. It does not compile against the current sources, so run it on the commit before the
//! time-domain rewrite (the merge base with `origin/main`):
//!
//! ```text
//! git worktree add --detach ../hrtf-base $(git merge-base HEAD origin/main)
//! mkdir ../hrtf-base/benches
//! cp benches/baseline/hrtf_fft.rs ../hrtf-base/benches/hrtf.rs
//! cp benches/IRC_1003_C.bin ../hrtf-base/benches/
//! sed -i 's|"../IRC_1003_C.bin"|"IRC_1003_C.bin"|' ../hrtf-base/benches/hrtf.rs
//! cd ../hrtf-base && cargo +nightly bench --bench hrtf
//! ```
//!
//! Use the same RUSTFLAGS (e.g. `-C target-cpu=x86-64-v3`) as for the new benchmark.
#![feature(test)]
extern crate test;

use hrtf::{HrirSphere, HrtfContext, HrtfProcessor, Vec3};
use test::{black_box, Bencher};

/// Block lengths: a Web Audio render quantum, and 0.1 s at 44.1 kHz.
const QUANTUM: usize = 128;
const LONG: usize = 4410;
const HRIR_BYTES: &[u8] = include_bytes!("../IRC_1003_C.bin");

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

fn direction(block: usize) -> Vec3 {
    let a = block as f32 * 0.07;
    Vec3::new(a.sin(), 0.4 * (a * 0.31).sin(), a.cos())
}

fn run(b: &mut Bencher, rate: u32, len: usize, active: impl Fn(usize) -> bool) {
    let sphere = HrirSphere::new(HRIR_BYTES, rate).unwrap();
    let mut processor = HrtfProcessor::new(sphere, 1, len);
    let mut rng = Lcg(1);
    let loud: Vec<Vec<f32>> = (0..8).map(|_| noise_block(&mut rng, len)).collect();
    let silent = vec![0.0f32; len];
    let mut output = vec![(0.0f32, 0.0f32); len];
    let mut prev_left = vec![];
    let mut prev_right = vec![];
    let mut block = 0usize;

    let mut step = |source: &[f32],
                    output: &mut [(f32, f32)],
                    block: usize,
                    prev_left: &mut Vec<f32>,
                    prev_right: &mut Vec<f32>| {
        processor.process_samples(HrtfContext {
            source,
            output,
            new_sample_vector: direction(block),
            prev_sample_vector: direction(block.saturating_sub(1)),
            prev_left_samples: prev_left,
            prev_right_samples: prev_right,
            new_distance_gain: 1.0,
            prev_distance_gain: 1.0,
        });
    };

    for _ in 0..16 {
        step(
            &loud[0],
            &mut output,
            block,
            &mut prev_left,
            &mut prev_right,
        );
        block += 1;
    }

    b.iter(|| {
        let source = if active(block) {
            &loud[block % 8]
        } else {
            &silent
        };
        output.fill((0.0, 0.0));
        step(
            black_box(source),
            black_box(&mut output),
            block,
            &mut prev_left,
            &mut prev_right,
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
