//! `std::simd` convolution kernel. Only compiled with `--cfg hrtf_simd` on a nightly compiler.
//!
//! The scalar processor *scatters* every input sample into the accumulator: for each sample it
//! loads the accumulator, three HRIRs, does three multiply-adds and stores the accumulator again.
//! That is five memory operations for three FMAs and it is limited by load/store ports.
//!
//! This kernel *gathers* instead. Consecutive input samples that fall in the same sphere triangle
//! use the same three HRIRs, so for such a "run" the contribution to output `n` is
//!
//! ```text
//! acc[n] += Σ_j Σ_v w[j][v] · hrir_v[n - j]
//! ```
//!
//! A tile of output vectors is kept in registers while looping over all samples `j` of the run,
//! so the accumulator is only touched once per tile and each FMA costs one (unaligned) load.

use std::simd::{Simd, StdFloat};

/// Extra zeros on both sides of every HRIR in the sphere's storage, in floats. It has to be at
/// least the number of floats in a tile (see [`accumulate`]).
pub(crate) const HRIR_PAD: usize = 8 * TILE_VECTORS;

/// How far past the logical end of the accumulator the kernel may read and write, in floats.
pub(crate) const ACC_OVERSHOOT: usize = 8 * TILE_VECTORS;

/// The number of vectors in a register tile.
const TILE_VECTORS: usize = 10;

/// How many samples, spaced a whole vector apart, share each loaded HRIR vector.
const BLOCK_GROUPS: usize = 2;

#[inline(always)]
fn fma<const LANES: usize, const FMA: bool>(
    a: Simd<f32, LANES>,
    b: Simd<f32, LANES>,
    c: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Whether fused operations are native on the compilation target.
#[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
const NATIVE_FMA: bool = cfg!(any(target_feature = "fma", target_arch = "aarch64"));

/// Adds the response of a run of consecutive input samples to `acc`.
///
/// * `rows` are the three HRIRs of the run's triangle, as stored in the sphere: `stride` floats
///   (interleaved left/right) with [`HRIR_PAD`] zeros before and after.
/// * `weights[j]` are the (already gain-scaled) weights of the three HRIRs for input sample
///   `first + j`.
/// * `acc` is the accumulator, which has two floats per output sample. Input sample `i` adds
///   the HRIR at `acc[2 * i..2 * i + stride]`.
///
/// `acc` must be at least `2 * (first + w.len() - 1) + stride + ACC_OVERSHOOT` long.
#[inline]
pub(crate) fn accumulate(
    acc: &mut [f32],
    rows: [&[f32]; 3],
    stride: usize,
    first: usize,
    weights: &[[f32; 3]],
) {
    // Ask for AVX2 + FMA at run time when the build was not already compiled for it.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: The CPU supports the features that were just checked.
            #[allow(unsafe_code)]
            unsafe {
                return accumulate_avx2_fma(acc, rows, stride, first, weights);
            }
        }
    }

    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
    accumulate_impl::<8, true>(acc, rows, stride, first, weights);
    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
    accumulate_impl::<4, NATIVE_FMA>(acc, rows, stride, first, weights);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
#[target_feature(enable = "avx2,fma")]
fn accumulate_avx2_fma(
    acc: &mut [f32],
    rows: [&[f32]; 3],
    stride: usize,
    first: usize,
    weights: &[[f32; 3]],
) {
    accumulate_impl::<8, true>(acc, rows, stride, first, weights);
}

/// `LANES` is the vector width in floats, and `FMA` whether to use fused multiply-add (which is
/// extremely slow when it is not a hardware instruction).
#[inline(always)]
fn accumulate_impl<const LANES: usize, const FMA: bool>(
    acc: &mut [f32],
    rows: [&[f32]; 3],
    stride: usize,
    first: usize,
    w: &[[f32; 3]],
) {
    type V<const L: usize> = Simd<f32, L>;
    let tile = LANES * TILE_VECTORS;
    assert!(tile <= HRIR_PAD && tile <= ACC_OVERSHOOT);
    if w.is_empty() {
        return;
    }
    let samples_per_vector = LANES / 2;
    let samples_per_block = samples_per_vector * BLOCK_GROUPS;

    // Output floats touched by this run, relative to the start of the accumulator.
    let start = 2 * first;
    let end = 2 * (first + w.len() - 1) + stride;
    assert!(acc.len() >= end + tile);
    for row in rows {
        assert!(row.len() == stride + 2 * HRIR_PAD);
    }

    let mut offset = 0;
    while start + offset < end {
        // Accumulator float `start + offset + k` pairs with HRIR float `offset + k - 2 * j` for
        // input sample `first + j`.
        let acc_tile = &mut acc[start + offset..][..tile];
        let mut sums = [V::<LANES>::splat(0.0); TILE_VECTORS];
        for (t, sum) in sums.iter_mut().enumerate() {
            *sum = V::<LANES>::from_slice(&acc_tile[t * LANES..]);
        }

        // Only samples whose HRIR window `[offset - 2j, offset - 2j + tile)` overlaps
        // `[0, stride)` contribute to this tile.
        let j_end = ((offset + tile).div_ceil(2)).min(w.len());
        let j_start = if offset >= stride {
            (offset - stride) / 2 + 1
        } else {
            0
        };

        let mut j = j_start;

        // Blocks of `SAMPLES_PER_BLOCK` samples. Sample `j + SAMPLES_PER_VECTOR * q + r` shifts
        // its HRIR by `2 * r` floats relative to sample `j + r`, plus a whole number of vectors
        // for `q`, so one loaded HRIR vector feeds up to `BLOCK_GROUPS` different tile vectors.
        // That needs far fewer loads per FMA than loading a vector for every product.
        while j + samples_per_block <= j_end {
            for r in 0..samples_per_vector {
                // Float index in the padded row that tile vector 0 reads for sample `j + r`.
                let origin = HRIR_PAD + offset - 2 * (j + r);
                for (v, row) in rows.iter().enumerate() {
                    let weights: [V<LANES>; BLOCK_GROUPS] = std::array::from_fn(|q| {
                        V::<LANES>::splat(w[j + samples_per_vector * q + r][v])
                    });
                    // Vector `e` of this window is read by tile vector `t` for group `q`
                    // when `e == t + BLOCK_GROUPS - 1 - q`.
                    let first = origin - (BLOCK_GROUPS - 1) * LANES;
                    let window = &row[first..first + (TILE_VECTORS + BLOCK_GROUPS - 1) * LANES];
                    for e in 0..TILE_VECTORS + BLOCK_GROUPS - 1 {
                        let h = V::<LANES>::from_slice(&window[e * LANES..]);
                        for (q, &weight) in weights.iter().enumerate() {
                            // `e == t + BLOCK_GROUPS - 1 - q`
                            if let Some(t) = (e + q).checked_sub(BLOCK_GROUPS - 1) {
                                if t < TILE_VECTORS {
                                    sums[t] = fma::<LANES, FMA>(h, weight, sums[t]);
                                }
                            }
                        }
                    }
                }
            }
            j += samples_per_block;
        }

        // Whatever is left (or all of it, for short runs).
        for (j, w) in w.iter().enumerate().take(j_end).skip(j) {
            if *w == [0.0; 3] {
                continue;
            }
            // Index into the padded row of the first float of the window. It is always in
            // `(PAD - tile, PAD + stride)` thanks to the bounds on `j` above.
            let base = HRIR_PAD + offset - 2 * j;
            for (row, &weight) in rows.iter().zip(w) {
                let window = &row[base..base + tile];
                let weight = V::<LANES>::splat(weight);
                for (t, sum) in sums.iter_mut().enumerate() {
                    let h = V::<LANES>::from_slice(&window[t * LANES..]);
                    *sum = fma::<LANES, FMA>(h, weight, *sum);
                }
            }
        }

        for (t, sum) in sums.iter().enumerate() {
            sum.copy_to_slice(&mut acc_tile[t * LANES..][..LANES]);
        }
        offset += tile;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// Compares the kernel against the definition, for many run lengths and positions.
    fn check<const LANES: usize, const FMA: bool>() {
        let stride = 2 * 37; // not a multiple of any tile size
        let mut rng = 12345;
        let rows: Vec<Vec<f32>> = (0..3)
            .map(|_| {
                let mut row = vec![0.0; stride + 2 * HRIR_PAD];
                for x in &mut row[HRIR_PAD..HRIR_PAD + stride] {
                    *x = lcg(&mut rng);
                }
                row
            })
            .collect();

        for len in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 127, 128] {
            for first in [0, 1, 5, 128 - len].map(|f| f.min(128 - len)) {
                let weights: Vec<[f32; 3]> = (0..len)
                    .map(|j| {
                        if j % 11 == 5 {
                            [0.0; 3]
                        } else {
                            [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)]
                        }
                    })
                    .collect();

                let acc_len = 2 * (stride / 2 + 128 - 1) + ACC_OVERSHOOT;
                let mut expected: Vec<f32> = (0..acc_len).map(|_| lcg(&mut rng)).collect();
                let mut actual = expected.clone();
                for j in 0..len {
                    for v in 0..3 {
                        for k in 0..stride {
                            expected[2 * (first + j) + k] += weights[j][v] * rows[v][HRIR_PAD + k];
                        }
                    }
                }

                accumulate_impl::<LANES, FMA>(
                    &mut actual,
                    [&rows[0], &rows[1], &rows[2]],
                    stride,
                    first,
                    &weights,
                );
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-4 * (1.0 + e.abs()),
                        "len {} first {} index {}: {} != {}",
                        len,
                        first,
                        i,
                        a,
                        e
                    );
                }
            }
        }
    }

    #[test]
    fn four_lanes_match_definition() {
        check::<4, false>();
    }

    #[test]
    fn eight_lanes_match_definition() {
        check::<8, false>();
    }

    #[test]
    fn selected_kernel_matches_definition() {
        // Whatever `accumulate` picks on this machine.
        let stride = 2 * 37;
        let mut row = vec![0.0; stride + 2 * HRIR_PAD];
        row[HRIR_PAD..HRIR_PAD + stride].fill(1.0);
        let mut acc = vec![0.0; 2 * (stride / 2 + 128) + ACC_OVERSHOOT];
        let w = vec![[1.0; 3]; 100];
        accumulate(&mut acc, [&row, &row, &row], stride, 3, &w);
        // Output 3 only gets sample 3's first tap, output 40 gets samples 4 through 40.
        assert_eq!(acc[2 * 3], 3.0);
        assert_eq!(acc[2 * 3 + 1], 3.0);
        assert_eq!(acc[2 * 40], 3.0 * 37.0);
    }
}
