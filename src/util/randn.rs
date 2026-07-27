// SPDX-License-Identifier: MIT OR Apache-2.0

//! Seeded Gaussian sampling for reproducible weight generation.
//!
//! Used by the random-model dead-salmon control
//! (`MIModel::from_pretrained_random_init`, `transformer` feature) and by the
//! from-scratch `OthelloGpt::init` recipe (`diffusion` feature).  Lives here so
//! both features share one deterministic sampler: a model is reproducible from
//! `(config, seed)` alone, independent of the device RNG.

/// `n` seeded `N(0, std)` samples via Box-Muller (deterministic given `rng`).
// The `f64 -> f32` sample narrowings below store each standard-normal at model
// (`F32`) precision; the lost mantissa bits are irrelevant to a random control
// or a from-scratch initialization.
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
pub fn randn_f32(rng: &mut rand::rngs::StdRng, n: usize, std: f64) -> Vec<f32> {
    use rand::Rng;
    let mut v: Vec<f32> = Vec::with_capacity(n);
    while v.len() < n {
        let u1 = rng.r#gen::<f64>().max(1e-12);
        let u2 = rng.r#gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = 2.0 * std::f64::consts::PI * u2;
        // CAST: f64 → f32, standard-normal sample stored at model precision.
        v.push((r * ang.cos() * std) as f32);
        if v.len() < n {
            // CAST: f64 → f32, as above (Box-Muller's second sample).
            v.push((r * ang.sin() * std) as f32);
        }
    }
    v.truncate(n);
    v
}

#[cfg(test)]
mod tests {
    // The seeded weights must be reproducible from the seed and correctly
    // scaled (moved here from `backend.rs` alongside `randn_f32` itself).
    // CAST: len -> f32 for a sample-statistic check; test-only, 4096 is exact.
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    #[test]
    fn randn_f32_is_seed_deterministic_and_scaled() {
        use rand::SeedableRng;

        let draw = |seed: u64, std: f64| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            super::randn_f32(&mut rng, 4096, std)
        };

        let a = draw(7, 0.02);
        let b = draw(7, 0.02);
        let c = draw(8, 0.02);
        assert_eq!(a.len(), 4096);
        assert_eq!(a, b, "same seed must reproduce the same weights");
        assert_ne!(a, c, "different seeds must differ");

        // Sample standard deviation should sit near the requested 0.02.
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / a.len() as f32;
        let sd = var.sqrt();
        assert!((0.015..0.025).contains(&sd), "std {sd} not near 0.02");
    }
}
