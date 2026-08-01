// SPDX-License-Identifier: MIT OR Apache-2.0

//! The crate's single construction point for a seeded, algorithm-frozen generator.
//!
//! Weight generation promises that a model is reproducible from `(config, seed)`
//! alone.  `rand`'s `StdRng` cannot back that promise: `rand` 0.8 documents it as
//! "deterministic but should not be considered reproducible due to dependence on
//! configuration and possible replacement in future library versions", and names
//! `rand_chacha` as the remedy.  The type has already changed implementation once
//! (`HC-128` to `ChaCha12`), so the hazard is not hypothetical.
//!
//! `ChaCha8` is a frozen specification, so `rand_chacha::ChaCha8Rng` does back it.
//! That would leave one unpinned link: `SeedableRng::seed_from_u64` is a
//! `rand_core` convenience carrying no stability guarantee of its own.  So the
//! 32-byte key is derived here instead, by `SplitMix64`, and handed to
//! `ChaCha8Rng::from_seed`.  Every step from `u64` seed to weight bytes is then
//! frozen inside this crate, and no dependency bump can move it.

use rand_chacha::ChaCha8Rng;

/// Expand a `u64` seed into the 32-byte key `ChaCha8Rng::from_seed` takes, using
/// `SplitMix64` (Steele, Lea & Flood 2014), the standard `u64` seed expander.
///
/// Four draws of eight little-endian bytes each.  Verified against the published
/// reference vector: the first draw for seed `0` is `0xE220_A839_7B1D_CDAF`, which
/// the `splitmix64_seed_matches_the_reference_vector` test pins.
fn splitmix64_seed(seed: u64) -> [u8; 32] {
    let mut state = seed;
    let mut key = [0_u8; 32];
    // EXPLICIT: `SplitMix64` is a stateful recurrence -- each draw advances
    // `state` for the next one -- so an iterator chain would hide the update.
    // `chunks_exact_mut` keeps this free of direct indexing.
    for chunk in key.chunks_exact_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    key
}

/// The crate's seeded generator: `ChaCha8`, keyed by the `SplitMix64` expansion
/// of `seed`.
///
/// Use this for anything whose output is promised to be reproducible from a seed
/// (weight initialization, the dead-salmon controls).  Two calls with the same
/// `seed` yield the same stream, on any platform, under any future `rand`
/// release.
#[must_use]
pub fn seeded(seed: u64) -> ChaCha8Rng {
    use rand::SeedableRng;

    ChaCha8Rng::from_seed(splitmix64_seed(seed))
}

#[cfg(test)]
mod tests {
    use super::{seeded, splitmix64_seed};
    use rand::Rng;

    /// The whole point of this module is that the derivation never moves, so pin
    /// it against the published `SplitMix64` vector rather than against whatever
    /// the implementation happens to produce.  The first draw for seed `0` is
    /// `0xE220_A839_7B1D_CDAF`; these are its little-endian bytes.
    #[test]
    fn splitmix64_seed_matches_the_reference_vector() {
        let key = splitmix64_seed(0);
        // INDEX: `key` is `[u8; 32]` by return type, so the first 8 bytes exist.
        assert_eq!(
            &key[..8],
            &[0xAF, 0xCD, 0x1D, 0x7B, 0x39, 0xA8, 0x20, 0xE2],
            "SplitMix64 seed derivation changed; this breaks every seeded model"
        );
    }

    /// A frozen generator must reproduce its stream from the seed alone, and
    /// distinct seeds must not collide.
    #[test]
    fn seeded_streams_are_reproducible_and_seed_dependent() {
        let draw = |seed: u64| {
            let mut rng = seeded(seed);
            (0..16).map(|_| rng.r#gen::<u64>()).collect::<Vec<u64>>()
        };

        assert_eq!(draw(0), draw(0), "same seed must reproduce the same stream");
        assert_ne!(draw(0), draw(1), "different seeds must differ");
    }
}
