// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parser for `mntss/gemma-scope-transcoders/config.yaml`.
//!
//! `GemmaScope` transcoder weights are split across two `HuggingFace` repos:
//! `mntss/gemma-scope-transcoders` holds only a `config.yaml` curating the
//! lowest-`L0` variant per layer, while the actual `.npz` weights live in
//! `google/gemma-scope-2b-pt-transcoders`. This module parses the curation
//! `YAML` so the loader in [`super`] can fetch each `.npz` from the Google
//! repo on demand.
//!
//! The `YAML` schema is intentionally minimal — a `transcoders:` list of
//! `hf://<repo>/<path>` URLs — so we hand-roll the parser rather than pull
//! in `serde_yaml`. The format reference lives at
//! `anamnesis/docs/formats/gemmascope.md`.
//!
//! # Encoder injection point
//!
//! `GemmaScope`'s `config.yaml` declares `feature_input_hook: "ln2.hook_normalized"`,
//! i.e. the encoder reads from the post-`LN2` (pre-MLP) normalized residual
//! stream. In candle-mi this corresponds to
//! [`HookPoint::MlpPre`](crate::HookPoint::MlpPre) — verified at
//! `src/transformer/mod.rs` where `MlpPre` is captured immediately after
//! `layer.mid_norm.forward(...)` and before the MLP. **No new `HookPoint`
//! variant is required for the v0.1.10 `GemmaScope` arm.** This differs from
//! the `CltSplit` schema, which reads from `hook_resid_mid`
//! ([`HookPoint::ResidMid`](crate::HookPoint::ResidMid)) — the residual
//! before `LN2`.
//!
//! The output side (`feature_output_hook: "hook_mlp_out"`) maps to
//! [`HookPoint::MlpOut`](crate::HookPoint::MlpOut) and is exercised only by
//! the Phase B intervention path in `clt_vs_plt_planning_site`.

use crate::error::{MIError, Result};

/// `HuggingFace` repository holding the actual `GemmaScope` weight `.npz`
/// files that `mntss/gemma-scope-transcoders/config.yaml` redirects to.
///
/// All entries in the curation `YAML` must point at this repo; the parser
/// rejects any divergent entry. If a future `mntss/*` config ever points
/// at a different `google/*` repo (e.g. `gemma-scope-9b-pt-transcoders`),
/// promote this constant to a `CltConfig` field — see the format reference
/// at `anamnesis/docs/formats/gemmascope.md` for the wider naming scheme.
pub(crate) const GEMMASCOPE_WEIGHTS_REPO: &str = "google/gemma-scope-2b-pt-transcoders";

/// Prefix that `transcoders:` entries carry in the `mntss` config — `hf://`.
const HF_URL_PREFIX: &str = "hf://";

/// Parse the `transcoders:` list from a `GemmaScope` curation `YAML`.
///
/// Returns the per-layer `.npz` paths relative to the
/// `google/gemma-scope-2b-pt-transcoders` weights repo (the same value
/// held by the crate-private `GEMMASCOPE_WEIGHTS_REPO` constant).
///
/// The expected `YAML` schema is:
///
/// ```yaml
/// transcoders:
///   - "hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz"
///   - "hf://google/gemma-scope-2b-pt-transcoders/layer_1/width_16k/average_l0_65/params.npz"
///   ...
/// ```
///
/// The parser is intentionally minimal — it only understands the subset of
/// `YAML` actually used by this single curation file. List entries may be
/// quoted with single or double quotes (or unquoted). Comment lines (`#…`)
/// and blank lines are skipped. The `transcoders:` block ends at the first
/// non-indented, non-comment, non-blank line.
///
/// # Errors
///
/// Returns [`MIError::Config`] if no `transcoders:` key is found, if the
/// list is empty, if a non-list line appears inside the block, if any
/// entry is missing the `hf://` prefix, or if any entry points at a repo
/// other than the curated `GemmaScope` weights repo.
pub fn parse_gemmascope_config(yaml: &str) -> Result<Vec<String>> {
    let mut npz_paths: Vec<String> = Vec::new();
    let mut in_transcoders_block = false;
    let expected_repo_prefix = format!("{GEMMASCOPE_WEIGHTS_REPO}/");

    for raw_line in yaml.lines() {
        let line = raw_line.trim_end();
        let stripped = line.trim_start();

        // Blank lines and YAML comments are ignored everywhere.
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }

        if !in_transcoders_block {
            if line.trim() == "transcoders:" {
                in_transcoders_block = true;
            }
            continue;
        }

        // Inside the `transcoders:` block. Any non-indented line ends the
        // block (the next top-level key has appeared).
        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }

        let after_dash = stripped.strip_prefix("- ").ok_or_else(|| {
            MIError::Config(format!(
                "unexpected non-list line inside transcoders block: {line}"
            ))
        })?;

        // Strip optional surrounding quotes (single or double).
        let unquoted = after_dash
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| {
                after_dash
                    .strip_prefix('\'')
                    .and_then(|s| s.strip_suffix('\''))
            })
            .unwrap_or(after_dash);

        let no_scheme = unquoted.strip_prefix(HF_URL_PREFIX).ok_or_else(|| {
            MIError::Config(format!(
                "transcoders entry missing '{HF_URL_PREFIX}' prefix: {unquoted}"
            ))
        })?;

        let relpath = no_scheme
            .strip_prefix(&expected_repo_prefix)
            .ok_or_else(|| {
                MIError::Config(format!(
                    "transcoders entry points at unexpected repo \
                 (expected {GEMMASCOPE_WEIGHTS_REPO}): {no_scheme}"
                ))
            })?;

        // BORROW: explicit .to_owned() — the parser hands the caller an owned Vec
        npz_paths.push(relpath.to_owned());
    }

    if !in_transcoders_block {
        return Err(MIError::Config(
            "no 'transcoders:' key found in gemmascope config.yaml".into(),
        ));
    }
    if npz_paths.is_empty() {
        return Err(MIError::Config(
            "'transcoders:' list is empty in gemmascope config.yaml".into(),
        ));
    }
    Ok(npz_paths)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{GEMMASCOPE_WEIGHTS_REPO, parse_gemmascope_config};
    use crate::error::MIError;

    const SAMPLE_YAML: &str = r#"# Transcoder Configuration Gemma Scope Transcoders (lowest L0)
model_name: "google/gemma-2-2b"
model_kind: "transcoder_set"
feature_input_hook: "ln2.hook_normalized"
feature_output_hook: 'hook_mlp_out'

transcoders:
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz"
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_1/width_16k/average_l0_65/params.npz"
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_2/width_16k/average_l0_49/params.npz"
"#;

    #[test]
    fn parses_well_formed_yaml() {
        let paths = parse_gemmascope_config(SAMPLE_YAML).unwrap();
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], "layer_0/width_16k/average_l0_76/params.npz");
        assert_eq!(paths[1], "layer_1/width_16k/average_l0_65/params.npz");
        assert_eq!(paths[2], "layer_2/width_16k/average_l0_49/params.npz");
    }

    #[test]
    fn weights_repo_constant_is_stable() {
        // If this constant ever needs to vary per config file, the parser
        // signature must change. Pin its current value here so a refactor
        // does not silently route weights to a different repo.
        assert_eq!(
            GEMMASCOPE_WEIGHTS_REPO,
            "google/gemma-scope-2b-pt-transcoders"
        );
    }

    #[test]
    fn parses_unquoted_entries() {
        let yaml = "transcoders:\n  - hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz\n";
        let paths = parse_gemmascope_config(yaml).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "layer_0/width_16k/average_l0_76/params.npz");
    }

    #[test]
    fn parses_single_quoted_entries() {
        let yaml = "transcoders:\n  - 'hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz'\n";
        let paths = parse_gemmascope_config(yaml).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "layer_0/width_16k/average_l0_76/params.npz");
    }

    #[test]
    fn rejects_missing_transcoders_key() {
        let yaml = "model_name: foo\nmodel_kind: bar\n";
        let err = parse_gemmascope_config(yaml).unwrap_err();
        assert!(
            matches!(&err, MIError::Config(msg) if msg.contains("no 'transcoders:'")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_empty_transcoders_list() {
        let yaml = "transcoders:\nmodel_name: foo\n";
        let err = parse_gemmascope_config(yaml).unwrap_err();
        assert!(
            matches!(&err, MIError::Config(msg) if msg.contains("empty")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_missing_hf_prefix() {
        let yaml = "transcoders:\n  - \"google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz\"\n";
        let err = parse_gemmascope_config(yaml).unwrap_err();
        assert!(
            matches!(&err, MIError::Config(msg) if msg.contains("hf://")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_wrong_repo() {
        let yaml = "transcoders:\n  - \"hf://google/some-other-repo/layer_0/width_16k/average_l0_76/params.npz\"\n";
        let err = parse_gemmascope_config(yaml).unwrap_err();
        assert!(
            matches!(&err, MIError::Config(msg) if msg.contains("unexpected repo")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let yaml = "# A leading comment\n\ntranscoders:\n  # comment inside the block\n  - \"hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz\"\n\n  - \"hf://google/gemma-scope-2b-pt-transcoders/layer_1/width_16k/average_l0_65/params.npz\"\n";
        let paths = parse_gemmascope_config(yaml).unwrap();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn stops_at_next_top_level_key() {
        let yaml = "transcoders:\n  - \"hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz\"\nnext_key: foo\n  - \"hf://google/gemma-scope-2b-pt-transcoders/layer_999/width_16k/average_l0_99/params.npz\"\n";
        let paths = parse_gemmascope_config(yaml).unwrap();
        // Only the first entry is parsed; entries after `next_key:` are ignored.
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "layer_0/width_16k/average_l0_76/params.npz");
    }

    #[test]
    fn rejects_non_list_line_inside_block() {
        let yaml = "transcoders:\n  not_a_list_item: foo\n";
        let err = parse_gemmascope_config(yaml).unwrap_err();
        assert!(
            matches!(&err, MIError::Config(msg) if msg.contains("non-list line")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parses_real_curation_yaml_shape() {
        // 26 entries, mixed average_l0 values per layer (lowest-L0 curation).
        // Mirrors the actual mntss/gemma-scope-transcoders/config.yaml snapshot
        // taken on 2026-04-30 — see anamnesis/docs/formats/gemmascope.md.
        let yaml = include_str!("gemmascope_test_fixture.yaml");
        let paths = parse_gemmascope_config(yaml).unwrap();
        assert_eq!(paths.len(), 26);
        assert_eq!(paths[0], "layer_0/width_16k/average_l0_76/params.npz");
        assert_eq!(paths[11], "layer_11/width_16k/average_l0_5/params.npz");
        assert_eq!(paths[25], "layer_25/width_16k/average_l0_41/params.npz");
    }
}
