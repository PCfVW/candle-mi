// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build-hygiene guard (test-only): every `tests/*.rs` and `examples/*.rs`
//! file must be registered as a `[[test]]` / `[[example]]` target in
//! `Cargo.toml`, so that feature-gated targets carry an explicit
//! `required-features`.
//!
//! Cargo auto-discovers any file under `tests/` or `examples/` as a target.
//! An unregistered file has no `required-features`, so it is compiled under
//! *every* feature lane and breaks any lane whose features it does not satisfy
//! (e.g. a `transformer`-only test failing `cargo test --features rwkv` with
//! `E0432: unresolved import`). These tests turn that drift into a failure on
//! the first push.
//!
//! Scope: top-level `*.rs` only. Cargo subdirectory examples
//! (`examples/foo/main.rs`) and `tests/common/mod.rs`-style shared helpers are
//! not modeled — none exist today (counts are 1:1). If one is added later the
//! guard flags it, forcing a conscious `Cargo.toml` decision.

use std::collections::BTreeSet;
use std::path::Path;

/// Collect the `name = "..."` values of every `section` (`"[[test]]"` or
/// `"[[example]]"`) target declared in a `Cargo.toml` string.
///
/// Plain line-scan (no `toml` dependency): tracks the current `[[...]]` header
/// and captures the first `name` key inside a matching section.
fn declared_names(cargo_toml: &str, section: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_section = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[[") {
            in_section = trimmed == section;
            continue;
        }
        if in_section {
            if let Some((key, value)) = trimmed.split_once('=') {
                if key.trim() == "name" {
                    // BORROW: own the parsed name slice for set storage
                    names.insert(value.trim().trim_matches('"').to_owned());
                }
            }
        }
    }
    names
}

/// File stems of the top-level `*.rs` files directly under `dir`.
fn rs_stems(dir: &Path) -> BTreeSet<String> {
    let mut stems = BTreeSet::new();
    for entry in std::fs::read_dir(dir)
        .expect("read target directory")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                // BORROW: own the file stem for set storage
                stems.insert(stem.to_owned());
            }
        }
    }
    stems
}

/// Compute registration drift between files on disk and declared targets.
///
/// Returns `(unregistered, orphan)`: `unregistered` are files present with no
/// declared target (`files − declared`); `orphan` are declared targets with no
/// matching file (`declared − files`).  Pure, so it can be exercised on
/// synthetic input.
fn drift(files: &BTreeSet<String>, declared: &BTreeSet<String>) -> (Vec<String>, Vec<String>) {
    let unregistered = files.difference(declared).cloned().collect();
    let orphan = declared.difference(files).cloned().collect();
    (unregistered, orphan)
}

/// Read the crate's `Cargo.toml` (located via `CARGO_MANIFEST_DIR`).
fn cargo_toml() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("read Cargo.toml")
}

#[test]
fn every_test_file_is_registered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (unregistered, orphan) = drift(
        &rs_stems(&root.join("tests")),
        &declared_names(&cargo_toml(), "[[test]]"),
    );
    assert!(
        unregistered.is_empty() && orphan.is_empty(),
        "tests/ registration drift:\n  \
         unregistered files (add a [[test]] entry with required-features): {unregistered:?}\n  \
         orphan [[test]] entries (no matching tests/*.rs file): {orphan:?}"
    );
}

#[test]
fn every_example_file_is_registered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (unregistered, orphan) = drift(
        &rs_stems(&root.join("examples")),
        &declared_names(&cargo_toml(), "[[example]]"),
    );
    assert!(
        unregistered.is_empty() && orphan.is_empty(),
        "examples/ registration drift:\n  \
         unregistered files (add an [[example]] entry with required-features): {unregistered:?}\n  \
         orphan [[example]] entries (no matching examples/*.rs file): {orphan:?}"
    );
}

#[test]
fn drift_detects_both_directions() {
    // Synthetic proof that the guard bites without perturbing real files.
    let files: BTreeSet<String> = ["a", "b", "new_unregistered"]
        .iter()
        .copied()
        .map(String::from)
        .collect();
    let declared: BTreeSet<String> = ["a", "b", "orphan_entry"]
        .iter()
        .copied()
        .map(String::from)
        .collect();
    let (unregistered, orphan) = drift(&files, &declared);
    assert_eq!(unregistered, vec![String::from("new_unregistered")]);
    assert_eq!(orphan, vec![String::from("orphan_entry")]);
}
