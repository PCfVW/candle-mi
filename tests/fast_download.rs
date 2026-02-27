// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for the `fast-download` feature.
//!
//! Downloads a small public test repository (`julien-c/dummy-unknown`)
//! and verifies the returned path is valid.
//!
//! Run with:
//!   `cargo test --test fast_download --features fast-download`

#![cfg(feature = "fast-download")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_docs_in_private_items,
    missing_docs
)]

/// Async download of a small test repo returns a valid directory path.
#[tokio::test]
async fn download_small_public_model() {
    let result = candle_mi::download_model("julien-c/dummy-unknown".to_owned()).await;

    let path = result.unwrap_or_else(|e| panic!("download failed: {e}"));

    assert!(
        path.exists(),
        "cache directory should exist: {}",
        path.display()
    );
    assert!(
        path.is_dir(),
        "cache path should be a directory: {}",
        path.display()
    );
}

/// Blocking download variant also works.
#[test]
fn download_blocking_small_public_model() {
    let result = candle_mi::download_model_blocking("julien-c/dummy-unknown".to_owned());

    let path = result.unwrap_or_else(|e| panic!("blocking download failed: {e}"));

    assert!(
        path.exists(),
        "cache directory should exist: {}",
        path.display()
    );
    assert!(
        path.is_dir(),
        "cache path should be a directory: {}",
        path.display()
    );
}
