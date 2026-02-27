// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast model download with progress via tracing.
//!
//! ```bash
//! cargo run --example fast_download --features fast-download
//! ```
//!
//! **What it does:**
//!
//! 1. Initialises a tracing subscriber (prints structured logs to stderr).
//! 2. Downloads a model using [`candle_mi::download_model_blocking()`],
//!    which uses `hf-fetch-model` for high-throughput parallel downloads.
//! 3. Reports the cache path where files were stored.
//!
//! Pass a model ID as the first argument, or defaults to
//! `julien-c/dummy-unknown` (a tiny test repository).

fn main() {
    // 1. Initialise tracing subscriber so progress events are visible.
    tracing_subscriber::fmt::init();

    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "julien-c/dummy-unknown".to_string());

    eprintln!("Downloading model: {model_id}");
    eprintln!("(Files will be cached in ~/.cache/huggingface/hub/)\n");

    // 2. Download the model (blocking — this is a sync main).
    match candle_mi::download_model_blocking(model_id) {
        Ok(path) => {
            eprintln!("\nDownload complete. Cache path: {}", path.display());
        }
        Err(e) => {
            eprintln!("\nDownload failed: {e}");
            std::process::exit(1);
        }
    }
}
