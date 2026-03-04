# Design: `hf-fetch-model` Single-File Download API

**Status:** Implemented (in `hf-fetch-model`)
**Relates to:** Roadmap §7 Phase 3 item 1, §8 item 8

## Question

Should `hf-fetch-model` expose a single-file download function, and what should its API look like?

## Motivation

`hf-fetch-model`'s current public API operates at **repo granularity**: `download`, `download_with_config`, `download_files`, and their blocking variants all take a `repo_id` and download a set of files (all, or glob-filtered). There is no way to download a single named file and get its cache path back.

This is the right model for **model weights** — a 1B-parameter model has 2–4 safetensor shards totaling ~2–4 GB, and you need all of them before inference can begin. But several candle-mi use cases need **per-file, on-demand downloads**:

- **CLT weights (Phase 3).** A cross-layer transcoder repository contains 52+ safetensor files totaling 40+ GB (26 encoder files + 26 decoder files for a 26-layer model). Any given CLT operation (encoding at one layer, steering at another) touches only 2–4 files. Downloading everything upfront wastes bandwidth, time, and disk space. plip-rs solves this with a lazy pattern: `open()` downloads only `config.yaml` + `W_enc_0.safetensors` (~75 MB) for dimension detection, then `ensure_encoder_path(layer)` / `ensure_decoder_path(layer)` download individual files on first access via `hf_hub::api::sync::Api::repo().get()`.

- **SAE weights (Phase 3b).** Sparse autoencoder repositories (Gemma Scope, SAELens) have a similar structure: many per-layer weight files, operations touch a subset.

- **Config-only downloads.** Reading a model's `config.json` (~2 KB) to inspect architecture metadata should not require downloading multi-GB weight files.

- **Tokenizer-only downloads.** Fetching `tokenizer.json` (~2 MB) for tokenization-only workflows (e.g., `EncodingWithOffsets` analysis without model inference).

## Proposed API

Two new public functions in `hf-fetch-model`, mirroring the existing `download` / `download_blocking` pair:

```rust
/// Download a single file from a HuggingFace model repository.
///
/// Returns the local cache path. If the file is already cached (and
/// checksums match when `verify_checksums` is enabled), the download
/// is skipped and the cached path is returned immediately.
///
/// Files at or above `FetchConfig::chunk_threshold` (default 100 MiB)
/// are downloaded using multiple parallel HTTP Range connections
/// (`connections_per_file`, default 8). Smaller files use a single
/// connection.
///
/// # Arguments
///
/// * `repo_id` — Repository identifier (e.g., `"mntss/clt-gemma-2-2b-426k"`).
/// * `filename` — Exact filename within the repository (e.g., `"W_enc_5.safetensors"`).
/// * `config` — Shared configuration for auth, progress, checksums, retries, and chunking.
///
/// # Errors
///
/// Returns `FetchError::Http` if the file does not exist in the repository.
/// Returns `FetchError::Api` on download failure (after retries).
/// Returns `FetchError::Checksum` if verification is enabled and fails.
pub async fn download_file(
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<PathBuf, FetchError>

/// Blocking version of [`download_file()`] for non-async callers.
///
/// Creates a Tokio runtime internally. Do **not** call from within an
/// existing async context.
///
/// # Errors
///
/// Same as [`download_file()`].
pub fn download_file_blocking(
    repo_id: String,
    filename: &str,
    config: &FetchConfig,
) -> Result<PathBuf, FetchError>
```

No new types are needed. The functions reuse `FetchConfig` for all configuration (token, progress callback, checksum verification, retry policy, chunk threshold, connections per file). These library functions also serve as the backend for the new `download-file` CLI subcommand (see [CLI section](#cli-download-file-subcommand) below).

## Behavior specification

```text
download_file(repo_id, filename, config)
  │
  ├─ Check HF cache for repo_id/filename
  │   └─ If cached (and checksum valid if verify_checksums=true) → return cached path
  │
  ├─ Resolve download URL via HF API (respects config.revision)
  │
  ├─ If file size ≥ chunk_threshold (default 100 MiB):
  │   └─ Download with connections_per_file parallel HTTP Range requests
  │      (default 8 connections), reassemble chunks
  │
  ├─ If file size < chunk_threshold:
  │   └─ Download with single HTTP connection
  │
  ├─ On failure: retry up to config.max_retries (default 3)
  │   with exponential backoff + jitter (base 300ms, cap 10s)
  │
  ├─ If verify_checksums=true:
  │   └─ Verify SHA256 against HF LFS metadata
  │
  ├─ Store in standard HF cache layout (~/.cache/huggingface/hub/)
  │
  └─ Return local cache path
```

## File size analysis for CLT weights

The `chunk_threshold` default of 100 MiB naturally partitions CLT files into the right download strategy:

| File | Typical size | Chunked? | Notes |
|------|-------------|----------|-------|
| `config.yaml` | < 1 KB | No | Instant; single connection |
| `W_enc_{l}.safetensors` | ~75 MB | No | Below threshold; single connection is fast enough |
| `W_dec_{l}.safetensors` (late layers, l ≥ 20) | 75–150 MB | Borderline | Near threshold; marginal benefit |
| `W_dec_{l}.safetensors` (early layers, l ≤ 5) | 500 MB – 1.6 GB | **Yes** | 8 parallel connections cut download time by 4–6× |

Early-layer decoder files are the bottleneck: a single `W_dec_0.safetensors` at 1.6 GB takes ~30–60s on a typical connection. With 8 parallel Range connections, this drops to ~5–10s. Since CLT operations often need exactly one decoder file at a time, the per-file speedup translates directly to user-perceived latency.

## Consumer-side usage sketch

How candle-mi's future `interp/clt.rs` would use the new API, replacing direct `hf_hub::api::sync::Api` calls:

```rust
use hf_fetch_model::FetchConfig;
use std::path::PathBuf;

pub struct CrossLayerTranscoder {
    repo_id: String,
    fetch_config: FetchConfig,
    encoder_paths: Vec<Option<PathBuf>>,
    decoder_paths: Vec<Option<PathBuf>>,
    clt_config: CltConfig,
    // ...
}

impl CrossLayerTranscoder {
    pub fn open(clt_repo: &str) -> Result<Self> {
        let fetch_config = FetchConfig::builder()
            .on_progress(|e| {
                tracing::info!(
                    filename = %e.filename,
                    percent = e.percent,
                    "CLT download",
                );
            })
            .build()?;

        // List repo files to detect n_layers (no download).
        // list_repo_files_with_metadata is async; use a runtime since open() is sync.
        let rt = tokio::runtime::Runtime::new()?;
        let files = rt.block_on(
            hf_fetch_model::repo::list_repo_files_with_metadata(clt_repo, None, None)
        )?;
        let n_layers = files.iter()
            .filter(|f| f.filename.starts_with("W_enc_")
                      && f.filename.ends_with(".safetensors"))
            .count();

        // Download only config.yaml + W_enc_0 for dimension detection
        let _cfg_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(), "config.yaml", &fetch_config,
        )?;
        let enc0_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(), "W_enc_0.safetensors", &fetch_config,
        )?;

        // ... detect dimensions from enc0_path ...

        Ok(Self { repo_id: clt_repo.to_owned(), fetch_config, /* ... */ })
    }

    fn ensure_encoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(ref path) = self.encoder_paths[layer] {
            return Ok(path.clone());
        }
        let filename = format!("W_enc_{layer}.safetensors");
        let path = hf_fetch_model::download_file_blocking(
            self.repo_id.clone(), &filename, &self.fetch_config,
        )?;
        self.encoder_paths[layer] = Some(path.clone());
        Ok(path)
    }

    fn ensure_decoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(ref path) = self.decoder_paths[layer] {
            return Ok(path.clone());
        }
        let filename = format!("W_dec_{layer}.safetensors");
        // Large decoder files (up to 1.6 GB) automatically get
        // chunked parallel download via FetchConfig defaults
        let path = hf_fetch_model::download_file_blocking(
            self.repo_id.clone(), &filename, &self.fetch_config,
        )?;
        self.decoder_paths[layer] = Some(path.clone());
        Ok(path)
    }
}
```

## CLI: `download-file` subcommand

The `hf-fetch-model` CLI binary (gated behind the `cli` feature flag) should expose the single-file download capability as a new subcommand. The existing CLI uses clap derive macros (`#[derive(Parser)]`, `#[derive(Subcommand)]`, `#[derive(Args)]`) with kebab-case subcommand names (`list-families`, `search`, `status`, `discover`).

### Subcommand definition

A new `DownloadFile` variant in the `Commands` enum:

```rust
/// Download a single file from a HuggingFace repository.
DownloadFile {
    /// The repository identifier (e.g., "mntss/clt-gemma-2-2b-426k").
    repo_id: String,

    /// Exact filename within the repository (e.g., "W_dec_0.safetensors").
    filename: String,

    /// Git revision (branch, tag, or commit SHA).
    #[arg(long)]
    revision: Option<String>,

    /// Authentication token (or set `HF_TOKEN` env var).
    #[arg(long)]
    token: Option<String>,

    /// Output directory (default: HF cache).
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Minimum file size (MiB) for parallel chunked download.
    #[arg(long, default_value = "100")]
    chunk_threshold_mib: u64,

    /// Number of parallel HTTP connections per large file.
    #[arg(long, default_value = "8")]
    connections_per_file: usize,
},
```

Arguments follow existing CLI conventions: `repo_id` is **positional** (matching the default download command and `status` subcommand), `filename` is a second positional argument, and optional flags use `#[arg(long)]` with kebab-case names (`--revision`, `--token`, `--output-dir`, `--chunk-threshold-mib`, `--connections-per-file`). The `--concurrency`, `--filter`, `--exclude`, and `--preset` flags are omitted — they are repo-level concepts that don't apply to single-file downloads.

### Usage example

```
hf-fetch-model download-file mntss/clt-gemma-2-2b-426k W_dec_0.safetensors
```

With optional flags:

```
hf-fetch-model download-file mntss/clt-gemma-2-2b-426k W_dec_0.safetensors \
    --revision main --chunk-threshold-mib 50
```

Or using the short alias:

```
hf-fm download-file mntss/clt-gemma-2-2b-426k config.yaml
```

### Expected output

On success, prints the local cache path to stdout, consistent with the existing download command's `"Downloaded to: {path}"` format:

```
Downloaded to: C:\Users\user\.cache\huggingface\hub\models--mntss--clt-gemma-2-2b-426k\snapshots\abc123\W_dec_0.safetensors
```

If the file is already cached, the output is identical (the download is silently skipped). Progress bars (via `IndicatifProgress`) are shown on stderr during download, matching existing CLI behavior.

On error, prints `"error: {message}"` to stderr and exits with a non-zero exit code, consistent with the existing error handling in `main()`.

### Implementation

The `run` function gains a new match arm that calls `download_file_blocking` internally:

```rust
Some(Commands::DownloadFile { repo_id, filename, revision, token, output_dir,
                              chunk_threshold_mib, connections_per_file })
    => run_download_file(repo_id.as_str(), filename.as_str(),
                         revision.as_deref(), token.as_deref(),
                         output_dir, chunk_threshold_mib, connections_per_file),
```

The `run_download_file` function constructs a `FetchConfig` from the CLI arguments (same pattern as `run_download`), sets up `IndicatifProgress`, and calls `hf_fetch_model::download_file_blocking`. This reuses the same `FetchConfig` construction logic as the existing CLI subcommands — no new configuration types are needed.

## Advantages over direct `hf_hub::Api::repo().get()`

plip-rs's CLT code calls `hf_hub::api::sync::Api::repo().get(filename)` directly. Routing through `hf-fetch-model::download_file` instead provides:

| Advantage | Detail |
|-----------|--------|
| **Faster large-file downloads** | Files ≥ 100 MiB get chunked parallel transfer (8 HTTP Range connections). `hf_hub::Api::get()` uses a single connection. |
| **Unified progress reporting** | The `on_progress` callback works identically for model downloads and CLT file downloads, giving users a consistent experience. |
| **Checksum verification** | `hf-fetch-model` verifies SHA256 against HF LFS metadata by default. `hf_hub::Api::get()` does not. |
| **Retry with backoff** | Built-in exponential backoff (3 retries, 300ms base, 10s cap). `hf_hub::Api::get()` has no retry logic. |
| **Cleaner dependency graph** | candle-mi depends only on `hf-fetch-model`, which wraps `hf-hub` internally. No direct `hf_hub` import needed in candle-mi source. |

## Other consumers

Beyond CLT weights, several other candle-mi use cases benefit from single-file download:

| Use case | Current approach | Benefit |
|----------|-----------------|---------|
| **SAE weights (Phase 3b)** | Not yet implemented | Same lazy per-layer pattern as CLT; avoids downloading full SAE repo upfront |
| **Config-only download** | `download_files_blocking()` downloads all files, then reads `config.json` | Download ~2 KB instead of multi-GB weight files for architecture inspection |
| **Tokenizer-only download** | Same as above | Download `tokenizer.json` (~2 MB) for tokenization-only workflows |
| **plip-rs forward files** | Each `PlipXxx::load()` calls `hf_hub::Api::repo().get()` for individual files | Gains chunked transfer for large safetensors + checksum verification + retry |
| **RWKV World tokenizer** | `hf_hub::Api::repo().get("rwkv_vocab_v20230424.txt")` | Unified API; no direct `hf_hub` dependency |
| **CLI users** | No single-file download command exists | `hf-fm download-file repo file` for scripting and manual downloads |

## Implementation notes

The implementation is straightforward: `download_file` is essentially the existing download pipeline applied to a file list of length 1. The same cache-check → URL resolution → chunked/single download → checksum verification → cache storage flow applies. Estimated scope: ~50–100 lines of new code, plus tests.

## Open questions

- Should `download_file` accept `&str` or `String` for `repo_id`? The existing repo-level functions take `String` (owned) because they spawn async tasks. A single-file download might not need ownership — but consistency with the existing API argues for `String`.
- Should there be a `download_files_list` variant that takes `&[&str]` (a specific list of filenames) and downloads them concurrently? This would serve the CLT `open()` case where we know upfront we need both `config.yaml` and `W_enc_0.safetensors`.
- Should `FetchError` gain a `FileNotFound` variant, or should a missing file map to the existing `Http` variant?
