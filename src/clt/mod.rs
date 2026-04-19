// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-Layer Transcoder (CLT) support.
//!
//! Loads pre-trained CLT weights from `HuggingFace` (circuit-tracer format),
//! encodes residual stream activations into sparse feature activations,
//! injects decoder vectors into the residual stream for steering, and
//! scores features by decoder projection for attribution graph construction.
//!
//! Memory-efficient: uses stream-and-free for encoders (~75 MB/layer on GPU)
//! and a micro-cache for steering vectors (~450 KB for 50 features).
//! Decoder scoring operates entirely on CPU (one file at a time, up to ~2 GB).
//!
//! # CLT Architecture
//!
//! A cross-layer transcoder at layer `l` implements:
//! ```text
//! Encode:  features = ReLU(W_enc[l] @ residual_mid[l] + b_enc[l])
//! Decode:  For each downstream layer l' >= l:
//!            mlp_out_hat[l'] += W_dec[l, l'] @ features + b_dec[l']
//! Inject:  residual[pos] += strength × W_dec[l, target_layer, feature_idx, :]
//! ```
//!
//! # Weight File Layout (circuit-tracer format)
//!
//! Each encoder file `W_enc_{l}.safetensors` contains:
//! - `W_enc_{l}`: shape `[n_features, d_model]` (BF16) — encoder weight matrix
//! - `b_enc_{l}`: shape `[n_features]` (BF16) — encoder bias
//! - `b_dec_{l}`: shape `[d_model]` (BF16) — decoder bias for target layer l
//!
//! Each decoder file `W_dec_{l}.safetensors` contains:
//! - `W_dec_{l}`: shape `[n_features, n_target_layers, d_model]` (BF16)
//!   where `n_target_layers = n_layers - l` (layer l writes to layers l..n_layers-1)

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use safetensors::tensor::SafeTensors;
use tracing::info;

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Identifies a single CLT feature by its source layer and index within that layer.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CltFeatureId {
    /// Source layer where this feature's encoder lives (`0..n_layers`).
    pub layer: usize,
    /// Feature index within the layer (`0..n_features_per_layer`).
    pub index: usize,
}

impl std::fmt::Display for CltFeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}:{}", self.layer, self.index)
    }
}

use crate::sparse::{FeatureId, SparseActivations};

impl FeatureId for CltFeatureId {}

/// A single edge in a CLT attribution graph.
///
/// Represents a feature's decoder projection score onto a target direction
/// at a specific downstream layer. Positive scores indicate alignment,
/// negative scores indicate opposition.
#[derive(Debug, Clone)]
pub struct AttributionEdge {
    /// The CLT feature contributing this edge.
    pub feature: CltFeatureId,
    /// Decoder projection score (dot product or cosine similarity).
    pub score: f32,
}

/// Attribution graph for CLT circuit analysis.
///
/// Represents a set of CLT features scored by how strongly their decoder
/// vectors project along a target direction at a specific layer. Built by
/// [`CrossLayerTranscoder::build_attribution_graph()`] or
/// [`CrossLayerTranscoder::build_attribution_graph_batch()`].
///
/// Edges are always sorted by score in descending order.
///
/// # Pruning
///
/// - [`top_k()`](Self::top_k): keep only the k highest-scoring features
/// - [`threshold()`](Self::threshold): keep features with |score| above a minimum
#[derive(Debug, Clone)]
pub struct AttributionGraph {
    /// Target layer these scores were computed for.
    target_layer: usize,
    /// Edges sorted by score descending.
    edges: Vec<AttributionEdge>,
}

impl AttributionGraph {
    /// Target layer this graph was scored against.
    #[must_use]
    pub const fn target_layer(&self) -> usize {
        self.target_layer
    }

    /// All edges, sorted by score descending.
    #[must_use]
    pub fn edges(&self) -> &[AttributionEdge] {
        &self.edges
    }

    /// Number of edges in the graph.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph has no edges.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Return a new graph with only the top-k highest-scoring edges.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Self {
        Self {
            target_layer: self.target_layer,
            edges: self.edges.iter().take(k).cloned().collect(),
        }
    }

    /// Return a new graph keeping only edges whose absolute score meets
    /// or exceeds `min_score`.
    #[must_use]
    pub fn threshold(&self, min_score: f32) -> Self {
        Self {
            target_layer: self.target_layer,
            edges: self
                .edges
                .iter()
                .filter(|e| e.score.abs() >= min_score)
                .cloned()
                .collect(),
        }
    }

    /// Extract the feature IDs from all edges in score order.
    #[must_use]
    pub fn features(&self) -> Vec<CltFeatureId> {
        self.edges.iter().map(|e| e.feature).collect()
    }

    /// Consume the graph and return its edges.
    #[must_use]
    pub fn into_edges(self) -> Vec<AttributionEdge> {
        self.edges
    }
}

/// On-disk layout of a transcoder repository.
///
/// Determines how weight files are named, which tensors they contain, and
/// whether `W_dec` is rank-2 (per-layer) or rank-3 (cross-layer). Auto-detected
/// from the repo file listing at [`CrossLayerTranscoder::open`] time —
/// see the schema-detection rules documented on each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscoderSchema {
    /// Cross-Layer Transcoder (`mntss/clt-*`): two files per layer,
    /// `W_enc_{l}.safetensors` + `W_dec_{l}.safetensors`, layer-suffixed tensor
    /// names (`W_enc_{l}`, `W_dec_{l}`). `W_dec` is rank-3
    /// `[n_features, n_target_layers, d_model]` — writes to multiple downstream
    /// layers.
    CltSplit,
    /// Per-Layer Transcoder bundle (`mntss/transcoder-*`, `mwhanna/qwen3-*-transcoders*`):
    /// one file per layer, `layer_{l}.safetensors`, un-suffixed tensor names
    /// (`W_enc`, `W_dec`, `W_skip`, `b_enc`, `b_dec`). `W_dec` is rank-2
    /// `[n_features, d_model]` — writes only to layer `l`. The `W_skip` linear
    /// path is loaded but not used by encode/inject/suppress.
    PltBundle,
    /// `GemmaScope` NPZ transcoder (`google/gemma-scope-2b-pt-transcoders`,
    /// pointed to by `mntss/gemma-scope-transcoders/config.yaml`). Per-layer
    /// NPZ file at `layer_N/width_16k/average_l0_X/params.npz`. Tensors:
    /// `W_enc [d_model, n_features]` (transposed vs `PltBundle`), `W_dec`,
    /// `b_enc`, `b_dec`, and `threshold [n_features]` for `JumpReLU` gating.
    /// No `W_skip`.
    ///
    /// **Not yet loadable in v0.1.9** — [`CrossLayerTranscoder::open`] returns
    /// a deferral error pointing to roadmap Step 1.6.
    GemmaScopeNpz,
}

impl TranscoderSchema {
    /// Whether this schema writes to multiple downstream layers per feature
    /// (`CltSplit`) or only to its own layer (`PltBundle`, `GemmaScopeNpz`).
    #[must_use]
    pub const fn is_cross_layer(self) -> bool {
        matches!(self, Self::CltSplit)
    }

    /// Whether this schema uses `JumpReLU` gating with a per-feature `threshold`
    /// tensor (`GemmaScopeNpz`) rather than plain `ReLU`.
    #[must_use]
    pub const fn is_jump_relu(self) -> bool {
        matches!(self, Self::GemmaScopeNpz)
    }
}

/// CLT configuration auto-detected from tensor shapes.
#[derive(Debug, Clone)]
pub struct CltConfig {
    /// Number of layers in the base model (26 for Gemma 2 2B).
    pub n_layers: usize,
    /// Hidden dimension of the base model (2304 for Gemma 2 2B).
    pub d_model: usize,
    /// Number of features per encoder layer (16384 for CLT-426K).
    pub n_features_per_layer: usize,
    /// Total feature count across all layers.
    pub n_features_total: usize,
    /// Base model name from config.yaml.
    pub model_name: String,
    /// Detected on-disk schema.
    pub schema: TranscoderSchema,
    /// Per-layer NPZ paths for `GemmaScopeNpz` repos routed via
    /// `mntss/gemma-scope-transcoders/config.yaml`. Empty for `CltSplit` /
    /// `PltBundle`; populated in v0.1.10 by the Step 1.6 loader.
    pub gemmascope_npz_paths: Vec<String>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Currently loaded encoder weights on GPU.
struct LoadedEncoder {
    /// Layer index this encoder corresponds to.
    layer: usize,
    /// Encoder weight matrix.
    ///
    /// # Shapes
    /// - `w_enc`: `[n_features, d_model]`
    w_enc: Tensor,
    /// Encoder bias vector.
    ///
    /// # Shapes
    /// - `b_enc`: `[n_features]`
    b_enc: Tensor,
}

// ---------------------------------------------------------------------------
// CrossLayerTranscoder
// ---------------------------------------------------------------------------

/// Cross-Layer Transcoder.
///
/// Loads CLT encoder/decoder weights on-demand from `HuggingFace` safetensors,
/// with memory-efficient streaming (only one encoder on GPU at a time)
/// and a micro-cache for steering vectors.
///
/// Downloads are lazy: [`open()`](Self::open) only fetches config and the first
/// encoder for dimension detection. Subsequent files are downloaded as needed by
/// [`load_encoder()`](Self::load_encoder), [`decoder_vector()`](Self::decoder_vector),
/// and [`cache_steering_vectors()`](Self::cache_steering_vectors).
///
/// # Example
///
/// ```no_run
/// # fn main() -> candle_mi::Result<()> {
/// use candle_mi::clt::CrossLayerTranscoder;
/// use candle_core::Device;
///
/// let mut clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k")?;
/// println!("CLT: {} layers, d_model={}", clt.config().n_layers, clt.config().d_model);
///
/// // Load encoder for layer 10
/// let device = Device::Cpu;
/// clt.load_encoder(10, &device)?;
/// # Ok(())
/// # }
/// ```
pub struct CrossLayerTranscoder {
    /// `HuggingFace` repository ID for on-demand downloads.
    repo_id: String,
    /// Fetch configuration for `hf-fetch-model` downloads.
    fetch_config: hf_fetch_model::FetchConfig,
    /// Local paths to already-downloaded encoder files (None = not yet downloaded).
    encoder_paths: Vec<Option<PathBuf>>,
    /// Local paths to already-downloaded decoder files (None = not yet downloaded).
    decoder_paths: Vec<Option<PathBuf>>,
    /// Auto-detected configuration.
    config: CltConfig,
    /// Currently loaded encoder (stream-and-free: only one at a time).
    loaded_encoder: Option<LoadedEncoder>,
    /// Micro-cache: pre-extracted steering vectors pinned on device.
    /// Key: (`feature_id`, `target_layer`), Value: decoder vector `[d_model]` on device.
    steering_cache: HashMap<(CltFeatureId, usize), Tensor>,
}

impl CrossLayerTranscoder {
    /// Open a CLT from `HuggingFace` and detect its configuration.
    ///
    /// Only downloads `config.yaml` and `W_enc_0.safetensors` (~75 MB).
    /// All other encoder/decoder files are downloaded lazily on first use.
    ///
    /// # Arguments
    /// * `clt_repo` — `HuggingFace` repository ID (e.g., `"mntss/clt-gemma-2-2b-426k"`)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the repository is inaccessible or files
    /// cannot be fetched. Returns [`MIError::Config`] if the weight format is
    /// unexpected.
    #[allow(clippy::too_many_lines)]
    pub fn open(clt_repo: &str) -> Result<Self> {
        let fetch_config = hf_fetch_model::FetchConfig::builder()
            .on_progress(|event| {
                tracing::info!(
                    filename = %event.filename,
                    percent = event.percent,
                    bytes_downloaded = event.bytes_downloaded,
                    bytes_total = event.bytes_total,
                    "CLT download progress",
                );
            })
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;

        // List repo files (no downloads needed) for schema detection and layer counting.
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| MIError::Download(format!("failed to create tokio runtime: {e}")))?;
        // hf-fetch-model 0.9.6 takes a shared reqwest client for the metadata listing.
        let http_client = hf_fetch_model::build_client(None)
            .map_err(|e| MIError::Download(format!("failed to build HTTP client: {e}")))?;
        let repo_files = rt
            .block_on(hf_fetch_model::repo::list_repo_files_with_metadata(
                clt_repo,
                None,
                None,
                &http_client,
            ))
            .map_err(|e| MIError::Download(format!("failed to list repo files: {e}")))?;

        // Classify schema from filename patterns — before any downloads.
        let has_clt_split = repo_files
            .iter()
            .any(|f| f.filename.starts_with("W_enc_") && f.filename.ends_with(".safetensors"));
        let has_plt_bundle = repo_files
            .iter()
            .any(|f| f.filename.starts_with("layer_") && f.filename.ends_with(".safetensors"));
        let has_gemmascope_npz_direct = repo_files.iter().any(|f| {
            f.filename.starts_with("layer_")
                && f.filename.contains("/width_")
                && f.filename.contains("/average_l0_")
                && f.filename.ends_with("/params.npz")
        });
        let has_config_yaml = repo_files.iter().any(|f| f.filename == "config.yaml");
        // Case-sensitive match is intentional here: `HuggingFace` filenames in the
        // `mntss/gemma-scope-transcoders` repo are always lowercase `.bin`; a
        // case-insensitive fallback would accept bogus matches.
        let has_gemmascope_bin_metadata = repo_files.iter().any(|f| {
            f.filename.starts_with("features/layer_")
                && std::path::Path::new(&f.filename)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        });
        let has_gemmascope_metadata_repo =
            has_config_yaml && has_gemmascope_bin_metadata && !has_clt_split && !has_plt_bundle;

        let schema = if has_clt_split {
            TranscoderSchema::CltSplit
        } else if has_plt_bundle {
            TranscoderSchema::PltBundle
        } else if has_gemmascope_npz_direct || has_gemmascope_metadata_repo {
            TranscoderSchema::GemmaScopeNpz
        } else {
            return Err(MIError::Config(format!(
                "unrecognised transcoder repo layout for {clt_repo}"
            )));
        };
        info!("Transcoder schema detected for {clt_repo}: {schema:?}");

        // v0.1.9: schema detection succeeds but loading of GemmaScope repos
        // is intentionally deferred. v0.1.10 implements the two-repo NPZ flow.
        if matches!(schema, TranscoderSchema::GemmaScopeNpz) {
            return Err(MIError::Config(
                "GemmaScope loader lands in v0.1.10 — see roadmap Step 1.6".into(),
            ));
        }

        // Count layers per schema. GemmaScopeNpz was rejected earlier, so the match
        // is effectively binary here but spelled exhaustively for safety.
        let n_layers = match schema {
            TranscoderSchema::CltSplit => repo_files
                .iter()
                .filter(|f| {
                    f.filename.starts_with("W_enc_") && f.filename.ends_with(".safetensors")
                })
                .count(),
            TranscoderSchema::PltBundle => repo_files
                .iter()
                .filter(|f| {
                    f.filename.starts_with("layer_") && f.filename.ends_with(".safetensors")
                })
                .count(),
            TranscoderSchema::GemmaScopeNpz => {
                // EXPLICIT: exhaustive match arm — variant rejected with the deferral error above.
                return Err(MIError::Config(
                    "GemmaScope loader lands in v0.1.10 — see roadmap Step 1.6".into(),
                ));
            }
        };
        if n_layers == 0 {
            return Err(MIError::Config(format!(
                "no encoder files found in {clt_repo} (schema={schema:?})"
            )));
        }

        // Parse config.yaml for model_name (simple line-by-line, no serde_yaml dep).
        let model_name = match hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            "config.yaml",
            &fetch_config,
        ) {
            Ok(outcome) => {
                let path = outcome.into_inner();
                let text = std::fs::read_to_string(&path)?;
                parse_yaml_value(&text, "model_name").unwrap_or_else(|| "unknown".to_owned())
            }
            Err(_) => "unknown".to_owned(),
        };

        // Download the first encoder file for dimension detection, branching on schema.
        // `CltSplit`: `W_enc_0.safetensors` (~75 MB) with tensor `"W_enc_0"`.
        // `PltBundle`: `layer_0.safetensors` (~1 GiB, bundle of
        //           `W_enc`/`W_dec`/`W_skip`/`b_enc`/`b_dec`) with un-suffixed tensor `"W_enc"`.
        let (enc0_filename, enc_tensor_name) = match schema {
            TranscoderSchema::CltSplit => ("W_enc_0.safetensors", "W_enc_0"),
            TranscoderSchema::PltBundle => ("layer_0.safetensors", "W_enc"),
            TranscoderSchema::GemmaScopeNpz => {
                // GemmaScopeNpz returns early above; this arm is kept for exhaustive matching.
                // EXPLICIT: exhaustive match on TranscoderSchema; GemmaScopeNpz rejected upstream.
                return Err(MIError::Config(
                    "GemmaScope loader lands in v0.1.10 — see roadmap Step 1.6".into(),
                ));
            }
        };
        // BORROW: explicit .to_owned() — hf_fetch_model download_file_blocking requires an owned repo ID.
        let enc0_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            enc0_filename,
            &fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {enc0_filename}: {e}")))?
        .into_inner();

        let data = std::fs::read(&enc0_path)?;
        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize {enc_tensor_name}: {e}"))
        })?;
        let w_enc_view = tensors
            .tensor(enc_tensor_name)
            .map_err(|e| MIError::Config(format!("tensor '{enc_tensor_name}' not found: {e}")))?;
        let shape = w_enc_view.shape();
        if shape.len() != 2 {
            return Err(MIError::Config(format!(
                "expected 2D encoder weight, got shape {shape:?}"
            )));
        }
        let n_features_per_layer = *shape
            .first()
            .ok_or_else(|| MIError::Config("encoder weight shape is empty".into()))?;
        let d_model = *shape.get(1).ok_or_else(|| {
            MIError::Config("encoder weight shape has fewer than 2 dimensions".into())
        })?;

        // Initialise paths: only first encoder known, rest downloaded lazily.
        let mut encoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];
        if let Some(slot) = encoder_paths.first_mut() {
            *slot = Some(enc0_path);
        }
        let decoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];

        let config = CltConfig {
            n_layers,
            d_model,
            n_features_per_layer,
            n_features_total: n_layers * n_features_per_layer,
            model_name,
            schema,
            gemmascope_npz_paths: Vec::new(),
        };
        info!(
            "CLT config: {} layers, d_model={}, features_per_layer={}, total={}, schema={:?}",
            config.n_layers,
            config.d_model,
            config.n_features_per_layer,
            config.n_features_total,
            config.schema,
        );

        Ok(Self {
            repo_id: clt_repo.to_owned(),
            fetch_config,
            encoder_paths,
            decoder_paths,
            config,
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        })
    }

    /// Access the auto-detected CLT configuration.
    #[must_use]
    pub const fn config(&self) -> &CltConfig {
        &self.config
    }

    /// Check whether an encoder is currently loaded and for which layer.
    #[must_use]
    pub fn loaded_encoder_layer(&self) -> Option<usize> {
        self.loaded_encoder.as_ref().map(|e| e.layer)
    }

    // --- Lazy download helpers ---

    /// Ensure the encoder file for a given layer is downloaded. Returns the path.
    fn ensure_encoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(path) = self
            .encoder_paths
            .get(layer)
            .and_then(std::option::Option::as_ref)
        {
            // BORROW: explicit .clone() — PathBuf from Vec
            return Ok(path.clone());
        }
        let filename = format!("W_enc_{layer}.safetensors");
        info!("Downloading {filename} from {}", self.repo_id);
        let path = hf_fetch_model::download_file_blocking(
            self.repo_id.clone(),
            &filename,
            &self.fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {filename}: {e}")))?
        .into_inner();
        if let Some(slot) = self.encoder_paths.get_mut(layer) {
            // BORROW: explicit .clone() — store PathBuf in cache
            *slot = Some(path.clone());
        }
        Ok(path)
    }

    /// Ensure the decoder file for a given layer is downloaded. Returns the path.
    fn ensure_decoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(path) = self
            .decoder_paths
            .get(layer)
            .and_then(std::option::Option::as_ref)
        {
            // BORROW: explicit .clone() — PathBuf from Vec
            return Ok(path.clone());
        }
        let filename = format!("W_dec_{layer}.safetensors");
        info!("Downloading {filename} from {}", self.repo_id);
        let path = hf_fetch_model::download_file_blocking(
            self.repo_id.clone(),
            &filename,
            &self.fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {filename}: {e}")))?
        .into_inner();
        if let Some(slot) = self.decoder_paths.get_mut(layer) {
            // BORROW: explicit .clone() — store PathBuf in cache
            *slot = Some(path.clone());
        }
        Ok(path)
    }

    // --- Encoder loading (stream-and-free) ---

    /// Load a single encoder's weights to the specified device.
    ///
    /// Frees any previously loaded encoder first (stream-and-free pattern).
    /// Peak GPU overhead: ~75 MB for CLT-426K, ~450 MB for CLT-2.5M.
    ///
    /// # Arguments
    /// * `layer` — Layer index (`0..n_layers`)
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the layer is out of range.
    /// Returns [`MIError::Download`] if the encoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor deserialization failure.
    pub fn load_encoder(&mut self, layer: usize, device: &Device) -> Result<()> {
        if layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "layer {layer} out of range (CLT has {} layers)",
                self.config.n_layers
            )));
        }

        // Skip if already loaded.
        if let Some(ref enc) = self.loaded_encoder
            && enc.layer == layer
        {
            return Ok(());
        }

        // Drop previous encoder (frees GPU memory).
        self.loaded_encoder = None;

        info!("Loading CLT encoder for layer {layer}");

        let enc_path = self.ensure_encoder_path(layer)?;
        let data = std::fs::read(&enc_path)?;
        let st = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize encoder layer {layer}: {e}"))
        })?;

        let w_enc_name = format!("W_enc_{layer}");
        let b_enc_name = format!("b_enc_{layer}");

        let w_enc = tensor_from_view(
            &st.tensor(&w_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{w_enc_name}' not found: {e}")))?,
            device,
        )?;
        let b_enc = tensor_from_view(
            &st.tensor(&b_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{b_enc_name}' not found: {e}")))?,
            device,
        )?;

        self.loaded_encoder = Some(LoadedEncoder {
            layer,
            w_enc,
            b_enc,
        });

        Ok(())
    }

    // --- Encoding ---

    /// Encode a residual stream activation into sparse CLT features.
    ///
    /// The residual should be the "residual mid" activation at the given layer
    /// (after attention, before MLP).
    ///
    /// Returns all features that pass the `ReLU` threshold, sorted by
    /// activation magnitude in descending order.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations<CltFeatureId>`] with `(CltFeatureId, f32)` pairs
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if no encoder is loaded or the wrong layer is loaded.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode(
        &self,
        residual: &Tensor,
        layer: usize,
    ) -> Result<SparseActivations<CltFeatureId>> {
        let enc = self.loaded_encoder.as_ref().ok_or_else(|| {
            MIError::Hook(format!(
                "no encoder loaded — call load_encoder({layer}) first"
            ))
        })?;
        if enc.layer != layer {
            return Err(MIError::Hook(format!(
                "loaded encoder is for layer {}, but layer {layer} was requested",
                enc.layer
            )));
        }

        // Compute pre-activations in F32 for numerical stability.
        // W_enc: [n_features, d_model], residual: [d_model]
        // pre_acts = W_enc @ residual + b_enc → [n_features]
        let residual_f32 = residual.flatten_all()?;
        // PROMOTE: matmul and bias add require F32 for numerical stability
        let residual_f32 = residual_f32.to_dtype(DType::F32)?;
        let w_enc_f32 = enc.w_enc.to_dtype(DType::F32)?;
        let b_enc_f32 = enc.b_enc.to_dtype(DType::F32)?;

        let pre_acts = w_enc_f32.matmul(&residual_f32.unsqueeze(1)?)?.squeeze(1)?;
        let pre_acts = (&pre_acts + &b_enc_f32)?;

        // ReLU activation.
        let acts = pre_acts.relu()?;

        // Transfer to CPU for sparse extraction.
        let acts_vec: Vec<f32> = acts.to_vec1()?;

        let mut features: Vec<(CltFeatureId, f32)> = acts_vec
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v > 0.0)
            .map(|(i, v)| (CltFeatureId { layer, index: i }, *v))
            .collect();

        // Sort by activation magnitude (descending).
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(SparseActivations { features })
    }

    /// Encode and return only the top-k most active features.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations<CltFeatureId>`] truncated to at most `k` entries
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Same as [`encode()`](Self::encode).
    pub fn top_k(
        &self,
        residual: &Tensor,
        layer: usize,
        k: usize,
    ) -> Result<SparseActivations<CltFeatureId>> {
        let mut sparse = self.encode(residual, layer)?;
        sparse.truncate(k);
        Ok(sparse)
    }

    // --- Decoder access ---

    /// Extract a single feature's decoder vector for a target downstream layer.
    ///
    /// Loads from safetensors on demand. Checks the steering cache first
    /// to avoid redundant file reads.
    ///
    /// # Shapes
    /// - returns: `[d_model]` — decoder vector on `device`
    ///
    /// # Arguments
    /// * `feature` — The CLT feature to extract the decoder for
    /// * `target_layer` — The downstream layer to decode to (must be >= feature.layer)
    /// * `device` — Device to place the resulting tensor on
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if layer indices are out of range.
    /// Returns [`MIError::Download`] if the decoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn decoder_vector(
        &mut self,
        feature: &CltFeatureId,
        target_layer: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if feature.layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "feature source layer {} out of range (CLT has {} layers)",
                feature.layer, self.config.n_layers
            )));
        }
        if target_layer < feature.layer || target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} must be >= source layer {} and < {}",
                feature.layer, self.config.n_layers
            )));
        }
        if feature.index >= self.config.n_features_per_layer {
            return Err(MIError::Config(format!(
                "feature index {} out of range (max {})",
                feature.index, self.config.n_features_per_layer
            )));
        }

        // Check steering cache first.
        let cache_key = (*feature, target_layer);
        if let Some(cached) = self.steering_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // W_dec_l has shape [n_features, n_layers - l, d_model]
        // target_offset = target_layer - feature.layer
        let target_offset = target_layer - feature.layer;

        let dec_path = self.ensure_decoder_path(feature.layer)?;
        let data = std::fs::read(&dec_path)?;
        let st = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!(
                "failed to deserialize decoder layer {}: {e}",
                feature.layer
            ))
        })?;

        let dec_name = format!("W_dec_{}", feature.layer);
        let w_dec = tensor_from_view(
            &st.tensor(&dec_name)
                .map_err(|e| MIError::Config(format!("tensor '{dec_name}' not found: {e}")))?,
            &Device::Cpu,
        )?;

        // w_dec[feature.index, target_offset, :] → [d_model]
        let column = w_dec.i((feature.index, target_offset))?;

        // Transfer to target device.
        let column = column.to_device(device)?;

        Ok(column)
    }

    // --- Micro-cache ---

    /// Pre-load decoder vectors into the steering micro-cache.
    ///
    /// Each entry is a `(CltFeatureId, target_layer)` pair. Vectors are
    /// loaded to the specified device and kept pinned for repeated injection.
    ///
    /// Uses an OOM-safe pattern: loads each decoder file to CPU, extracts needed
    /// columns as independent F32 tensors, drops the large file, then moves
    /// small tensors to the target device.
    ///
    /// Memory: 50 features × 2304 × 4 bytes = ~450 KB (negligible).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn cache_steering_vectors(
        &mut self,
        features: &[(CltFeatureId, usize)],
        device: &Device,
    ) -> Result<()> {
        // Group by source layer to batch decoder file reads.
        let mut by_source: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (fid, target_layer) in features {
            by_source
                .entry(fid.layer)
                .or_default()
                .push((fid.index, *target_layer));
        }

        let mut loaded = 0_usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, entries)) in by_source.iter().enumerate() {
            info!(
                "cache_steering_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );

            // Group by target_layer to identify needed offsets.
            let mut by_target: HashMap<usize, Vec<usize>> = HashMap::new();
            for &(index, target_layer) in entries {
                by_target.entry(target_layer).or_default().push(index);
            }

            // Load decoder file, extract needed columns as independent CPU
            // tensors, then drop the large file data BEFORE any GPU transfer.
            // This prevents OOM when early-layer decoders can be >1.6 GB each.
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let data = std::fs::read(&dec_path)?;
                info!(
                    "cache_steering_vectors: loaded {} MB for layer {}",
                    data.len() / (1024 * 1024),
                    source_layer
                );
                let st = SafeTensors::deserialize(&data).map_err(|e| {
                    MIError::Config(format!(
                        "failed to deserialize decoder layer {source_layer}: {e}"
                    ))
                })?;
                let dec_name = format!("W_dec_{source_layer}");
                let w_dec = tensor_from_view(
                    &st.tensor(&dec_name).map_err(|e| {
                        MIError::Config(format!("tensor '{dec_name}' not found: {e}"))
                    })?,
                    &Device::Cpu,
                )?;

                for (target_layer, indices) in &by_target {
                    let target_offset = target_layer - source_layer;
                    for &index in indices {
                        let fid = CltFeatureId {
                            layer: *source_layer,
                            index,
                        };
                        let cache_key = (fid, *target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            // Extract as independent F32 tensor: to_dtype +
                            // to_vec1 copies data OUT of candle's Arc storage,
                            // so dropping w_dec truly frees the ~1.6 GB decoder.
                            let view = w_dec.i((index, target_offset))?;
                            let dims = view.dims().to_vec();
                            // PROMOTE: F32 for numerical stability in accumulation
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, *target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Now move the small independent columns to the target device.
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let device_tensor = cpu_tensor.to_device(device)?;
                    e.insert(device_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Cache steering vectors for ALL downstream layers of each feature.
    ///
    /// For each feature at source layer `l`, caches decoder vectors for every
    /// downstream target layer `l..n_layers`. This enables multi-layer
    /// "clamping" injection where the steering signal propagates through all
    /// downstream transformer layers.
    ///
    /// Same OOM-safe pattern as [`cache_steering_vectors()`](Self::cache_steering_vectors).
    ///
    /// # Arguments
    /// * `features` — Feature IDs to cache (all downstream layers are cached automatically)
    /// * `device` — Device to store cached tensors on (typically GPU)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any feature layer is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn cache_steering_vectors_all_downstream(
        &mut self,
        features: &[CltFeatureId],
        device: &Device,
    ) -> Result<()> {
        let n_layers = self.config.n_layers;

        // Group by source layer to batch decoder file reads.
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            if fid.layer >= n_layers {
                return Err(MIError::Config(format!(
                    "feature source layer {} out of range (max {})",
                    fid.layer,
                    n_layers - 1
                )));
            }
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut loaded = 0_usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            let n_target_layers = n_layers - source_layer;
            info!(
                "cache_steering_vectors_all_downstream: loading decoder for source layer {} \
                 ({}/{}, {} downstream layers)",
                source_layer,
                layer_idx + 1,
                n_source_layers,
                n_target_layers
            );

            // Load decoder file, extract ALL offsets as independent CPU tensors, then drop.
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let data = std::fs::read(&dec_path)?;
                info!(
                    "cache_steering_vectors_all_downstream: loaded {} MB for layer {}",
                    data.len() / (1024 * 1024),
                    source_layer
                );
                let st = SafeTensors::deserialize(&data).map_err(|e| {
                    MIError::Config(format!(
                        "failed to deserialize decoder layer {source_layer}: {e}"
                    ))
                })?;
                let dec_name = format!("W_dec_{source_layer}");
                let w_dec = tensor_from_view(
                    &st.tensor(&dec_name).map_err(|e| {
                        MIError::Config(format!("tensor '{dec_name}' not found: {e}"))
                    })?,
                    &Device::Cpu,
                )?;

                for &index in indices {
                    let fid = CltFeatureId {
                        layer: *source_layer,
                        index,
                    };
                    for target_offset in 0..n_target_layers {
                        let target_layer = source_layer + target_offset;
                        let cache_key = (fid, target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            let view = w_dec.i((index, target_offset))?;
                            let dims = view.dims().to_vec();
                            // PROMOTE: F32 for numerical stability in accumulation
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Move small independent columns to the target device.
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let device_tensor = cpu_tensor.to_device(device)?;
                    e.insert(device_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors across all downstream layers ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Clear all cached steering vectors, freeing device memory.
    pub fn clear_steering_cache(&mut self) {
        let count = self.steering_cache.len();
        self.steering_cache.clear();
        if count > 0 {
            info!("Cleared {count} steering vectors from cache");
        }
    }

    /// Number of vectors currently in the steering cache.
    #[must_use]
    pub fn steering_cache_len(&self) -> usize {
        self.steering_cache.len()
    }

    // --- Injection ---

    /// Build a [`crate::HookSpec`] that injects CLT decoder vectors into the residual stream.
    ///
    /// Groups cached steering vectors by target layer, accumulates them per layer,
    /// scales by `strength`, and creates [`crate::Intervention::Add`] entries on
    /// [`crate::HookPoint::ResidPost`] for each target layer. The resulting `HookSpec`
    /// can be passed directly to [`MIModel::forward()`](crate::MIModel::forward).
    ///
    /// # Shapes
    /// - Internally constructs `[1, seq_len, d_model]` tensors with the steering
    ///   vector placed at `position` and zeros elsewhere.
    ///
    /// # Arguments
    /// * `features` — List of `(feature_id, target_layer)` pairs (must be cached)
    /// * `position` — Token position in the sequence to inject at
    /// * `seq_len` — Total sequence length (needed to construct position-specific tensors)
    /// * `strength` — Scalar multiplier for the accumulated steering vectors
    /// * `device` — Device to construct injection tensors on
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if any feature is not in the steering cache.
    /// Returns [`MIError::Model`] on tensor construction failure.
    pub fn prepare_hook_injection(
        &self,
        features: &[(CltFeatureId, usize)],
        position: usize,
        seq_len: usize,
        strength: f32,
        device: &Device,
    ) -> Result<crate::hooks::HookSpec> {
        use crate::hooks::{HookPoint, HookSpec, Intervention};

        // Group features by target layer and accumulate their decoder vectors.
        let mut per_layer: HashMap<usize, Tensor> = HashMap::new();
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                MIError::Hook(format!(
                    "feature {feature} for target layer {target_layer} not in steering cache \
                     — call cache_steering_vectors() first"
                ))
            })?;
            // PROMOTE: accumulate in F32 for numerical stability
            let cached_f32 = cached.to_dtype(DType::F32)?;
            if let Some(acc) = per_layer.get_mut(target_layer) {
                let acc_ref: &Tensor = acc;
                *acc = (acc_ref + &cached_f32)?;
            } else {
                per_layer.insert(*target_layer, cached_f32);
            }
        }

        // Build HookSpec with Intervention::Add at each target layer.
        let mut hooks = HookSpec::new();
        let d_model = self.config.d_model;

        for (target_layer, accumulated) in &per_layer {
            // Scale by strength.
            let scaled = (accumulated * f64::from(strength))?;

            // Build a [1, seq_len, d_model] tensor with the vector at `position`.
            let mut injection = Tensor::zeros((1, seq_len, d_model), DType::F32, device)?;

            // Place the scaled vector at the target position.
            let scaled_3d = scaled.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, d_model]
            let before = if position > 0 {
                Some(injection.narrow(1, 0, position)?)
            } else {
                None
            };
            let after = if position + 1 < seq_len {
                Some(injection.narrow(1, position + 1, seq_len - position - 1)?)
            } else {
                None
            };

            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            if let Some(b) = before {
                parts.push(b);
            }
            parts.push(scaled_3d);
            if let Some(a) = after {
                parts.push(a);
            }

            injection = Tensor::cat(&parts, 1)?;

            hooks.intervene(
                HookPoint::ResidPost(*target_layer),
                Intervention::Add(injection),
            );
        }

        Ok(hooks)
    }

    /// Inject cached steering vectors directly into a residual stream tensor.
    ///
    /// Convenience method for use outside the forward pass (e.g., in analysis
    /// scripts). Returns a new tensor with the injection applied:
    /// `residual[:, position, :] += strength × Σ decoder_vectors`
    ///
    /// # Shapes
    /// - `residual`: `[batch, seq_len, d_model]` — hidden states
    /// - returns: `[batch, seq_len, d_model]` — modified hidden states
    ///
    /// # Arguments
    /// * `residual` — Hidden states tensor
    /// * `features` — List of `(feature, target_layer)` pairs to inject (must be cached)
    /// * `position` — Token position in the sequence to inject at
    /// * `strength` — Scalar multiplier for the steering vectors
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if any feature is not in the steering cache.
    /// Returns [`MIError::Config`] if dimensions don't match.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn inject(
        &self,
        residual: &Tensor,
        features: &[(CltFeatureId, usize)],
        position: usize,
        strength: f32,
    ) -> Result<Tensor> {
        let (batch, seq_len, d_model) = residual.dims3()?;
        if position >= seq_len {
            return Err(MIError::Config(format!(
                "injection position {position} out of range (seq_len={seq_len})"
            )));
        }
        if d_model != self.config.d_model {
            return Err(MIError::Config(format!(
                "residual d_model={d_model} doesn't match CLT d_model={}",
                self.config.d_model
            )));
        }

        // Accumulate all steering vectors into one vector (F32 for stability).
        let mut accumulated = Tensor::zeros((d_model,), DType::F32, residual.device())?;
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                MIError::Hook(format!(
                    "feature {feature} for target layer {target_layer} not in steering cache"
                ))
            })?;
            // PROMOTE: accumulate in F32 for numerical stability
            let cached_f32 = cached.to_dtype(DType::F32)?;
            accumulated = (&accumulated + &cached_f32)?;
        }

        // Scale by strength.
        let accumulated = (accumulated * f64::from(strength))?;

        // Convert to residual dtype.
        let accumulated = accumulated.to_dtype(residual.dtype())?;

        // Build steering tensor and inject at position.
        let pos_slice = residual.narrow(1, position, 1)?; // [batch, 1, d_model]
        let steering_expanded = accumulated
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((batch, 1, d_model))?; // [batch, 1, d_model]
        let pos_updated = (&pos_slice + &steering_expanded)?;

        // Reassemble: before + updated_position + after.
        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        if position > 0 {
            parts.push(residual.narrow(1, 0, position)?);
        }
        parts.push(pos_updated);
        if position + 1 < seq_len {
            parts.push(residual.narrow(1, position + 1, seq_len - position - 1)?);
        }

        let result = Tensor::cat(&parts, 1)?;
        Ok(result)
    }

    // --- Attribution / decoder scoring ---

    /// Score all CLT features by how strongly their decoder vector at
    /// `target_layer` projects along a given direction vector.
    ///
    /// For each source layer `0..n_layers` where `source_layer <= target_layer`:
    /// loads the decoder file to CPU, extracts the target layer slice
    /// `[n_features, d_model]`, and computes `scores = slice @ direction`.
    ///
    /// When `cosine` is true, scores are normalized by both the direction
    /// vector norm and each decoder row norm (cosine similarity).
    ///
    /// # Shapes
    /// - `direction`: `[d_model]` — target direction vector (e.g., token embedding)
    /// - returns: top-k `(CltFeatureId, f32)` pairs, sorted by score descending
    ///
    /// # Arguments
    /// * `direction` — `[d_model]` direction vector to project decoders onto
    /// * `target_layer` — downstream layer to examine decoders at
    /// * `top_k` — number of top-scoring features to return
    /// * `cosine` — whether to use cosine similarity instead of dot product
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `direction` shape is wrong or `target_layer`
    /// is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Processes one decoder file at a time on CPU (up to ~2 GB for layer 0).
    /// No GPU memory required.
    pub fn score_features_by_decoder_projection(
        &mut self,
        direction: &Tensor,
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<(CltFeatureId, f32)>> {
        let d_model = self.config.d_model;
        if direction.dims() != [d_model] {
            return Err(MIError::Config(format!(
                "direction must have shape [{d_model}], got {:?}",
                direction.dims()
            )));
        }
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // PROMOTE: F32 for dot-product precision matching Python reference
        let direction_f32 = direction.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        // Optionally normalize direction to unit length for cosine similarity.
        let direction_norm = if cosine {
            let norm: f32 = direction_f32.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
            if norm > 1e-10 {
                direction_f32.broadcast_div(&Tensor::new(norm, &Device::Cpu)?)?
            } else {
                direction_f32
            }
        } else {
            direction_f32
        };

        let mut all_scores: Vec<(CltFeatureId, f32)> = Vec::new();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue; // This source layer cannot decode to target_layer.
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU.
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let data = std::fs::read(&dec_path)?;
            info!(
                "score_features_by_decoder_projection: loaded {} MB for layer {}",
                data.len() / (1024 * 1024),
                source_layer
            );
            let st = SafeTensors::deserialize(&data).map_err(|e| {
                MIError::Config(format!(
                    "failed to deserialize decoder layer {source_layer}: {e}"
                ))
            })?;

            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(
                &st.tensor(&dec_name)
                    .map_err(|e| MIError::Config(format!("tensor '{dec_name}' not found: {e}")))?,
                &Device::Cpu,
            )?;
            // PROMOTE: decoder weights are BF16 on disk; F32 for matmul precision
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;

            // Extract target layer slice: [n_features, d_model]
            let dec_slice = w_dec_f32.i((.., target_offset, ..))?;

            // raw_scores = dec_slice @ direction_norm → [n_features]
            let raw_scores = dec_slice
                .matmul(&direction_norm.unsqueeze(1)?)?
                .squeeze(1)?;

            let scores_vec: Vec<f32> = if cosine {
                // Divide by each decoder row's L2 norm → cosine similarity.
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?;
                let cosine_scores = raw_scores.broadcast_div(&dec_norms)?;
                cosine_scores.to_vec1()?
            } else {
                raw_scores.to_vec1()?
            };

            for (idx, &score) in scores_vec.iter().enumerate() {
                if score.is_finite() {
                    all_scores.push((
                        CltFeatureId {
                            layer: source_layer,
                            index: idx,
                        },
                        score,
                    ));
                }
            }

            info!(
                "Scored {} features at source layer {source_layer} (target layer {target_layer})",
                scores_vec.len()
            );
        }

        // Sort by score descending, take top-k.
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_scores.truncate(top_k);

        Ok(all_scores)
    }

    /// Batch version of [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection).
    ///
    /// Scores multiple direction vectors against all decoder files in a single
    /// pass. Each decoder file is loaded **once** for all directions, reducing
    /// I/O from `n_words × n_layers` file reads to just `n_layers`.
    ///
    /// # Shapes
    /// - `directions`: slice of `[d_model]` tensors (one per word/direction)
    /// - returns: one `Vec<(CltFeatureId, f32)>` per direction (top-k per word)
    ///
    /// # Arguments
    /// * `directions` — slice of `[d_model]` direction vectors
    /// * `target_layer` — downstream layer to examine decoders at
    /// * `top_k` — number of top-scoring features to return per direction
    /// * `cosine` — whether to use cosine similarity
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any direction has wrong shape, directions is
    /// empty, or `target_layer` is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Stacks directions to `[n_words, d_model]` on CPU. Each decoder file
    /// loaded one at a time (up to ~2 GB for layer 0). No GPU memory required.
    pub fn score_features_by_decoder_projection_batch(
        &mut self,
        directions: &[Tensor],
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<Vec<(CltFeatureId, f32)>>> {
        let d_model = self.config.d_model;
        let n_words = directions.len();
        if n_words == 0 {
            return Err(MIError::Config(
                "at least one direction vector required".into(),
            ));
        }
        for (i, dir) in directions.iter().enumerate() {
            if dir.dims() != [d_model] {
                return Err(MIError::Config(format!(
                    "direction vector {i} must have shape [{d_model}], got {:?}",
                    dir.dims()
                )));
            }
        }
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // PROMOTE: directions may arrive as BF16; F32 for matmul precision
        let dirs_f32: Vec<Tensor> = directions
            .iter()
            .map(|d| d.to_dtype(DType::F32)?.to_device(&Device::Cpu))
            .collect::<std::result::Result<_, _>>()?;
        let stacked = Tensor::stack(&dirs_f32, 0)?; // [n_words, d_model]

        // For cosine: row-normalize direction vectors to unit length.
        let stacked_norm = if cosine {
            let norms = stacked.sqr()?.sum(1)?.sqrt()?; // [n_words]
            let ones = Tensor::ones_like(&norms)?;
            let safe_norms = norms.maximum(&(&ones * 1e-10f64)?)?; // [n_words]
            stacked.broadcast_div(&safe_norms.unsqueeze(1)?)?
        } else {
            stacked
        };
        let directions_t = stacked_norm.t()?; // [d_model, n_words]

        // Per-word score accumulators.
        let mut all_scores: Vec<Vec<(CltFeatureId, f32)>> =
            (0..n_words).map(|_| Vec::new()).collect();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue;
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file ONCE for all words.
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let data = std::fs::read(&dec_path)?;
            info!(
                "score_features_batch: loaded {} MB for layer {}",
                data.len() / (1024 * 1024),
                source_layer
            );
            let st = SafeTensors::deserialize(&data).map_err(|e| {
                MIError::Config(format!(
                    "failed to deserialize decoder layer {source_layer}: {e}"
                ))
            })?;
            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(
                &st.tensor(&dec_name)
                    .map_err(|e| MIError::Config(format!("tensor '{dec_name}' not found: {e}")))?,
                &Device::Cpu,
            )?;
            // PROMOTE: decoder weights are BF16 on disk; F32 for matmul precision
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;
            let dec_slice = w_dec_f32.i((.., target_offset, ..))?; // [n_features, d_model]

            // Batch matmul: [n_features, d_model] × [d_model, n_words] = [n_features, n_words]
            let raw_scores = dec_slice.matmul(&directions_t)?;

            // Transpose to [n_words, n_features] for easy extraction.
            let scores_2d: Vec<Vec<f32>> = if cosine {
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?; // [n_features]
                let cosine_scores = raw_scores.broadcast_div(&dec_norms.unsqueeze(1)?)?;
                cosine_scores.t()?.to_vec2()?
            } else {
                raw_scores.t()?.to_vec2()?
            };

            for (w, word_scores) in scores_2d.iter().enumerate() {
                for (idx, &score) in word_scores.iter().enumerate() {
                    if score.is_finite()
                        && let Some(word_vec) = all_scores.get_mut(w)
                    {
                        word_vec.push((
                            CltFeatureId {
                                layer: source_layer,
                                index: idx,
                            },
                            score,
                        ));
                    }
                }
            }

            info!(
                "Batch scored {} words × {} features at source layer {} (target layer {})",
                n_words,
                scores_2d.first().map_or(0, Vec::len),
                source_layer,
                target_layer
            );
        }

        // Sort and truncate per word.
        for word_scores in &mut all_scores {
            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            word_scores.truncate(top_k);
        }

        Ok(all_scores)
    }

    /// Extract decoder vectors for a set of features at a specific target layer.
    ///
    /// Groups features by source layer, loads each decoder file once, and
    /// extracts the decoder vector at the target layer offset as an independent
    /// F32 CPU tensor. Uses the OOM-safe `to_vec1` + `from_vec` pattern to
    /// ensure large decoder files are freed before processing the next layer.
    ///
    /// # Shapes
    /// - returns: `HashMap<CltFeatureId, Tensor>` where each tensor is `[d_model]` (F32, CPU)
    ///
    /// # Arguments
    /// * `features` — feature IDs to extract decoder vectors for
    /// * `target_layer` — downstream layer to extract decoders at
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any feature layer or `target_layer` is out
    /// of range, or if `target_layer < feature.layer` for any feature.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Loads each decoder to CPU (up to ~2 GB), extracts independent F32
    /// tensors, then drops the large file before processing the next layer.
    pub fn extract_decoder_vectors(
        &mut self,
        features: &[CltFeatureId],
        target_layer: usize,
    ) -> Result<HashMap<CltFeatureId, Tensor>> {
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // Group by source layer.
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            if fid.layer >= self.config.n_layers {
                return Err(MIError::Config(format!(
                    "feature source layer {} out of range (max {})",
                    fid.layer,
                    self.config.n_layers - 1
                )));
            }
            if target_layer < fid.layer {
                return Err(MIError::Config(format!(
                    "target layer {target_layer} must be >= source layer {}",
                    fid.layer
                )));
            }
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut result: HashMap<CltFeatureId, Tensor> = HashMap::new();
        let n_source_layers = by_source.len();

        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            info!(
                "extract_decoder_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU, extract needed rows as independent tensors.
            let dec_path = self.ensure_decoder_path(*source_layer)?;
            let data = std::fs::read(&dec_path)?;
            let st = SafeTensors::deserialize(&data).map_err(|e| {
                MIError::Config(format!(
                    "failed to deserialize decoder layer {source_layer}: {e}"
                ))
            })?;
            let dec_name = format!("W_dec_{source_layer}");
            let w_dec = tensor_from_view(
                &st.tensor(&dec_name)
                    .map_err(|e| MIError::Config(format!("tensor '{dec_name}' not found: {e}")))?,
                &Device::Cpu,
            )?;

            for &index in indices {
                let fid = CltFeatureId {
                    layer: *source_layer,
                    index,
                };
                if let std::collections::hash_map::Entry::Vacant(e) = result.entry(fid) {
                    // Extract as independent F32 tensor (OOM-safe copy).
                    let view = w_dec.i((index, target_offset))?;
                    let dims = view.dims().to_vec();
                    // PROMOTE: decoder weights are BF16 on disk; extract as F32
                    let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let independent = Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                    e.insert(independent);
                }
            }
            // data, st, w_dec drop here — freeing the large decoder file.
        }

        info!(
            "Extracted {} decoder vectors across {} source layers",
            result.len(),
            n_source_layers
        );

        Ok(result)
    }

    /// Build an attribution graph by scoring features against a direction.
    ///
    /// Convenience wrapper around
    /// [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection)
    /// that returns an [`AttributionGraph`] instead of a raw Vec.
    ///
    /// # Shapes
    /// - `direction`: `[d_model]`
    ///
    /// # Errors
    ///
    /// Same as [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection).
    pub fn build_attribution_graph(
        &mut self,
        direction: &Tensor,
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<AttributionGraph> {
        let scored =
            self.score_features_by_decoder_projection(direction, target_layer, top_k, cosine)?;
        Ok(AttributionGraph {
            target_layer,
            edges: scored
                .into_iter()
                .map(|(feature, score)| AttributionEdge { feature, score })
                .collect(),
        })
    }

    /// Build attribution graphs for multiple directions in a single pass.
    ///
    /// Convenience wrapper around
    /// [`score_features_by_decoder_projection_batch`](Self::score_features_by_decoder_projection_batch)
    /// that returns `Vec<AttributionGraph>`.
    ///
    /// # Shapes
    /// - `directions`: slice of `[d_model]` tensors
    ///
    /// # Errors
    ///
    /// Same as [`score_features_by_decoder_projection_batch`](Self::score_features_by_decoder_projection_batch).
    pub fn build_attribution_graph_batch(
        &mut self,
        directions: &[Tensor],
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<AttributionGraph>> {
        let batch = self.score_features_by_decoder_projection_batch(
            directions,
            target_layer,
            top_k,
            cosine,
        )?;
        Ok(batch
            .into_iter()
            .map(|scored| AttributionGraph {
                target_layer,
                edges: scored
                    .into_iter()
                    .map(|(feature, score)| AttributionEdge { feature, score })
                    .collect(),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert a safetensors `TensorView` to a candle `Tensor`.
///
/// # Shapes
/// - Preserves the original tensor shape from safetensors.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the tensor dtype is not supported (BF16, F16, F32).
/// Returns [`MIError::Model`] on tensor construction failure.
fn tensor_from_view(view: &safetensors::tensor::TensorView<'_>, device: &Device) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    #[allow(clippy::wildcard_enum_match_arm)]
    // EXHAUSTIVE: safetensors exposes many dtypes; CLTs only use float types
    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::F32 => DType::F32,
        other => {
            return Err(MIError::Config(format!(
                "unsupported CLT tensor dtype: {other:?}"
            )));
        }
    };
    let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)?;
    Ok(tensor)
}

/// Parse a value from a simple YAML file by key.
///
/// No `serde_yaml` dependency — uses line-by-line matching.
fn parse_yaml_value(yaml_text: &str, key: &str) -> Option<String> {
    for line in yaml_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key)
            && let Some(rest) = rest.strip_prefix(':')
        {
            let value = rest.trim().trim_matches('"');
            return Some(value.to_owned());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn clt_feature_id_display() {
        let fid = CltFeatureId {
            layer: 5,
            index: 42,
        };
        assert_eq!(fid.to_string(), "L5:42");
    }

    #[test]
    fn clt_feature_id_ordering() {
        let a = CltFeatureId {
            layer: 0,
            index: 10,
        };
        let b = CltFeatureId {
            layer: 0,
            index: 20,
        };
        let c = CltFeatureId { layer: 1, index: 0 };
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn sparse_activations_basics() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let sparse = SparseActivations { features };
        assert_eq!(sparse.len(), 3);
        assert!(!sparse.is_empty());
    }

    #[test]
    fn sparse_activations_truncate() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let mut sparse = SparseActivations { features };
        sparse.truncate(2);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.features[0].0.index, 5);
        assert_eq!(sparse.features[1].0.index, 2);
    }

    #[test]
    fn parse_yaml_value_basic() {
        let yaml = "model_name: \"google/gemma-2-2b\"\nmodel_kind: cross_layer_transcoder\n";
        assert_eq!(
            parse_yaml_value(yaml, "model_name"),
            Some("google/gemma-2-2b".to_owned())
        );
        assert_eq!(
            parse_yaml_value(yaml, "model_kind"),
            Some("cross_layer_transcoder".to_owned())
        );
        assert_eq!(parse_yaml_value(yaml, "missing_key"), None);
    }

    #[test]
    fn encode_synthetic() {
        // Create a small synthetic encoder: 4 features, d_model=8
        let device = Device::Cpu;
        let d_model = 8;
        let n_features = 4;

        // W_enc: [4, 8] — identity-like rows so we can predict output
        #[rustfmt::skip]
        let w_enc_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 0: picks up residual[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 1: picks up residual[1]
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 2: picks up residual[2]
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // feature 3: picks up residual[3]
        ];
        let w_enc = Tensor::from_vec(w_enc_data, (n_features, d_model), &device).unwrap();

        // b_enc: [4] — bias shifts to test ReLU
        let b_enc_data: Vec<f32> = vec![0.0, -0.5, 0.0, -2.0]; // feature 3 will need residual[3] > 2.0
        let b_enc = Tensor::from_vec(b_enc_data, (n_features,), &device).unwrap();

        // Residual: [8] — values: [1.5, 0.3, 0.0, 1.0, ...]
        let residual_data: Vec<f32> = vec![1.5, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let residual = Tensor::from_vec(residual_data, (d_model,), &device).unwrap();

        // Expected pre_acts = W_enc @ residual + b_enc
        // = [1.5, 0.3, 0.0, 1.0] + [0.0, -0.5, 0.0, -2.0]
        // = [1.5, -0.2, 0.0, -1.0]
        // After ReLU: [1.5, 0.0, 0.0, 0.0]
        // Only feature 0 is active with activation 1.5

        // Create a fake loaded encoder
        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
            }),
            steering_cache: HashMap::new(),
        };

        let sparse = clt.encode(&residual, 0).unwrap();
        assert_eq!(sparse.len(), 1, "only feature 0 should be active");
        assert_eq!(sparse.features[0].0.index, 0);
        assert!((sparse.features[0].1 - 1.5).abs() < 1e-5);
    }

    #[test]
    fn encode_wrong_layer_errors() {
        let device = Device::Cpu;
        let w_enc = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let b_enc = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let residual = Tensor::zeros((8,), DType::F32, &device).unwrap();

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![None; 2],
            config: CltConfig {
                n_layers: 2,
                d_model: 8,
                n_features_per_layer: 4,
                n_features_total: 8,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
            }),
            steering_cache: HashMap::new(),
        };

        // Requesting layer 1 when layer 0 is loaded should error.
        let result = clt.encode(&residual, 1);
        assert!(result.is_err());
    }

    #[test]
    fn inject_position() {
        let device = Device::Cpu;
        let d_model = 4;

        // Residual: [1, 3, 4] — batch=1, seq_len=3, d_model=4
        let residual = Tensor::ones((1, 3, d_model), DType::F32, &device).unwrap();

        // Create a CLT with a pre-cached steering vector.
        let fid = CltFeatureId { layer: 0, index: 0 };
        let target_layer = 1;
        let steering_vec =
            Tensor::from_vec(vec![10.0_f32, 20.0, 30.0, 40.0], (d_model,), &device).unwrap();

        let mut steering_cache = HashMap::new();
        steering_cache.insert((fid, target_layer), steering_vec);

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![None; 2],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: 1,
                n_features_total: 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache,
        };

        // Inject at position 1 with strength 1.0
        let result = clt
            .inject(&residual, &[(fid, target_layer)], 1, 1.0)
            .unwrap();

        // Position 0 should be unchanged (all 1.0)
        let pos0: Vec<f32> = result.i((0, 0)).unwrap().to_vec1().unwrap();
        assert_eq!(pos0, vec![1.0, 1.0, 1.0, 1.0]);

        // Position 1 should have the steering vector added (1 + [10, 20, 30, 40])
        let pos1: Vec<f32> = result.i((0, 1)).unwrap().to_vec1().unwrap();
        assert_eq!(pos1, vec![11.0, 21.0, 31.0, 41.0]);

        // Position 2 should be unchanged
        let pos2: Vec<f32> = result.i((0, 2)).unwrap().to_vec1().unwrap();
        assert_eq!(pos2, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn prepare_hook_injection_creates_correct_hooks() {
        use crate::hooks::HookPoint;

        let device = Device::Cpu;
        let d_model = 4;

        let fid = CltFeatureId { layer: 0, index: 0 };
        let target_layer = 5;
        let steering_vec =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (d_model,), &device).unwrap();

        let mut steering_cache = HashMap::new();
        steering_cache.insert((fid, target_layer), steering_vec);

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 10],
            decoder_paths: vec![None; 10],
            config: CltConfig {
                n_layers: 10,
                d_model,
                n_features_per_layer: 1,
                n_features_total: 10,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache,
        };

        let hooks = clt
            .prepare_hook_injection(&[(fid, target_layer)], 2, 5, 1.0, &device)
            .unwrap();

        // Should have an intervention at ResidPost(5).
        assert!(hooks.has_intervention_at(&HookPoint::ResidPost(target_layer)));
        // Should NOT have interventions at other layers.
        assert!(!hooks.has_intervention_at(&HookPoint::ResidPost(0)));
        assert!(!hooks.has_intervention_at(&HookPoint::ResidPost(4)));
    }

    // ====================================================================
    // Attribution graph — pure type tests
    // ====================================================================

    #[test]
    fn attribution_edge_basics() {
        let edge = AttributionEdge {
            feature: CltFeatureId {
                layer: 3,
                index: 42,
            },
            score: 0.75,
        };
        assert_eq!(edge.feature.layer, 3);
        assert_eq!(edge.feature.index, 42);
        assert!((edge.score - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn attribution_graph_empty() {
        let graph = AttributionGraph {
            target_layer: 5,
            edges: Vec::new(),
        };
        assert_eq!(graph.target_layer(), 5);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(graph.features().is_empty());
        assert!(graph.into_edges().is_empty());
    }

    #[test]
    fn attribution_graph_top_k() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 0 },
                score: 5.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 1 },
                score: 3.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 0 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 1 },
                score: -1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 0 },
                score: -4.0,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 3,
            edges,
        };

        assert_eq!(graph.len(), 5);

        let top3 = graph.top_k(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3.target_layer(), 3);
        assert!((top3.edges()[0].score - 5.0).abs() < f32::EPSILON);
        assert!((top3.edges()[1].score - 3.0).abs() < f32::EPSILON);
        assert!((top3.edges()[2].score - 1.0).abs() < f32::EPSILON);

        // top_k larger than graph size returns all edges.
        let top10 = graph.top_k(10);
        assert_eq!(top10.len(), 5);
    }

    #[test]
    fn attribution_graph_threshold() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 0 },
                score: 5.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 1 },
                score: 3.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 0 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 1 },
                score: -1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 0 },
                score: -4.0,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 3,
            edges,
        };

        // Threshold at 2.0 keeps |score| >= 2.0: 5.0, 3.0, -4.0
        let pruned = graph.threshold(2.0);
        assert_eq!(pruned.len(), 3);
        assert!((pruned.edges()[0].score - 5.0).abs() < f32::EPSILON);
        assert!((pruned.edges()[1].score - 3.0).abs() < f32::EPSILON);
        assert!((pruned.edges()[2].score - -4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn attribution_graph_features() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 7 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 3 },
                score: 0.5,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 5,
            edges,
        };

        let features = graph.features();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0], CltFeatureId { layer: 2, index: 7 });
        assert_eq!(features[1], CltFeatureId { layer: 0, index: 3 });
    }

    // ====================================================================
    // Attribution graph — synthetic decoder file tests
    // ====================================================================

    /// Create a synthetic decoder safetensors file and return its path.
    fn create_synthetic_decoder(
        dir: &std::path::Path,
        layer: usize,
        n_features: usize,
        n_target_layers: usize,
        d_model: usize,
        values: &[f32],
    ) -> PathBuf {
        assert_eq!(values.len(), n_features * n_target_layers * d_model);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let name = format!("W_dec_{layer}");
        let shape = vec![n_features, n_target_layers, d_model];
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &bytes).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(name, view);
        let serialized = safetensors::serialize(&tensors, &None).unwrap();
        let path = dir.join(format!("W_dec_{layer}.safetensors"));
        std::fs::write(&path, serialized).unwrap();
        path
    }

    #[test]
    fn score_decoder_projection_synthetic() {
        // 2 layers, 4 features/layer, d_model=4.
        // Layer 0 can decode to layers 0 and 1. Layer 1 can decode to layer 1.
        // Target layer = 1.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 4;

        // W_dec_0: [4 features, 2 target_layers, 4 d_model]
        // Feature 0, offset 1 (target layer 1): [1, 0, 0, 0]
        // Feature 1, offset 1: [0, 1, 0, 0]
        // Feature 2, offset 1: [0, 0, 1, 0]
        // Feature 3, offset 1: [0, 0, 0, 1]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,
            // feature 1
            0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,
            // feature 2
            0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,
            // feature 3
            0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        // W_dec_1: [4 features, 1 target_layer, 4 d_model]
        // Feature 0, offset 0 (target layer 1): [2, 0, 0, 0]  (strong on dim 0)
        // Feature 1: [0, 0, 0, 0]
        // Feature 2: [0, 0, 0, 0]
        // Feature 3: [0, 3, 0, 0]  (strong on dim 1)
        #[rustfmt::skip]
        let dec1_values: Vec<f32> = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
        ];
        let path1 = create_synthetic_decoder(dir.path(), 1, n_features, 1, d_model, &dec1_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![Some(path0), Some(path1)],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // Direction: [1, 0, 0, 0] — should pick up L0:0 (score=1) and L1:0 (score=2).
        let direction =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let scores = clt
            .score_features_by_decoder_projection(&direction, 1, 10, false)
            .unwrap();

        // Top scorer should be L1:0 (score=2), then L0:0 (score=1).
        assert!(scores.len() >= 2, "expected at least 2 non-zero scores");
        assert_eq!(scores[0].0, CltFeatureId { layer: 1, index: 0 });
        assert!((scores[0].1 - 2.0).abs() < 1e-5);
        assert_eq!(scores[1].0, CltFeatureId { layer: 0, index: 0 });
        assert!((scores[1].1 - 1.0).abs() < 1e-5);

        // Direction: [0, 1, 0, 0] — should pick up L1:3 (score=3) and L0:1 (score=1).
        let direction2 =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let scores2 = clt
            .score_features_by_decoder_projection(&direction2, 1, 10, false)
            .unwrap();

        assert_eq!(scores2[0].0, CltFeatureId { layer: 1, index: 3 });
        assert!((scores2[0].1 - 3.0).abs() < 1e-5);
        assert_eq!(scores2[1].0, CltFeatureId { layer: 0, index: 1 });
        assert!((scores2[1].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_decoder_projection_cosine_synthetic() {
        // Same setup: verify cosine normalization.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // W_dec_0: [2 features, 1 target_layer, 4 d_model]
        // Feature 0: [3, 0, 0, 0]  (length 3, aligned with [1,0,0,0])
        // Feature 1: [1, 1, 0, 0]  (length sqrt(2), partially aligned)
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            3.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let direction =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        // Dot product: feature 0 = 3.0, feature 1 = 1.0.
        let dot_scores = clt
            .score_features_by_decoder_projection(&direction, 0, 10, false)
            .unwrap();
        assert!((dot_scores[0].1 - 3.0).abs() < 1e-5);
        assert!((dot_scores[1].1 - 1.0).abs() < 1e-5);

        // Cosine: feature 0 = 1.0 (perfectly aligned), feature 1 = 1/sqrt(2) ≈ 0.707.
        let cos_scores = clt
            .score_features_by_decoder_projection(&direction, 0, 10, true)
            .unwrap();
        assert!(
            (cos_scores[0].1 - 1.0).abs() < 1e-4,
            "expected ~1.0, got {}",
            cos_scores[0].1
        );
        let expected_cos = 1.0 / 2.0_f32.sqrt();
        assert!(
            (cos_scores[1].1 - expected_cos).abs() < 1e-4,
            "expected ~{expected_cos}, got {}",
            cos_scores[1].1
        );
    }

    #[test]
    fn score_decoder_projection_batch_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // W_dec_0: feature 0 = [1,0,0,0], feature 1 = [0,1,0,0]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // Two directions: [1,0,0,0] and [0,1,0,0].
        let dir0 =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();
        let dir1 =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let batch = clt
            .score_features_by_decoder_projection_batch(&[dir0, dir1], 0, 10, false)
            .unwrap();

        assert_eq!(batch.len(), 2);

        // Direction 0 should score feature 0 highest.
        assert_eq!(batch[0][0].0, CltFeatureId { layer: 0, index: 0 });
        assert!((batch[0][0].1 - 1.0).abs() < 1e-5);

        // Direction 1 should score feature 1 highest.
        assert_eq!(batch[1][0].0, CltFeatureId { layer: 0, index: 1 });
        assert!((batch[1][0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn extract_decoder_vectors_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 3;

        // W_dec_0: [3 features, 2 target_layers, 4 d_model]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            // feature 1
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
            // feature 2
            17.0, 18.0, 19.0, 20.0,  21.0, 22.0, 23.0, 24.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![Some(path0), None],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let features = vec![
            CltFeatureId { layer: 0, index: 0 },
            CltFeatureId { layer: 0, index: 2 },
        ];

        // Extract at target_layer=1 (offset 1 for source layer 0).
        let vectors = clt.extract_decoder_vectors(&features, 1).unwrap();
        assert_eq!(vectors.len(), 2);

        // Feature 0, offset 1: [5, 6, 7, 8]
        let v0: Vec<f32> = vectors[&CltFeatureId { layer: 0, index: 0 }]
            .to_vec1()
            .unwrap();
        assert_eq!(v0, vec![5.0, 6.0, 7.0, 8.0]);

        // Feature 2, offset 1: [21, 22, 23, 24]
        let v2: Vec<f32> = vectors[&CltFeatureId { layer: 0, index: 2 }]
            .to_vec1()
            .unwrap();
        assert_eq!(v2, vec![21.0, 22.0, 23.0, 24.0]);
    }

    #[test]
    fn build_attribution_graph_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let direction =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let graph = clt
            .build_attribution_graph(&direction, 0, 10, false)
            .unwrap();

        assert_eq!(graph.target_layer(), 0);
        assert!(!graph.is_empty());
        // Feature 1 has score 2.0, feature 0 has score 0.0.
        assert_eq!(
            graph.edges()[0].feature,
            CltFeatureId { layer: 0, index: 1 }
        );
        assert!((graph.edges()[0].score - 2.0).abs() < 1e-5);

        // Pruning: threshold at 1.0 should keep only feature 1.
        let pruned = graph.threshold(1.0);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned.features()[0], CltFeatureId { layer: 0, index: 1 });
    }
}
