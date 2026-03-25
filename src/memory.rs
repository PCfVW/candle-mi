// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process and GPU memory reporting.
//!
//! Provides [`MemorySnapshot`] to capture current RAM and VRAM usage,
//! and [`MemoryReport`] to measure deltas between two snapshots.
//!
//! # VRAM measurement strategy
//!
//! VRAM is measured using a three-tier approach:
//!
//! 1. **Windows primary — DXGI** (per-process): Uses
//!    `IDXGIAdapter3::QueryVideoMemoryInfo` (DXGI 1.4, Windows 10+) to get
//!    true per-process GPU memory. This is the only reliable method on Windows
//!    because WDDM means the Windows kernel manages GPU memory, not the NVIDIA
//!    driver — so NVML returns `NOT_AVAILABLE` for per-process queries.
//! 2. **Linux primary — NVML** (per-process): Dynamically loads
//!    `libnvidia-ml.so.1` via `libloading` and calls
//!    `nvmlDeviceGetComputeRunningProcesses` for true per-process GPU memory.
//! 3. **Fallback — `nvidia-smi`** (device-wide): If both DXGI and NVML fail,
//!    spawns `nvidia-smi` as a subprocess. Reports device-wide VRAM; the delta
//!    between two snapshots is accurate on single-user machines.
//!
//! # Platform support
//!
//! | Metric | Windows | Linux |
//! |--------|---------|-------|
//! | RAM (RSS) | `K32GetProcessMemoryInfo` (per-process, exact) | `/proc/self/status` `VmRSS` (per-process, exact) |
//! | VRAM (DXGI) | `IDXGIAdapter3` (per-process, exact) | N/A |
//! | VRAM (NVML) | `NOT_AVAILABLE` under WDDM | `libnvidia-ml.so.1` (per-process, exact) |
//! | VRAM (fallback) | `nvidia-smi` (device-wide) | `nvidia-smi` (device-wide) |
//!
//! # Feature gates
//!
//! - **`memory`**: Enables this module. Relaxes `#![forbid(unsafe_code)]` to
//!   `#![deny(unsafe_code)]` for the Windows FFI calls (`K32GetProcessMemoryInfo`,
//!   DXGI COM calls) and for NVML dynamic symbol loading via `libloading`.
//!   On Linux RAM measurement, no unsafe code is used.
//! - **`memory-debug`** (implies `memory`): Prints raw DXGI query results and
//!   per-chunk VRAM measurements to stderr for diagnosing GPU memory issues.

use crate::{MIError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Memory snapshot at a point in time.
///
/// Captures process RAM (resident set size) and optionally GPU VRAM.
/// Use [`MemorySnapshot::now`] to take a measurement, and
/// [`MemoryReport::new`] to compute deltas between two snapshots.
///
/// # Example
///
/// ```no_run
/// use candle_mi::MemorySnapshot;
///
/// let before = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// // ... load a model ...
/// let after = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// let report = candle_mi::MemoryReport::new(before, after);
/// println!("RAM delta: {:+.1} MB", report.ram_delta_mb());
/// # Ok::<(), candle_mi::MIError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Process resident set size (working set on Windows) in bytes.
    pub ram_bytes: u64,
    /// GPU memory used in bytes.
    /// Per-process when measured via DXGI/NVML, device-wide when via `nvidia-smi` fallback.
    /// `None` if no GPU is present or measurement failed.
    pub vram_bytes: Option<u64>,
    /// Total GPU memory on the active device in bytes.
    /// `None` if no GPU is present or measurement failed.
    pub vram_total_bytes: Option<u64>,
    /// Whether the VRAM measurement is per-process (`true`) or device-wide (`false`).
    /// `None` if no VRAM data is available.
    pub vram_per_process: Option<bool>,
    /// GPU adapter name (e.g., `NVIDIA GeForce RTX 5060 Ti`).
    /// `None` if not available (non-DXGI path or no GPU).
    pub gpu_name: Option<String>,
}

/// Memory delta between two snapshots.
///
/// Computed from a `before` and `after` [`MemorySnapshot`].
/// Positive deltas mean memory increased; negative means freed.
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Snapshot taken before the operation.
    pub before: MemorySnapshot,
    /// Snapshot taken after the operation.
    pub after: MemorySnapshot,
}

impl MemorySnapshot {
    /// Capture current memory state.
    ///
    /// RAM is always measured (per-process RSS). VRAM is measured only if
    /// `device` is CUDA — first via DXGI (Windows, per-process), then NVML
    /// (Linux, per-process), falling back to `nvidia-smi` (device-wide).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Memory`] if the RAM query fails (platform API error).
    /// VRAM measurement failures are non-fatal — `vram_bytes` is set to `None`.
    pub fn now(device: &candle_core::Device) -> Result<Self> {
        let ram_bytes = process_rss()?;
        let (vram_bytes, vram_total_bytes, per_process, gpu_name) = if device.is_cuda() {
            gpu_memory_used()
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            ram_bytes,
            vram_bytes,
            vram_total_bytes,
            vram_per_process: per_process,
            gpu_name,
        })
    }

    /// Format RAM usage as megabytes.
    #[must_use]
    pub fn ram_mb(&self) -> f64 {
        // CAST: u64 → f64, value is memory in bytes — fits in f64 mantissa
        // for any realistic process size (< 2^53 bytes = 8 PB)
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let mb = self.ram_bytes as f64 / 1_048_576.0;
        mb
    }

    /// Format VRAM usage as megabytes, if available.
    #[must_use]
    pub fn vram_mb(&self) -> Option<f64> {
        // CAST: u64 → f64, same justification as ram_mb
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        self.vram_bytes.map(|b| b as f64 / 1_048_576.0)
    }
}

/// Synchronize the CUDA device and trim its memory pool.
///
/// On a CUDA device this:
/// 1. Calls `cuCtxSynchronize` so all pending async frees complete.
/// 2. Calls `cuMemPoolTrimTo(pool, 0)` to release all unused reserved
///    VRAM back to the device.
///
/// cudarc's stream-ordered allocator (`malloc_async` / `free_async`)
/// keeps freed blocks in a pool for reuse. Over many forward passes
/// with varying tensor sizes the pool grows monotonically — DXGI and
/// `nvidia-smi` report this reserved memory as "in use", eventually
/// causing OOM even though no live tensors need it.
///
/// This function is a no-op on CPU and on Metal.
///
/// # Example
///
/// ```no_run
/// # use candle_mi::sync_and_trim_gpu;
/// # let device = candle_core::Device::Cpu;
/// // After dropping all GPU tensors from a forward pass:
/// sync_and_trim_gpu(&device);
/// ```
pub fn sync_and_trim_gpu(device: &candle_core::Device) {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(cuda_dev) = device {
        use candle_core::backend::BackendDevice;
        // Synchronize so all pending async frees complete.
        let _ = cuda_dev.synchronize();

        // Trim the default memory pool to release all unused reserved VRAM.
        // SAFETY: cuDeviceGetDefaultMemPool and cuMemPoolTrimTo are
        // documented CUDA driver APIs for pool management. The CUdevice
        // handle comes from candle's CudaContext (valid after synchronize).
        // cuMemPoolTrimTo(pool, 0) releases all unused memory — it cannot
        // free memory that is still in use by live tensors.
        #[allow(unsafe_code)]
        {
            use candle_core::cuda_backend::cudarc::driver::sys;

            let stream = cuda_dev.cuda_stream();
            // Allocate a zero-length slice just to access the CudaContext
            // (CudaStream.ctx is pub(crate), but CudaSlice.context() is pub).
            if let Ok(probe) = stream.null::<u8>() {
                let ctx = probe.context();
                let cu_device = ctx.cu_device();
                unsafe {
                    let mut pool = std::mem::zeroed();
                    let rc = sys::cuDeviceGetDefaultMemPool(&raw mut pool, cu_device);
                    if rc == sys::CUresult::CUDA_SUCCESS {
                        let _ = sys::cuMemPoolTrimTo(pool, 0);
                    }
                }
            }
        }
    }

    // Suppress unused-variable warning on non-CUDA builds.
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

impl MemoryReport {
    /// Create a report from two snapshots.
    #[must_use]
    pub const fn new(before: MemorySnapshot, after: MemorySnapshot) -> Self {
        Self { before, after }
    }

    /// RAM delta in megabytes (positive = increased).
    #[must_use]
    pub fn ram_delta_mb(&self) -> f64 {
        self.after.ram_mb() - self.before.ram_mb()
    }

    /// VRAM delta in megabytes (positive = increased).
    /// Returns `None` if either snapshot lacks VRAM data.
    #[must_use]
    pub fn vram_delta_mb(&self) -> Option<f64> {
        match (self.after.vram_mb(), self.before.vram_mb()) {
            (Some(after), Some(before)) => Some(after - before),
            (Some(_) | None, None) | (None, Some(_)) => None,
        }
    }

    /// Print a one-line summary of the delta.
    pub fn print_delta(&self, label: &str) {
        let ram = self.ram_delta_mb();
        print!("  {label}: RAM {ram:+.0} MB");
        if let Some(vram) = self.vram_delta_mb() {
            let qualifier = self.vram_qualifier();
            print!("  |  VRAM {vram:+.0} MB{qualifier}");
        }
        println!();
    }

    /// Print a two-line summary showing before → after for both RAM and VRAM.
    pub fn print_before_after(&self, label: &str) {
        println!(
            "  {label}: RAM {:.0} MB → {:.0} MB ({:+.0} MB)",
            self.before.ram_mb(),
            self.after.ram_mb(),
            self.ram_delta_mb(),
        );
        if let (Some(before), Some(after)) = (self.before.vram_mb(), self.after.vram_mb()) {
            // CAST: u64 → f64, same justification as ram_mb
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let total = self.after.vram_total_bytes.map_or(String::new(), |t| {
                format!(" / {:.0} MB", t as f64 / 1_048_576.0)
            });
            let qualifier = self.vram_qualifier();
            let gpu = self
                .after
                .gpu_name
                .as_deref()
                .map_or(String::new(), |name| format!(" [{name}]"));
            println!(
                "  {label}: VRAM {before:.0} MB → {after:.0} MB ({:+.0} MB{total}){qualifier}{gpu}",
                after - before,
            );
        }
    }

    /// Return a short qualifier string indicating VRAM measurement quality.
    #[must_use]
    const fn vram_qualifier(&self) -> &'static str {
        match self.after.vram_per_process {
            Some(true) => " [per-process]",
            Some(false) => " [device-wide]",
            None => "",
        }
    }
}

// ---------------------------------------------------------------------------
// RAM measurement — per-process RSS
// ---------------------------------------------------------------------------

/// Query the current process's resident set size (RSS) in bytes.
///
/// # Platform
///
/// - **Windows**: `K32GetProcessMemoryInfo` → `WorkingSetSize` (exact, per-process).
/// - **Linux**: `/proc/self/status` → `VmRSS` (exact, per-process, no unsafe).
///
/// # Errors
///
/// Returns [`MIError::Memory`] if the platform API call fails.
fn process_rss() -> Result<u64> {
    #[cfg(target_os = "windows")]
    {
        windows_rss()
    }
    #[cfg(target_os = "linux")]
    {
        linux_rss()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err(MIError::Memory(
            "RAM measurement not supported on this platform".into(),
        ))
    }
}

// -- Windows ----------------------------------------------------------------

/// Windows FFI types and functions for `K32GetProcessMemoryInfo`.
#[cfg(target_os = "windows")]
mod win_ffi {
    /// `PROCESS_MEMORY_COUNTERS` structure from the Windows API.
    ///
    /// See: <https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters>
    #[repr(C)]
    pub(super) struct ProcessMemoryCounters {
        /// Size of this structure in bytes.
        pub cb: u32,
        /// Number of page faults.
        pub page_fault_count: u32,
        /// Peak working set size in bytes.
        pub peak_working_set_size: usize,
        /// Current working set size in bytes (= RSS).
        pub working_set_size: usize,
        /// Peak paged pool usage in bytes.
        pub quota_peak_paged_pool_usage: usize,
        /// Current paged pool usage in bytes.
        pub quota_paged_pool_usage: usize,
        /// Peak non-paged pool usage in bytes.
        pub quota_peak_non_paged_pool_usage: usize,
        /// Current non-paged pool usage in bytes.
        pub quota_non_paged_pool_usage: usize,
        /// Current pagefile usage in bytes.
        pub pagefile_usage: usize,
        /// Peak pagefile usage in bytes.
        pub peak_pagefile_usage: usize,
    }

    // SAFETY: These are stable Windows API functions with well-defined ABI.
    // GetCurrentProcess always returns a valid pseudo-handle.
    // K32GetProcessMemoryInfo writes to caller-provided memory of known size.
    #[allow(unsafe_code)]
    unsafe extern "system" {
        /// Returns a pseudo-handle to the current process (always valid, never null).
        pub(super) safe fn GetCurrentProcess() -> isize;

        /// Retrieves memory usage information for the specified process.
        pub(super) unsafe fn K32GetProcessMemoryInfo(
            process: isize,
            ppsmem_counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
}

/// Query RSS on Windows via `K32GetProcessMemoryInfo`.
#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
fn windows_rss() -> Result<u64> {
    let mut counters = win_ffi::ProcessMemoryCounters {
        cb: 0,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // CAST: usize → u32, struct size is 80 bytes on x64 — fits in u32
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let cb = std::mem::size_of::<win_ffi::ProcessMemoryCounters>() as u32;
    counters.cb = cb;

    let handle = win_ffi::GetCurrentProcess();

    // SAFETY: K32GetProcessMemoryInfo writes into the stack-allocated
    // `counters` struct, which is correctly sized (cb field set to struct
    // size). The process handle from GetCurrentProcess is a pseudo-handle
    // that is always valid for the lifetime of the process.
    let ok = unsafe { win_ffi::K32GetProcessMemoryInfo(handle, &raw mut counters, cb) };

    if ok != 0 {
        // CAST: usize → u64, working set size in bytes — always fits
        #[allow(clippy::as_conversions)]
        let rss = counters.working_set_size as u64;
        Ok(rss)
    } else {
        Err(MIError::Memory("K32GetProcessMemoryInfo failed".into()))
    }
}

// -- Linux ------------------------------------------------------------------

/// Query RSS on Linux via `/proc/self/status`.
#[cfg(target_os = "linux")]
fn linux_rss() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| MIError::Memory(format!("failed to read /proc/self/status: {e}")))?;

    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.trim().trim_end_matches(" kB").trim();
            let kb: u64 = kb_str.parse().map_err(|e| {
                MIError::Memory(format!("failed to parse VmRSS value '{kb_str}': {e}"))
            })?;
            return Ok(kb * 1024);
        }
    }

    Err(MIError::Memory(
        "VmRSS not found in /proc/self/status".into(),
    ))
}

// ---------------------------------------------------------------------------
// VRAM measurement — DXGI (Windows), NVML, nvidia-smi fallback
// ---------------------------------------------------------------------------

/// Result of a GPU memory query: `(used_bytes, total_bytes, per_process, gpu_name)`.
type GpuMemoryResult = (Option<u64>, Option<u64>, Option<bool>, Option<String>);

/// NVML shared library path (stable across driver versions).
#[cfg(target_os = "linux")]
const NVML_LIB_PATH: &str = "libnvidia-ml.so.1";

/// NVML shared library path (stable across driver versions).
#[cfg(target_os = "windows")]
const NVML_LIB_PATH: &str = "nvml.dll";

/// NVML return code: success.
const NVML_SUCCESS: u32 = 0;

/// NVML return code: buffer too small (need to retry with larger buffer).
const NVML_ERROR_INSUFFICIENT_SIZE: u32 = 7;

/// Maximum number of processes to query from NVML in a single call.
/// 64 is generous — most machines have fewer than 10 GPU processes.
const NVML_MAX_PROCESSES: usize = 64;

/// Per-process GPU memory info returned by NVML.
///
/// Matches the C struct `nvmlProcessInfo_v2_t` (24 bytes) used by both
/// `nvmlDeviceGetComputeRunningProcesses_v2` and `_v3` (the `_v3` suffix
/// is a function version, not a struct version).
/// See: <https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__v2__t.html>
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmlProcessInfo {
    /// Process ID.
    pid: u32,
    /// GPU memory used by this process in bytes.
    /// `u64::MAX` (`0xFFFF_FFFF_FFFF_FFFF`) means "not available".
    used_gpu_memory: u64,
    /// GPU instance ID (MIG). Unused in non-MIG mode.
    gpu_instance_id: u32,
    /// Compute instance ID (MIG). Unused in non-MIG mode.
    compute_instance_id: u32,
}

/// NVML memory info for a device.
///
/// Matches the C struct `nvmlMemory_t` from the NVML API.
/// See: <https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html>
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvmlMemoryInfo {
    /// Total GPU memory in bytes.
    total: u64,
    /// Free GPU memory in bytes.
    free: u64,
    /// Used GPU memory in bytes.
    used: u64,
}

/// Opaque NVML device handle.
type NvmlDevice = *mut std::ffi::c_void;

/// Function signature: `nvmlInit_v2() -> nvmlReturn_t`.
type NvmlInitFn = unsafe extern "C" fn() -> u32;

/// Function signature: `nvmlShutdown() -> nvmlReturn_t`.
type NvmlShutdownFn = unsafe extern "C" fn() -> u32;

/// Function signature: `nvmlDeviceGetHandleByIndex_v2(index, *mut device) -> nvmlReturn_t`.
type NvmlDeviceGetHandleByIndexFn = unsafe extern "C" fn(u32, *mut NvmlDevice) -> u32;

/// Function signature: `nvmlDeviceGetMemoryInfo(device, *mut memory) -> nvmlReturn_t`.
type NvmlDeviceGetMemoryInfoFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemoryInfo) -> u32;

/// Function signature:
/// `nvmlDeviceGetComputeRunningProcesses_v3(device, *mut count, *mut infos) -> nvmlReturn_t`.
type NvmlDeviceGetComputeRunningProcessesFn =
    unsafe extern "C" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> u32;

/// Query GPU memory — DXGI (Windows), NVML, or `nvidia-smi` fallback.
///
/// Returns `(used_bytes, total_bytes, per_process, gpu_name)`:
/// - `per_process = Some(true)` when per-process query succeeded (DXGI or NVML).
/// - `per_process = Some(false)` when falling back to `nvidia-smi` (device-wide).
/// - `gpu_name` is set when DXGI provides the adapter description.
/// - All `None` if all methods fail.
fn gpu_memory_used() -> GpuMemoryResult {
    // Windows: try DXGI first (per-process, works under WDDM)
    #[cfg(windows)]
    if let Some(result) = dxgi_query_process_vram() {
        return result;
    }

    // Try NVML (per-process on Linux, NOT_AVAILABLE on Windows WDDM)
    if let Some(result) = nvml_query_process_vram() {
        let (used, total, per_process) = result;
        return (used, total, per_process, None);
    }

    // Fallback to nvidia-smi (device-wide)
    let (used, total) = nvidia_smi_query();
    if used.is_some() {
        (used, total, Some(false), None)
    } else {
        (None, None, None, None)
    }
}

/// Attempt to query per-process VRAM via NVML.
///
/// Returns `None` if NVML cannot be loaded or any API call fails,
/// signaling the caller to try the fallback path.
#[allow(unsafe_code)]
fn nvml_query_process_vram() -> Option<(Option<u64>, Option<u64>, Option<bool>)> {
    // SAFETY: libloading::Library::new dynamically loads a shared library.
    // The NVML library is a stable NVIDIA driver component with a well-defined
    // C ABI. We load it, call functions, and unload it within this scope.
    let lib = unsafe { libloading::Library::new(NVML_LIB_PATH) }.ok()?;

    // SAFETY: Loading function symbols from the NVML library. Each symbol
    // name matches the documented NVML C API exactly. The function signatures
    // (type aliases above) match the NVML header definitions.
    let init: libloading::Symbol<'_, NvmlInitFn> = unsafe { lib.get(b"nvmlInit_v2\0") }.ok()?;
    let shutdown: libloading::Symbol<'_, NvmlShutdownFn> =
        unsafe { lib.get(b"nvmlShutdown\0") }.ok()?;
    let get_handle: libloading::Symbol<'_, NvmlDeviceGetHandleByIndexFn> =
        unsafe { lib.get(b"nvmlDeviceGetHandleByIndex_v2\0") }.ok()?;
    let get_memory: libloading::Symbol<'_, NvmlDeviceGetMemoryInfoFn> =
        unsafe { lib.get(b"nvmlDeviceGetMemoryInfo\0") }.ok()?;
    let get_processes: libloading::Symbol<'_, NvmlDeviceGetComputeRunningProcessesFn> =
        unsafe { lib.get(b"nvmlDeviceGetComputeRunningProcesses_v3\0") }.ok()?;

    // Initialize NVML
    // SAFETY: nvmlInit_v2 is safe to call from any thread; it initializes
    // internal NVML state. Returns NVML_SUCCESS (0) on success.
    let ret = unsafe { init() };
    if ret != NVML_SUCCESS {
        return None;
    }

    // Get device handle for GPU 0 (primary GPU)
    let mut device: NvmlDevice = std::ptr::null_mut();
    // SAFETY: nvmlDeviceGetHandleByIndex_v2 writes a valid opaque handle
    // into `device` when it returns NVML_SUCCESS. Index 0 = primary GPU.
    let ret = unsafe { get_handle(0, &raw mut device) };
    if ret != NVML_SUCCESS {
        // SAFETY: nvmlShutdown is always safe after a successful nvmlInit.
        unsafe { shutdown() };
        return None;
    }

    // Get total memory for the device
    let mut mem_info = NvmlMemoryInfo {
        total: 0,
        free: 0,
        used: 0,
    };
    // SAFETY: nvmlDeviceGetMemoryInfo writes into the caller-provided
    // NvmlMemoryInfo struct. The device handle is valid (obtained above).
    let ret = unsafe { get_memory(device, &raw mut mem_info) };
    let total_bytes = if ret == NVML_SUCCESS {
        Some(mem_info.total)
    } else {
        None
    };

    // Get per-process memory
    // CAST: usize → u32, NVML_MAX_PROCESSES is 64 — fits in u32
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let mut count = NVML_MAX_PROCESSES as u32;
    let mut infos = [NvmlProcessInfo {
        pid: 0,
        used_gpu_memory: 0,
        gpu_instance_id: 0,
        compute_instance_id: 0,
    }; NVML_MAX_PROCESSES];

    // SAFETY: nvmlDeviceGetComputeRunningProcesses_v3 fills `infos` with
    // up to `count` entries and updates `count` to the actual number written.
    // The buffer is stack-allocated with NVML_MAX_PROCESSES slots, which is
    // sufficient for typical workloads.
    let ret = unsafe { get_processes(device, &raw mut count, infos.as_mut_ptr()) };

    // SAFETY: nvmlShutdown pairs with nvmlInit; always called before return.
    unsafe { shutdown() };

    if ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE {
        return None;
    }

    // Find our process in the list
    let my_pid = std::process::id();
    // CAST: u32 → usize, count is a small process count — always fits
    #[allow(clippy::as_conversions)]
    let actual_count = count as usize;
    let my_vram = infos
        .get(..actual_count)?
        .iter()
        .find(|info| info.pid == my_pid)
        .map(|info| info.used_gpu_memory);

    // NVML uses u64::MAX as "not available" sentinel — some drivers (e.g., R570
    // on RTX 5060 Ti) return this for all processes. Fall back to nvidia-smi.
    if my_vram == Some(u64::MAX) {
        return None;
    }

    // Sanity check: if per-process VRAM exceeds total device memory, the value
    // is likely garbage. Fall back to nvidia-smi.
    if let (Some(used), Some(total)) = (my_vram, total_bytes)
        && used > total
    {
        return None;
    }

    // Our PID might not be in the list (no active CUDA context yet?) — return None to trigger fallback
    my_vram.map(|used| (Some(used), total_bytes, Some(true)))
}

/// Query GPU memory via `nvidia-smi` (device-wide fallback).
///
/// Returns `(Some(used_bytes), Some(total_bytes))` on success,
/// or `(None, None)` if `nvidia-smi` is not available or fails.
fn nvidia_smi_query() -> (Option<u64>, Option<u64>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return (None, None),
    };

    // BORROW: explicit String::from_utf8_lossy — nvidia-smi output is ASCII
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = match stdout.lines().next() {
        Some(l) => l.trim(),
        None => return (None, None),
    };

    // Format: "1234, 16384" (used MiB, total MiB)
    let mut parts = line.split(',');
    let used_str = match parts.next() {
        Some(s) => s.trim(),
        None => return (None, None),
    };
    let total_str = match parts.next() {
        Some(s) => s.trim(),
        None => return (None, None),
    };

    let used_mb: u64 = match used_str.parse() {
        Ok(v) => v,
        Err(_) => return (None, None),
    };
    let total_mb: u64 = match total_str.parse() {
        Ok(v) => v,
        Err(_) => return (None, None),
    };

    // nvidia-smi reports in MiB — convert to bytes
    (Some(used_mb * 1_048_576), Some(total_mb * 1_048_576))
}

// ---------------------------------------------------------------------------
// DXGI per-process VRAM (Windows only)
// ---------------------------------------------------------------------------

/// Query per-process GPU VRAM via DXGI (`IDXGIAdapter3::QueryVideoMemoryInfo`).
///
/// This is the only reliable way to get per-process GPU memory on Windows
/// (WDDM). NVML returns `NVML_VALUE_NOT_AVAILABLE` under WDDM because the
/// Windows kernel memory manager owns GPU memory, not the NVIDIA driver.
///
/// DXGI 1.4 (Windows 10+) provides `QueryVideoMemoryInfo` which returns:
/// - `CurrentUsage`: per-process GPU memory in bytes (exactly what we want).
/// - `Budget`: OS-assigned memory budget for this process.
///
/// We query `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` (dedicated VRAM on discrete GPUs).
/// Total VRAM comes from `IDXGIAdapter::GetDesc` → `DedicatedVideoMemory`.
///
/// Returns `None` if DXGI is not available or the query fails,
/// signaling the caller to try NVML or nvidia-smi fallback.
#[cfg(windows)]
#[allow(unsafe_code)]
fn dxgi_query_process_vram() -> Option<GpuMemoryResult> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
        IDXGIAdapter, IDXGIAdapter3, IDXGIFactory1,
    };
    use windows::core::Interface;

    // SAFETY: CreateDXGIFactory1 is a well-documented COM factory function.
    // It initializes COM internally if needed. The returned IDXGIFactory1
    // is reference-counted and released automatically when dropped.
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;

    // Enumerate adapters — find the first one with dedicated VRAM > 0
    // (skip software/render-only adapters like Microsoft Basic Render Driver).
    let mut adapter_idx = 0u32;
    loop {
        // SAFETY: EnumAdapters1 returns S_OK with a valid adapter, or
        // DXGI_ERROR_NOT_FOUND when idx is out of range.
        let adapter: IDXGIAdapter = unsafe { factory.EnumAdapters1(adapter_idx) }
            .ok()?
            .cast()
            .ok()?;

        // SAFETY: GetDesc writes a valid DXGI_ADAPTER_DESC into the
        // caller-provided struct. The adapter handle is valid (obtained above).
        let desc = unsafe { adapter.GetDesc() }.ok()?;
        let dedicated_vram = desc.DedicatedVideoMemory;

        if dedicated_vram == 0 {
            adapter_idx += 1;
            continue;
        }

        // Cast to IDXGIAdapter3 for QueryVideoMemoryInfo (DXGI 1.4)
        let adapter3: IDXGIAdapter3 = adapter.cast().ok()?;

        // SAFETY: QueryVideoMemoryInfo fills a DXGI_QUERY_VIDEO_MEMORY_INFO
        // struct with per-process memory stats. Node 0 = primary GPU node.
        // DXGI_MEMORY_SEGMENT_GROUP_LOCAL = dedicated VRAM on discrete GPUs.
        let mut mem_info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &raw mut mem_info)
        }
        .ok()?;

        // CAST: usize → u64, DedicatedVideoMemory is usize on Windows
        #[allow(clippy::as_conversions)]
        let total = dedicated_vram as u64;

        // Trim trailing null characters from the UTF-16 adapter description
        // BORROW: explicit from_utf16_lossy — DXGI Description is a fixed-size UTF-16 array
        let raw_name = String::from_utf16_lossy(&desc.Description);
        // BORROW: to_owned — trim returns a &str slice; we need an owned String
        let gpu_name = raw_name.trim_end_matches('\0').to_owned();

        #[cfg(feature = "memory-debug")]
        eprintln!(
            "[DXGI debug] adapter={gpu_name}, dedicated_vram={total}, \
             current_usage={}, budget={}",
            mem_info.CurrentUsage, mem_info.Budget,
        );

        return Some((
            Some(mem_info.CurrentUsage),
            Some(total),
            Some(true),
            Some(gpu_name),
        ));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_cpu_has_ram() {
        let snap = MemorySnapshot::now(&candle_core::Device::Cpu).unwrap();
        // Process must be using > 0 bytes of RAM
        assert!(snap.ram_bytes > 0, "RAM should be non-zero");
        // CPU device should not have VRAM
        assert!(snap.vram_bytes.is_none(), "CPU should have no VRAM");
        assert!(
            snap.vram_per_process.is_none(),
            "CPU should have no VRAM qualifier"
        );
    }

    #[test]
    fn report_delta_positive_for_allocation() {
        let before = MemorySnapshot {
            ram_bytes: 100 * 1_048_576, // 100 MB
            vram_bytes: Some(500 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
            gpu_name: None,
        };
        let after = MemorySnapshot {
            ram_bytes: 200 * 1_048_576, // 200 MB
            vram_bytes: Some(1_000 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
            gpu_name: None,
        };
        let report = MemoryReport::new(before, after);

        let ram_delta = report.ram_delta_mb();
        assert!(
            (ram_delta - 100.0).abs() < 0.01,
            "RAM delta should be ~100 MB, got {ram_delta}"
        );

        let vram_delta = report.vram_delta_mb().unwrap();
        assert!(
            (vram_delta - 500.0).abs() < 0.01,
            "VRAM delta should be ~500 MB, got {vram_delta}"
        );
    }

    #[test]
    fn report_delta_none_when_no_vram() {
        let before = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
        };
        let after = MemorySnapshot {
            ram_bytes: 200,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
        };
        let report = MemoryReport::new(before, after);
        assert!(report.vram_delta_mb().is_none());
    }

    #[test]
    fn ram_mb_conversion() {
        let snap = MemorySnapshot {
            ram_bytes: 1_048_576, // exactly 1 MB
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
        };
        assert!((snap.ram_mb() - 1.0).abs() < 0.001);
    }

    #[test]
    fn vram_qualifier_per_process() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(true),
            gpu_name: None,
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [per-process]");
    }

    #[test]
    fn vram_qualifier_device_wide() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(false),
            gpu_name: None,
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [device-wide]");
    }
}
