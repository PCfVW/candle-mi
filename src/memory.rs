// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process and GPU memory reporting.
//!
//! Provides [`MemorySnapshot`] to capture current RAM and VRAM usage,
//! and [`MemoryReport`] to measure deltas between two snapshots.
//!
//! # Platform support
//!
//! | Metric | Windows | Linux |
//! |--------|---------|-------|
//! | RAM (RSS) | `K32GetProcessMemoryInfo` (per-process, exact) | `/proc/self/status` `VmRSS` (per-process, exact) |
//! | VRAM | `nvidia-smi` (device-wide) | `nvidia-smi` (device-wide) |
//!
//! VRAM measurement is device-wide (not per-process) because CUDA does not
//! expose per-process memory accounting. On single-user development machines
//! the delta between two snapshots is effectively per-process.
//!
//! # Feature gate
//!
//! This module requires `features = ["memory"]`. The `memory` feature relaxes
//! `#![forbid(unsafe_code)]` to `#![deny(unsafe_code)]` for the Windows FFI
//! call to `K32GetProcessMemoryInfo`. On Linux, no unsafe code is used.

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
    /// GPU memory used on the active device in bytes.
    /// `None` if no GPU is present or measurement failed.
    pub vram_bytes: Option<u64>,
    /// Total GPU memory on the active device in bytes.
    /// `None` if no GPU is present or measurement failed.
    pub vram_total_bytes: Option<u64>,
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
    /// `device` is CUDA, using `nvidia-smi`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Memory`] if the RAM query fails (platform API error).
    /// VRAM measurement failures are non-fatal — `vram_bytes` is set to `None`.
    pub fn now(device: &candle_core::Device) -> Result<Self> {
        let ram_bytes = process_rss()?;
        let (vram_bytes, vram_total_bytes) = if device.is_cuda() {
            gpu_memory_used()
        } else {
            (None, None)
        };
        Ok(Self {
            ram_bytes,
            vram_bytes,
            vram_total_bytes,
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
            print!("  |  VRAM {vram:+.0} MB");
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
            println!(
                "  {label}: VRAM {before:.0} MB → {after:.0} MB ({:+.0} MB{total})",
                after - before,
            );
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
// VRAM measurement — nvidia-smi (device-wide)
// ---------------------------------------------------------------------------

/// Query GPU memory via `nvidia-smi`.
///
/// Returns `(Some(used_bytes), Some(total_bytes))` on success,
/// or `(None, None)` if `nvidia-smi` is not available or fails.
///
/// This measures **device-wide** VRAM, not per-process. The delta between
/// two snapshots is accurate on single-user machines.
fn gpu_memory_used() -> (Option<u64>, Option<u64>) {
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
    }

    #[test]
    fn report_delta_positive_for_allocation() {
        let before = MemorySnapshot {
            ram_bytes: 100 * 1_048_576, // 100 MB
            vram_bytes: Some(500 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
        };
        let after = MemorySnapshot {
            ram_bytes: 200 * 1_048_576, // 200 MB
            vram_bytes: Some(1_000 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
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
        };
        let after = MemorySnapshot {
            ram_bytes: 200,
            vram_bytes: None,
            vram_total_bytes: None,
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
        };
        assert!((snap.ram_mb() - 1.0).abs() < 0.001);
    }
}
