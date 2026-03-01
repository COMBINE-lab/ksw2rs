//! Native Rust port of the KSW2 `ksw_extz2_sse` extension aligner.
//!
//! The original C implementation uses byte-packed anti-diagonal dynamic
//! programming with SSE2/SSE4.1 intrinsics. This Rust port preserves the same
//! algorithmic structure and scoring semantics while staying on stable Rust.
//! The implementation below is intentionally heavily documented because the DP
//! transform is compact but non-obvious.

/// Sentinel used by KSW2 for impossible/uninitialized scores.
pub const KSW_NEG_INF: i32 = -0x4000_0000;

/// Do not emit CIGAR/backtrack matrix.
pub const KSW_EZ_SCORE_ONLY: u32 = 0x01;
/// Prefer right-aligned gaps when tie-breaking.
pub const KSW_EZ_RIGHT: u32 = 0x02;
/// Use full `m x m` scoring matrix instead of match/mismatch fast path.
pub const KSW_EZ_GENERIC_SC: u32 = 0x04;
/// Use approximate max tracking (faster, less exact).
pub const KSW_EZ_APPROX_MAX: u32 = 0x08;
/// Use approximate z-drop tracking.
pub const KSW_EZ_APPROX_DROP: u32 = 0x10;
/// Extension-only mode.
pub const KSW_EZ_EXTZ_ONLY: u32 = 0x40;
/// Keep CIGAR reversed.
pub const KSW_EZ_REV_CIGAR: u32 = 0x80;

/// BAM-like CIGAR op code: match/mismatch.
pub const KSW_CIGAR_MATCH: u32 = 0;
/// BAM-like CIGAR op code: insertion.
pub const KSW_CIGAR_INS: u32 = 1;
/// BAM-like CIGAR op code: deletion.
pub const KSW_CIGAR_DEL: u32 = 2;
/// BAM-like CIGAR op code: reference skip.
pub const KSW_CIGAR_N_SKIP: u32 = 3;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimdBackend {
    Scalar,
    Avx2,
    Sse41,
    Neon,
}

/// Alignment result (shape compatible with `ksw_extz_t`).
#[derive(Debug, Clone, Default)]
pub struct Extz {
    /// Best score seen during extension.
    pub max: u32,
    /// Whether extension stopped due to z-drop.
    pub zdropped: bool,
    /// Query coordinate of `max`.
    pub max_q: i32,
    /// Target coordinate of `max`.
    pub max_t: i32,
    /// Best score that reaches query end.
    pub mqe: i32,
    /// Target coordinate for `mqe`.
    pub mqe_t: i32,
    /// Best score that reaches target end.
    pub mte: i32,
    /// Query coordinate for `mte`.
    pub mte_q: i32,
    /// End-to-end score (or `KSW_NEG_INF` if not reached).
    pub score: i32,
    /// Whether extension-only mode backtracked to query end.
    pub reach_end: bool,
    /// BAM-style packed CIGAR (len<<4 | op).
    pub cigar: Vec<u32>,
}

impl Extz {
    #[inline]
    pub fn reset(&mut self) {
        self.max_q = -1;
        self.max_t = -1;
        self.mqe_t = -1;
        self.mte_q = -1;
        self.max = 0;
        self.score = KSW_NEG_INF;
        self.mqe = KSW_NEG_INF;
        self.mte = KSW_NEG_INF;
        self.zdropped = false;
        self.reach_end = false;
        self.cigar.clear();
    }
}

/// Input bundle for `extz2`.
#[derive(Debug, Clone)]
pub struct Extz2Input<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
    pub m: i8,
    pub mat: &'a [i8],
    pub q: i8,
    pub e: i8,
    pub w: i32,
    pub zdrop: i32,
    pub end_bonus: i32,
    pub flag: u32,
}

/// Reusable scratch buffers for repeated `extz2` calls.
///
/// Reusing one workspace across many alignments avoids repeated heap
/// allocations and improves short/medium input throughput.
///
/// The seven byte-sized DP arrays (u, v, x, y, s, sf, qr) are packed into a
/// single contiguous allocation (`buf`) in the same layout that the C reference
/// uses: `[u | v | x | y | s | sf | qr+16_pad]`.  This gives better cache and
/// TLB behaviour than seven separate `Vec` allocations, and the 16-byte
/// overshoot beyond `qr` means the SIMD score-fill kernels never need a
/// per-chunk bounds check.
#[derive(Debug, Default, Clone)]
pub struct Workspace {
    /// Flat buffer: `6 * tlen_pad + qlen_pad + 16` bytes, zeroed on resize.
    buf: Vec<u8>,
    h: Vec<i32>,
    off: Vec<i32>,
    off_end: Vec<i32>,
    p: Vec<u8>,
    /// Cached layout dimensions set by the most recent `prepare_*` call.
    tlen_pad: usize,
    qlen_pad: usize,
}

impl Workspace {
    /// Prepare the flat buffer for a score-only alignment.
    ///
    /// Layout: `[u | v | x | y | s | sf | qr+16]`, all zero-initialised once.
    /// The trailing 16-byte pad on `qr` (and the implicit pad provided by `qr`
    /// after `sf`) means SIMD score-fill kernels can always load 16 bytes
    /// without a per-chunk bounds check.
    #[inline]
    fn prepare_score_only(&mut self, qlen: usize, tlen: usize, approx: bool) {
        let tlen_pad = tlen.div_ceil(16) * 16;
        let qlen_pad = qlen.div_ceil(16) * 16;
        // 6 tlen_pad-wide arrays (u,v,x,y,s,sf) plus qr with a 16-byte pad.
        let total = 6 * tlen_pad + qlen_pad + 16;
        self.buf.clear();
        self.buf.resize(total, 0u8);
        self.tlen_pad = tlen_pad;
        self.qlen_pad = qlen_pad;
        if !approx {
            self.h.clear();
            self.h.resize(tlen, KSW_NEG_INF);
        } else {
            self.h.clear();
        }
    }

    #[inline]
    fn prepare_traceback(
        &mut self,
        qlen: usize,
        tlen: usize,
        approx: bool,
        rows: usize,
        n_col: usize,
    ) {
        self.prepare_score_only(qlen, tlen, approx);
        self.off.clear();
        self.off.resize(rows, 0);
        self.off_end.clear();
        self.off_end.resize(rows, 0);
        let p_len = rows * n_col;
        // The traceback matrix is written per active row span before reads.
        if self.p.capacity() < p_len {
            self.p.reserve(p_len - self.p.capacity());
        }
        // SAFETY: readers only access row spans written by DP row kernels.
        unsafe { self.p.set_len(p_len) };
    }
}

/// Public entry point. The current implementation is stable-Rust and platform
/// neutral; it preserves KSW2 semantics and dispatches to architecture-specific
/// SIMD score-preparation kernels where available.
pub fn extz2(input: &Extz2Input<'_>, ez: &mut Extz) {
    let mut ws = Workspace::default();
    extz2_dispatch(input, ez, &mut ws);
}

/// Public entry point that reuses caller-provided scratch buffers.
#[inline]
pub fn extz2_with_workspace(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    extz2_dispatch(input, ez, ws);
}

/// Force the scalar backend.
///
/// This is primarily useful for benchmarking and differential debugging.
pub fn extz2_scalar(input: &Extz2Input<'_>, ez: &mut Extz) {
    let mut ws = Workspace::default();
    core::extz2_core::<core::ScalarOps>(input, ez, &mut ws);
}

/// Force the scalar backend while reusing caller-provided workspace.
#[inline]
pub fn extz2_scalar_with_workspace(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    core::extz2_core::<core::ScalarOps>(input, ez, ws);
}

#[inline]
fn extz2_dispatch(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // Safe because runtime detection guarantees AVX2 availability.
            unsafe {
                extz2_avx2(input, ez, ws);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            // Safe because runtime detection guarantees SSE4.1 availability.
            unsafe {
                extz2_sse41(input, ez, ws);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Safe because runtime detection guarantees NEON availability.
            unsafe {
                extz2_neon(input, ez, ws);
            }
            return;
        }
    }

    core::extz2_core::<core::ScalarOps>(input, ez, ws);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extz2_avx2(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    core::extz2_core::<core::Avx2Ops>(input, ez, ws);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn extz2_sse41(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    core::extz2_core::<core::Sse41Ops>(input, ez, ws);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn extz2_neon(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    core::extz2_core::<core::NeonOps>(input, ez, ws);
}

mod core;
