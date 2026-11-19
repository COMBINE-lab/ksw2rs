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
/// Exact score path only (no [`KSW_EZ_APPROX_MAX`]): do not track the overall
/// maximum. `mqe`, `mte` and `score` are computed exactly as without it, but
/// `max`, `max_q` and `max_t` are left untouched, which skips a horizontal
/// maximum per anti-diagonal. Applies to score-only alignment
/// ([`KSW_EZ_SCORE_ONLY`]) without z-drop (`zdrop < 0`), the cases that never
/// read the maximum; otherwise the flag is ignored.
pub const KSW_EZ_NO_MAX: u32 = 0x1000;

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
///
/// Cell values are stored in 8 bits, as in KSW2, which bounds the scoring
/// scheme: with `max_sc` and `min_sc` the largest and smallest entries of
/// `mat`, an alignment is computed only if `-min_sc <= 2 * (q + e)` and
/// `max_sc + 2 * (q + e) <= 255`. Otherwise the result is left reset
/// (`score == KSW_NEG_INF`, empty CIGAR). Schemes with
/// `max_sc + 2 * (q + e) > 127` run on the scalar kernel.
///
/// A negative `w` disables banding, as does any `w` at least as large as the
/// longer sequence.
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
/// uses: `[u | v | x | y | s+16 | sf | qr+32_pad]`.  This gives better cache and
/// TLB behaviour than seven separate `Vec` allocations, and the 32-byte
/// overshoot beyond `qr` means the SIMD score-fill kernels never need a
/// per-chunk bounds check.
#[derive(Debug, Default, Clone)]
pub struct Workspace {
    /// Flat buffer: `6 * tlen_pad + 16 + qlen_pad + 32` bytes, zeroed on resize.
    buf: Vec<u8>,
    /// Exact-path scores, `tlen` cells plus padding to a whole 4-cell group.
    h: Vec<i32>,
    off: Vec<i32>,
    off_end: Vec<i32>,
    /// Traceback matrix. Only ever grown: every cell that is read was written
    /// by the same alignment, so stale contents are never observed.
    p: Vec<u8>,
    /// Cached layout dimensions set by the most recent `prepare_*` call.
    tlen_pad: usize,
    qlen_pad: usize,
}

impl Workspace {
    /// Prepare the flat buffer for a score-only alignment.
    ///
    /// Layout: `[u | v | x | y | s+16 | sf | qr+32]`, all zero-initialised once.
    /// The trailing 32-byte pad on `qr` (and the implicit pad provided by `qr`
    /// after `sf`) means SIMD score-fill kernels can always load whole blocks
    /// without a per-chunk bounds check.
    #[inline]
    fn prepare_score_only(&mut self, qlen: usize, tlen: usize, approx: bool) {
        let tlen_pad = tlen.div_ceil(16) * 16;
        let qlen_pad = qlen.div_ceil(16) * 16;
        // 6 tlen_pad-wide arrays (u,v,x,y,s,sf), the slack after s, and qr
        // with a 32-byte pad.
        let total = 6 * tlen_pad + core::S_SLACK + qlen_pad + 32;
        self.buf.clear();
        self.buf.resize(total, 0u8);
        self.tlen_pad = tlen_pad;
        self.qlen_pad = qlen_pad;
        if !approx {
            self.h.clear();
            // padded so the SIMD row update can work on whole 4-cell groups
            self.h.resize(tlen + 3, KSW_NEG_INF);
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
        // The traceback matrix is fully written per active row span by the DP
        // kernels before any read, so it is only grown, never cleared.
        let p_len = rows * n_col;
        if self.p.len() < p_len {
            self.p.resize(p_len, 0);
        }
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
    extz2_scalar_with_workspace(input, ez, &mut ws);
}

/// Force the scalar backend while reusing caller-provided workspace.
#[inline]
pub fn extz2_scalar_with_workspace(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    // SAFETY: the scalar kernel needs no target feature.
    unsafe { extz2_scalar_kernel(input, ez, ws) }
}

/// A resolved alignment kernel: the score-fill implementation specialized for a
/// particular CPU's SIMD capability. Obtain one with [`resolve_kernel`].
///
/// Calling a kernel is `unsafe` because it may use target features the calling
/// context cannot statically guarantee; it is sound to call exactly the kernel
/// that [`resolve_kernel`] returned on the same machine, since the resolver picks
/// it from runtime CPU feature detection.
pub type KernelFn = unsafe fn(&Extz2Input<'_>, &mut Extz, &mut Workspace);

/// Scalar kernel, wrapped to share the [`KernelFn`] signature with the SIMD ones.
/// The one copy of the scalar core: the SIMD kernels fall back to it too.
#[inline(never)]
unsafe fn extz2_scalar_kernel(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    core::extz2_core::<core::ScalarOps>(input, ez, ws);
}

/// Detect the running CPU's SIMD capability **once** and return the best
/// available kernel (AVX2 > SSE4.1 > scalar on x86-64; NEON > scalar on aarch64).
///
/// Resolve this a single time — at program start or in [`Aligner::new`] — and
/// store the returned pointer; then every alignment dispatches through one
/// indirect call with no per-call feature check. (It is cheap to call more than
/// once, but resolving once is the intended pattern.)
pub fn resolve_kernel() -> KernelFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return extz2_avx2;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return extz2_sse41;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return extz2_neon;
        }
    }

    extz2_scalar_kernel
}

/// The kernel used by the free-function API ([`extz2`], [`extz2_with_workspace`]),
/// resolved once on first use and cached for the life of the process.
#[inline]
fn cached_kernel() -> KernelFn {
    use std::sync::OnceLock;
    static KERNEL: OnceLock<KernelFn> = OnceLock::new();
    *KERNEL.get_or_init(resolve_kernel)
}

#[inline]
fn extz2_dispatch(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    // SAFETY: `cached_kernel()` selected this kernel from this CPU's detected
    // features, so any target feature it requires is present.
    unsafe { (cached_kernel())(input, ez, ws) }
}

// The SIMD kernels compare transformed scores as signed bytes. Scoring schemes
// outside that range (see `core::fits_simd`) take the scalar kernel instead.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extz2_avx2(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    if !core::fits_simd(input) {
        // SAFETY: the scalar kernel needs no target feature.
        return unsafe { extz2_scalar_kernel(input, ez, ws) };
    }
    core::extz2_core::<core::Avx2Ops>(input, ez, ws);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn extz2_sse41(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    if !core::fits_simd(input) {
        // SAFETY: the scalar kernel needs no target feature.
        return unsafe { extz2_scalar_kernel(input, ez, ws) };
    }
    core::extz2_core::<core::Sse41Ops>(input, ez, ws);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn extz2_neon(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    if !core::fits_simd(input) {
        // SAFETY: the scalar kernel needs no target feature.
        return unsafe { extz2_scalar_kernel(input, ez, ws) };
    }
    core::extz2_core::<core::NeonOps>(input, ez, ws);
}

mod core;
