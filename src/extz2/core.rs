use super::*;
use ::core::cmp::{max, min};

fn push_cigar(cigar: &mut Vec<u32>, op: u32, len: u32) {
    if let Some(last) = cigar.last_mut() {
        if (*last & 0x0f) == op {
            *last = last.saturating_add(len << 4);
            return;
        }
    }
    cigar.push((len << 4) | op);
}

/// Backtrack through encoded predecessor states.
///
/// Layout matches `ksw_backtrack()` with `is_rot=1` (anti-diagonal coordinates).
fn backtrack(
    is_rev: bool,
    min_intron_len: i32,
    p: &[u8],
    off: &[i32],
    off_end: &[i32],
    n_col: usize,
    i0: i32,
    j0: i32,
    cigar: &mut Vec<u32>,
) {
    let mut i = i0;
    let mut j = j0;
    let mut state: u8 = 0;

    while i >= 0 && j >= 0 {
        let r = i + j;
        let row = r as usize;
        let mut force_state = -1i32;
        if i < off[row] {
            force_state = 2;
        }
        if i > off_end[row] {
            force_state = 1;
        }
        let tmp = if force_state < 0 {
            let col = (i - off[row]) as usize;
            p[row * n_col + col]
        } else {
            0
        };

        if state == 0 {
            state = tmp & 7;
        } else if ((tmp >> (state + 2)) & 1) == 0 {
            state = 0;
        }
        if state == 0 {
            state = tmp & 7;
        }
        if force_state >= 0 {
            state = force_state as u8;
        }

        if state == 0 {
            push_cigar(cigar, KSW_CIGAR_MATCH, 1);
            i -= 1;
            j -= 1;
        } else if state == 1 || (state == 3 && min_intron_len <= 0) {
            push_cigar(cigar, KSW_CIGAR_DEL, 1);
            i -= 1;
        } else if state == 3 && min_intron_len > 0 {
            push_cigar(cigar, KSW_CIGAR_N_SKIP, 1);
            i -= 1;
        } else {
            push_cigar(cigar, KSW_CIGAR_INS, 1);
            j -= 1;
        }
    }

    if i >= 0 {
        let op = if min_intron_len > 0 && i >= min_intron_len {
            KSW_CIGAR_N_SKIP
        } else {
            KSW_CIGAR_DEL
        };
        push_cigar(cigar, op, (i + 1) as u32);
    }
    if j >= 0 {
        push_cigar(cigar, KSW_CIGAR_INS, (j + 1) as u32);
    }

    if !is_rev {
        cigar.reverse();
    }
}

#[inline]
fn apply_zdrop(ez: &mut Extz, h: i32, a: i32, b: i32, zdrop: i32, e: i8) -> bool {
    let r = a;
    let t = b;
    if h > ez.max as i32 {
        ez.max = h as u32;
        ez.max_t = t;
        ez.max_q = r - t;
    } else if t >= ez.max_t && r - t >= ez.max_q {
        let tl = t - ez.max_t;
        let ql = (r - t) - ez.max_q;
        let l = (tl - ql).abs();
        if zdrop >= 0 && (ez.max as i32 - h) > zdrop + l * i32::from(e) {
            ez.zdropped = true;
            return true;
        }
    }
    false
}

/// True when the transformed scores fit the signed 8-bit comparisons of the
/// SIMD kernels. The scalar kernel computes in 32 bits and stores unsigned
/// bytes, so it is exact up to 255.
pub(super) fn fits_simd(input: &Extz2Input<'_>) -> bool {
    let max_sc = input.mat.iter().copied().max().map_or(0, i32::from);
    max_sc + 2 * (i32::from(input.q) + i32::from(input.e)) <= 127
}

// Fast-path score fill. Every backend writes exactly the cells
// `[st0, st0 + 16k)`, with `k` the number of 16-cell chunks needed to reach
// `en0`, as the C reference does: the cells past `en0` are padding, but the
// next anti-diagonal reads them when the band edge moves, so the padding is
// part of the result. `sfqr` is `sf` followed by `qr` (`sf` is `tlen_pad`
// long): target cell `t` is `sfqr[t]`, and past the end of `sf` the fill runs
// on into `qr`, again as in C. `s` is `tlen_pad + S_SLACK` long; a chunk that
// starts below `tlen_pad` may run on into the slack, or stop at `tlen_pad`.

#[inline]
fn fill_scores_fast_row_scalar(
    sfqr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    let tlen_pad = s.len() - S_SLACK;
    let end = (st0 + ((en0 - st0) / 16 + 1) * 16).min(tlen_pad);
    for idx in st0..end {
        let sq = sfqr[idx];
        // qpos >= 0 is guaranteed for idx >= st0; the cast is safe.
        let qq = sfqr[tlen_pad + (qrr_base + idx as i32) as usize];
        s[idx] = if sq == wildcard || qq == wildcard {
            sc_n
        } else if sq == qq {
            sc_mch
        } else {
            sc_mis
        };
    }
}

// The SIMD fills work on the 16-cell blocks the DP kernels load rather than on
// chunks from `st0`: a load of `s` then reads back exactly one store and is
// forwarded from it. A load spanning two chunk stores has to wait for both to
// leave the store buffer, which stalled every anti-diagonal. When `st0` is
// not a multiple of 16, the first block starts before `st0` and the last one
// ends after `st0 + 16k`; the cells outside keep their previous contents.

/// `LOW_BYTES[16 - n..]` starts with `n` bytes set, for `n` in `0..=16`.
#[cfg(target_arch = "x86_64")]
static LOW_BYTES: [i8; 32] = {
    let mut m = [0i8; 32];
    let mut i = 0;
    while i < 16 {
        m[i] = -1;
        i += 1;
    }
    m
};

/// Scores of the 16 cells of the block at `t` (a multiple of 16). With
/// `KEEP_LOW` the first `n` cells of the block keep their contents, without
/// it the first `n` cells are the ones written (`n` in `0..=16`).
///
/// # Safety
/// `t + 16 <= tlen_pad` (`s.len() - S_SLACK`), and `sfqr` holds 16 bytes at
/// both `t` and `tlen_pad + qrr_base + t` (the latter may start up to 15 bytes
/// before `qr`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn fill_scores_partial_block_sse41<const KEEP_LOW: bool>(
    sfqr: &[u8],
    qrr_base: i32,
    t: usize,
    n: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::x86_64::*;

    let wc = _mm_set1_epi8(wildcard as i8);
    let tlen_pad = s.len() - S_SLACK;
    let qpos = (tlen_pad as isize + qrr_base as isize + t as isize) as usize;
    debug_assert!(n <= 16 && t + 16 <= tlen_pad && qpos + 16 <= sfqr.len());
    // SAFETY: both loads are inside `sfqr`, `sp` inside `s` (see above).
    let sq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(t) as *const __m128i) };
    let qq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(qpos) as *const __m128i) };
    let sp = unsafe { s.as_mut_ptr().add(t) } as *mut __m128i;
    let mask_wc = _mm_or_si128(_mm_cmpeq_epi8(sq, wc), _mm_cmpeq_epi8(qq, wc));
    let eq = _mm_cmpeq_epi8(sq, qq);
    let mut tmp = _mm_blendv_epi8(_mm_set1_epi8(sc_mis), _mm_set1_epi8(sc_mch), eq);
    tmp = _mm_blendv_epi8(tmp, _mm_set1_epi8(sc_n), mask_wc);
    // SAFETY: `16 - n..16 - n + 16` is inside the 32-byte table.
    let low = unsafe { _mm_loadu_si128(LOW_BYTES.as_ptr().add(16 - n) as *const __m128i) };
    // A volatile read, so that the compiler cannot fold load, blend and store
    // into one masked store (it does with AVX-512): the DP kernel's load is
    // not forwarded from a masked store.
    let old = unsafe { sp.read_volatile() };
    tmp = if KEEP_LOW {
        _mm_blendv_epi8(tmp, old, low)
    } else {
        _mm_blendv_epi8(old, tmp, low)
    };
    unsafe { _mm_storeu_si128(sp, tmp) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fill_scores_fast_row_avx2(
    sfqr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::x86_64::*;

    let tlen_pad = s.len() - S_SLACK;
    let end = st0 + ((en0 - st0) / 16 + 1) * 16;
    let lim = end.min(tlen_pad);
    // The last byte any block below reads from `qr` is at most cell
    // `qrr_base + en0 + 30 <= qlen + 29`, inside the 32-byte pad; the first
    // is at least 15 bytes before `qr`, inside `sf`.
    assert!(
        st0 <= en0
            && en0 < tlen_pad
            && qrr_base + st0 as i32 >= 0
            && tlen_pad + (qrr_base + en0 as i32) as usize + 31 <= sfqr.len()
    );

    let mut t = st0 & !15;
    if t < st0 {
        // SAFETY: `t < st0 <= en0 < tlen_pad` and the bounds asserted above.
        unsafe {
            fill_scores_partial_block_sse41::<true>(
                sfqr,
                qrr_base,
                t,
                st0 - t,
                wildcard,
                sc_mch,
                sc_mis,
                sc_n,
                s,
            )
        };
        t += 16;
    }
    // Whole blocks, 32 cells at a time while they last: a 16-byte load is
    // forwarded from a wider store too.
    let wc = _mm256_set1_epi8(wildcard as i8);
    let mch = _mm256_set1_epi8(sc_mch);
    let mis = _mm256_set1_epi8(sc_mis);
    let nsc = _mm256_set1_epi8(sc_n);
    while t + 32 <= lim {
        let qpos = (tlen_pad as isize + qrr_base as isize + t as isize) as usize;
        // SAFETY: `st0 <= t` and `t + 32 <= lim`: all 32 cells are inside
        // `[st0, end)` and `s`, and the loads inside `sfqr` (asserted above).
        let sq = unsafe { _mm256_loadu_si256(sfqr.as_ptr().add(t) as *const __m256i) };
        let qq = unsafe { _mm256_loadu_si256(sfqr.as_ptr().add(qpos) as *const __m256i) };
        let mask_wc = _mm256_or_si256(_mm256_cmpeq_epi8(sq, wc), _mm256_cmpeq_epi8(qq, wc));
        let eq = _mm256_cmpeq_epi8(sq, qq);
        let mut tmp = _mm256_blendv_epi8(mis, mch, eq);
        tmp = _mm256_blendv_epi8(tmp, nsc, mask_wc);
        unsafe { _mm256_storeu_si256(s.as_mut_ptr().add(t) as *mut __m256i, tmp) };
        t += 32;
    }
    if t + 16 <= lim {
        let qpos = (tlen_pad as isize + qrr_base as isize + t as isize) as usize;
        // SAFETY: as above, for 16 cells.
        let sq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(t) as *const __m128i) };
        let qq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(qpos) as *const __m128i) };
        let (wc, eq) = (_mm256_castsi256_si128(wc), _mm_cmpeq_epi8(sq, qq));
        let mask_wc = _mm_or_si128(_mm_cmpeq_epi8(sq, wc), _mm_cmpeq_epi8(qq, wc));
        let mut tmp = _mm_blendv_epi8(_mm256_castsi256_si128(mis), _mm256_castsi256_si128(mch), eq);
        tmp = _mm_blendv_epi8(tmp, _mm256_castsi256_si128(nsc), mask_wc);
        unsafe { _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp) };
        t += 16;
    }
    if t < lim {
        // SAFETY: `t` is a multiple of 16 below `lim <= tlen_pad`.
        unsafe {
            fill_scores_partial_block_sse41::<false>(
                sfqr,
                qrr_base,
                t,
                end - t,
                wildcard,
                sc_mch,
                sc_mis,
                sc_n,
                s,
            )
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn fill_scores_fast_row_sse41(
    sfqr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::x86_64::*;

    let tlen_pad = s.len() - S_SLACK;
    let end = st0 + ((en0 - st0) / 16 + 1) * 16;
    let lim = end.min(tlen_pad);
    // see `fill_scores_fast_row_avx2`
    assert!(
        st0 <= en0
            && en0 < tlen_pad
            && qrr_base + st0 as i32 >= 0
            && tlen_pad + (qrr_base + en0 as i32) as usize + 31 <= sfqr.len()
    );

    let mut t = st0 & !15;
    if t < st0 {
        // SAFETY: `t < st0 <= en0 < tlen_pad` and the bounds asserted above.
        unsafe {
            fill_scores_partial_block_sse41::<true>(
                sfqr,
                qrr_base,
                t,
                st0 - t,
                wildcard,
                sc_mch,
                sc_mis,
                sc_n,
                s,
            )
        };
        t += 16;
    }
    let wc = _mm_set1_epi8(wildcard as i8);
    let mch = _mm_set1_epi8(sc_mch);
    let mis = _mm_set1_epi8(sc_mis);
    let nsc = _mm_set1_epi8(sc_n);
    while t + 16 <= lim {
        let qpos = (tlen_pad as isize + qrr_base as isize + t as isize) as usize;
        // SAFETY: `st0 <= t` and `t + 16 <= lim`: all 16 cells are inside
        // `[st0, end)` and `s`, and the loads inside `sfqr` (asserted above).
        let sq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(t) as *const __m128i) };
        let qq = unsafe { _mm_loadu_si128(sfqr.as_ptr().add(qpos) as *const __m128i) };
        let mask_wc = _mm_or_si128(_mm_cmpeq_epi8(sq, wc), _mm_cmpeq_epi8(qq, wc));
        let eq = _mm_cmpeq_epi8(sq, qq);
        let mut tmp = _mm_blendv_epi8(mis, mch, eq);
        tmp = _mm_blendv_epi8(tmp, nsc, mask_wc);
        unsafe { _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp) };
        t += 16;
    }
    if t < lim {
        // SAFETY: `t` is a multiple of 16 below `lim <= tlen_pad`.
        unsafe {
            fill_scores_partial_block_sse41::<false>(
                sfqr,
                qrr_base,
                t,
                end - t,
                wildcard,
                sc_mch,
                sc_mis,
                sc_n,
                s,
            )
        };
    }
}

/// NEON score fill: writes the cells `[st0, st0 + 16k)` (see the notes above
/// `fill_scores_fast_row_scalar`) in whole 16-cell stores, the last of which
/// may run on into the slack of `s`.
///
/// Narrow rows (at most two chunks, the banded case) are chunks from `st0`
/// while `t <= en0`, with no per-row setup. Wider rows store the chunk at
/// `st0`, the one ending at `st0 + 16k`, then the whole 16-cell blocks between
/// them, so that each block the DP kernel loads is forwarded from a single
/// store. On Apple cores the per-row instruction count of narrow rows matters
/// more than that forwarding; on wide rows the aligned blocks win.
///
/// # Safety
/// With `tlen_pad = s.len() - S_SLACK`: `st0 <= en0 < tlen_pad`,
/// `qrr_base + st0 >= 0`, and `tlen_pad + qrr_base + en0 + 31 <= sfqr.len()`
/// (as for the x86 fills; the score-only and traceback cores guarantee it by
/// construction).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fill_scores_fast_row_neon(
    sfqr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::aarch64::*;

    let tlen_pad = s.len() - S_SLACK;
    // the caller guarantees these (see the safety section above)
    debug_assert!(
        st0 <= en0
            && en0 < tlen_pad
            && qrr_base + st0 as i32 >= 0
            && tlen_pad + (qrr_base + en0 as i32) as usize + 31 <= sfqr.len()
    );
    let wc = vdupq_n_u8(wildcard);
    let mch = vdupq_n_s8(sc_mch);
    let mis = vdupq_n_s8(sc_mis);
    let nsc = vdupq_n_s8(sc_n);
    let sp = s.as_mut_ptr();
    let tp = sfqr.as_ptr();
    // `qp + t` is the query cell of target cell `t` (`qrr_base + t >= 0`)
    let qp = unsafe { sfqr.as_ptr().add(tlen_pad).offset(qrr_base as isize) };
    // SAFETY: called with `st0 <= t <= en0 < tlen_pad` or `t + 16 <= end`,
    // and `end <= en0 + 16`: the store ends inside `s` (`S_SLACK` is 16), the
    // loads inside `sfqr` (safety section above).
    let chunk = |t: usize| unsafe {
        let sq = vld1q_u8(tp.add(t));
        let qq = vld1q_u8(qp.add(t));
        let mask_wc = vorrq_u8(vceqq_u8(sq, wc), vceqq_u8(qq, wc));
        let tmp = vbslq_s8(vceqq_u8(sq, qq), mch, mis);
        vst1q_s8(sp.add(t), vbslq_s8(mask_wc, nsc, tmp));
    };
    if en0 - st0 >= 32 {
        // A wide row: the chunk at `st0`, the one ending at `end`, then the
        // whole 16-cell blocks between them (cells covered twice get the same
        // value twice).
        let end = st0 + ((en0 - st0) / 16 + 1) * 16;
        chunk(st0);
        chunk(end - 16);
        let mut t = (st0 + 16) & !15;
        while t + 16 < end {
            chunk(t);
            t += 16;
        }
        return;
    }
    let mut t = st0;
    while t <= en0 {
        chunk(t);
        t += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dp_row_score_only_sse41(
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::x86_64::*;

    let zero = _mm_set1_epi8(0);
    let qv = _mm_set1_epi8(q as i8);
    let qe2v = _mm_set1_epi8(qe2 as i8);
    let maxv = _mm_set1_epi8(max_sc_transformed as i8);

    let mut t = stv;
    while t <= env {
        // SAFETY: `t` traverses valid rounded 16-byte blocks inside padded buffers.
        let mut z = _mm_add_epi8(
            unsafe { _mm_loadu_si128(s.as_ptr().add(t) as *const __m128i) },
            qe2v,
        );
        // SAFETY: valid padded loads/stores.
        let xt_load = unsafe { _mm_loadu_si128(x.as_ptr().add(t) as *const __m128i) };
        let xt_hi = _mm_srli_si128(xt_load, 15);
        let x1v = _mm_cvtsi32_si128(i32::from(x1));
        let xt1 = _mm_or_si128(_mm_slli_si128(xt_load, 1), x1v);
        x1 = _mm_cvtsi128_si32(xt_hi) as u8;

        // SAFETY: valid padded loads/stores.
        let vt_load = unsafe { _mm_loadu_si128(v.as_ptr().add(t) as *const __m128i) };
        let vt_hi = _mm_srli_si128(vt_load, 15);
        let v1v = _mm_cvtsi32_si128(i32::from(v1));
        let vt1 = _mm_or_si128(_mm_slli_si128(vt_load, 1), v1v);
        v1 = _mm_cvtsi128_si32(vt_hi) as u8;

        let mut a = _mm_add_epi8(xt1, vt1);
        // SAFETY: valid padded loads/stores.
        let ut = unsafe { _mm_loadu_si128(u.as_ptr().add(t) as *const __m128i) };
        // SAFETY: valid padded loads/stores.
        let mut b = _mm_add_epi8(
            unsafe { _mm_loadu_si128(y.as_ptr().add(t) as *const __m128i) },
            ut,
        );

        z = _mm_max_epi8(z, a);
        z = _mm_max_epu8(z, b);
        z = _mm_min_epu8(z, maxv);

        // SAFETY: valid padded loads/stores.
        unsafe { _mm_storeu_si128(u.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, vt1)) };
        // SAFETY: valid padded loads/stores.
        unsafe { _mm_storeu_si128(v.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, ut)) };

        let zq = _mm_sub_epi8(z, qv);
        a = _mm_sub_epi8(a, zq);
        b = _mm_sub_epi8(b, zq);

        // SAFETY: valid padded loads/stores.
        unsafe { _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_max_epi8(a, zero)) };
        // SAFETY: valid padded loads/stores.
        unsafe { _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_max_epi8(b, zero)) };

        t += 16;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dp_row_score_only_neon(
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_s8(0);
    let qv = vdupq_n_u8(q as u8);
    let qe2v = vdupq_n_s8(qe2 as i8);
    let maxv = vdupq_n_u8(max_sc_transformed as u8);

    let mut t = stv;
    while t <= env {
        // SAFETY: `t` traverses valid rounded 16-byte blocks inside padded buffers.
        let svec = unsafe { vld1q_s8(s.as_ptr().add(t)) };
        let mut z_s = vaddq_s8(svec, qe2v);

        // SAFETY: valid padded loads.
        let xt_load = unsafe { vld1q_u8(x.as_ptr().add(t)) };
        let xt1 = vextq_u8(vdupq_n_u8(x1), xt_load, 15);
        x1 = vgetq_lane_u8(xt_load, 15);

        // SAFETY: valid padded loads.
        let vt_load = unsafe { vld1q_u8(v.as_ptr().add(t)) };
        let vt1 = vextq_u8(vdupq_n_u8(v1), vt_load, 15);
        v1 = vgetq_lane_u8(vt_load, 15);

        let mut a = vaddq_u8(xt1, vt1);
        // SAFETY: valid padded loads.
        let ut = unsafe { vld1q_u8(u.as_ptr().add(t)) };
        // SAFETY: valid padded loads.
        let mut b = vaddq_u8(unsafe { vld1q_u8(y.as_ptr().add(t)) }, ut);

        z_s = vmaxq_s8(z_s, vreinterpretq_s8_u8(a));
        let mut z_u = vreinterpretq_u8_s8(z_s);
        z_u = vmaxq_u8(z_u, b);
        z_u = vminq_u8(z_u, maxv);

        // SAFETY: valid padded stores.
        unsafe { vst1q_u8(u.as_mut_ptr().add(t), vsubq_u8(z_u, vt1)) };
        // SAFETY: valid padded stores.
        unsafe { vst1q_u8(v.as_mut_ptr().add(t), vsubq_u8(z_u, ut)) };

        let zq = vsubq_u8(z_u, qv);
        a = vsubq_u8(a, zq);
        b = vsubq_u8(b, zq);

        // SAFETY: valid padded stores.
        unsafe {
            vst1q_u8(
                x.as_mut_ptr().add(t),
                vreinterpretq_u8_s8(vmaxq_s8(vreinterpretq_s8_u8(a), zero)),
            )
        };
        // SAFETY: valid padded stores.
        unsafe {
            vst1q_u8(
                y.as_mut_ptr().add(t),
                vreinterpretq_u8_s8(vmaxq_s8(vreinterpretq_s8_u8(b), zero)),
            )
        };

        t += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dp_row_score_only_right_sse41(
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::x86_64::*;
    let zero = _mm_set1_epi8(0);
    let qv = _mm_set1_epi8(q as i8);
    let qe2v = _mm_set1_epi8(qe2 as i8);
    let maxv = _mm_set1_epi8(max_sc_transformed as i8);
    let mut t = stv;
    while t <= env {
        let mut z = _mm_add_epi8(
            unsafe { _mm_loadu_si128(s.as_ptr().add(t) as *const __m128i) },
            qe2v,
        );
        let xt_load = unsafe { _mm_loadu_si128(x.as_ptr().add(t) as *const __m128i) };
        let xt_hi = _mm_srli_si128(xt_load, 15);
        let x1v = _mm_cvtsi32_si128(i32::from(x1));
        let xt1 = _mm_or_si128(_mm_slli_si128(xt_load, 1), x1v);
        x1 = _mm_cvtsi128_si32(xt_hi) as u8;
        let vt_load = unsafe { _mm_loadu_si128(v.as_ptr().add(t) as *const __m128i) };
        let vt_hi = _mm_srli_si128(vt_load, 15);
        let v1v = _mm_cvtsi32_si128(i32::from(v1));
        let vt1 = _mm_or_si128(_mm_slli_si128(vt_load, 1), v1v);
        v1 = _mm_cvtsi128_si32(vt_hi) as u8;
        let mut a = _mm_add_epi8(xt1, vt1);
        let ut = unsafe { _mm_loadu_si128(u.as_ptr().add(t) as *const __m128i) };
        let mut b = _mm_add_epi8(
            unsafe { _mm_loadu_si128(y.as_ptr().add(t) as *const __m128i) },
            ut,
        );
        z = _mm_max_epi8(z, a);
        z = _mm_max_epu8(z, b);
        z = _mm_min_epu8(z, maxv);
        unsafe { _mm_storeu_si128(u.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, vt1)) };
        unsafe { _mm_storeu_si128(v.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, ut)) };
        let zq = _mm_sub_epi8(z, qv);
        a = _mm_sub_epi8(a, zq);
        b = _mm_sub_epi8(b, zq);
        let am = _mm_cmpgt_epi8(a, _mm_sub_epi8(zero, _mm_set1_epi8(1))); // a >= 0
        let bm = _mm_cmpgt_epi8(b, _mm_sub_epi8(zero, _mm_set1_epi8(1))); // b >= 0
        unsafe { _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(a, am)) };
        unsafe { _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(b, bm)) };
        t += 16;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dp_row_score_only_right_neon(
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_s8(0);
    let qv = vdupq_n_u8(q as u8);
    let qe2v = vdupq_n_s8(qe2 as i8);
    let maxv = vdupq_n_u8(max_sc_transformed as u8);
    let mut t = stv;
    while t <= env {
        let svec = unsafe { vld1q_s8(s.as_ptr().add(t)) };
        let mut z_s = vaddq_s8(svec, qe2v);
        let xt_load = unsafe { vld1q_u8(x.as_ptr().add(t)) };
        let xt1 = vextq_u8(vdupq_n_u8(x1), xt_load, 15);
        x1 = vgetq_lane_u8(xt_load, 15);
        let vt_load = unsafe { vld1q_u8(v.as_ptr().add(t)) };
        let vt1 = vextq_u8(vdupq_n_u8(v1), vt_load, 15);
        v1 = vgetq_lane_u8(vt_load, 15);
        let mut a = vaddq_u8(xt1, vt1);
        let ut = unsafe { vld1q_u8(u.as_ptr().add(t)) };
        let mut b = vaddq_u8(unsafe { vld1q_u8(y.as_ptr().add(t)) }, ut);
        z_s = vmaxq_s8(z_s, vreinterpretq_s8_u8(a));
        let mut z_u = vreinterpretq_u8_s8(z_s);
        z_u = vmaxq_u8(z_u, b);
        z_u = vminq_u8(z_u, maxv);
        unsafe { vst1q_u8(u.as_mut_ptr().add(t), vsubq_u8(z_u, vt1)) };
        unsafe { vst1q_u8(v.as_mut_ptr().add(t), vsubq_u8(z_u, ut)) };
        let zq = vsubq_u8(z_u, qv);
        a = vsubq_u8(a, zq);
        b = vsubq_u8(b, zq);
        let am = vcgeq_s8(vreinterpretq_s8_u8(a), zero);
        let bm = vcgeq_s8(vreinterpretq_s8_u8(b), zero);
        unsafe { vst1q_u8(x.as_mut_ptr().add(t), vandq_u8(a, am)) };
        unsafe { vst1q_u8(y.as_mut_ptr().add(t), vandq_u8(b, bm)) };
        t += 16;
    }
}

#[inline(always)]
fn dp_row_traceback_scalar(
    right_align: bool,
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    p_row: &mut [u8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    for t in stv..=env {
        let mut z = i32::from(s[t]) + qe2;
        let xt1 = x1;
        let vt1 = v1;
        x1 = x[t];
        v1 = v[t];
        let a = i32::from(xt1) + i32::from(vt1);
        let ut = u[t];
        let b = i32::from(y[t]) + i32::from(ut);
        let mut d = 0u8;
        if !right_align {
            if a > z {
                d = 1;
                z = a;
            }
            if b > z {
                d = 2;
                z = b;
            }
        } else {
            if z <= a {
                d = 1;
                z = a;
            }
            if z <= b {
                d = 2;
                z = b;
            }
        }
        z = min(z, max_sc_transformed);
        u[t] = (z - i32::from(vt1)) as u8;
        v[t] = (z - i32::from(ut)) as u8;
        let zq = z - q;
        let na = a - zq;
        let nb = b - zq;
        if !right_align {
            if na > 0 {
                x[t] = na as u8;
                d |= 0x08;
            } else {
                x[t] = 0;
            }
            if nb > 0 {
                y[t] = nb as u8;
                d |= 0x10;
            } else {
                y[t] = 0;
            }
        } else {
            if na >= 0 {
                x[t] = na as u8;
                d |= 0x08;
            } else {
                x[t] = 0;
            }
            if nb >= 0 {
                y[t] = nb as u8;
                d |= 0x10;
            } else {
                y[t] = 0;
            }
        }
        p_row[t - stv] = d;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dp_row_traceback_sse41(
    right_align: bool,
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    p_row: &mut [u8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::x86_64::*;
    let zero = _mm_set1_epi8(0);
    let qv = _mm_set1_epi8(q as i8);
    let qe2v = _mm_set1_epi8(qe2 as i8);
    let maxv = _mm_set1_epi8(max_sc_transformed as i8);
    let flag1 = _mm_set1_epi8(1);
    let flag2 = _mm_set1_epi8(2);
    let flag8 = _mm_set1_epi8(0x08);
    let flag16 = _mm_set1_epi8(0x10);
    let neg1 = _mm_set1_epi8(-1);
    let mut t = stv;
    while t <= env {
        let mut z = _mm_add_epi8(
            unsafe { _mm_loadu_si128(s.as_ptr().add(t) as *const __m128i) },
            qe2v,
        );
        let xt_load = unsafe { _mm_loadu_si128(x.as_ptr().add(t) as *const __m128i) };
        let xt_hi = _mm_srli_si128(xt_load, 15);
        let x1v = _mm_cvtsi32_si128(i32::from(x1));
        let xt1 = _mm_or_si128(_mm_slli_si128(xt_load, 1), x1v);
        x1 = _mm_cvtsi128_si32(xt_hi) as u8;
        let vt_load = unsafe { _mm_loadu_si128(v.as_ptr().add(t) as *const __m128i) };
        let vt_hi = _mm_srli_si128(vt_load, 15);
        let v1v = _mm_cvtsi32_si128(i32::from(v1));
        let vt1 = _mm_or_si128(_mm_slli_si128(vt_load, 1), v1v);
        v1 = _mm_cvtsi128_si32(vt_hi) as u8;
        let mut a = _mm_add_epi8(xt1, vt1);
        let ut = unsafe { _mm_loadu_si128(u.as_ptr().add(t) as *const __m128i) };
        let mut b = _mm_add_epi8(
            unsafe { _mm_loadu_si128(y.as_ptr().add(t) as *const __m128i) },
            ut,
        );
        let mut d;
        if !right_align {
            d = _mm_and_si128(_mm_cmpgt_epi8(a, z), flag1);
            z = _mm_max_epi8(z, a);
            let tmp = _mm_cmpgt_epi8(b, z);
            d = _mm_blendv_epi8(d, flag2, tmp);
        } else {
            d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), flag1);
            z = _mm_max_epi8(z, a);
            let tmp = _mm_cmpgt_epi8(z, b);
            d = _mm_blendv_epi8(flag2, d, tmp);
        }
        z = _mm_max_epu8(z, b);
        z = _mm_min_epu8(z, maxv);
        unsafe { _mm_storeu_si128(u.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, vt1)) };
        unsafe { _mm_storeu_si128(v.as_mut_ptr().add(t) as *mut __m128i, _mm_sub_epi8(z, ut)) };
        let zq = _mm_sub_epi8(z, qv);
        a = _mm_sub_epi8(a, zq);
        b = _mm_sub_epi8(b, zq);
        if !right_align {
            let am = _mm_cmpgt_epi8(a, zero);
            let bm = _mm_cmpgt_epi8(b, zero);
            unsafe {
                _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(a, am))
            };
            unsafe {
                _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(b, bm))
            };
            d = _mm_or_si128(d, _mm_and_si128(am, flag8));
            d = _mm_or_si128(d, _mm_and_si128(bm, flag16));
        } else {
            let am = _mm_cmpgt_epi8(a, neg1); // a >= 0
            let bm = _mm_cmpgt_epi8(b, neg1); // b >= 0
            unsafe {
                _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(a, am))
            };
            unsafe {
                _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(b, bm))
            };
            d = _mm_or_si128(d, _mm_and_si128(am, flag8));
            d = _mm_or_si128(d, _mm_and_si128(bm, flag16));
        }
        unsafe { _mm_storeu_si128(p_row.as_mut_ptr().add(t - stv) as *mut __m128i, d) };
        t += 16;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dp_row_traceback_neon(
    right_align: bool,
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    p_row: &mut [u8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    mut x1: u8,
    mut v1: u8,
) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_s8(0);
    let qv = vdupq_n_u8(q as u8);
    let qe2v = vdupq_n_s8(qe2 as i8);
    let maxv = vdupq_n_u8(max_sc_transformed as u8);
    let flag1 = vdupq_n_u8(1);
    let flag2 = vdupq_n_u8(2);
    let flag8 = vdupq_n_u8(0x08);
    let flag16 = vdupq_n_u8(0x10);
    let mut t = stv;
    while t <= env {
        let svec = unsafe { vld1q_s8(s.as_ptr().add(t)) };
        let mut z_s = vaddq_s8(svec, qe2v);
        let xt_load = unsafe { vld1q_u8(x.as_ptr().add(t)) };
        let xt1 = vextq_u8(vdupq_n_u8(x1), xt_load, 15);
        x1 = vgetq_lane_u8(xt_load, 15);
        let vt_load = unsafe { vld1q_u8(v.as_ptr().add(t)) };
        let vt1 = vextq_u8(vdupq_n_u8(v1), vt_load, 15);
        v1 = vgetq_lane_u8(vt_load, 15);
        let mut a = vaddq_u8(xt1, vt1);
        let ut = unsafe { vld1q_u8(u.as_ptr().add(t)) };
        let mut b = vaddq_u8(unsafe { vld1q_u8(y.as_ptr().add(t)) }, ut);
        let mut d;
        if !right_align {
            let am = vcgtq_s8(vreinterpretq_s8_u8(a), z_s);
            d = vandq_u8(am, flag1);
            z_s = vmaxq_s8(z_s, vreinterpretq_s8_u8(a));
            let bm = vcgtq_s8(vreinterpretq_s8_u8(b), z_s);
            d = vbslq_u8(bm, flag2, d);
        } else {
            let zm = vcgtq_s8(z_s, vreinterpretq_s8_u8(a));
            d = vbicq_u8(flag1, zm); // z>a ?0:1
            z_s = vmaxq_s8(z_s, vreinterpretq_s8_u8(a));
            let zm2 = vcgtq_s8(z_s, vreinterpretq_s8_u8(b));
            d = vbslq_u8(zm2, d, flag2); // z>b ? d : 2
        }
        let mut z_u = vreinterpretq_u8_s8(z_s);
        z_u = vmaxq_u8(z_u, b);
        z_u = vminq_u8(z_u, maxv);
        unsafe { vst1q_u8(u.as_mut_ptr().add(t), vsubq_u8(z_u, vt1)) };
        unsafe { vst1q_u8(v.as_mut_ptr().add(t), vsubq_u8(z_u, ut)) };
        let zq = vsubq_u8(z_u, qv);
        a = vsubq_u8(a, zq);
        b = vsubq_u8(b, zq);
        if !right_align {
            let am = vcgtq_s8(vreinterpretq_s8_u8(a), zero);
            let bm = vcgtq_s8(vreinterpretq_s8_u8(b), zero);
            unsafe { vst1q_u8(x.as_mut_ptr().add(t), vandq_u8(a, am)) };
            unsafe { vst1q_u8(y.as_mut_ptr().add(t), vandq_u8(b, bm)) };
            d = vorrq_u8(d, vandq_u8(am, flag8));
            d = vorrq_u8(d, vandq_u8(bm, flag16));
        } else {
            let am = vcgeq_s8(vreinterpretq_s8_u8(a), zero);
            let bm = vcgeq_s8(vreinterpretq_s8_u8(b), zero);
            unsafe { vst1q_u8(x.as_mut_ptr().add(t), vandq_u8(a, am)) };
            unsafe { vst1q_u8(y.as_mut_ptr().add(t), vandq_u8(b, bm)) };
            d = vorrq_u8(d, vandq_u8(am, flag8));
            d = vorrq_u8(d, vandq_u8(bm, flag16));
        }
        unsafe { vst1q_u8(p_row.as_mut_ptr().add(t - stv), d) };
        t += 16;
    }
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn dp_row_traceback_dispatch(
    backend: SimdBackend,
    right_align: bool,
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    p_row: &mut [u8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    x1: u8,
    v1: u8,
) {
    match backend {
        SimdBackend::Scalar => dp_row_traceback_scalar(
            right_align,
            u,
            v,
            x,
            y,
            s,
            p_row,
            stv,
            env,
            q,
            qe2,
            max_sc_transformed,
            x1,
            v1,
        ),
        SimdBackend::Avx2 | SimdBackend::Sse41 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                dp_row_traceback_sse41(
                    right_align,
                    u,
                    v,
                    x,
                    y,
                    s,
                    p_row,
                    stv,
                    env,
                    q,
                    qe2,
                    max_sc_transformed,
                    x1,
                    v1,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            dp_row_traceback_scalar(
                right_align,
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                dp_row_traceback_neon(
                    right_align,
                    u,
                    v,
                    x,
                    y,
                    s,
                    p_row,
                    stv,
                    env,
                    q,
                    qe2,
                    max_sc_transformed,
                    x1,
                    v1,
                );
            }
            #[cfg(not(target_arch = "aarch64"))]
            dp_row_traceback_scalar(
                right_align,
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
    }
}

/// [`update_h_exact_scalar`] without the maximum: the same updates to `h`, in
/// the same order, for [`super::KSW_EZ_NO_MAX`]. A plain widening add: the
/// compiler vectorizes it well enough that per-backend versions measured no
/// faster.
// On aarch64 the call, and the vector constants it makes the anti-diagonal
// loop spill and reload, cost NO_MAX 5-8% (measured on an M5 Max).
#[cfg_attr(target_arch = "aarch64", inline(always))]
#[cfg_attr(not(target_arch = "aarch64"), inline)]
fn update_h_no_max(h: &mut [i32], v: &[u8], u: &[u8], st0: usize, en0: usize, qe: i32) {
    if en0 > 0 {
        h[en0] = h[en0 - 1] + i32::from(u[en0]) - qe;
    } else {
        h[en0] = h[en0] + i32::from(v[en0]) - qe;
    }
    for (ht, &vt) in h[st0..en0].iter_mut().zip(&v[st0..en0]) {
        *ht += i32::from(vt) - qe;
    }
}

#[inline]
fn update_h_exact_scalar(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    if en0 > 0 {
        h[en0] = h[en0 - 1] + i32::from(u[en0]) - qe;
    } else {
        h[en0] = h[en0] + i32::from(v[en0]) - qe;
    }
    let mut max_h = h[en0];
    let mut max_t = en0 as i32;
    for t in st0..en0 {
        h[t] += i32::from(v[t]) - qe;
        if h[t] > max_h {
            max_h = h[t];
            max_t = t as i32;
        }
    }
    (max_h, max_t)
}

// Exact-max row update, `update_h_exact_scalar` without per-block branches:
// the same `h` updates and the same `(max_h, max_t)`: the strictly greatest
// interior score at its first position if it beats `h[en0]`, else `h[en0]` at
// `en0`. The first per-backend versions reduced each block horizontally and
// branched on the running maximum, a loop-carried scalar dependency that cost
// about half the time of an exact score-only alignment. Here every lane keeps
// its own maximum and the first position reaching it (strict `>`, so within a
// lane the earliest position wins), with independent accumulators so no step
// waits on the previous one, and the lanes are reduced once per row: the
// maximum is the largest lane maximum, its first position the smallest
// position among the lanes holding it.

/// SSE4.1 lane-wise exact-max row update (also used by the AVX2 backend).
///
/// `h` is walked in absolute 4-cell groups (`g % 4 == 0`), whatever `st0` is:
/// every anti-diagonal then loads and stores the same 16 bytes of `h`, and
/// reads `v` inside one 16-byte store of the DP kernel, so each load is
/// forwarded from a single store. (Groups anchored at `st0` move by one cell
/// every other row, and their loads then straddle two stores of the previous
/// row.) The groups holding `st0` and `en0` are masked; `h[en0]` is written
/// with its group. Whole 16-cell blocks in between go four groups at a time
/// into separate accumulators.
///
/// `h` must hold whole groups: `h.len() >= (en0 | 3) + 1`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn update_h_exact_lanes_sse41(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    use std::arch::x86_64::*;

    let h_en = if en0 > 0 {
        h[en0 - 1] + i32::from(u[en0]) - qe
    } else {
        h[en0] + i32::from(v[en0]) - qe
    };
    let lo = st0 & !3;
    let hi = en0 & !3;
    assert!(st0 <= en0 && hi + 4 <= h.len() && hi + 4 <= v.len());

    let qe_v = _mm_set1_epi32(qe);
    let lane = _mm_setr_epi32(0, 1, 2, 3);
    let st_m1 = _mm_set1_epi32(st0 as i32 - 1);
    let en_v = _mm_set1_epi32(en0 as i32);
    let hen_v = _mm_set1_epi32(h_en);
    let min_v = _mm_set1_epi32(i32::MIN);
    let mut mx = [min_v; 4];
    let mut mt = [_mm_set1_epi32(i32::MAX); 4];

    // The group at `g` when it holds `st0` or `en0`: the cells in
    // `[st0, en0)` are updated and enter accumulator 0, `h[en0]` is set, the
    // others are stored back as they were.
    macro_rules! masked_group {
        ($g:expr) => {{
            let g: usize = $g;
            // SAFETY: `g <= hi`, so the group is inside `h` and `v` (asserted).
            let hp = unsafe { h.as_mut_ptr().add(g) } as *mut __m128i;
            // volatile: keeps the store below a plain one, see
            // `fill_scores_partial_block_sse41`
            let hv = unsafe { hp.read_volatile() };
            let v4 = unsafe { (v.as_ptr().add(g) as *const i32).read_unaligned() };
            let vw = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(v4));
            let nh = _mm_sub_epi32(_mm_add_epi32(hv, vw), qe_v);
            let pos = _mm_add_epi32(_mm_set1_epi32(g as i32), lane);
            let inside = _mm_and_si128(_mm_cmpgt_epi32(pos, st_m1), _mm_cmpgt_epi32(en_v, pos));
            let mut out = _mm_blendv_epi8(hv, nh, inside);
            out = _mm_blendv_epi8(out, hen_v, _mm_cmpeq_epi32(pos, en_v));
            unsafe { _mm_storeu_si128(hp, out) };
            let take = _mm_and_si128(inside, _mm_cmpgt_epi32(nh, mx[0]));
            mx[0] = _mm_blendv_epi8(mx[0], nh, take);
            mt[0] = _mm_blendv_epi8(mt[0], pos, take);
        }};
    }
    // A group wholly inside `[st0, en0)`, into accumulator 0.
    macro_rules! whole_group {
        ($g:expr) => {{
            let g: usize = $g;
            // SAFETY: `g + 4 <= en0`, within both `h` and `v` (asserted above).
            let hp = unsafe { h.as_mut_ptr().add(g) } as *mut __m128i;
            let v4 = unsafe { (v.as_ptr().add(g) as *const i32).read_unaligned() };
            let vw = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(v4));
            let nh = _mm_sub_epi32(_mm_add_epi32(unsafe { _mm_loadu_si128(hp) }, vw), qe_v);
            unsafe { _mm_storeu_si128(hp, nh) };
            let take = _mm_cmpgt_epi32(nh, mx[0]);
            mx[0] = _mm_blendv_epi8(mx[0], nh, take);
            mt[0] = _mm_blendv_epi8(mt[0], _mm_add_epi32(_mm_set1_epi32(g as i32), lane), take);
        }};
    }

    let mut g = lo;
    if g < st0 {
        masked_group!(g);
        g += 4;
    }
    // Whole groups up to the first whole 16-cell block...
    while g + 4 <= en0 && (g & 15 != 0 || g + 16 > en0) {
        whole_group!(g);
        g += 4;
    }
    // ...the whole blocks, four groups at a time...
    let wide = g + 16 <= en0;
    if wide {
        let base = _mm_add_epi32(_mm_set1_epi32(g as i32), lane);
        let mut idx = [
            base,
            _mm_add_epi32(base, _mm_set1_epi32(4)),
            _mm_add_epi32(base, _mm_set1_epi32(8)),
            _mm_add_epi32(base, _mm_set1_epi32(12)),
        ];
        let sixteen = _mm_set1_epi32(16);
        while g + 16 <= en0 {
            // SAFETY: `g + 16 <= en0`, within both `h` and `v` (asserted above).
            let vb = unsafe { _mm_loadu_si128(v.as_ptr().add(g) as *const __m128i) };
            let vw = [
                _mm_cvtepu8_epi32(vb),
                _mm_cvtepu8_epi32(_mm_srli_si128(vb, 4)),
                _mm_cvtepu8_epi32(_mm_srli_si128(vb, 8)),
                _mm_cvtepu8_epi32(_mm_srli_si128(vb, 12)),
            ];
            for k in 0..4 {
                let hp = unsafe { h.as_mut_ptr().add(g + 4 * k) } as *mut __m128i;
                let nh = _mm_sub_epi32(_mm_add_epi32(unsafe { _mm_loadu_si128(hp) }, vw[k]), qe_v);
                unsafe { _mm_storeu_si128(hp, nh) };
                let take = _mm_cmpgt_epi32(nh, mx[k]);
                mx[k] = _mm_blendv_epi8(mx[k], nh, take);
                mt[k] = _mm_blendv_epi8(mt[k], idx[k], take);
                idx[k] = _mm_add_epi32(idx[k], sixteen);
            }
            g += 16;
        }
        // ...and the whole groups after them. Accumulator 0 still sees its
        // positions in increasing order, so it still keeps the first maximum.
        while g + 4 <= en0 {
            whole_group!(g);
            g += 4;
        }
    }
    // The group of `en0`, unless it was also the group of `st0`.
    if g <= hi {
        masked_group!(g);
    }

    let none = _mm_set1_epi32(i32::MAX);
    let (mut m4, mut first);
    if wide {
        let mall = _mm_max_epi32(_mm_max_epi32(mx[0], mx[1]), _mm_max_epi32(mx[2], mx[3]));
        m4 = _mm_max_epi32(mall, _mm_shuffle_epi32(mall, 0b01_00_11_10));
        m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0b10_11_00_01));
        first = none;
        for k in 0..4 {
            first = _mm_min_epi32(
                first,
                _mm_blendv_epi8(none, mt[k], _mm_cmpeq_epi32(mx[k], m4)),
            );
        }
    } else {
        m4 = _mm_max_epi32(mx[0], _mm_shuffle_epi32(mx[0], 0b01_00_11_10));
        m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0b10_11_00_01));
        first = _mm_blendv_epi8(none, mt[0], _mm_cmpeq_epi32(mx[0], m4));
    }
    first = _mm_min_epi32(first, _mm_shuffle_epi32(first, 0b01_00_11_10));
    first = _mm_min_epi32(first, _mm_shuffle_epi32(first, 0b10_11_00_01));
    let best = _mm_cvtsi128_si32(m4);
    if best > h_en {
        (best, _mm_cvtsi128_si32(first))
    } else {
        (h_en, en0 as i32)
    }
}

/// NEON lane-wise exact-max row update, lanes anchored at `st0`: whole
/// 16-cell blocks go four groups at a time into separate accumulators, then
/// 4-cell groups into accumulator 0, then a scalar tail. Measured faster on
/// Apple cores than the absolute 4-cell groups of the x86 kernels: the masked
/// end groups cost more than the store-forwarding they save.
// NEON is part of the aarch64 baseline: no `target_feature` is needed, and
// without one the update can be forced inline into the anti-diagonal loop
// (worth about 6% at w=15: the call made the loop spill its vector constants).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn update_h_exact_lanes_neon(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    // SAFETY: NEON is always present on aarch64; the memory accesses are
    // justified where they happen.
    #[allow(unused_unsafe)]
    unsafe {
        use std::arch::aarch64::*;

        let h_en = if en0 > 0 {
            h[en0 - 1] + i32::from(u[en0]) - qe
        } else {
            h[en0] + i32::from(v[en0]) - qe
        };
        h[en0] = h_en;
        let qe_v = vdupq_n_s32(qe);
        let lane = unsafe { vld1q_s32([0i32, 1, 2, 3].as_ptr()) };
        let base = vaddq_s32(vdupq_n_s32(st0 as i32), lane);
        let mut idx = [
            base,
            vaddq_s32(base, vdupq_n_s32(4)),
            vaddq_s32(base, vdupq_n_s32(8)),
            vaddq_s32(base, vdupq_n_s32(12)),
        ];
        let sixteen = vdupq_n_s32(16);
        let mut mx = [vdupq_n_s32(i32::MIN); 4];
        let mut mt = [vdupq_n_s32(i32::MAX); 4];
        let mut t = st0;
        assert!(en0 <= v.len() && en0 <= h.len());
        while t + 16 <= en0 {
            // SAFETY: `t + 16 <= en0`, within both `h` and `v` (asserted above).
            let vb = unsafe { vld1q_u8(v.as_ptr().add(t)) };
            let (w_lo, w_hi) = (vmovl_u8(vget_low_u8(vb)), vmovl_high_u8(vb));
            let vw = [
                vmovl_u16(vget_low_u16(w_lo)),
                vmovl_high_u16(w_lo),
                vmovl_u16(vget_low_u16(w_hi)),
                vmovl_high_u16(w_hi),
            ];
            for k in 0..4 {
                let hp = unsafe { h.as_mut_ptr().add(t + 4 * k) };
                let nh = vsubq_s32(
                    vaddq_s32(unsafe { vld1q_s32(hp) }, vreinterpretq_s32_u32(vw[k])),
                    qe_v,
                );
                unsafe { vst1q_s32(hp, nh) };
                let take = vcgtq_s32(nh, mx[k]);
                mx[k] = vbslq_s32(take, nh, mx[k]);
                mt[k] = vbslq_s32(take, idx[k], mt[k]);
                idx[k] = vaddq_s32(idx[k], sixteen);
            }
            t += 16;
        }
        // Then 4 cells at a time into the first accumulator: each lane still
        // sees its positions in increasing order, so it keeps the first maximum.
        let mut idx4 = vaddq_s32(vdupq_n_s32(t as i32), lane);
        while t + 4 <= en0 {
            // SAFETY: `t + 4 <= en0`, within both `h` and `v` (asserted above).
            let v4 = unsafe { (v.as_ptr().add(t) as *const u32).read_unaligned() };
            let vw = vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(v4)))));
            let hp = unsafe { h.as_mut_ptr().add(t) };
            let nh = vsubq_s32(
                vaddq_s32(unsafe { vld1q_s32(hp) }, vreinterpretq_s32_u32(vw)),
                qe_v,
            );
            unsafe { vst1q_s32(hp, nh) };
            let take = vcgtq_s32(nh, mx[0]);
            mx[0] = vbslq_s32(take, nh, mx[0]);
            mt[0] = vbslq_s32(take, idx4, mt[0]);
            idx4 = vaddq_s32(idx4, vdupq_n_s32(4));
            t += 4;
        }
        let (mut best, mut best_t) = (i32::MIN, i32::MAX);
        if t > st0 {
            best = vmaxvq_s32(vmaxq_s32(vmaxq_s32(mx[0], mx[1]), vmaxq_s32(mx[2], mx[3])));
            let mv = vdupq_n_s32(best);
            let none = vdupq_n_s32(i32::MAX);
            let mut first = none;
            for k in 0..4 {
                first = vminq_s32(first, vbslq_s32(vceqq_s32(mx[k], mv), mt[k], none));
            }
            best_t = vminvq_s32(first);
        }
        // The tail follows every position above, so strict `>` keeps them first.
        for k in t..en0 {
            h[k] += i32::from(v[k]) - qe;
            if h[k] > best {
                best = h[k];
                best_t = k as i32;
            }
        }
        if best > h_en {
            (best, best_t)
        } else {
            (h_en, en0 as i32)
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn dp_row_score_only_dispatch(
    backend: SimdBackend,
    u: &mut [u8],
    v: &mut [u8],
    x: &mut [u8],
    y: &mut [u8],
    s: &[i8],
    stv: usize,
    env: usize,
    q: i32,
    qe2: i32,
    max_sc_transformed: i32,
    x1: u8,
    v1: u8,
    right_align: bool,
) {
    if right_align {
        match backend {
            SimdBackend::Scalar => {
                let mut x1 = x1;
                let mut v1 = v1;
                for t in stv..=env {
                    let mut z = i32::from(s[t]) + qe2;
                    let xt1 = x1;
                    let vt1 = v1;
                    x1 = x[t];
                    v1 = v[t];
                    let a = i32::from(xt1) + i32::from(vt1);
                    let ut = u[t];
                    let b = i32::from(y[t]) + i32::from(ut);
                    if z < 0 {
                        z = 0;
                    }
                    if z <= a {
                        z = a;
                    }
                    if z <= b {
                        z = b;
                    }
                    z = min(z, max_sc_transformed);
                    u[t] = (z - i32::from(vt1)) as u8;
                    v[t] = (z - i32::from(ut)) as u8;
                    let zq = z - q;
                    let na = a - zq;
                    let nb = b - zq;
                    x[t] = if na >= 0 { na as u8 } else { 0 };
                    y[t] = if nb >= 0 { nb as u8 } else { 0 };
                }
            }
            SimdBackend::Avx2 | SimdBackend::Sse41 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    dp_row_score_only_right_sse41(
                        u,
                        v,
                        x,
                        y,
                        s,
                        stv,
                        env,
                        q,
                        qe2,
                        max_sc_transformed,
                        x1,
                        v1,
                    )
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
            SimdBackend::Neon => {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    dp_row_score_only_right_neon(
                        u,
                        v,
                        x,
                        y,
                        s,
                        stv,
                        env,
                        q,
                        qe2,
                        max_sc_transformed,
                        x1,
                        v1,
                    )
                }
                #[cfg(not(target_arch = "aarch64"))]
                unreachable!()
            }
        }
        return;
    }

    match backend {
        SimdBackend::Scalar => {
            let mut x1 = x1;
            let mut v1 = v1;
            for t in stv..=env {
                let mut z = i32::from(s[t]) + qe2;
                let xt1 = x1;
                let vt1 = v1;
                x1 = x[t];
                v1 = v[t];
                let a = i32::from(xt1) + i32::from(vt1);
                let ut = u[t];
                let b = i32::from(y[t]) + i32::from(ut);
                if z < 0 {
                    z = 0;
                }
                if a > z {
                    z = a;
                }
                if b > z {
                    z = b;
                }
                z = min(z, max_sc_transformed);
                u[t] = (z - i32::from(vt1)) as u8;
                v[t] = (z - i32::from(ut)) as u8;
                let zq = z - q;
                let na = a - zq;
                let nb = b - zq;
                x[t] = if na > 0 { na as u8 } else { 0 };
                y[t] = if nb > 0 { nb as u8 } else { 0 };
            }
        }
        SimdBackend::Avx2 | SimdBackend::Sse41 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                dp_row_score_only_sse41(u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1)
            }
            #[cfg(not(target_arch = "x86_64"))]
            unreachable!()
        }
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                dp_row_score_only_neon(u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1)
            }
            #[cfg(not(target_arch = "aarch64"))]
            unreachable!()
        }
    }
}

pub(super) trait BackendOps {
    unsafe fn fill_fast_row(
        sfqr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    );
    unsafe fn dp_row_score_only<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    );
    unsafe fn dp_row_traceback<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        p_row: &mut [u8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    );
    unsafe fn update_h_exact(
        h: &mut [i32],
        v: &[u8],
        u: &[u8],
        st0: usize,
        en0: usize,
        qe: i32,
    ) -> (i32, i32);
}

/// Cells of `s` after `tlen_pad`: room for a whole 16-cell store starting at
/// any cell below `tlen_pad`, so a score fill that works in chunks from `st0`
/// need not clip its last one. Never read.
pub(super) const S_SLACK: usize = 16;

/// Carve the flat `Workspace::buf` into the named DP slices.
///
/// Layout: `[u | v | x | y | s+16 | sf | qr+32]`: the C reference's, plus
/// `S_SLACK` cells after `s`.
/// The `s` slice is re-interpreted as `&mut [i8]` (same size/alignment as
/// `u8`).  All returned slices are non-overlapping sub-slices of `buf`.
/// `sf` and `qr` come back as one slice, `sfqr`: the score fill reads across
/// their boundary (`sf` is `tlen_pad` long).
///
/// `s` is `tlen_pad + S_SLACK` long.
#[inline]
pub(super) fn split_main_buf(
    buf: &mut [u8],
    tlen_pad: usize,
    qlen_pad: usize,
) -> (
    &mut [u8],
    &mut [u8],
    &mut [u8],
    &mut [u8],
    &mut [i8],
    &mut [u8],
) {
    let (u, rest) = buf.split_at_mut(tlen_pad);
    let (v, rest) = rest.split_at_mut(tlen_pad);
    let (x, rest) = rest.split_at_mut(tlen_pad);
    let (y, rest) = rest.split_at_mut(tlen_pad);
    let (s_raw, rest) = rest.split_at_mut(tlen_pad + S_SLACK);
    let (sfqr, _) = rest.split_at_mut(tlen_pad + qlen_pad + 32);
    // SAFETY: i8 has the same size and alignment as u8; the slice is valid.
    let s = unsafe { std::slice::from_raw_parts_mut(s_raw.as_mut_ptr() as *mut i8, s_raw.len()) };
    (u, v, x, y, s, sfqr)
}

pub(super) struct ScalarOps;
#[allow(dead_code)]
pub(super) struct Sse41Ops;
#[allow(dead_code)]
pub(super) struct Avx2Ops;
#[allow(dead_code)] // constructed only on aarch64; dead on other targets (cf. Sse41Ops/Avx2Ops)
pub(super) struct NeonOps;

impl BackendOps for ScalarOps {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sfqr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    ) {
        fill_scores_fast_row_scalar(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
    }

    #[inline(always)]
    unsafe fn dp_row_score_only<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        mut x1: u8,
        mut v1: u8,
    ) {
        for t in stv..=env {
            let mut z = i32::from(s[t]) + qe2;
            let xt1 = x1;
            let vt1 = v1;
            x1 = x[t];
            v1 = v[t];
            let a = i32::from(xt1) + i32::from(vt1);
            let ut = u[t];
            let b = i32::from(y[t]) + i32::from(ut);
            if z < 0 {
                z = 0;
            }
            if RIGHT {
                if z <= a {
                    z = a;
                }
                if z <= b {
                    z = b;
                }
            } else {
                if a > z {
                    z = a;
                }
                if b > z {
                    z = b;
                }
            }
            z = min(z, max_sc_transformed);
            u[t] = (z - i32::from(vt1)) as u8;
            v[t] = (z - i32::from(ut)) as u8;
            let zq = z - q;
            let na = a - zq;
            let nb = b - zq;
            x[t] = if RIGHT {
                if na >= 0 { na as u8 } else { 0 }
            } else if na > 0 {
                na as u8
            } else {
                0
            };
            y[t] = if RIGHT {
                if nb >= 0 { nb as u8 } else { 0 }
            } else if nb > 0 {
                nb as u8
            } else {
                0
            };
        }
    }

    #[inline(always)]
    unsafe fn dp_row_traceback<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        p_row: &mut [u8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        dp_row_traceback_scalar(
            RIGHT,
            u,
            v,
            x,
            y,
            s,
            p_row,
            stv,
            env,
            q,
            qe2,
            max_sc_transformed,
            x1,
            v1,
        );
    }

    #[inline(always)]
    unsafe fn update_h_exact(
        h: &mut [i32],
        v: &[u8],
        u: &[u8],
        st0: usize,
        en0: usize,
        qe: i32,
    ) -> (i32, i32) {
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

impl BackendOps for Sse41Ops {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sfqr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fill_scores_fast_row_sse41(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
        }
        #[cfg(not(target_arch = "x86_64"))]
        fill_scores_fast_row_scalar(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
    }

    #[inline(always)]
    unsafe fn dp_row_score_only<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if RIGHT {
                dp_row_score_only_right_sse41(
                    u,
                    v,
                    x,
                    y,
                    s,
                    stv,
                    env,
                    q,
                    qe2,
                    max_sc_transformed,
                    x1,
                    v1,
                );
            } else {
                dp_row_score_only_sse41(
                    u,
                    v,
                    x,
                    y,
                    s,
                    stv,
                    env,
                    q,
                    qe2,
                    max_sc_transformed,
                    x1,
                    v1,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            ScalarOps::dp_row_score_only::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
    }

    #[inline(always)]
    unsafe fn dp_row_traceback<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        p_row: &mut [u8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            dp_row_traceback_sse41(
                RIGHT,
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        dp_row_traceback_scalar(
            RIGHT,
            u,
            v,
            x,
            y,
            s,
            p_row,
            stv,
            env,
            q,
            qe2,
            max_sc_transformed,
            x1,
            v1,
        );
    }

    #[inline(always)]
    unsafe fn update_h_exact(
        h: &mut [i32],
        v: &[u8],
        u: &[u8],
        st0: usize,
        en0: usize,
        qe: i32,
    ) -> (i32, i32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            update_h_exact_lanes_sse41(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "x86_64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

impl BackendOps for Avx2Ops {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sfqr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fill_scores_fast_row_avx2(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
        }
        #[cfg(not(target_arch = "x86_64"))]
        fill_scores_fast_row_scalar(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
    }

    #[inline(always)]
    unsafe fn dp_row_score_only<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        unsafe {
            Sse41Ops::dp_row_score_only::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
    }

    #[inline(always)]
    unsafe fn dp_row_traceback<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        p_row: &mut [u8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        unsafe {
            Sse41Ops::dp_row_traceback::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
    }

    #[inline(always)]
    unsafe fn update_h_exact(
        h: &mut [i32],
        v: &[u8],
        u: &[u8],
        st0: usize,
        en0: usize,
        qe: i32,
    ) -> (i32, i32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            update_h_exact_lanes_sse41(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "x86_64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

impl BackendOps for NeonOps {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sfqr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            fill_scores_fast_row_neon(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
        }
        #[cfg(not(target_arch = "aarch64"))]
        fill_scores_fast_row_scalar(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
    }

    #[inline(always)]
    unsafe fn dp_row_score_only<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if RIGHT {
                dp_row_score_only_right_neon(
                    u,
                    v,
                    x,
                    y,
                    s,
                    stv,
                    env,
                    q,
                    qe2,
                    max_sc_transformed,
                    x1,
                    v1,
                );
            } else {
                dp_row_score_only_neon(u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            ScalarOps::dp_row_score_only::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
    }

    #[inline(always)]
    unsafe fn dp_row_traceback<const RIGHT: bool>(
        u: &mut [u8],
        v: &mut [u8],
        x: &mut [u8],
        y: &mut [u8],
        s: &[i8],
        p_row: &mut [u8],
        stv: usize,
        env: usize,
        q: i32,
        qe2: i32,
        max_sc_transformed: i32,
        x1: u8,
        v1: u8,
    ) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            dp_row_traceback_neon(
                RIGHT,
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv,
                env,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        dp_row_traceback_scalar(
            RIGHT,
            u,
            v,
            x,
            y,
            s,
            p_row,
            stv,
            env,
            q,
            qe2,
            max_sc_transformed,
            x1,
            v1,
        );
    }

    #[inline(always)]
    unsafe fn update_h_exact(
        h: &mut [i32],
        v: &[u8],
        u: &[u8],
        st0: usize,
        en0: usize,
        qe: i32,
    ) -> (i32, i32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            update_h_exact_lanes_neon(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "aarch64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

// The core is generic over the backend and carries no target feature itself:
// it has to be inlined into the `#[target_feature]` entry points in `mod.rs`,
// or the kernels (which do carry one) cannot be inlined into it and every
// anti-diagonal pays three calls, with the glue compiled for the baseline.

#[inline(always)]
fn extz2_core_score_only<B: BackendOps>(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    if (input.flag & KSW_EZ_APPROX_MAX) != 0 {
        if (input.flag & KSW_EZ_RIGHT) != 0 {
            extz2_core_score_only_impl::<B, true, true>(input, ez, ws);
        } else {
            extz2_core_score_only_impl::<B, true, false>(input, ez, ws);
        }
    } else {
        if (input.flag & KSW_EZ_RIGHT) != 0 {
            extz2_core_score_only_impl::<B, false, true>(input, ez, ws);
        } else {
            extz2_core_score_only_impl::<B, false, false>(input, ez, ws);
        }
    }
}

#[inline(always)]
fn extz2_core_score_only_impl<B: BackendOps, const APPROX: bool, const RIGHT: bool>(
    input: &Extz2Input<'_>,
    ez: &mut Extz,
    ws: &mut Workspace,
) {
    ez.reset();

    let qlen = input.query.len();
    let tlen = input.target.len();
    let m = input.m as i32;
    if m <= 0 || qlen == 0 || tlen == 0 {
        return;
    }

    let q = i32::from(input.q);
    let e = i32::from(input.e);
    let qe = q + e;
    // see KSW_EZ_NO_MAX: the maximum is only skipped when z-drop is off
    let no_max = (input.flag & super::KSW_EZ_NO_MAX) != 0 && input.zdrop < 0;
    let qe2 = qe * 2;

    // A band at least as wide as the longer sequence never binds; clamping
    // it also keeps `r + w` from overflowing.
    let mut w = input.w;
    if w < 0 || w as usize > qlen.max(tlen) {
        w = qlen.max(tlen) as i32;
    }
    let (wl, wr) = (w, w);

    let mut max_sc = i32::from(input.mat[0]);
    let mut min_sc = i32::from(input.mat[0]);
    for &x in input.mat {
        let xi = i32::from(x);
        max_sc = max(max_sc, xi);
        min_sc = min(min_sc, xi);
    }
    // outside the 8-bit cell range, see `Extz2Input`
    if -min_sc > 2 * qe || max_sc + 2 * qe > 255 {
        return;
    }

    let generic_sc = (input.flag & KSW_EZ_GENERIC_SC) != 0;
    ws.prepare_score_only(qlen, tlen, APPROX);
    let tlen_pad = ws.tlen_pad;
    let qlen_pad = ws.qlen_pad;
    let h = &mut ws.h;
    let (u, v, x, y, s, sfqr) = split_main_buf(&mut ws.buf, tlen_pad, qlen_pad);

    sfqr[..tlen].copy_from_slice(input.target);
    for (i, qv) in input.query.iter().enumerate() {
        sfqr[tlen_pad + qlen - 1 - i] = *qv;
    }

    let sc_mch = input.mat[0];
    let sc_mis = input.mat[1];
    let sc_n = if input.mat[(m * m - 1) as usize] == 0 {
        -(input.e)
    } else {
        input.mat[(m * m - 1) as usize]
    };
    let wildcard = (m - 1) as u8;
    let max_sc_transformed = max_sc + qe2;

    let rows = qlen + tlen - 1;
    let mut last_st = -1i32;
    let mut last_en = -1i32;
    let mut h0 = 0i32;
    let mut last_h0_t = 0i32;

    for r in 0..rows as i32 {
        let mut st = 0i32;
        let mut en = tlen as i32 - 1;
        if st < r - qlen as i32 + 1 {
            st = r - qlen as i32 + 1;
        }
        if en > r {
            en = r;
        }
        if st < (r - wr + 1) >> 1 {
            st = (r - wr + 1) >> 1;
        }
        if en > (r + wl) >> 1 {
            en = (r + wl) >> 1;
        }
        if st > en {
            ez.zdropped = true;
            break;
        }

        let st0 = st as usize;
        let en0 = en as usize;
        let stv = (st0 / 16 * 16) as i32;
        let env = ((en0 + 16) / 16 * 16 - 1) as i32;
        let qrr_base = qlen as i32 - 1 - r;

        if stv > 0 {
            let s1 = (stv - 1) as usize;
            if !((stv - 1) >= last_st && (stv - 1) <= last_en) {
                x[s1] = 0;
                v[s1] = 0;
            }
        }
        if env >= r {
            let rr = r as usize;
            y[rr] = 0;
            u[rr] = if r != 0 { input.q as u8 } else { 0 };
        }

        if !generic_sc {
            // SAFETY: selected once from runtime-dispatched backend.
            unsafe {
                B::fill_fast_row(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
            }
        } else {
            // Only `[st0, en0]`, as in C: the padding keeps what earlier
            // anti-diagonals left there.
            for t in st0..=en0 {
                let sq = sfqr[t] as usize;
                let qidx = (qrr_base + t as i32) as usize;
                let qq = sfqr[tlen_pad + qidx] as usize;
                s[t] = input.mat[sq * m as usize + qq];
            }
        }

        let x1 = if stv > 0 {
            let idx = (stv - 1) as usize;
            if (stv - 1) >= last_st && (stv - 1) <= last_en {
                x[idx]
            } else {
                0
            }
        } else {
            0
        };
        let v1 = if stv > 0 {
            let idx = (stv - 1) as usize;
            if (stv - 1) >= last_st && (stv - 1) <= last_en {
                v[idx]
            } else {
                0
            }
        } else if r != 0 {
            input.q as u8
        } else {
            0
        };

        unsafe {
            B::dp_row_score_only::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                stv as usize,
                env as usize,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }

        if !APPROX {
            let max_h;
            let max_t;
            if r > 0 && no_max {
                update_h_no_max(h, v, u, st0, en0, qe);
                (max_h, max_t) = (KSW_NEG_INF, 0);
            } else if r > 0 {
                unsafe {
                    (max_h, max_t) = B::update_h_exact(h, v, u, st0, en0, qe);
                }
            } else {
                h[0] = i32::from(v[0]) - qe - qe;
                max_h = h[0];
                max_t = 0;
            }

            if en0 + 1 == tlen && h[en0] > ez.mte {
                ez.mte = h[en0];
                ez.mte_q = r - en0 as i32;
            }
            if r - st0 as i32 == qlen as i32 - 1 && h[st0] > ez.mqe {
                ez.mqe = h[st0];
                ez.mqe_t = st0 as i32;
            }
            if !no_max && apply_zdrop(ez, max_h, r, max_t, input.zdrop, input.e) {
                break;
            }
            if r == rows as i32 - 1 && en0 + 1 == tlen {
                ez.score = h[tlen - 1];
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0 as i32
                    && last_h0_t <= en0 as i32
                    && last_h0_t + 1 >= st0 as i32
                    && last_h0_t + 1 <= en0 as i32
                {
                    let d0 = i32::from(v[last_h0_t as usize]) - qe;
                    let d1 = i32::from(u[last_h0_t as usize + 1]) - qe;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 as i32 && last_h0_t <= en0 as i32 {
                    h0 += i32::from(v[last_h0_t as usize]) - qe;
                } else {
                    last_h0_t += 1;
                    h0 += i32::from(u[last_h0_t as usize]) - qe;
                }
                if (input.flag & KSW_EZ_APPROX_DROP) != 0
                    && apply_zdrop(ez, h0, r, last_h0_t, input.zdrop, input.e)
                {
                    break;
                }
            } else {
                h0 = i32::from(v[0]) - qe - qe;
                last_h0_t = 0;
            }
            if r == rows as i32 - 1 && en0 + 1 == tlen {
                ez.score = h0;
            }
        }

        last_st = stv;
        last_en = env;
    }
}

#[inline(always)]
fn extz2_core_traceback_impl<B: BackendOps, const APPROX: bool, const RIGHT: bool>(
    input: &Extz2Input<'_>,
    ez: &mut Extz,
    ws: &mut Workspace,
) {
    ez.reset();

    let qlen = input.query.len();
    let tlen = input.target.len();
    let m = input.m as i32;
    if m <= 0 || qlen == 0 || tlen == 0 {
        return;
    }

    let q = i32::from(input.q);
    let e = i32::from(input.e);
    let qe = q + e;
    // see KSW_EZ_NO_MAX: the maximum is only skipped when z-drop is off
    // KSW_EZ_NO_MAX applies to score-only alignment: the traceback start
    // below depends on the tracked maximum
    let no_max = false;
    let qe2 = qe * 2;

    // A band at least as wide as the longer sequence never binds; clamping
    // it also keeps `r + w` from overflowing.
    let mut w = input.w;
    if w < 0 || w as usize > qlen.max(tlen) {
        w = qlen.max(tlen) as i32;
    }
    let (wl, wr) = (w, w);

    let mut max_sc = i32::from(input.mat[0]);
    let mut min_sc = i32::from(input.mat[0]);
    for &x in input.mat {
        let xi = i32::from(x);
        max_sc = max(max_sc, xi);
        min_sc = min(min_sc, xi);
    }
    // outside the 8-bit cell range, see `Extz2Input`
    if -min_sc > 2 * qe || max_sc + 2 * qe > 255 {
        return;
    }

    let generic_sc = (input.flag & KSW_EZ_GENERIC_SC) != 0;

    let rows = qlen + tlen - 1;
    // Row stride of the traceback matrix, in whole 16-cell blocks as in C: a
    // row of `n` cells spans at most `ceil(n / 16) + 1` blocks. A narrower
    // stride lets the padding of one row run into the next, and the
    // backtrack reads that padding at a band edge.
    let n_col = qlen.min(tlen).min(w as usize + 1);
    let n_col = (n_col.div_ceil(16) + 1) * 16;
    ws.prepare_traceback(qlen, tlen, APPROX, rows, n_col);
    let tlen_pad = ws.tlen_pad;
    let qlen_pad = ws.qlen_pad;
    let h = &mut ws.h;
    let off = &mut ws.off;
    let off_end = &mut ws.off_end;
    let p = &mut ws.p;
    let (u, v, x, y, s, sfqr) = split_main_buf(&mut ws.buf, tlen_pad, qlen_pad);

    sfqr[..tlen].copy_from_slice(input.target);
    for (i, qv) in input.query.iter().enumerate() {
        sfqr[tlen_pad + qlen - 1 - i] = *qv;
    }

    let sc_mch = input.mat[0];
    let sc_mis = input.mat[1];
    let sc_n = if input.mat[(m * m - 1) as usize] == 0 {
        -(input.e)
    } else {
        input.mat[(m * m - 1) as usize]
    };
    let wildcard = (m - 1) as u8;
    let max_sc_transformed = max_sc + qe2;

    let mut last_st = -1i32;
    let mut last_en = -1i32;
    let mut h0 = 0i32;
    let mut last_h0_t = 0i32;

    for r in 0..rows as i32 {
        let mut st = 0i32;
        let mut en = tlen as i32 - 1;

        if st < r - qlen as i32 + 1 {
            st = r - qlen as i32 + 1;
        }
        if en > r {
            en = r;
        }
        if st < (r - wr + 1) >> 1 {
            st = (r - wr + 1) >> 1;
        }
        if en > (r + wl) >> 1 {
            en = (r + wl) >> 1;
        }
        if st > en {
            ez.zdropped = true;
            break;
        }

        let st0 = st as usize;
        let en0 = en as usize;
        let stv = (st0 / 16 * 16) as i32;
        let env = ((en0 + 16) / 16 * 16 - 1) as i32;
        let qrr_base = qlen as i32 - 1 - r;

        if stv > 0 {
            let s1 = (stv - 1) as usize;
            if !((stv - 1) >= last_st && (stv - 1) <= last_en) {
                x[s1] = 0;
                v[s1] = 0;
            }
        }
        if env >= r {
            let rr = r as usize;
            y[rr] = 0;
            u[rr] = if r != 0 { input.q as u8 } else { 0 };
        }

        // Match the C loop-fission semantics: in fast scoring mode the loop
        // stores 16-byte chunks and may spill past `en0` up to the rounded
        // vector boundary.
        if !generic_sc {
            // SAFETY: selected once from runtime-dispatched backend.
            unsafe {
                B::fill_fast_row(sfqr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s);
            }
        } else {
            // Only `[st0, en0]`, as in C: the padding keeps what earlier
            // anti-diagonals left there.
            for t in st0..=en0 {
                let sq = sfqr[t] as usize;
                let qidx = (qrr_base + t as i32) as usize;
                let qq = sfqr[tlen_pad + qidx] as usize;
                s[t] = input.mat[sq * m as usize + qq];
            }
        }

        let x1 = if stv > 0 {
            let idx = (stv - 1) as usize;
            if (stv - 1) >= last_st && (stv - 1) <= last_en {
                x[idx]
            } else {
                0
            }
        } else {
            0
        };
        let v1 = if stv > 0 {
            let idx = (stv - 1) as usize;
            if (stv - 1) >= last_st && (stv - 1) <= last_en {
                v[idx]
            } else {
                0
            }
        } else if r != 0 {
            input.q as u8
        } else {
            0
        };

        off[r as usize] = stv;
        off_end[r as usize] = env;

        let row_off = r as usize * n_col;
        let row_len = env as usize - stv as usize + 1;
        let p_row = &mut p[row_off..row_off + row_len];
        unsafe {
            B::dp_row_traceback::<RIGHT>(
                u,
                v,
                x,
                y,
                s,
                p_row,
                stv as usize,
                env as usize,
                q,
                qe2,
                max_sc_transformed,
                x1,
                v1,
            );
        }

        if !APPROX {
            let max_h;
            let max_t;

            if r > 0 && no_max {
                update_h_no_max(h, v, u, st0, en0, qe);
                (max_h, max_t) = (KSW_NEG_INF, 0);
            } else if r > 0 {
                // SAFETY: selected once from runtime-dispatched backend.
                unsafe {
                    (max_h, max_t) = B::update_h_exact(h, v, u, st0, en0, qe);
                }
            } else {
                h[0] = i32::from(v[0]) - qe - qe;
                max_h = h[0];
                max_t = 0;
            }

            if en0 + 1 == tlen && h[en0] > ez.mte {
                ez.mte = h[en0];
                ez.mte_q = r - en0 as i32;
            }
            if r - st0 as i32 == qlen as i32 - 1 && h[st0] > ez.mqe {
                ez.mqe = h[st0];
                ez.mqe_t = st0 as i32;
            }
            if !no_max && apply_zdrop(ez, max_h, r, max_t, input.zdrop, input.e) {
                break;
            }
            if r == rows as i32 - 1 && en0 + 1 == tlen {
                ez.score = h[tlen - 1];
            }
        } else {
            if r > 0 {
                if last_h0_t >= st0 as i32
                    && last_h0_t <= en0 as i32
                    && last_h0_t + 1 >= st0 as i32
                    && last_h0_t + 1 <= en0 as i32
                {
                    let d0 = i32::from(v[last_h0_t as usize]) - qe;
                    let d1 = i32::from(u[last_h0_t as usize + 1]) - qe;
                    if d0 > d1 {
                        h0 += d0;
                    } else {
                        h0 += d1;
                        last_h0_t += 1;
                    }
                } else if last_h0_t >= st0 as i32 && last_h0_t <= en0 as i32 {
                    h0 += i32::from(v[last_h0_t as usize]) - qe;
                } else {
                    last_h0_t += 1;
                    h0 += i32::from(u[last_h0_t as usize]) - qe;
                }

                if (input.flag & KSW_EZ_APPROX_DROP) != 0
                    && apply_zdrop(ez, h0, r, last_h0_t, input.zdrop, input.e)
                {
                    break;
                }
            } else {
                h0 = i32::from(v[0]) - qe - qe;
                last_h0_t = 0;
            }

            if r == rows as i32 - 1 && en0 + 1 == tlen {
                ez.score = h0;
            }
        }

        last_st = stv;
        last_en = env;
    }

    let rev_cigar = (input.flag & KSW_EZ_REV_CIGAR) != 0;
    if !ez.zdropped && (input.flag & KSW_EZ_EXTZ_ONLY) == 0 {
        backtrack(
            rev_cigar,
            0,
            &p,
            &off,
            &off_end,
            n_col,
            tlen as i32 - 1,
            qlen as i32 - 1,
            &mut ez.cigar,
        );
    } else if !ez.zdropped
        && (input.flag & KSW_EZ_EXTZ_ONLY) != 0
        && ez.mqe + input.end_bonus > ez.max as i32
    {
        ez.reach_end = true;
        backtrack(
            rev_cigar,
            0,
            &p,
            &off,
            &off_end,
            n_col,
            ez.mqe_t,
            qlen as i32 - 1,
            &mut ez.cigar,
        );
    } else if ez.max_t >= 0 && ez.max_q >= 0 {
        backtrack(
            rev_cigar,
            0,
            &p,
            &off,
            &off_end,
            n_col,
            ez.max_t,
            ez.max_q,
            &mut ez.cigar,
        );
    }
}

#[inline(always)]
pub(super) fn extz2_core<B: BackendOps>(input: &Extz2Input<'_>, ez: &mut Extz, ws: &mut Workspace) {
    let with_cigar = (input.flag & KSW_EZ_SCORE_ONLY) == 0;
    if !with_cigar {
        extz2_core_score_only::<B>(input, ez, ws);
        return;
    }
    let approx = (input.flag & KSW_EZ_APPROX_MAX) != 0;
    let right = (input.flag & KSW_EZ_RIGHT) != 0;
    match (approx, right) {
        (false, false) => extz2_core_traceback_impl::<B, false, false>(input, ez, ws),
        (false, true) => extz2_core_traceback_impl::<B, false, true>(input, ez, ws),
        (true, false) => extz2_core_traceback_impl::<B, true, false>(input, ez, ws),
        (true, true) => extz2_core_traceback_impl::<B, true, true>(input, ez, ws),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane-wise exact-max update must match `update_h_exact_scalar`
    /// exactly, ties included (the first position of the maximum wins, and
    /// `h[en0]` wins unless strictly beaten). Small value ranges force ties.
    #[test]
    fn lane_exact_max_matches_scalar() {
        #[allow(unused_mut)]
        let mut kernels: Vec<(
            &str,
            fn(&mut [i32], &[u8], &[u8], usize, usize, i32) -> (i32, i32),
        )> = Vec::new();
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            kernels.push(("neon", |h, v, u, st0, en0, qe| unsafe {
                update_h_exact_lanes_neon(h, v, u, st0, en0, qe)
            }));
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", |h, v, u, st0, en0, qe| unsafe {
                update_h_exact_lanes_sse41(h, v, u, st0, en0, qe)
            }));
        }
        let mut seed = 7u64;
        let mut next = move |m: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        for (name, kernel) in kernels {
            for _ in 0..20_000 {
                let len = 1 + next(120) as usize;
                let st0 = next(len as u64) as usize;
                let en0 = st0 + next((len - st0) as u64) as usize;
                let span = 1 + next(6) as i32;
                // `h` and `v` hold whole 4-cell groups, as in a `Workspace`
                let h0: Vec<i32> = (0..len + 3)
                    .map(|_| next(span as u64) as i32 - span / 2)
                    .collect();
                let v: Vec<u8> = (0..len + 3).map(|_| next(4) as u8).collect();
                let u: Vec<u8> = (0..len).map(|_| next(4) as u8).collect();
                let qe = next(4) as i32;
                let (mut ha, mut hb) = (h0.clone(), h0);
                let want = update_h_exact_scalar(&mut ha, &v, &u, st0, en0, qe);
                let got = kernel(&mut hb, &v, &u, st0, en0, qe);
                assert_eq!((got, &hb), (want, &ha), "{name}: st0={st0} en0={en0}");
            }
        }
    }

    /// Every x86 kernel the CPU supports gives the scalar result, not only
    /// the one runtime dispatch picks (which hides SSE4.1 behind AVX2).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn each_x86_kernel_matches_scalar() {
        let mut kernels: Vec<(&str, KernelFn)> = Vec::new();
        if std::is_x86_feature_detected!("sse4.1") {
            kernels.push(("sse4.1", extz2_sse41));
        }
        if std::is_x86_feature_detected!("avx2") {
            kernels.push(("avx2", extz2_avx2));
        }
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % n
        };
        let mat = mm_mat(2, -4);
        let flags = [
            KSW_EZ_SCORE_ONLY,
            KSW_EZ_SCORE_ONLY | KSW_EZ_RIGHT,
            0,
            KSW_EZ_RIGHT,
            KSW_EZ_EXTZ_ONLY,
            KSW_EZ_SCORE_ONLY | KSW_EZ_APPROX_MAX | KSW_EZ_APPROX_DROP,
            KSW_EZ_GENERIC_SC,
            KSW_EZ_SCORE_ONLY | KSW_EZ_NO_MAX,
        ];
        let (mut ws, mut ws_scalar) = (Workspace::default(), Workspace::default());
        let (mut got, mut want) = (Extz::default(), Extz::default());
        for case in 0..10_000 {
            let ql = 1 + rnd(if case % 50 == 0 { 700 } else { 200 }) as usize;
            let tl = (ql as i64 + rnd(31) as i64 - 15).max(1) as usize;
            let t: Vec<u8> = (0..tl).map(|_| rnd(5) as u8).collect();
            let q: Vec<u8> = (0..ql)
                .map(|i| {
                    if i < tl && rnd(8) != 0 {
                        t[i]
                    } else {
                        rnd(5) as u8
                    }
                })
                .collect();
            let input = Extz2Input {
                query: &q,
                target: &t,
                m: 5,
                mat: &mat,
                q: 4 + rnd(4) as i8,
                e: 1 + rnd(2) as i8,
                w: [-1, 1, 4, 15, 31][rnd(5) as usize],
                zdrop: [-1, 30, 100][rnd(3) as usize],
                end_bonus: 0,
                flag: flags[rnd(flags.len() as u64) as usize],
            };
            extz2_scalar_with_workspace(&input, &mut want, &mut ws_scalar);
            for &(name, kernel) in &kernels {
                // SAFETY: the kernel's target feature was detected above.
                unsafe { kernel(&input, &mut got, &mut ws) };
                let what = format!(
                    "{name} case {case}: ql={ql} tl={tl} w={} flag={:#x}",
                    input.w, input.flag
                );
                assert_eq!(
                    (got.max, got.zdropped, got.max_q, got.max_t),
                    (want.max, want.zdropped, want.max_q, want.max_t),
                    "{what}"
                );
                assert_eq!(
                    (got.mqe, got.mqe_t, got.mte, got.mte_q, got.score),
                    (want.mqe, want.mqe_t, want.mte, want.mte_q, want.score),
                    "{what}"
                );
                assert_eq!(
                    (got.reach_end, &got.cigar),
                    (want.reach_end, &want.cigar),
                    "{what}"
                );
            }
        }
    }

    fn mm_mat(match_score: i8, mismatch_score: i8) -> [i8; 25] {
        let mut mat = [mismatch_score; 25];
        for i in 0..5 {
            mat[i * 5 + i] = match_score;
        }
        mat[24] = 0;
        mat
    }

    #[test]
    fn identical_sequences_produce_full_match() {
        let mat = mm_mat(2, -4);
        let q = [0, 1, 2, 3, 0, 1, 2, 3];
        let t = q;

        let input = Extz2Input {
            query: &q,
            target: &t,
            m: 5,
            mat: &mat,
            q: 4,
            e: 2,
            w: -1,
            zdrop: -1,
            end_bonus: 0,
            flag: 0,
        };

        let mut ez = Extz::default();
        extz2(&input, &mut ez);

        assert!(!ez.zdropped);
        assert!(ez.score > 0);
        assert_eq!(ez.cigar.len(), 1);
        assert_eq!(ez.cigar[0] & 0x0f, KSW_CIGAR_MATCH);
        assert_eq!(ez.cigar[0] >> 4, q.len() as u32);
    }

    /// `KSW_EZ_NO_MAX` leaves `mqe`, `mte` and `score` exactly as the exact
    /// path computes them in score-only mode, for both gap alignments and on
    /// the dispatched and scalar backends; with a CIGAR requested the flag is
    /// ignored, so the whole result is unchanged.
    #[test]
    fn no_max_keeps_mqe_mte_and_score() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        let mat = mm_mat(2, -4);
        let mut aligner = crate::Aligner::new();
        for case in 0..5_000 {
            let ql = 1 + rnd(160) as usize;
            let tl = 1 + rnd(200) as usize;
            let t: Vec<u8> = (0..tl).map(|_| rnd(5) as u8).collect();
            let q: Vec<u8> = (0..ql)
                .map(|i| {
                    if i < tl && rnd(8) != 0 {
                        t[i]
                    } else {
                        rnd(5) as u8
                    }
                })
                .collect();
            let w = if case % 3 == 0 { -1 } else { rnd(40) as i32 };
            for base in [
                0,
                KSW_EZ_SCORE_ONLY,
                KSW_EZ_RIGHT,
                KSW_EZ_SCORE_ONLY | KSW_EZ_RIGHT,
            ] {
                let input = Extz2Input {
                    query: &q,
                    target: &t,
                    m: 5,
                    mat: &mat,
                    q: 6,
                    e: 2,
                    w,
                    zdrop: -1,
                    end_bonus: 0,
                    flag: base,
                };
                let fast_input = Extz2Input {
                    flag: base | KSW_EZ_NO_MAX,
                    ..input
                };
                for scalar in [false, true] {
                    let (want, got) = if scalar {
                        let a = aligner.align_scalar(&input).clone();
                        (a, aligner.align_scalar(&fast_input).clone())
                    } else {
                        let a = aligner.align(&input).clone();
                        (a, aligner.align(&fast_input).clone())
                    };
                    let what = format!("case {case} flag {base:#x} w {w} scalar {scalar}");
                    assert_eq!(got.mqe, want.mqe, "{what}");
                    assert_eq!(got.mqe_t, want.mqe_t, "{what}");
                    assert_eq!(got.mte, want.mte, "{what}");
                    assert_eq!(got.mte_q, want.mte_q, "{what}");
                    assert_eq!(got.score, want.score, "{what}");
                    if base & KSW_EZ_SCORE_ONLY == 0 {
                        assert_eq!(got.cigar, want.cigar, "{what}");
                    }
                }
            }
        }
    }

    #[test]
    fn score_only_mode_skips_cigar() {
        let mat = mm_mat(2, -4);
        let q = [0, 1, 2, 3];
        let t = [0, 1, 2, 3];
        let input = Extz2Input {
            query: &q,
            target: &t,
            m: 5,
            mat: &mat,
            q: 4,
            e: 2,
            w: -1,
            zdrop: -1,
            end_bonus: 0,
            flag: KSW_EZ_SCORE_ONLY,
        };

        let mut ez = Extz::default();
        extz2(&input, &mut ez);

        assert!(ez.cigar.is_empty());
        assert!(ez.score > 0);
    }

    /// Verify that reusing an Aligner across calls with varying sequence
    /// lengths produces the same results as fresh one-shot calls.
    #[test]
    fn aligner_reuse_matches_fresh_calls() {
        use crate::Aligner;

        let mat = mm_mat(1, -4);
        // Pairs with different lengths to exercise workspace resizing.
        let cases: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (vec![0, 1, 2, 3, 0, 1, 2, 3], vec![0, 1, 2, 3, 0, 1, 2, 3]),
            (vec![0, 1, 2], vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
            (
                vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                vec![0, 1, 2, 3],
            ),
            (vec![0; 50], vec![0; 50]),
            (vec![0, 1, 2], vec![3, 3, 3]),
            (vec![0, 1, 2, 3, 0, 1, 2, 3], vec![0, 1, 2, 3, 0, 1, 2, 3]),
        ];

        let flags = [
            0,
            KSW_EZ_EXTZ_ONLY,
            KSW_EZ_EXTZ_ONLY | KSW_EZ_APPROX_MAX | KSW_EZ_APPROX_DROP,
        ];

        for flag in flags {
            let mut aligner = Aligner::new();
            for (q, t) in &cases {
                let input = Extz2Input {
                    query: q,
                    target: t,
                    m: 5,
                    mat: &mat,
                    q: 4,
                    e: 1,
                    w: -1,
                    zdrop: 40,
                    end_bonus: 0,
                    flag,
                };

                // Fresh call
                let mut fresh_ez = Extz::default();
                extz2(&input, &mut fresh_ez);

                // Reused aligner
                let reused_ez = aligner.align(&input);

                assert_eq!(
                    reused_ez.score,
                    fresh_ez.score,
                    "score mismatch: flag={flag} qlen={} tlen={}",
                    q.len(),
                    t.len(),
                );
                assert_eq!(
                    reused_ez.max,
                    fresh_ez.max,
                    "max mismatch: flag={flag} qlen={} tlen={}",
                    q.len(),
                    t.len(),
                );
                assert_eq!(
                    reused_ez.cigar,
                    fresh_ez.cigar,
                    "cigar mismatch: flag={flag} qlen={} tlen={}",
                    q.len(),
                    t.len(),
                );
                assert_eq!(
                    reused_ez.zdropped,
                    fresh_ez.zdropped,
                    "zdropped mismatch: flag={flag} qlen={} tlen={}",
                    q.len(),
                    t.len(),
                );
            }
        }
    }
}
