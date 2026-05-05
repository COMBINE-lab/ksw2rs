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

#[inline]
fn fill_scores_fast_row_scalar(
    sf: &[u8],
    qr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    for idx in st0..=en0 {
        let sq = sf[idx];
        let qidx = qrr_base + idx as i32;
        // qpos >= 0 is guaranteed for idx in [st0, en0]; the cast is safe.
        let qq = qr[qidx as usize];
        s[idx] = if sq == wildcard || qq == wildcard {
            sc_n
        } else if sq == qq {
            sc_mch
        } else {
            sc_mis
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_scores_fast_row_avx2(
    sf: &[u8],
    qr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    _tlen_pad: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::x86_64::*;

    let wc = _mm256_set1_epi8(wildcard as i8);
    let mch = _mm256_set1_epi8(sc_mch);
    let mis = _mm256_set1_epi8(sc_mis);
    let nsc = _mm256_set1_epi8(sc_n);

    // The flat Workspace buffer guarantees sf is followed by at least 16 bytes
    // of qr, and qr has a 16-byte trailing pad, so SIMD loads are always safe
    // for t in [st0, en0].  qpos >= 0 is analytically guaranteed for valid t.
    let mut t = st0;
    while t + 32 <= en0 + 1 {
        let qpos = (qrr_base + t as i32) as usize;
        // SAFETY: flat buffer layout ensures sf[t..t+31] and qr[qpos..qpos+31] are in-bounds.
        let sq = unsafe { _mm256_loadu_si256(sf.as_ptr().add(t) as *const __m256i) };
        let qq = unsafe { _mm256_loadu_si256(qr.as_ptr().add(qpos) as *const __m256i) };
        let mask_wc = _mm256_or_si256(_mm256_cmpeq_epi8(sq, wc), _mm256_cmpeq_epi8(qq, wc));
        let eq = _mm256_cmpeq_epi8(sq, qq);
        let mut tmp = _mm256_blendv_epi8(mis, mch, eq);
        tmp = _mm256_blendv_epi8(tmp, nsc, mask_wc);
        // SAFETY: s has tlen_pad bytes; t+31 <= en0+30 < tlen_pad+31, sf overshoot reads into qr pad.
        unsafe { _mm256_storeu_si256(s.as_mut_ptr().add(t) as *mut __m256i, tmp) };
        t += 32;
    }
    // Handle the tail (0, 1, or 2 remaining 16-byte chunks) with SSE.
    let sse_wc = _mm_set1_epi8(wildcard as i8);
    let sse_mch = _mm_set1_epi8(sc_mch);
    let sse_mis = _mm_set1_epi8(sc_mis);
    let sse_nsc = _mm_set1_epi8(sc_n);
    while t <= en0 {
        let qpos = (qrr_base + t as i32) as usize;
        // SAFETY: same flat-buffer guarantee, 16-byte load.
        let sq = unsafe { _mm_loadu_si128(sf.as_ptr().add(t) as *const __m128i) };
        let qq = unsafe { _mm_loadu_si128(qr.as_ptr().add(qpos) as *const __m128i) };
        let mask_wc = _mm_or_si128(_mm_cmpeq_epi8(sq, sse_wc), _mm_cmpeq_epi8(qq, sse_wc));
        let eq = _mm_cmpeq_epi8(sq, qq);
        let mut tmp = _mm_blendv_epi8(sse_mis, sse_mch, eq);
        tmp = _mm_blendv_epi8(tmp, sse_nsc, mask_wc);
        unsafe { _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp) };
        t += 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn fill_scores_fast_row_sse41(
    sf: &[u8],
    qr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    _tlen_pad: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::x86_64::*;

    let wc = _mm_set1_epi8(wildcard as i8);
    let mch = _mm_set1_epi8(sc_mch);
    let mis = _mm_set1_epi8(sc_mis);
    let nsc = _mm_set1_epi8(sc_n);

    // Flat buffer layout guarantees all 16-byte SIMD loads within [st0, en0]
    // are in-bounds; qpos >= 0 is analytically guaranteed for valid t.
    let mut t = st0;
    while t <= en0 {
        let qpos = (qrr_base + t as i32) as usize;
        // SAFETY: flat buffer ensures sf[t..t+15] and qr[qpos..qpos+15] are in-bounds.
        let sq = unsafe { _mm_loadu_si128(sf.as_ptr().add(t) as *const __m128i) };
        let qq = unsafe { _mm_loadu_si128(qr.as_ptr().add(qpos) as *const __m128i) };
        let mask_wc = _mm_or_si128(_mm_cmpeq_epi8(sq, wc), _mm_cmpeq_epi8(qq, wc));
        let eq = _mm_cmpeq_epi8(sq, qq);
        let mut tmp = _mm_blendv_epi8(mis, mch, eq);
        tmp = _mm_blendv_epi8(tmp, nsc, mask_wc);
        unsafe { _mm_storeu_si128(s.as_mut_ptr().add(t) as *mut __m128i, tmp) };
        t += 16;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_scores_fast_row_neon(
    sf: &[u8],
    qr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    _tlen_pad: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    use std::arch::aarch64::*;

    let wc = vdupq_n_u8(wildcard);
    let mch = vdupq_n_s8(sc_mch);
    let mis = vdupq_n_s8(sc_mis);
    let nsc = vdupq_n_s8(sc_n);

    // Flat buffer layout guarantees all 16-byte SIMD loads within [st0, en0]
    // are in-bounds; qpos >= 0 is analytically guaranteed for valid t.
    let mut t = st0;
    while t <= en0 {
        let qpos = (qrr_base + t as i32) as usize;
        // SAFETY: flat buffer ensures sf[t..t+15] and qr[qpos..qpos+15] are in-bounds.
        let sq = unsafe { vld1q_u8(sf.as_ptr().add(t)) };
        let qq = unsafe { vld1q_u8(qr.as_ptr().add(qpos)) };
        let mask_wc = vorrq_u8(vceqq_u8(sq, wc), vceqq_u8(qq, wc));
        let eq = vceqq_u8(sq, qq);
        let tmp = vbslq_s8(eq, mch, mis);
        let tmp = vbslq_s8(mask_wc, nsc, tmp);
        unsafe { vst1q_s8(s.as_mut_ptr().add(t), tmp) };
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
        unsafe { vst1q_u8(x.as_mut_ptr().add(t), vreinterpretq_u8_s8(vmaxq_s8(vreinterpretq_s8_u8(a), zero))) };
        // SAFETY: valid padded stores.
        unsafe { vst1q_u8(y.as_mut_ptr().add(t), vreinterpretq_u8_s8(vmaxq_s8(vreinterpretq_s8_u8(b), zero))) };

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
        let mut z = _mm_add_epi8(unsafe { _mm_loadu_si128(s.as_ptr().add(t) as *const __m128i) }, qe2v);
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
        let mut b = _mm_add_epi8(unsafe { _mm_loadu_si128(y.as_ptr().add(t) as *const __m128i) }, ut);
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
        let mut z = _mm_add_epi8(unsafe { _mm_loadu_si128(s.as_ptr().add(t) as *const __m128i) }, qe2v);
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
        let mut b = _mm_add_epi8(unsafe { _mm_loadu_si128(y.as_ptr().add(t) as *const __m128i) }, ut);
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
            unsafe { _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(a, am)) };
            unsafe { _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(b, bm)) };
            d = _mm_or_si128(d, _mm_and_si128(am, flag8));
            d = _mm_or_si128(d, _mm_and_si128(bm, flag16));
        } else {
            let am = _mm_cmpgt_epi8(a, neg1); // a >= 0
            let bm = _mm_cmpgt_epi8(b, neg1); // b >= 0
            unsafe { _mm_storeu_si128(x.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(a, am)) };
            unsafe { _mm_storeu_si128(y.as_mut_ptr().add(t) as *mut __m128i, _mm_and_si128(b, bm)) };
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn update_h_exact_sse41(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    use ::core::ptr;
    use std::arch::x86_64::*;

    if en0 > 0 {
        h[en0] = h[en0 - 1] + i32::from(u[en0]) - qe;
    } else {
        h[en0] = h[en0] + i32::from(v[en0]) - qe;
    }
    let mut max_h = h[en0];
    let mut max_t = en0 as i32;

    let qe_v = _mm_set1_epi32(qe);
    let mut t = st0;
    while t + 4 <= en0 {
        // SAFETY: bounds ensured by loop condition.
        let hvec = unsafe { _mm_loadu_si128(h.as_ptr().add(t) as *const __m128i) };
        // SAFETY: reading 4 bytes at `t` is in-bounds by loop condition.
        let v4 = unsafe { ptr::read_unaligned(v.as_ptr().add(t) as *const u32) };
        let vbytes = _mm_cvtsi32_si128(v4 as i32);
        let v32 = _mm_cvtepu8_epi32(vbytes);
        let updated = _mm_sub_epi32(_mm_add_epi32(hvec, v32), qe_v);
        // SAFETY: bounds ensured by loop condition.
        unsafe { _mm_storeu_si128(h.as_mut_ptr().add(t) as *mut __m128i, updated) };
        let mx1 = _mm_max_epi32(updated, _mm_shuffle_epi32(updated, 0b10_11_00_01));
        let mx2 = _mm_max_epi32(mx1, _mm_shuffle_epi32(mx1, 0b01_00_11_10));
        let block_max = _mm_cvtsi128_si32(mx2);
        if block_max > max_h {
            max_h = block_max;
            let mut vals = [0i32; 4];
            // SAFETY: local fixed-size buffer.
            unsafe { _mm_storeu_si128(vals.as_mut_ptr() as *mut __m128i, updated) };
            for (i, &val) in vals.iter().enumerate() {
                if val == block_max {
                    max_t = (t + i) as i32;
                    break;
                }
            }
        }
        t += 4;
    }
    for idx in t..en0 {
        h[idx] += i32::from(v[idx]) - qe;
        if h[idx] > max_h {
            max_h = h[idx];
            max_t = idx as i32;
        }
    }
    (max_h, max_t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_h_exact_avx2(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    use ::core::ptr;
    use std::arch::x86_64::*;

    if en0 > 0 {
        h[en0] = h[en0 - 1] + i32::from(u[en0]) - qe;
    } else {
        h[en0] = h[en0] + i32::from(v[en0]) - qe;
    }
    let mut max_h = h[en0];
    let mut max_t = en0 as i32;

    let qe_v = _mm256_set1_epi32(qe);
    let mut t = st0;
    while t + 8 <= en0 {
        // SAFETY: bounds ensured by loop condition.
        let hvec = unsafe { _mm256_loadu_si256(h.as_ptr().add(t) as *const __m256i) };
        // SAFETY: reading 8 bytes at `t` is in-bounds by loop condition.
        let v8 = unsafe { ptr::read_unaligned(v.as_ptr().add(t) as *const u64) };
        let vbytes = _mm_cvtsi64_si128(v8 as i64);
        let v32 = _mm256_cvtepu8_epi32(vbytes);
        let updated = _mm256_sub_epi32(_mm256_add_epi32(hvec, v32), qe_v);
        // SAFETY: bounds ensured by loop condition.
        unsafe { _mm256_storeu_si256(h.as_mut_ptr().add(t) as *mut __m256i, updated) };
        let hi = _mm256_extracti128_si256(updated, 1);
        let lo = _mm256_castsi256_si128(updated);
        let fold = _mm_max_epi32(lo, hi);
        let mx1 = _mm_max_epi32(fold, _mm_shuffle_epi32(fold, 0b10_11_00_01));
        let mx2 = _mm_max_epi32(mx1, _mm_shuffle_epi32(mx1, 0b01_00_11_10));
        let block_max = _mm_cvtsi128_si32(mx2);
        if block_max > max_h {
            max_h = block_max;
            let mut vals = [0i32; 8];
            // SAFETY: local fixed-size buffer.
            unsafe { _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, updated) };
            for (i, &val) in vals.iter().enumerate() {
                if val == block_max {
                    max_t = (t + i) as i32;
                    break;
                }
            }
        }
        t += 8;
    }
    for idx in t..en0 {
        h[idx] += i32::from(v[idx]) - qe;
        if h[idx] > max_h {
            max_h = h[idx];
            max_t = idx as i32;
        }
    }
    (max_h, max_t)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn update_h_exact_neon(
    h: &mut [i32],
    v: &[u8],
    u: &[u8],
    st0: usize,
    en0: usize,
    qe: i32,
) -> (i32, i32) {
    use std::arch::aarch64::*;

    if en0 > 0 {
        h[en0] = h[en0 - 1] + i32::from(u[en0]) - qe;
    } else {
        h[en0] = h[en0] + i32::from(v[en0]) - qe;
    }
    let mut max_h = h[en0];
    let mut max_t = en0 as i32;

    let qe_v = vdupq_n_s32(qe);
    let mut t = st0;
    while t + 8 <= en0 {
        // SAFETY: bounds ensured by loop condition.
        let v8 = unsafe { vld1_u8(v.as_ptr().add(t)) };
        let v16 = vmovl_u8(v8);
        let v32_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v16)));
        let v32_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v16)));

        // SAFETY: bounds ensured by loop condition.
        let h_lo = unsafe { vld1q_s32(h.as_ptr().add(t)) };
        // SAFETY: bounds ensured by loop condition.
        let h_hi = unsafe { vld1q_s32(h.as_ptr().add(t + 4)) };
        let out_lo = vsubq_s32(vaddq_s32(h_lo, v32_lo), qe_v);
        let out_hi = vsubq_s32(vaddq_s32(h_hi, v32_hi), qe_v);
        // SAFETY: bounds ensured by loop condition.
        unsafe { vst1q_s32(h.as_mut_ptr().add(t), out_lo) };
        // SAFETY: bounds ensured by loop condition.
        unsafe { vst1q_s32(h.as_mut_ptr().add(t + 4), out_hi) };
        let block_max = vmaxvq_s32(vmaxq_s32(out_lo, out_hi));
        if block_max > max_h {
            // Avoid spilling vectors to memory: probe lanes directly.
            max_h = block_max;
            if vgetq_lane_s32(out_lo, 0) == block_max {
                max_t = t as i32;
            } else if vgetq_lane_s32(out_lo, 1) == block_max {
                max_t = (t + 1) as i32;
            } else if vgetq_lane_s32(out_lo, 2) == block_max {
                max_t = (t + 2) as i32;
            } else if vgetq_lane_s32(out_lo, 3) == block_max {
                max_t = (t + 3) as i32;
            } else if vgetq_lane_s32(out_hi, 0) == block_max {
                max_t = (t + 4) as i32;
            } else if vgetq_lane_s32(out_hi, 1) == block_max {
                max_t = (t + 5) as i32;
            } else if vgetq_lane_s32(out_hi, 2) == block_max {
                max_t = (t + 6) as i32;
            } else {
                max_t = (t + 7) as i32;
            }
        }
        t += 8;
    }
    for idx in t..en0 {
        h[idx] += i32::from(v[idx]) - qe;
        if h[idx] > max_h {
            max_h = h[idx];
            max_t = idx as i32;
        }
    }
    (max_h, max_t)
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn fill_fast_row_dispatch(
    backend: SimdBackend,
    sf: &[u8],
    qr: &[u8],
    qrr_base: i32,
    st0: usize,
    en0: usize,
    wildcard: u8,
    sc_mch: i8,
    sc_mis: i8,
    sc_n: i8,
    s: &mut [i8],
) {
    match backend {
        SimdBackend::Scalar => {
            fill_scores_fast_row_scalar(sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s)
        }
        SimdBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                fill_scores_fast_row_avx2(
                    sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            fill_scores_fast_row_scalar(
                sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
            )
        }
        SimdBackend::Sse41 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                fill_scores_fast_row_sse41(
                    sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
                )
            }
            #[cfg(not(target_arch = "x86_64"))]
            fill_scores_fast_row_scalar(
                sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
            )
        }
        SimdBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                fill_scores_fast_row_neon(
                    sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
                )
            }
            #[cfg(not(target_arch = "aarch64"))]
            fill_scores_fast_row_scalar(
                sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
            )
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
        sf: &[u8],
        qr: &[u8],
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

/// Carve the flat `Workspace::buf` into the seven named DP slices.
///
/// Layout: `[u | v | x | y | s | sf | qr+16]`, matching the C reference.
/// The `s` slice is re-interpreted as `&mut [i8]` (same size/alignment as
/// `u8`).  All returned slices are non-overlapping sub-slices of `buf`.
#[inline]
pub(super) fn split_main_buf(
    buf: &mut [u8],
    tlen_pad: usize,
    qlen_pad: usize,
) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8], &mut [i8], &mut [u8], &mut [u8]) {
    let (u, rest) = buf.split_at_mut(tlen_pad);
    let (v, rest) = rest.split_at_mut(tlen_pad);
    let (x, rest) = rest.split_at_mut(tlen_pad);
    let (y, rest) = rest.split_at_mut(tlen_pad);
    let (s_raw, rest) = rest.split_at_mut(tlen_pad);
    let (sf, rest) = rest.split_at_mut(tlen_pad);
    let (qr, _) = rest.split_at_mut(qlen_pad + 16);
    // SAFETY: i8 has the same size and alignment as u8; the slice is valid.
    let s = unsafe { std::slice::from_raw_parts_mut(s_raw.as_mut_ptr() as *mut i8, tlen_pad) };
    (u, v, x, y, s, sf, qr)
}

pub(super) struct ScalarOps;
#[allow(dead_code)]
pub(super) struct Sse41Ops;
#[allow(dead_code)]
pub(super) struct Avx2Ops;
pub(super) struct NeonOps;

impl BackendOps for ScalarOps {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sf: &[u8],
        qr: &[u8],
        qrr_base: i32,
        st0: usize,
        en0: usize,
        wildcard: u8,
        sc_mch: i8,
        sc_mis: i8,
        sc_n: i8,
        s: &mut [i8],
    ) {
        fill_scores_fast_row_scalar(
            sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
        );
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
        sf: &[u8],
        qr: &[u8],
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
            fill_scores_fast_row_sse41(
                sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        fill_scores_fast_row_scalar(
            sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
        );
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
                    u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1,
                );
            } else {
                dp_row_score_only_sse41(u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            ScalarOps::dp_row_score_only::<RIGHT>(
                u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1,
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
            update_h_exact_sse41(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "x86_64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

impl BackendOps for Avx2Ops {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sf: &[u8],
        qr: &[u8],
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
            fill_scores_fast_row_avx2(
                sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        fill_scores_fast_row_scalar(
            sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
        );
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
                u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1,
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
                u, v, x, y, s, p_row, stv, env, q, qe2, max_sc_transformed, x1, v1,
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
            update_h_exact_avx2(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "x86_64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

impl BackendOps for NeonOps {
    #[inline(always)]
    unsafe fn fill_fast_row(
        sf: &[u8],
        qr: &[u8],
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
            fill_scores_fast_row_neon(
                sf, qr, qrr_base, st0, en0, 0, wildcard, sc_mch, sc_mis, sc_n, s,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        fill_scores_fast_row_scalar(
            sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
        );
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
                    u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1,
                );
            } else {
                dp_row_score_only_neon(u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            ScalarOps::dp_row_score_only::<RIGHT>(
                u, v, x, y, s, stv, env, q, qe2, max_sc_transformed, x1, v1,
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
            update_h_exact_neon(h, v, u, st0, en0, qe)
        }
        #[cfg(not(target_arch = "aarch64"))]
        update_h_exact_scalar(h, v, u, st0, en0, qe)
    }
}

fn extz2_core_score_only<B: BackendOps>(
    input: &Extz2Input<'_>,
    ez: &mut Extz,
    ws: &mut Workspace,
) {
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
    let qe2 = qe * 2;

    let mut w = input.w;
    if w < 0 {
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
    if -min_sc > 2 * qe {
        return;
    }

    let generic_sc = (input.flag & KSW_EZ_GENERIC_SC) != 0;
    ws.prepare_score_only(qlen, tlen, APPROX);
    let tlen_pad = ws.tlen_pad;
    let qlen_pad = ws.qlen_pad;
    let h = &mut ws.h;
    let (u, v, x, y, s, sf, qr) = split_main_buf(&mut ws.buf, tlen_pad, qlen_pad);

    sf[..tlen].copy_from_slice(input.target);
    for (i, qv) in input.query.iter().enumerate() {
        qr[qlen - 1 - i] = *qv;
    }
    debug_assert_eq!(qr.len(), qlen_pad + 16);

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
                B::fill_fast_row(
                    sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
                );
            }
        } else {
            s[stv as usize..=env as usize].fill(0);
            for t in st0..=en0 {
                let sq = sf[t] as usize;
                let qidx = (qrr_base + t as i32) as usize;
                let qq = qr[qidx] as usize;
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
            if r > 0 {
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
            if apply_zdrop(ez, max_h, r, max_t, input.zdrop, input.e) {
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
    let qe2 = qe * 2;

    let mut w = input.w;
    if w < 0 {
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
    if -min_sc > 2 * qe {
        return;
    }

    let generic_sc = (input.flag & KSW_EZ_GENERIC_SC) != 0;

    let rows = qlen + tlen - 1;
    let mut n_col = qlen.min(tlen);
    n_col = n_col.min((w + 1) as usize);
    let n_col = n_col + 16;
    ws.prepare_traceback(qlen, tlen, APPROX, rows, n_col);
    let tlen_pad = ws.tlen_pad;
    let qlen_pad = ws.qlen_pad;
    let h = &mut ws.h;
    let off = &mut ws.off;
    let off_end = &mut ws.off_end;
    let p = &mut ws.p;
    let (u, v, x, y, s, sf, qr) = split_main_buf(&mut ws.buf, tlen_pad, qlen_pad);

    sf[..tlen].copy_from_slice(input.target);
    for (i, qv) in input.query.iter().enumerate() {
        qr[qlen - 1 - i] = *qv;
    }
    debug_assert_eq!(qr.len(), qlen_pad + 16);

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
                B::fill_fast_row(
                    sf, qr, qrr_base, st0, en0, wildcard, sc_mch, sc_mis, sc_n, s,
                );
            }
        } else {
            s[stv as usize..=env as usize].fill(0);
            for t in st0..=en0 {
                let sq = sf[t] as usize;
                let qidx = (qrr_base + t as i32) as usize;
                let qq = qr[qidx] as usize;
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

            if r > 0 {
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
            if apply_zdrop(ez, max_h, r, max_t, input.zdrop, input.e) {
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
            (vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], vec![0, 1, 2, 3]),
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
                    reused_ez.score, fresh_ez.score,
                    "score mismatch: flag={flag} qlen={} tlen={}",
                    q.len(), t.len(),
                );
                assert_eq!(
                    reused_ez.max, fresh_ez.max,
                    "max mismatch: flag={flag} qlen={} tlen={}",
                    q.len(), t.len(),
                );
                assert_eq!(
                    reused_ez.cigar, fresh_ez.cigar,
                    "cigar mismatch: flag={flag} qlen={} tlen={}",
                    q.len(), t.len(),
                );
                assert_eq!(
                    reused_ez.zdropped, fresh_ez.zdropped,
                    "zdropped mismatch: flag={flag} qlen={} tlen={}",
                    q.len(), t.len(),
                );
            }
        }
    }
}
