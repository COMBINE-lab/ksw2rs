//! Checks that need no C reference: the backends agree with each other, and
//! global scores and CIGARs agree with a plain affine-gap DP.

use ksw2rs::{
    Extz, Extz2Input, KSW_EZ_APPROX_DROP, KSW_EZ_APPROX_MAX, KSW_EZ_EXTZ_ONLY, KSW_EZ_GENERIC_SC,
    KSW_EZ_NO_MAX, KSW_EZ_REV_CIGAR, KSW_EZ_RIGHT, KSW_EZ_SCORE_ONLY, KSW_NEG_INF, extz2,
    extz2_scalar,
};

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) % n
    }
}

fn dna5_mat(match_score: i8, mismatch_score: i8, n_score: i8) -> [i8; 25] {
    let mut mat = [mismatch_score; 25];
    for i in 0..4 {
        mat[i * 5 + i] = match_score;
    }
    for i in 0..5 {
        mat[i * 5 + 4] = n_score;
        mat[20 + i] = n_score;
    }
    mat
}

/// A random target and a query derived from it with substitutions and short
/// indels (`div == 0`: unrelated), so that alignments run along the band.
fn related_pair(rng: &mut Rng, qlen: usize, tlen: usize, div: u64) -> (Vec<u8>, Vec<u8>) {
    let t: Vec<u8> = (0..tlen).map(|_| rng.below(4) as u8).collect();
    let mut q = Vec::with_capacity(qlen + 4);
    let mut i = 0;
    while q.len() < qlen {
        if div == 0 || i >= tlen {
            q.push(rng.below(4) as u8);
            i += 1;
            continue;
        }
        match rng.below(div * 3) {
            0 => {
                q.push(rng.below(4) as u8);
                i += 1;
            }
            1 => i += 1 + rng.below(4) as usize,
            2 => {
                for _ in 0..1 + rng.below(4) {
                    q.push(rng.below(4) as u8);
                }
            }
            _ => {
                q.push(t[i]);
                i += 1;
            }
        }
    }
    q.truncate(qlen);
    (q, t)
}

fn fields(ez: &Extz) -> (u32, bool, [i32; 7], bool, &[u32]) {
    (
        ez.max,
        ez.zdropped,
        [
            ez.max_q, ez.max_t, ez.mqe, ez.mqe_t, ez.mte, ez.mte_q, ez.score,
        ],
        ez.reach_end,
        &ez.cigar,
    )
}

/// The scalar kernel and the dispatched one return the same result, banded
/// alignments included: at a band edge the result depends on the padding
/// cells past the band, which every backend has to fill alike.
#[test]
fn scalar_matches_dispatched_kernel() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let flags = [
        KSW_EZ_SCORE_ONLY,
        KSW_EZ_SCORE_ONLY | KSW_EZ_RIGHT,
        0,
        KSW_EZ_RIGHT,
        KSW_EZ_EXTZ_ONLY,
        KSW_EZ_EXTZ_ONLY | KSW_EZ_RIGHT | KSW_EZ_REV_CIGAR,
        KSW_EZ_APPROX_MAX | KSW_EZ_APPROX_DROP,
        KSW_EZ_SCORE_ONLY | KSW_EZ_APPROX_MAX | KSW_EZ_APPROX_DROP,
        KSW_EZ_GENERIC_SC,
        KSW_EZ_SCORE_ONLY | KSW_EZ_GENERIC_SC,
        KSW_EZ_SCORE_ONLY | KSW_EZ_NO_MAX,
    ];
    let bands = [-1, 0, 1, 4, 15, 15, 31, 100];
    let zdrops = [-1, -1, 20, 100, 400];
    for case in 0..30_000 {
        let long = rng.below(40) == 0;
        let qlen = 1 + rng.below(if long { 900 } else { 220 }) as usize;
        let tlen = if rng.below(3) == 0 {
            1 + rng.below(if long { 900 } else { 220 }) as usize
        } else {
            (qlen as i64 + rng.below(31) as i64 - 15).max(1) as usize
        };
        let div = [0, 4, 10, 25][rng.below(4) as usize];
        let (q, t) = related_pair(&mut rng, qlen, tlen, div);
        let (a, b) = [(2, -4), (1, -4), (1, -1), (5, -4)][rng.below(4) as usize];
        let (gq, ge) = [(4, 2), (6, 2), (5, 3), (4, 1), (0, 1), (12, 3)][rng.below(6) as usize];
        let mat = dna5_mat(a, b, [0, -1][rng.below(2) as usize]);
        let input = Extz2Input {
            query: &q,
            target: &t,
            m: 5,
            mat: &mat,
            q: gq,
            e: ge,
            w: bands[rng.below(bands.len() as u64) as usize],
            zdrop: zdrops[rng.below(zdrops.len() as u64) as usize],
            end_bonus: [0, 5][rng.below(2) as usize],
            flag: flags[rng.below(flags.len() as u64) as usize],
        };
        let (mut simd, mut scalar) = (Extz::default(), Extz::default());
        extz2(&input, &mut simd);
        extz2_scalar(&input, &mut scalar);
        assert_eq!(
            fields(&simd),
            fields(&scalar),
            "case {case}: qlen={qlen} tlen={tlen} w={} zdrop={} flag={:#x} q={gq} e={ge}",
            input.w,
            input.zdrop,
            input.flag,
        );
    }
}

/// Global alignment score with a gap of length `k` costing `q + k * e`.
fn reference_global_score(q: &[u8], t: &[u8], mat: &[i8; 25], gq: i32, ge: i32) -> i32 {
    const NEG: i32 = i32::MIN / 2;
    let (n, m) = (t.len(), q.len());
    // h: best score; e: ending in a gap along the target; f: along the query
    let mut h = vec![vec![NEG; m + 1]; n + 1];
    let mut e = vec![vec![NEG; m + 1]; n + 1];
    let mut f = vec![vec![NEG; m + 1]; n + 1];
    h[0][0] = 0;
    for i in 0..=n {
        for j in 0..=m {
            if i > 0 {
                e[i][j] = (e[i - 1][j] - ge).max(h[i - 1][j] - gq - ge);
            }
            if j > 0 {
                f[i][j] = (f[i][j - 1] - ge).max(h[i][j - 1] - gq - ge);
            }
            if i > 0 && j > 0 {
                let s = i32::from(mat[t[i - 1] as usize * 5 + q[j - 1] as usize]);
                h[i][j] = h[i - 1][j - 1] + s;
            }
            h[i][j] = h[i][j].max(e[i][j]).max(f[i][j]);
        }
    }
    h[n][m]
}

/// Score of the alignment a CIGAR describes, and the lengths it consumes.
fn cigar_score(
    cigar: &[u32],
    q: &[u8],
    t: &[u8],
    mat: &[i8; 25],
    gq: i32,
    ge: i32,
) -> (i32, usize, usize) {
    let (mut i, mut j, mut score) = (0, 0, 0);
    for &c in cigar {
        let len = (c >> 4) as usize;
        match c & 0xf {
            0 => {
                for _ in 0..len {
                    score += i32::from(mat[t[i] as usize * 5 + q[j] as usize]);
                    i += 1;
                    j += 1;
                }
            }
            1 => {
                score -= gq + ge * len as i32;
                j += len;
            }
            _ => {
                score -= gq + ge * len as i32;
                i += len;
            }
        }
    }
    (score, i, j)
}

/// Unbanded scores equal the textbook DP, and the CIGAR is an alignment of
/// that score, on both kernels and across the scoring range: schemes the SIMD
/// kernels handle, schemes only the scalar one does, and schemes outside the
/// 8-bit cells, which are refused rather than computed wrongly.
#[test]
fn global_score_and_cigar_match_reference_dp() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    // (match, mismatch, q, e): `match + 2 * (q + e)` is 14, 127, 128, 162,
    // 254, 255 and 256
    let schemes = [
        (2i8, -4i8, 4i8, 2i8),
        (1, -4, 50, 13),
        (2, -4, 50, 13),
        (2, -4, 60, 20),
        (2, -4, 100, 26),
        (1, -4, 100, 27),
        (2, -4, 100, 27),
    ];
    for (a, b, gq, ge) in schemes {
        let mat = dna5_mat(a, b, 0);
        let supported = i32::from(a) + 2 * (i32::from(gq) + i32::from(ge)) <= 255;
        for case in 0..300 {
            let qlen = 1 + rng.below(90) as usize;
            let tlen = 1 + rng.below(90) as usize;
            let (q, t) = related_pair(&mut rng, qlen, tlen, [0, 6, 20][case % 3]);
            let want = reference_global_score(&q, &t, &mat, gq.into(), ge.into());
            for flag in [0, KSW_EZ_RIGHT, KSW_EZ_SCORE_ONLY] {
                let input = Extz2Input {
                    query: &q,
                    target: &t,
                    m: 5,
                    mat: &mat,
                    q: gq,
                    e: ge,
                    w: -1,
                    zdrop: -1,
                    end_bonus: 0,
                    flag,
                };
                for scalar in [false, true] {
                    let mut ez = Extz::default();
                    if scalar {
                        extz2_scalar(&input, &mut ez);
                    } else {
                        extz2(&input, &mut ez);
                    }
                    let what = format!(
                        "scheme ({a},{b},{gq},{ge}) case {case} flag {flag:#x} scalar {scalar}"
                    );
                    if !supported {
                        assert_eq!(ez.score, KSW_NEG_INF, "{what}");
                        assert!(ez.cigar.is_empty(), "{what}");
                        continue;
                    }
                    assert_eq!(ez.score, want, "{what}");
                    if flag & KSW_EZ_SCORE_ONLY == 0 {
                        let got = cigar_score(&ez.cigar, &q, &t, &mat, gq.into(), ge.into());
                        assert_eq!(got, (want, tlen, qlen), "{what}");
                    }
                }
            }
        }
    }
}

/// A band at least as wide as the sequences is no band, however large `w` is.
#[test]
fn huge_band_is_unbanded() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mat = dna5_mat(2, -4, 0);
    for case in 0..200 {
        let qlen = 1 + rng.below(150) as usize;
        let tlen = 1 + rng.below(150) as usize;
        let (q, t) = related_pair(&mut rng, qlen, tlen, 10);
        for flag in [0, KSW_EZ_SCORE_ONLY, KSW_EZ_EXTZ_ONLY] {
            let unbanded = Extz2Input {
                query: &q,
                target: &t,
                m: 5,
                mat: &mat,
                q: 4,
                e: 2,
                w: -1,
                zdrop: 100,
                end_bonus: 0,
                flag,
            };
            let mut want = Extz::default();
            extz2(&unbanded, &mut want);
            for w in [qlen.max(tlen) as i32, 1 << 20, i32::MAX - 1, i32::MAX] {
                let mut got = Extz::default();
                extz2(
                    &Extz2Input {
                        w,
                        ..unbanded.clone()
                    },
                    &mut got,
                );
                assert_eq!(
                    fields(&got),
                    fields(&want),
                    "case {case} flag {flag:#x} w {w}"
                );
            }
        }
    }
}
