#![cfg(has_c_ref)]

use ksw2rs::{
    Extz, Extz2Input, KSW_EZ_EXTZ_ONLY, KSW_EZ_GENERIC_SC, KSW_EZ_REV_CIGAR, KSW_EZ_RIGHT,
    KSW_EZ_SCORE_ONLY, extz2,
};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct CRefExtz {
    max: u32,
    zdropped: i32,
    max_q: i32,
    max_t: i32,
    mqe: i32,
    mqe_t: i32,
    mte: i32,
    mte_q: i32,
    score: i32,
    reach_end: i32,
    n_cigar: u32,
    cigar_hash: u64,
}

unsafe extern "C" {
    fn ksw2rs_extz2_sse_ref(
        qlen: i32,
        query: *const u8,
        tlen: i32,
        target: *const u8,
        m: i8,
        mat: *const i8,
        q: i8,
        e: i8,
        w: i32,
        zdrop: i32,
        end_bonus: i32,
        flag: i32,
        out: *mut CRefExtz,
    );
}

fn dna5_mat(match_score: i8, mismatch_score: i8) -> [i8; 25] {
    let mut mat = [mismatch_score; 25];
    for i in 0..5 {
        mat[i * 5 + i] = match_score;
    }
    mat[24] = 0;
    mat
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed >> 32) as u32
}

fn run_c_ref(input: &Extz2Input<'_>) -> CRefExtz {
    let mut out = CRefExtz::default();
    // SAFETY: pointers are valid for call duration and `out` is writable.
    unsafe {
        ksw2rs_extz2_sse_ref(
            input.query.len() as i32,
            input.query.as_ptr(),
            input.target.len() as i32,
            input.target.as_ptr(),
            input.m,
            input.mat.as_ptr(),
            input.q,
            input.e,
            input.w,
            input.zdrop,
            input.end_bonus,
            input.flag as i32,
            &mut out,
        );
    }
    out
}

#[test]
fn differential_score_only_randomized() {
    let mat = dna5_mat(2, -4);
    let mut seed = 0xDEADBEEFCAFEBABEu64;

    for _ in 0..64 {
        let qlen = (lcg(&mut seed) % 60 + 20) as usize;
        let tlen = (lcg(&mut seed) % 60 + 20) as usize;

        let mut q = vec![0u8; qlen];
        let mut t = vec![0u8; tlen];
        for x in &mut q {
            *x = (lcg(&mut seed) % 5) as u8;
        }
        for x in &mut t {
            *x = (lcg(&mut seed) % 5) as u8;
        }

        let w = (lcg(&mut seed) % 40) as i32;
        let zdrop = 50 + (lcg(&mut seed) % 120) as i32;

        let input = Extz2Input {
            query: &q,
            target: &t,
            m: 5,
            mat: &mat,
            q: 4,
            e: 2,
            w,
            zdrop,
            end_bonus: 0,
            flag: KSW_EZ_SCORE_ONLY,
        };

        let mut rz = Extz::default();
        extz2(&input, &mut rz);
        let cz = run_c_ref(&input);

        if rz.score != cz.score
            || rz.max != cz.max
            || rz.max_q != cz.max_q
            || rz.max_t != cz.max_t
            || rz.mqe != cz.mqe
            || rz.mqe_t != cz.mqe_t
            || rz.mte != cz.mte
            || rz.mte_q != cz.mte_q
            || rz.zdropped as i32 != cz.zdropped
        {
            panic!(
                "mismatch\\nqlen={} tlen={} w={} zdrop={}\\nq={:?}\\nt={:?}\\nrust={:?}\\nc={:?}",
                qlen, tlen, w, zdrop, q, t, rz, cz
            );
        }
    }
}

#[test]
fn differential_cigar_simple_case() {
    let mat = dna5_mat(2, -4);
    let q = [0u8, 1, 2, 3, 0, 1, 2, 3, 4, 4];
    let t = [0u8, 1, 2, 3, 0, 1, 2, 3, 4, 4];

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

    let mut rz = Extz::default();
    extz2(&input, &mut rz);
    let cz = run_c_ref(&input);

    let rh = rz.cigar.iter().fold(1469598103934665603u64, |acc, &x| {
        (acc ^ x as u64).wrapping_mul(1099511628211)
    });

    assert_eq!(rz.score, cz.score);
    assert_eq!(rz.cigar.len() as u32, cz.n_cigar);
    assert_eq!(rh, cz.cigar_hash);
}

/// Banded alignments, with and without CIGAR and with the full scoring
/// matrix, against C. At a band edge both implementations read padding cells
/// past the band, so the padding has to match too.
///
/// Compared: everything that does not depend on which of several equal
/// maxima is reported. C reduces its four SIMD lanes lowest lane first,
/// ksw2rs reports the first position, so `max_t`/`max_q` (and what follows
/// from them: z-drop, the CIGAR of an extension that stops at the maximum)
/// may differ on ties. Z-drop is therefore off here.
#[test]
fn differential_banded_randomized() {
    let mut seed = 0x0123_4567_89AB_CDEFu64;
    let flags = [
        KSW_EZ_SCORE_ONLY,
        KSW_EZ_SCORE_ONLY | KSW_EZ_RIGHT,
        0,
        KSW_EZ_RIGHT,
        KSW_EZ_REV_CIGAR,
        KSW_EZ_EXTZ_ONLY,
        KSW_EZ_GENERIC_SC,
        KSW_EZ_GENERIC_SC | KSW_EZ_RIGHT,
        KSW_EZ_SCORE_ONLY | KSW_EZ_GENERIC_SC,
    ];
    for case in 0..20_000 {
        let qlen = (lcg(&mut seed) % 200 + 1) as usize;
        let tlen = (qlen as i64 + (lcg(&mut seed) % 31) as i64 - 15).max(1) as usize;
        let t: Vec<u8> = (0..tlen).map(|_| (lcg(&mut seed) % 4) as u8).collect();
        // the query follows the target, with substitutions and short indels
        let mut q = Vec::with_capacity(qlen + 4);
        let mut i = 0;
        while q.len() < qlen {
            match if i < tlen { lcg(&mut seed) % 30 } else { 0 } {
                0 => {
                    q.push((lcg(&mut seed) % 4) as u8);
                    i += 1;
                }
                1 => i += 1 + (lcg(&mut seed) % 4) as usize,
                2 => {
                    for _ in 0..1 + lcg(&mut seed) % 4 {
                        q.push((lcg(&mut seed) % 4) as u8);
                    }
                }
                _ => {
                    q.push(t[i]);
                    i += 1;
                }
            }
        }
        q.truncate(qlen);

        let mat = dna5_mat([2, 1, 5][(lcg(&mut seed) % 3) as usize], -4);
        let flag = flags[(lcg(&mut seed) % flags.len() as u32) as usize];
        let input = Extz2Input {
            query: &q,
            target: &t,
            m: 5,
            mat: &mat,
            q: [4, 6, 12][(lcg(&mut seed) % 3) as usize],
            e: [1, 2, 3][(lcg(&mut seed) % 3) as usize],
            w: [-1, 1, 4, 15, 15, 31][(lcg(&mut seed) % 6) as usize],
            zdrop: -1,
            end_bonus: 0,
            flag,
        };

        let mut rz = Extz::default();
        extz2(&input, &mut rz);
        let cz = run_c_ref(&input);

        let what = format!(
            "case {case}: qlen={qlen} tlen={tlen} w={} flag={flag:#x}",
            input.w
        );
        assert_eq!(
            (
                rz.score,
                rz.mqe,
                rz.mqe_t,
                rz.mte,
                rz.mte_q,
                rz.zdropped as i32
            ),
            (cz.score, cz.mqe, cz.mqe_t, cz.mte, cz.mte_q, cz.zdropped),
            "{what}"
        );
        // A full alignment is traced back from the end cell, whatever the
        // maximum was. (An alignment that left the band is `zdropped`.)
        if flag & (KSW_EZ_SCORE_ONLY | KSW_EZ_EXTZ_ONLY) == 0 && !rz.zdropped {
            let rh = rz.cigar.iter().fold(1469598103934665603u64, |acc, &x| {
                (acc ^ x as u64).wrapping_mul(1099511628211)
            });
            assert_eq!(
                (rz.cigar.len() as u32, rh),
                (cz.n_cigar, cz.cigar_hash),
                "{what}"
            );
        }
    }
}
