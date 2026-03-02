#![cfg(has_c_ref)]

use ksw2rs::{extz2, Extz, Extz2Input, KSW_EZ_SCORE_ONLY};

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

    let rh = rz
        .cigar
        .iter()
        .fold(1469598103934665603u64, |acc, &x| {
            (acc ^ x as u64).wrapping_mul(1099511628211)
        });

    assert_eq!(rz.score, cz.score);
    assert_eq!(rz.cigar.len() as u32, cz.n_cigar);
    assert_eq!(rh, cz.cigar_hash);
}
