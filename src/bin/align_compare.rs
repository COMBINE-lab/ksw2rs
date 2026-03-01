use std::fs;
use std::path::Path;
use std::time::Instant;

use ksw2rs::{Extz, Extz2Input, KSW_EZ_SCORE_ONLY, extz2};

#[cfg(has_c_ref)]
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

#[cfg(has_c_ref)]
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

fn fnv1a_u32_words(words: &[u32]) -> u64 {
    let mut h = 1469598103934665603u64;
    for &x in words {
        h ^= x as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn read_fasta_one<P: AsRef<Path>>(path: P) -> Result<String, String> {
    let p = path.as_ref();
    let text = fs::read_to_string(p).map_err(|e| format!("{}: {}", p.display(), e))?;
    let mut seq = String::new();
    for line in text.lines() {
        if line.starts_with('>') {
            continue;
        }
        seq.push_str(line.trim());
    }
    if seq.is_empty() {
        return Err(format!("{}: empty FASTA sequence", p.display()));
    }
    Ok(seq)
}

fn encode_dna5(seq: &str) -> Vec<u8> {
    seq.bytes()
        .map(|b| match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' | b'U' => 3,
            _ => 4,
        })
        .collect()
}

fn run_rust(input: &Extz2Input<'_>) -> Extz {
    let mut ez = Extz::default();
    extz2(input, &mut ez);
    ez
}

#[cfg(has_c_ref)]
fn run_c(input: &Extz2Input<'_>) -> CRefExtz {
    let mut out = CRefExtz::default();
    // SAFETY: all pointers are valid for call duration and `out` is writable.
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

fn bench_rust(input: &Extz2Input<'_>, iters: usize) -> f64 {
    for _ in 0..5 {
        let _ = run_rust(input);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_rust(input);
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

#[cfg(has_c_ref)]
fn bench_c(input: &Extz2Input<'_>, iters: usize) -> f64 {
    for _ in 0..5 {
        let _ = run_c(input);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = run_c(input);
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn choose_iters(qlen: usize, tlen: usize) -> usize {
    let cells = (qlen as u128) * (tlen as u128);
    if cells >= 1_000_000_000 {
        1
    } else if cells >= 250_000_000 {
        2
    } else if cells >= 50_000_000 {
        5
    } else if cells >= 10_000_000 {
        10
    } else {
        300
    }
}

fn run_pair(name: &str, q_path: &str, t_path: &str) -> Result<(), String> {
    let q_raw = read_fasta_one(q_path)?;
    let t_raw = read_fasta_one(t_path)?;
    let q = encode_dna5(&q_raw);
    let t = encode_dna5(&t_raw);
    let mat = dna5_mat(2, -4);

    println!("=== {} ===", name);
    println!("query_len={} target_len={}", q.len(), t.len());

    for (mode_name, flag) in [("traceback", 0u32), ("score_only", KSW_EZ_SCORE_ONLY)] {
        let input = Extz2Input {
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

        let rz = run_rust(&input);
        let rust_hash = fnv1a_u32_words(&rz.cigar);

        #[cfg(has_c_ref)]
        {
            let cz = run_c(&input);
            let base_parity = rz.score == cz.score
                && rz.max == cz.max
                && rz.max_q == cz.max_q
                && rz.max_t == cz.max_t
                && rz.mqe == cz.mqe
                && rz.mqe_t == cz.mqe_t
                && rz.mte == cz.mte
                && rz.mte_q == cz.mte_q
                && (rz.zdropped as i32) == cz.zdropped;
            let cigar_parity =
                (rz.cigar.len() as u32) == cz.n_cigar && rust_hash == cz.cigar_hash;
            let parity = if (flag & KSW_EZ_SCORE_ONLY) != 0 {
                base_parity
            } else {
                base_parity && cigar_parity
            };

            let iters = choose_iters(q.len(), t.len());
            let rust_us = bench_rust(&input, iters);
            let c_us = bench_c(&input, iters);
            let cells = (q.len() as f64) * (t.len() as f64);
            let rust_gcups = cells / (rust_us * 1e3);
            let c_gcups = cells / (c_us * 1e3);

            println!(
                "mode={:<10} parity={} iters={} rust_us={:.3} ({:.3} ms) c_us={:.3} ({:.3} ms) ratio={:.2}x rust_gcups={:.2} c_gcups={:.2} score={} cigar_n={}",
                mode_name,
                parity,
                iters,
                rust_us,
                rust_us / 1000.0,
                c_us,
                c_us / 1000.0,
                rust_us / c_us,
                rust_gcups,
                c_gcups,
                rz.score,
                rz.cigar.len()
            );

            if !parity {
                println!(
                    "  mismatch: rust(max={},max_q={},max_t={},mqe={},mte={},zd={},cigar_n={},cigar_hash={:#x}) c(max={},max_q={},max_t={},mqe={},mte={},zd={},cigar_n={},cigar_hash={:#x})",
                    rz.max,
                    rz.max_q,
                    rz.max_t,
                    rz.mqe,
                    rz.mte,
                    rz.zdropped,
                    rz.cigar.len(),
                    rust_hash,
                    cz.max,
                    cz.max_q,
                    cz.max_t,
                    cz.mqe,
                    cz.mte,
                    cz.zdropped,
                    cz.n_cigar,
                    cz.cigar_hash
                );
            }
        }

        #[cfg(not(has_c_ref))]
        {
            let iters = choose_iters(q.len(), t.len());
            let rust_us = bench_rust(&input, iters);
            let cells = (q.len() as f64) * (t.len() as f64);
            let rust_gcups = cells / (rust_us * 1e3);
            println!(
                "mode={:<10} rust_only iters={} rust_us={:.3} ({:.3} ms) rust_gcups={:.2} score={} cigar_n={}",
                mode_name,
                iters,
                rust_us,
                rust_us / 1000.0,
                rust_gcups,
                rz.score,
                rz.cigar.len()
            );
        }
    }

    println!();
    Ok(())
}

fn main() {
    let pairs = [
        ("MT-human_vs_MT-orang", "test/MT-human.fa", "test/MT-orang.fa"),
        ("q2_vs_t2", "test/q2.fa", "test/t2.fa"),
    ];

    for (name, q, t) in pairs {
        if let Err(e) = run_pair(name, q, t) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}
