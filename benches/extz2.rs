use std::time::Duration;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ksw2rs::{
    Extz, Extz2Input, KSW_EZ_APPROX_MAX, KSW_EZ_SCORE_ONLY, Workspace, extz2_with_workspace,
    extz2_scalar_with_workspace,
};

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

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed >> 32) as u32
}

fn make_input(qlen: usize, tlen: usize) -> (Vec<u8>, Vec<u8>, [i8; 25]) {
    let mut seed = (qlen as u64) << 32 | (tlen as u64);
    let mut q = vec![0u8; qlen];
    let mut t = vec![0u8; tlen];
    for x in &mut q {
        *x = (lcg(&mut seed) % 5) as u8;
    }
    for x in &mut t {
        *x = (lcg(&mut seed) % 5) as u8;
    }
    (q, t, dna5_mat(2, -4))
}

#[cfg(has_c_ref)]
fn c_ref(input: &Extz2Input<'_>) {
    let mut out = CRefExtz::default();
    // SAFETY: pointers are valid for the call duration; `out` is writable.
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
    black_box(out);
}

fn bench_extz2(c: &mut Criterion) {
    let mut group = c.benchmark_group("extz2_score_only");
    group.sample_size(250);
    group.warm_up_time(Duration::from_millis(800));
    group.measurement_time(Duration::from_secs(4));

    for (qlen, tlen) in [(128usize, 128usize), (512usize, 512usize), (1024usize, 1024usize)] {
        let (q, t, mat) = make_input(qlen, tlen);
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
            flag: KSW_EZ_SCORE_ONLY,
        };

        let bytes = (qlen + tlen) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("rust_auto", format!("{}x{}", qlen, tlen)), &input, |b, i| {
            let mut ez = Extz::default();
            let mut ws = Workspace::default();
            b.iter(|| {
                extz2_with_workspace(black_box(i), &mut ez, &mut ws);
                black_box(ez.score);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("rust_scalar", format!("{}x{}", qlen, tlen)),
            &input,
            |b, i| {
                let mut ez = Extz::default();
                let mut ws = Workspace::default();
                b.iter(|| {
                    extz2_scalar_with_workspace(black_box(i), &mut ez, &mut ws);
                    black_box(ez.score);
                });
            },
        );

        #[cfg(has_c_ref)]
        group.bench_with_input(BenchmarkId::new("c_ref", format!("{}x{}", qlen, tlen)), &input, |b, i| {
            b.iter(|| c_ref(black_box(i)));
        });
    }

    group.finish();

    let mut approx = c.benchmark_group("extz2_score_only_approx");
    approx.sample_size(200);
    approx.warm_up_time(Duration::from_millis(800));
    approx.measurement_time(Duration::from_secs(4));
    let (q, t, mat) = make_input(1024, 1024);
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
        flag: KSW_EZ_SCORE_ONLY | KSW_EZ_APPROX_MAX,
    };
    approx.throughput(Throughput::Bytes((q.len() + t.len()) as u64));

    approx.bench_function("rust_auto_1024x1024", |b| {
        let mut ez = Extz::default();
        let mut ws = Workspace::default();
        b.iter(|| {
            extz2_with_workspace(black_box(&input), &mut ez, &mut ws);
            black_box(ez.score);
        });
    });

    #[cfg(has_c_ref)]
    approx.bench_function("c_ref_1024x1024", |b| {
        b.iter(|| c_ref(black_box(&input)));
    });
    approx.finish();

    let mut tb = c.benchmark_group("extz2_traceback");
    tb.sample_size(140);
    tb.warm_up_time(Duration::from_millis(800));
    tb.measurement_time(Duration::from_secs(4));
    let (q, t, mat) = make_input(1024, 1024);
    let input_left = Extz2Input {
        query: &q,
        target: &t,
        m: 5,
        mat: &mat,
        q: 4,
        e: 2,
        w: -1,
        zdrop: 100,
        end_bonus: 0,
        flag: 0,
    };
    let input_right = Extz2Input {
        flag: 0x02, // KSW_EZ_RIGHT
        ..input_left.clone()
    };
    tb.throughput(Throughput::Bytes((q.len() + t.len()) as u64));

    tb.bench_function("rust_auto_left_1024x1024", |b| {
        let mut ez = Extz::default();
        let mut ws = Workspace::default();
        b.iter(|| {
            extz2_with_workspace(black_box(&input_left), &mut ez, &mut ws);
            black_box(ez.score);
            black_box(ez.cigar.len());
        });
    });
    tb.bench_function("rust_auto_right_1024x1024", |b| {
        let mut ez = Extz::default();
        let mut ws = Workspace::default();
        b.iter(|| {
            extz2_with_workspace(black_box(&input_right), &mut ez, &mut ws);
            black_box(ez.score);
            black_box(ez.cigar.len());
        });
    });

    #[cfg(has_c_ref)]
    tb.bench_function("c_ref_left_1024x1024", |b| {
        b.iter(|| c_ref(black_box(&input_left)));
    });
    #[cfg(has_c_ref)]
    tb.bench_function("c_ref_right_1024x1024", |b| {
        b.iter(|| c_ref(black_box(&input_right)));
    });
    tb.finish();
}

criterion_group!(benches, bench_extz2);
criterion_main!(benches);
