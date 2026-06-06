# ksw2rs

`ksw2rs` is a native Rust port of [ksw2](https://github.com/lh3/ksw2), focused on preserving ksw2 behavior and performance characteristics as directly as possible.

## Building for optimal performance

Just build normally:

```sh
cargo build --release
```

`ksw2rs` uses **runtime feature detection** to dispatch to the best available
SIMD backend (AVX2, SSE4.1, or NEON). The SIMD kernels live in
`#[target_feature(enable = "…")]` functions, so the compiler emits the
AVX2/SSE4.1/NEON instructions for them *regardless of the global target CPU*,
and `std::is_x86_feature_detected!` / `is_aarch64_feature_detected!` selects the
right one at run time. A plain `cargo build --release` therefore already
produces a **portable** binary that runs the appropriate backend on whatever CPU
executes it — no special flags required.

### Do you need `-C target-cpu=native`?

No — and for distributed binaries you should *not* use it:

- **Not needed for correctness or backend selection.** The `#[target_feature]`
  kernels already emit and dispatch to AVX2/SSE4.1/NEON under a default release
  build.
- **No measurable speedup.** The SIMD kernels operate per DP row, so the only
  thing `target-cpu=native` could add — inlining those per-row kernels into the
  driver — is negligible against the per-row SIMD work. (Measured: a plain
  release build and `-C target-cpu=native` are within ~0.5% — i.e. noise —
  across 150–2000 bp alignments.)
- **It makes the binary non-portable.** `target-cpu=native` pins codegen to the
  *build* machine's instruction set, so a binary built on (say) an AVX-512 host
  will `SIGILL` (illegal instruction) on an older CPU. Do not use it for any
  binary you ship.

If you are building strictly for one machine, `RUSTFLAGS="-C target-cpu=native"`
is harmless and lets the compiler auto-vectorize the surrounding scalar glue,
but it will not meaningfully change the alignment kernels' throughput.

This project is closely related to [minimap2](https://github.com/lh3/minimap2), where ksw2 is used as a core alignment component.

## Scope

At this time, `ksw2rs` implements only the `ksw2_extz2_sse` variant (ported to stable Rust with scalar + SIMD backends).

This is intentional: the current primary client is [`bramble-rs`](https://github.com/zrudnick/bramble/tree/new-main/bramble-rs), and this is the specific kernel variant required there.

Additional ksw2 variants may be added in the future.

## Design goal

The guiding goal is not to redesign the algorithm, but to make a faithful, direct Rust port of the original C kernel:

- preserve scoring and traceback semantics,
- preserve anti-diagonal DP structure,
- preserve SIMD-oriented data flow,
- support stable Rust and modern SIMD backends.

## Provenance

This codebase was produced almost entirely via automated conversion and iterative optimization using an AI agen (Codex 5.3), with human review and direction.

## Usage

`ksw2rs` expects sequences encoded in the same compact DNA5 alphabet used by ksw2:

- `A=0`
- `C=1`
- `G=2`
- `T=3`
- `N/other=4`

### One-shot alignment

```rust
use ksw2rs::{Extz, Extz2Input, extz2};

fn dna5_mat(match_score: i8, mismatch_score: i8) -> [i8; 25] {
    let mut mat = [mismatch_score; 25];
    for i in 0..5 {
        mat[i * 5 + i] = match_score;
    }
    mat[24] = 0;
    mat
}

let query = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
let target = query.clone();
let mat = dna5_mat(2, -4);

let input = Extz2Input {
    query: &query,
    target: &target,
    m: 5,
    mat: &mat,
    q: 4,
    e: 2,
    w: -1,
    zdrop: 100,
    end_bonus: 0,
    flag: 0, // traceback enabled
};

let mut ez = Extz::default();
extz2(&input, &mut ez);
println!("score={}, cigar_len={}", ez.score, ez.cigar.len());
```

### High-throughput API (`Aligner`)

For repeated alignments, prefer `Aligner`. It reuses both DP scratch buffers and the result object to reduce per-call overhead.

```rust
use ksw2rs::{Aligner, Extz2Input, KSW_EZ_SCORE_ONLY};

fn dna5_mat(match_score: i8, mismatch_score: i8) -> [i8; 25] {
    let mut mat = [mismatch_score; 25];
    for i in 0..5 {
        mat[i * 5 + i] = match_score;
    }
    mat[24] = 0;
    mat
}

let mat = dna5_mat(2, -4);
let mut aligner = Aligner::new();

for (query, target) in [
    (vec![0u8, 1, 2, 3], vec![0u8, 1, 2, 3]),
    (vec![0u8, 0, 1, 1], vec![0u8, 1, 1, 2]),
] {
    let input = Extz2Input {
        query: &query,
        target: &target,
        m: 5,
        mat: &mat,
        q: 4,
        e: 2,
        w: -1,
        zdrop: 100,
        end_bonus: 0,
        flag: KSW_EZ_SCORE_ONLY, // score-only mode
    };
    let ez = aligner.align(&input);
    println!("score={}", ez.score);
}
```

### Manual workspace reuse

If you prefer functional-style calls, you can reuse `Workspace` directly:

```rust
use ksw2rs::{Extz, Extz2Input, Workspace, extz2_with_workspace};

let query = vec![0u8, 1, 2, 3];
let target = vec![0u8, 1, 2, 3];
let mat = [
     2, -4, -4, -4, -4,
    -4,  2, -4, -4, -4,
    -4, -4,  2, -4, -4,
    -4, -4, -4,  2, -4,
    -4, -4, -4, -4,  0,
];

let input = Extz2Input {
    query: &query,
    target: &target,
    m: 5,
    mat: &mat,
    q: 4,
    e: 2,
    w: -1,
    zdrop: 100,
    end_bonus: 0,
    flag: 0,
};

let mut ws = Workspace::default();
let mut ez = Extz::default();
extz2_with_workspace(&input, &mut ez, &mut ws);
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](./LICENSE).
