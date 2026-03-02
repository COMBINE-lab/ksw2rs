# ksw2rs

`ksw2rs` is a native Rust port of [ksw2](https://github.com/lh3/ksw2), focused on preserving ksw2 behavior and performance characteristics as directly as possible.

## Building for optimal performance

`ksw2rs` uses runtime feature detection to dispatch to the best available SIMD backend (AVX2, SSE4.1, or NEON). However, to ensure the compiler can generate optimal code for all backends, you should compile with native target CPU support:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Alternatively, add this to your project's `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

This ensures the compiler is aware of all SIMD instruction sets your CPU supports, enabling the best runtime dispatch path. Without this, the compiler may not emit AVX2 or other advanced instruction variants even though the runtime detection selects them.

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
