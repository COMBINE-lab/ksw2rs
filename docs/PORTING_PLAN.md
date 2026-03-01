# KSW2 C-to-Rust Porting Plan

## Goal
Translate `c/ksw2_extz2_sse.c` and supporting pieces from `c/ksw2.h` into a native Rust library that:

- stays on stable Rust,
- preserves core algorithmic behavior and score/CIGAR semantics,
- works on x86_64 (SSE4.1-capable systems) and aarch64 (NEON),
- keeps a path to SIMD-grade performance.

## Required Workstreams

1. API and data model parity
- Port constants/flags and result structure from `ksw2.h`.
- Keep BAM-like packed CIGAR encoding (`len << 4 | op`).

2. Algorithmic parity
- Port anti-diagonal DP recurrence with transformed byte states (`u`, `v`, `x`, `y`).
- Preserve z-drop logic, exact/approximate max modes, and extension-only behavior.
- Port backtracking state encoding and tie-break behavior (left/right gap alignment).

3. Platform and dispatch strategy
- Keep stable Rust only (`std` + `#[target_feature]` + runtime detection).
- x86_64 dispatch for SSE4.1-capable CPUs.
- aarch64 dispatch for NEON-capable CPUs.
- Scalar fallback for all other targets.

4. Correctness validation
- Unit tests for common score/CIGAR invariants.
- Differential tests versus C reference implementation (next step).

5. Performance hardening
- Preserve memory layout locality and compact state footprint.
- Add architecture-specific hand-written intrinsic kernels for SSE4.1/NEON (next step).
- Benchmark against original C on representative read lengths.

## Status (Current Commit)

Completed:
- Native Rust `extz2` implementation with extensive inline/rustdoc documentation.
- Ported flags/constants/result/CIGAR and backtracking logic.
- Preserved exact and approximate score tracking behavior.
- Runtime dispatch stubs for SSE4.1 and NEON using stable Rust feature detection.
- Unit tests validating basic behavior.

Remaining for full performance parity:
- Hand-written SSE4.1 and NEON intrinsics for the hot DP loop.
- Differential fuzz/fixture tests against the C implementation.
- Criterion benchmarks and tuning of memory alignment/blocking.
