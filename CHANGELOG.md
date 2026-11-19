# Changelog

## Unreleased

NEON only; results and the x86-64 kernels are unchanged.

0.3.0 was 4–8% slower than 0.2.1 on Apple Silicon with `KSW_EZ_NO_MAX`. That
is fixed, and the exact path is faster. Time relative to 0.2.1 on an Apple M5
Max (native build, one core, workspace at a fixed page offset):

| workload | 0.3.0 | this version |
|---|---|---|
| score-only exact, 150×165, w=15 | 1.05× | 1.20× |
| CIGAR, 150×165, w=15 | 1.02× | 1.15× |
| score-only exact, 150×165, full band | 1.56× | 1.71× |
| score-only, z-drop, 1000×1100, full band | 1.41× | 1.43× |
| `KSW_EZ_NO_MAX`, 150×165, w=15 | 0.94× | 1.01× |
| `KSW_EZ_NO_MAX`, 150×165, full band | 0.92× | 0.99× |
| `KSW_EZ_APPROX_MAX`, 150×165, full band | 0.96× | 0.98× |

- The `KSW_EZ_NO_MAX` row update is inlined into the anti-diagonal loop on
  aarch64; the call made the loop spill and reload its vector constants on
  every row.
- The exact row update is inlined too, and its lanes start at the first cell
  of the row instead of at a multiple of four (the update of #4).
- The score fill no longer masks its first and last block. Narrow rows use a
  plain 16-cell loop; wide rows keep aligned stores between an unaligned first
  and last chunk.
- The score row has 16 cells of slack after it, so the fill never clips its
  last chunk at the end of the row. On short targets (salmon's default
  extensions are a few dozen bases) most rows ended in a clipped, masked store
  (found and fixed by @BenjaminDEMAILLE).

## 0.3.0 (2026-11-18)

No API signatures change, but some results do; see "Changed results".

### Faster

Time relative to 0.2.1 on x86-64 (AMD EPYC 9555, AVX2 kernel, one core).
"Portable" is a build without `target-cpu=native`, as a dependent crate gets
by default.

| workload | native | portable |
|---|---|---|
| score-only exact, 150×165, w=15 | 1.21× | 1.41× |
| CIGAR, 150×165, w=15 | 1.18× | 1.37× |
| score-only exact, 150×165, full band | 1.60× | 1.73× |
| score-only, z-drop, 1000×1100, full band | 1.47× | 1.32× |
| CIGAR, right-aligned gaps, z-drop, 150×165 | 1.55× | 1.63× |
| `KSW_EZ_NO_MAX`, 150×165, w=15 | 1.20× | 1.31× |
| `KSW_EZ_NO_MAX`, 150×165, full band | 1.05× | 1.09× |
| `KSW_EZ_APPROX_MAX`, 150×165, full band | 1.12× | 1.30× |

The NEON kernel received the same changes and passes the test suite on Apple
and Linux arm64, but has not been benchmarked yet.

- The exact path (no `KSW_EZ_APPROX_MAX`) tracks the row maximum with
  per-lane accumulators reduced once per row, instead of a horizontal
  reduction and a branch per block (#4, by @BenjaminDEMAILLE).
- The score fill and the exact row update work on the same aligned blocks
  from one anti-diagonal to the next, so their loads are forwarded from a
  single store. Before, a load of the score row spanned two stores on every
  anti-diagonal and waited for both (#5).
- The alignment core is inlined into the per-CPU entry points, so the SIMD
  kernels inline into it in portable builds too (#5).

### Fixed

- Banded CIGARs could differ from KSW2: the rows of the traceback matrix were
  narrower than in C, and the backtrack read the wrong cell at a band edge.
- With `KSW_EZ_GENERIC_SC`, banded results could differ from KSW2.
- The scalar kernel (`extz2_scalar`, and any CPU without SSE4.1 or NEON)
  could differ from the SIMD kernels on banded input.
- A very large `w` (near `i32::MAX`) overflowed: a panic in debug builds, a
  spurious z-dropped result in release builds. Any `w` at least as large as
  the longer sequence now means "no band".
- Scoring schemes with `max_sc + 2 * (q + e) > 127` returned wrong scores from
  the SIMD kernels. They now run on the scalar kernel, which is exact up to
  255; beyond that the result is left reset (`score == KSW_NEG_INF`, empty
  CIGAR) rather than computed wrongly. The limits are documented on
  `Extz2Input`.
- `Workspace` no longer exposes uninitialized memory through `Debug` and
  `Clone`.

### Changed results

- Banded alignments with a CIGAR, and banded alignments with
  `KSW_EZ_GENERIC_SC`, can differ from 0.2.1. They now match KSW2.
- Banded results of the scalar kernel can differ from 0.2.1. They now match
  the SIMD kernels.
- Unbanded results, and banded score-only results on the default scoring
  path, are unchanged.

### Known difference from KSW2

When several cells share the maximum score, ksw2rs reports the first one
along the anti-diagonal; KSW2 reports whichever its SIMD lane order finds.
`max_t` and `max_q` can therefore differ, and with them a z-drop decision or
the CIGAR of an extension that stops at the maximum. This was already the
case in earlier versions.

### Tests and CI

- New tests: scalar against SIMD on banded input, scores and CIGARs against a
  plain affine-gap DP across the scoring range, banded differential tests
  against the C reference, and each x86 kernel run directly.
- CI runs the suite on x86-64 Linux, arm64 Linux and arm64 macOS, in native
  and portable builds.

## 0.2.1 (2026-10-31)

- `KSW_EZ_NO_MAX`: skip tracking the overall maximum in score-only alignment
  without z-drop (#2).

## 0.2.0 (2026-06-12)

- The SIMD kernel is resolved once (`resolve_kernel`, `Aligner::new`) and
  called through a stored pointer, with no feature check per alignment.
- Minimum supported Rust version is 1.91.
