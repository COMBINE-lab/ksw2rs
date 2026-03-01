//! Rust port of `ksw2_extz2_sse` with stable-Rust SIMD backends.
//!
//! The main alignment API is [`extz2`] or [`extz2_with_workspace`]. For
//! high-throughput repeated alignments, prefer [`Aligner`], which reuses both
//! DP scratch buffers and result storage across calls.

#![forbid(unsafe_op_in_unsafe_fn)]

mod aligner;
mod extz2;

pub use aligner::Aligner;
pub use extz2::*;
