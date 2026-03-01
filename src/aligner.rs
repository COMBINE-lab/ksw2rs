use crate::{Extz, Extz2Input, Workspace, extz2_scalar_with_workspace, extz2_with_workspace};

/// Reusable alignment engine for repeated calls.
///
/// `Aligner` is a stateful convenience wrapper around `extz2_with_workspace`.
/// It owns:
/// - a reusable [`Workspace`] for DP scratch memory,
/// - a reusable [`Extz`] result object.
///
/// This avoids frequent heap allocations and repeated initialization costs in
/// workloads that align many sequence pairs (for example, read mapping or
/// overlap detection pipelines).
///
/// # Typical usage
/// ```
/// use ksw2rs::{Aligner, Extz2Input};
///
/// let query = [0u8, 1, 2, 3];
/// let target = [0u8, 1, 2, 3];
/// let mat = [
///      2, -4, -4, -4, -4,
///     -4,  2, -4, -4, -4,
///     -4, -4,  2, -4, -4,
///     -4, -4, -4,  2, -4,
///     -4, -4, -4, -4,  0,
/// ];
///
/// let mut aligner = Aligner::new();
/// let input = Extz2Input {
///     query: &query,
///     target: &target,
///     m: 5,
///     mat: &mat,
///     q: 4,
///     e: 2,
///     w: -1,
///     zdrop: 100,
///     end_bonus: 0,
///     flag: 0,
/// };
///
/// let result = aligner.align(&input);
/// assert!(result.score > 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Aligner {
    workspace: Workspace,
    result: Extz,
}

impl Aligner {
    /// Create a new aligner with empty reusable buffers.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new aligner from a caller-provided workspace.
    ///
    /// This is useful when the caller wants explicit control over workspace
    /// lifetime or wants to move pre-warmed capacities between components.
    #[inline]
    pub fn with_workspace(workspace: Workspace) -> Self {
        Self {
            workspace,
            result: Extz::default(),
        }
    }

    /// Run alignment and return a shared reference to the internally stored
    /// result.
    ///
    /// The returned reference remains valid until the next mutating call on the
    /// same `Aligner`.
    #[inline]
    pub fn align(&mut self, input: &Extz2Input<'_>) -> &Extz {
        extz2_with_workspace(input, &mut self.result, &mut self.workspace);
        &self.result
    }

    /// Run alignment with scalar backend only and return the internal result.
    ///
    /// This is mostly intended for benchmarking and differential debugging.
    #[inline]
    pub fn align_scalar(&mut self, input: &Extz2Input<'_>) -> &Extz {
        extz2_scalar_with_workspace(input, &mut self.result, &mut self.workspace);
        &self.result
    }

    /// Run alignment into a caller-provided output while still reusing this
    /// aligner's internal workspace.
    #[inline]
    pub fn align_into(&mut self, input: &Extz2Input<'_>, out: &mut Extz) {
        extz2_with_workspace(input, out, &mut self.workspace);
    }

    /// Immutable access to the reusable scratch workspace.
    #[inline]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Mutable access to the reusable scratch workspace.
    #[inline]
    pub fn workspace_mut(&mut self) -> &mut Workspace {
        &mut self.workspace
    }

    /// Borrow the most recent internally stored result.
    #[inline]
    pub fn result(&self) -> &Extz {
        &self.result
    }

    /// Mutable access to the internal result object.
    ///
    /// This can be useful for integration code that wants to inspect or reset
    /// fields without allocating a separate output object.
    #[inline]
    pub fn result_mut(&mut self) -> &mut Extz {
        &mut self.result
    }
}
