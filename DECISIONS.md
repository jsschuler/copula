# Implementation Decisions Log

Tracks choices made while implementing `high_dimensional_copula_claude_code_spec.md`
that the spec left open, plus any deviations from it. Ordered by milestone.
Update this file whenever a milestone introduces a new judgment call or a
deviation from the spec — don't let it fall behind the code.

---

## Milestone 1 — core numerics and marginal smoother

**Build system: plain Makefile, not CMake.**
CMake is not installed in the development sandbox and an unverified build
config is worse than none. A plain `Makefile` (using only `make`/`cc`, both
present) builds the static library, all tests, and the examples. Worth
revisiting once language bindings (Milestones 10–12) are underway, since
CMake integrates more naturally with pybind11/Rcpp/BinaryBuilder-style
toolchains than a hand-rolled Makefile would.

**Default bandwidth search bounds.**
The spec (§2.1) requires bounds "derived from robust scale measures such as
IQR or MAD" but does not pin an exact formula. Chosen:
```
robust_scale = min(IQR / 1.349, MAD)         (falls back to sample sd if both are 0)
sigma_min    = 0.05 * robust_scale
sigma_max    = 3.0  * robust_scale
```
`IQR/1.349` and MAD (scaled by 1.4826) are both consistent estimators of the
standard deviation under normality; taking the smaller of the two is a
standard robustness trick (guards against one measure being distorted by a
particular data pathology). The `0.05`–`3.0` multiplier range is a wide net
intended to comfortably contain a good LOO-CV optimum without user input.

**LOO bandwidth objective is O(n²) per evaluation.**
Leave-one-out log-likelihood recomputes a full mixture sum per held-out
point, with no incremental/cached updates across optimizer iterations.
Acceptable per spec §31 Milestone 14 ("only after numerical correctness"
should performance work happen). Revisit if this becomes a bottleneck once
real dataset sizes are exercised.

**Golden-section search for the 1D optimizer.**
The spec requires "a deterministic bounded one-dimensional optimizer"
without naming an algorithm. Golden-section search was chosen because it
needs no derivatives, is exactly reproducible (no randomness, no
line-search heuristics with platform-dependent tie-breaking), and is simple
enough to keep the O(n²) LOO objective as its only real cost.

---

## Milestone 2 — copula transform

**`hdcd_marginal_t` introduced as a per-dimension opaque handle, ahead of
the full `hdcd_model_t` from spec §21.**
The spec's C ABI sketch centers on a single aggregate `hdcd_model_t` that
holds all `d` marginals plus the DAG etc. Milestone 2 only asked for "a
fitted marginal object," so a smaller single-dimension handle
(`hdcd_marginal_t`, owning a copy of its observed training values `O_j`
plus its selected bandwidth) was introduced now and will be composed into
`hdcd_model_t` at whichever later milestone assembles the full model,
rather than building the aggregate type prematurely.

**Default copula clipping epsilon.**
`HDCD_DEFAULT_COPULA_EPSILON = 1e-9`. Spec §4 mandates a clipping rule and
that the constant be identical across C/Python/R/Julia, but leaves the
value itself to the implementation. `1e-9` keeps `U` numerically distinct
from 0/1 in double precision while being small enough not to distort
non-extreme observations.

**Missing values write `NaN` into the transform output.**
Spec §23 is explicit that missingness must be carried by the explicit
`observed_mask`, never solely by `NaN`. `hdcd_transform_to_copula` honors
that: the mask is the source of truth and is never modified by the call.
`NaN` is written to `u_out[i]` at missing positions purely as a defensive
sentinel, so that a caller who forgets to consult the mask gets an obvious
garbage value rather than stale/uninitialized memory — it is not itself
the missingness signal.

**Test-only deterministic PRNG (xorshift64\*).**
Used only inside `tests/test_copula_transform.c` to generate a reproducible
synthetic normal sample for the "approximately uniform after transform"
check (spec §29.2). This is not part of the public API and is unrelated to
the seeded RNG module (`rng/rng.c`) the spec schedules for the sampling and
simulated-annealing milestones — that module doesn't exist yet and
shouldn't be built before it's needed. The same small block is duplicated
(not shared via a header) in the Milestone 3 test files and the Milestone 3
example for the same reason — see below.

---

## Milestone 3 — pairwise distance correlation

**`dependence_matrix.c` added as a new file beyond the two the spec names
under `dcor/`.**
Spec §22 lists only `dcor_exact.c` and `dcor_fast.c` under `dcor/`, but §5
("Pairwise Dependence Matrix") describes a distinct concern from raw
two-vector dCor: extracting the pairwise-complete set `O_jk = O_j ∩ O_k`
per pair, looping over all `(j,k)`, and storing both the `d x d` matrix and
the effective-sample-size matrix. Keeping `dcor_exact.c` focused purely on
"dCor of two paired 1D arrays" and putting that orchestration in a
sibling `dependence_matrix.c` mirrors how `marginal_model.c` was kept
separate from the raw Gaussian-mixture math in Milestone 1/2.

**Approximate ("fast") distance-correlation backend not implemented.**
Spec §5 says an accelerated backend "may be added for large n" and "must
be selectable" if it exists — it does not mandate one for v1. Only the
exact O(n²) backend (`hdcd_dcor_exact`) exists; `dcor_fast.c` is deferred
until a milestone actually needs it (consistent with §31 M14 deferring all
performance work).

**Degenerate cases return a value rather than an error.**
`hdcd_dcor_exact` returns `dCor = 0` (not an error) for a constant input
series, since the 0/0 the raw formula produces is a well-defined limiting
case, not a caller mistake. `hdcd_compute_dependence_matrix` returns `NaN`
(not an error) for a pair with fewer than 2 pairwise-complete rows, since
dCor is mathematically undefined there — the caller is expected to check
the stored effective sample size (`hdcd_dependence_matrix_n_effective`)
rather than have a fabricated number silently masquerade as a real
correlation.

**Diagonal is hardcoded to exactly 1.0, not computed via `dCor(U_j, U_j)`.**
Floating-point self-correlation would land extremely close to 1 but not
bit-exact, and there's no ambiguity about what the diagonal should be by
definition. This also sidesteps a degenerate edge case: a dimension with
zero variance would otherwise make `dCor(U_j, U_j)` itself hit the 0/0
guard and incorrectly report 0 instead of a self-correlation of 1.

---

## Milestone 4 — MST and persistent-topology ordering

**Module split follows the spec's suggested layout (§22) closely, with
five internal files under `topology/` plus one public orchestrator.**
`union_find.{h,c}`, `mst.{h,c}`, `persistent_affinity.{h,c}`,
`merge_tree.{h,c}`, and `ordering.{h,c}` are all private (headers live in
`src/topology/`, not `include/hdcd/`) and are wired together by
`topology.c`, which implements the single public header
`include/hdcd/topology.h`. This mirrors how `dependence_matrix.c` sits
on top of `dcor_exact.c` in Milestone 3 — tests only ever exercise the
public `hdcd_compute_topology` surface, never the internal modules
directly.

**`S(C)` (component score) is computed via an O(1) closed-form recurrence
during merge-tree construction, not by summing the materialized affinity
matrix over member pairs.**
At the moment two components `C1`, `C2` merge at MST edge weight `w`,
*every* cross pair `(x in C1, y in C2)` has merge level `tau_xy = w`
exactly — the same bottleneck-path fact `persistent_affinity.c` uses
directly. So:
```
S(C1 ∪ C2) = S(C1) + S(C2) + 2 * |C1| * |C2| * (1 - w)
```
is exact, not an approximation, and lets the whole tree be scored in
O(d) additional work with no member-list bookkeeping in `merge_tree.c` at
all (`persistent_affinity.c` still explicitly materializes the full
`tau`/affinity matrix, via its own member-list bookkeeping, purely as
the public diagnostic surface spec §33 asks for — the two computations
are independent and were cross-checked against each other during
development on the hand-worked example below, not merely assumed to
agree).

**Important, spec-faithful, occasionally counterintuitive consequence:
`S(C)` is an *extensive* (size-sensitive) sum over all ordered pairs in
`C`, not a per-pair average.** A large supercluster formed by merging two
individually looser clusters can outscore — and so be visited before — a
smaller but individually tighter cluster, purely because it has more
pairs to sum over. Confirmed this is correct (not a bug) by hand-tracing
`examples/example_topology.c`'s actual MST edges through the recurrence
independently in Python and matching the program's output exactly; the
example now prints a note explaining this so the result doesn't look
like a defect. This is what spec §6.3's literal formula
(`S(C) = sum_j sum_{k != j} A_jk`, an unnormalized double sum) specifies;
the spec separately defines a size-normalized per-variable quantity
`S_j^(C)` but that is not what's used for component-vs-component
ordering decisions.

**Hand-worked correctness check.** Before writing `tests/test_topology.c`,
a 4-node scenario was worked by hand: groups `{0,1}` (looser,
`delta=0.3`) and `{2,3}` (tighter, `delta=0.05`), weak cross-group
dependence (`delta=0.9`). MST edges `(2,3,0.05), (0,1,0.3), (0,2,0.9)`;
scores `S({0,1})=1.4 < S({2,3})=1.9`; predicted final ordering
`[2,3,0,1]` (tighter group first, both blocks contiguous). The test
builds data with that same qualitative shape end-to-end (marginal fit →
copula transform → dependence matrix → topology) and asserts exactly
that structural outcome — this is the primary correctness anchor for the
whole milestone, everything else (three-cluster contiguity,
reproducibility, disconnected-graph failure) checks a narrower property.

**Disconnected dependence graph is a hard failure
(`HDCD_ERROR_NUMERICAL`), not a partial/best-effort tree.**
If some variable's dCor is NaN (< 2 pairwise-complete rows, spec §5)
against every other variable, no MST can span all `d` nodes. Per spec
§24 ("never silently continue after NaN/Inf creation") and §36 rule 13,
this fails clearly rather than returning a forest or silently dropping
the disconnected variable from the ordering.
