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

---

## Milestone 5 — Bernstein basis and roughness penalty

**Binomial coefficient via the multiplicative recurrence
`C(m,r+1) = C(m,r)*(m-r)/(r+1)`, not factorials.**
Spec §24 explicitly calls this out ("avoid direct factorial evaluation
for Bernstein coefficients; use recurrence relations where appropriate").
Direct factorials overflow `double` around `m≈170`; the recurrence stays
well-behaved far beyond any "modest" degree the spec anticipates (§9).

**`hdcd_bernstein_basis` domain is the closed interval `[0,1]`, not just
the open copula-scale interior `(epsilon, 1-epsilon)`.**
Basis-identity tests (partition of unity, boundary values `B_0(0)=1`,
`B_m(1)=1`) need the exact endpoints; rejecting them would make those
tests impossible to write cleanly. Values actually flowing through the
pipeline are already clipped to `(epsilon, 1-epsilon)` by
`hdcd_transform_to_copula` (Milestone 2), so this wider domain never
admits values the rest of the system wouldn't otherwise accept.

**Roughness-penalty gradient implemented as direct residual-by-residual
backprop (scatter `2*d*[1,-2,1]` per second-difference triple), not via a
closed-form `D^T D` band-matrix kernel.**
Both are mathematically equivalent, but the scatter approach is a direct,
easy-to-verify transcription of "differentiate a sum of squared linear
residuals," rather than requiring a separately-derived (and separately
bug-prone) closed form. Verified against finite differences in tests
either way, so this is a legibility choice, not a correctness hedge.

**Discovered while writing the example, not a bug:** a `Theta` surface
that is additively separable (`Theta[r][s] = f(r) + g(s)`) always
produces `g(u,z) ≡ 0`, because the *centered* basis sums to exactly zero
along each axis (`sum_r B~_r(u) = 0` for any `u`, since the raw basis is
a partition of unity and centering subtracts the constant `1/(m+1)`
`m+1` times). A bilinear term (`r*s`) is not additively separable and
survives centering, so `example_bernstein.c` uses
`Theta[r][s] = a*r*s + b*r + c*s` (still zero roughness penalty, since
second differences are linear-in-each-index and this form is affine in
each index separately) to show a non-degenerate `g(u,z)` alongside
`R(Theta)=0`. Worth remembering for later milestones: only the
non-additively-separable part of an edge's `Theta` actually contributes
to the conditional kernel at all.

---

## Milestone 6 — copula-preserving Sinkhorn normalization

**Sinkhorn is a standalone module over an abstract raw-kernel callback,
not wired to the Bernstein kernel or any DAG structure.**
`hdcd_raw_kernel_fn` is `double (*)(double u, const double *z, size_t
z_dim, void *userdata)` — Milestone 5's Bernstein tensor kernel is one
possible caller of this, exercised in tests/the example via
`exp(hdcd_bernstein_tensor_interaction(...))`, but `normalize.c` itself
has no dependency on `bernstein.h` or any notion of parents/edges/DAGs
(those don't exist until Milestone 7+). This mirrors the same
"orchestration file stays generic, concrete kernel is the caller's
problem" split already used for `dcor_exact.c` vs. `dependence_matrix.c`
and `merge_tree.c`'s kernel-agnostic recurrence.

**`z_samples` are caller-supplied, not drawn by this module.**
Spec §11.2 says the `q_j(z)` expectation "may use Monte Carlo samples
... permit cached parent samples" — read as license to accept
externally-supplied draws rather than a mandate to implement sampling
here. There is no machinery yet (Milestone 6) for sampling from a
partially-built joint copula model, which is what `q_j(z)` will actually
be once multi-parent nodes exist; building a sampler prematurely, before
anything needs it, would be scope creep. Tests use i.i.d. Uniform(0,1)
draws for `z_samples`, which is not an arbitrary test choice: for a
single parent, `q_j(z)` is exactly the previously-fitted parent's own
copula-scale marginal, which is uniform by construction (spec §12's
inductive invariant) — so this is the *correct* `q_j` for the one-parent
case the whole milestone is scoped to, not a simplification of it.

**`sinkhorn/quadrature.c` and `sinkhorn/monte_carlo.c` (named in spec
§22) were not created as separate files.** The generic Simpson-rule
utility lives in `numerics/quadrature.c` (reusable outside Sinkhorn, and
spec §22 lists `numerics/quadrature.c` too) and is called directly from
`sinkhorn/normalize.c`; the "Monte Carlo" side is literally an unweighted
average over caller-supplied samples with no separate logic worth its
own file. Same reasoning as skipping `dcor_fast.c` in Milestone 3: no
empty wrapper files.

**Evaluating `c_j(u|z)` at an arbitrary, possibly out-of-sample, `(u,z)`
recomputes `a(u)` and `b(z)` from their closed-form update rules against
the fitted state, rather than interpolating a cached grid.** Both
closed forms are already O(small) closures over the fitted vectors
(`a(u)` needs the fitted `b` + `z_samples`; `b(z)` needs the fitted `a` +
`u_nodes`/weights), so this is exact given the fitted discretization, not
an approximation layered on top of it — and it costs no more than a
handful of extra kernel evaluations per query. Confirmed in
`test_conditional_integral_error_below_tolerance`: `∫c(u|z)du ≈ 1` holds
to `1e-4` (via an independently re-run Simpson quadrature through the
public API) at both a training-sample `z` and a fresh `z` never seen
during fitting, because `b(z)`'s closed form only involves a
deterministic quadrature over `u`, not a Monte Carlo term — the only
place genuine Monte Carlo noise can appear is `a(u)`'s dependence on the
finite `z_samples`, which is exactly why `test_marginal_preservation_
error_below_tolerance` uses a much looser tolerance (`0.08`) when
checking against a *fresh, independent* `z` sample, versus `1e-6` for
the internal diagnostic that's self-consistent against the training
`z_samples` by construction.

**Convergence metric evaluated jointly at the current `(a,b)` after
*both* half-steps, not as either half-step's own (trivially near-zero)
residual.** Each Sinkhorn half-step exactly satisfies the constraint it
was just solving for, given the *other* variable's previous value — so
checking the conditional-integral constraint immediately after the
`b`-update (using the `a` that produced it) would always read ~0 and
tell you nothing about whether the joint fixed point has been reached.
The implementation always recomputes both error terms from the latest
`a` and `b` together, matching spec §11.2's metric literally.

---

## Milestone 7 — fixed-DAG fitting

The largest milestone so far, tying together everything since Milestone
1. Several of the entries below were only discovered by actually running
the pipeline against real dependent data — not designed in up front —
and are logged in the order they were found, since later fixes depend on
understanding earlier ones.

**RNG module (`hdcd/rng.h`, `src/rng/rng.c`) pulled forward from its
originally-scheduled later milestone.** Spec §22 lists `rng/rng.c`
alongside the annealing/sampling modules, but §16's held-out scoring
("compare parent sets on a common held-out subset... store effective
sample size") needs a reproducible train/holdout split *now*, which
needs a real seeded RNG, not another test-only PRNG copy-pasted into
library code. Minimal by design: seed, uniform-in-(0,1), and a
Fisher-Yates shuffle — nothing annealing/sampling will need later has
been speculatively added.

**DAG acyclicity is enforced incrementally, at `hdcd_dag_add_edge` time,
via reachability (BFS), not deferred to a validation pass.** A DAG built
exclusively through the public API is therefore always acyclic by
construction — cheaper to reason about downstream (no code path ever
needs to handle "this `hdcd_dag_t` might secretly be cyclic"). A general
`hdcd_dag_topological_order` (Kahn's algorithm) is still provided as an
independent validator, since spec §19 (a later milestone) needs to
validate arbitrary externally-supplied DAGs, not just ones built via
`add_edge`.

**`dag/validation.c` (named in spec §22) was folded into `graph.c`,
same reasoning as prior "no empty wrapper file" decisions** (Milestone
3's `dcor_fast.c`, Milestone 6's `sinkhorn/quadrature.c` and
`monte_carlo.c`): topological-order validation is a few dozen lines that
belongs next to the structure it validates, not split into its own file
for the sake of matching the spec's suggested layout exactly.

**Theta-fitting and Sinkhorn normalization are two separate SEQUENTIAL
steps — fit Θ once via a raw-kernel surrogate objective, then
Sinkhorn-normalize once — not an alternating/joint optimization through
Sinkhorn's implicit fixed point.** This is a direct reading of spec
§28's literal, numbered step list ("3. optimize edge coefficient
matrices; 4. apply Sinkhorn normalization" — two distinct, ordered
steps, not one). It also sidesteps implicit differentiation through an
iterative fixed-point procedure (Sinkhorn), which spec doesn't ask for
and which would be substantial additional machinery for a "version 1"
system. The Θ-fit surrogate objective is `sum_i g_jk(u_i,z_i;Θ_jk) -
λ_R·R(Θ_jk)` (the raw log-kernel fit to the data, ignoring how `a,b`
*would* renormalize if refit) — this is well-posed to optimize on its
own because `g_jk` is *linear* in Θ_jk (a fixed outer-product
statistic `M_jk` dotted with Θ_jk), making the surrogate a simple
linear-plus-quadratic (hence concave, for `λ_R>0`) function, solvable by
plain gradient ascent without any risk of local optima.

**`q_j(z)` for Sinkhorn is exactly the training rows' own observed
parent values — no model sampler is built or needed.** This resolved
what looked, in the Milestone 6 planning notes, like it might require
sampling from a partially-built joint model (real machinery that
doesn't exist until much later milestones). It doesn't: the empirical
distribution of a node's *training* parent-vector rows already **is**
(an empirical approximation of) `q_j(z)`, so passing those rows directly
as Sinkhorn's `z_samples` is both the simplest and the mathematically
correct choice, not a shortcut — matching spec §11.2's own phrasing
("Monte Carlo samples from the fitted parent distribution").

**M_jk (the Θ-fit sufficient statistic) is the MEAN, not the SUM, of
per-row outer products.** Discovered via actual divergence: an early
version summed `hdcd_bernstein_tensor_gradient(u_i,z_i,m)` over all
training rows without dividing by `n_train`. Since the roughness penalty
`λ_R·R(Θ)` doesn't scale with `n_train`, a fixed `λ_R` regularizes less
and less as the dataset grows — Theta diverged, `R(Theta)` reached
~75,000, and both the Θ-fit and Sinkhorn failed to converge. Averaging
keeps `λ_R`'s effective strength independent of dataset size, the way
ridge regression divides the squared-error term by `n` for the same
reason.

**A small fixed-fraction L2 "ridge backstop"
(`HDCD_LOCAL_FIT_RIDGE_FRACTION = 0.02`, i.e. `0.02·λ_R`) is added to
the Θ-fit objective and gradient, beyond what spec §10 literally
specifies.** After fixing the mean-vs-sum bug above, Θ *still* failed to
converge, because `R(Θ)` has a non-trivial null space: any
`Θ[r][s] = a·r·s + b·r + c·s + d` surface has **zero** second difference
along both axes (the exact fact exploited deliberately in
`examples/example_bernstein.c` back in Milestone 5 — a bilinear surface
is "smooth" by this penalty's own definition but still contributes a
nonzero kernel). Whenever the data's sufficient statistic `M_jk` has any
nonzero component along that ~4-dimensional null space — generic for
real data — the raw objective is **unbounded** along it: gradient ascent
never converges, and more iterations make the held-out score *worse*
(confirmed empirically: sweeping iteration budgets from 300 to 2000
made held-out log-likelihood strictly worse for most `λ_R` values before
this fix). The ridge backstop makes the objective globally strictly
concave (a bounded, unique optimum exists) without materially changing
`R`'s intended smoothing behavior for typical `λ_R`, since it's two
orders of magnitude smaller. This is a necessary well-posedness fix, not
a modeling choice — it is not exposed as a user-facing option, and spec
§10's literal `R(Θ)` formula is unchanged; only the *optimizer* adds this
term internally.

**Θ-fit convergence is judged by relative objective-improvement, not
raw gradient norm.** Direct consequence of the ridge backstop being
deliberately tiny: it makes the objective bounded, but also genuinely
ill-conditioned (curvature ratio ~50:1 between the roughness-penalized
and ridge-only directions), so the gradient norm along the near-flat
ridge direction shrinks extremely slowly even long after the objective
value itself has clearly stopped moving in any way that matters.
Confirmed empirically: gradient-norm convergence never triggered even at
2000 iterations for weaker `λ_R`, while the objective had visibly
plateaued within a few dozen. Relative objective improvement
(`|Δobjective| < tol·(1+|objective|)`) is standard practice for exactly
this "bounded but ill-conditioned" situation, and is what
`hdcd_local_fit_options_t.theta_tol` now means (documented as such in
the header, not left implicit).

**Default `theta_max_iterations` raised from an initial guess of 300 to
2000.** Empirically bisected: with the fixes above, `λ_R=0.1` (a
moderate, reasonable regularization strength) needs roughly 1200
iterations of plain backtracking gradient ascent to satisfy the relative
objective-improvement criterion at the default tolerance. 2000 gives
comfortable headroom without being wastefully large. Weaker `λ_R` needs
more (the ill-conditioning above scales with `1/λ_R`); a caller fitting
with unusually weak regularization should raise `theta_max_iterations`
explicitly.

**Node-wise Bernstein degree tuning (spec §9: "may optionally be tuned
node-wise over a small discrete grid") is not implemented.**
`hdcd_dag_fit` uses the same `hdcd_local_fit_options_t` — including the
same `bernstein_degree` — for every node. Spec marks node-wise tuning as
optional; implementing it would mean a grid search wrapped around
`hdcd_local_fit_node` per node, which is a straightforward addition for
a later pass but adds real scope (needs its own held-out comparison
logic) not required for M7's acceptance criteria.

**Conditional CDF evaluation (spec §13) was not implemented in this
milestone.** M7's explicit "Implement" list (spec §31) is DAG
validation, local parent-set fitting, composite missing-data score,
held-out KL/cross-entropy, parameter storage, and factorized log
density — conditional CDF evaluation isn't on it, and isn't needed by
anything M7 tests. Deferred to whenever it's actually consumed (likely
alongside sampling).

**Joint log-density evaluation (`hdcd_dag_fit_joint_log_density`)
requires every dimension of the query point to be observed; it does not
integrate out missing dimensions.** Spec §16 explicitly places full
observed-data likelihood (which would require integrating over missing
coordinates) out of scope for v1 ("`L_i = ∫ c(u_obs, u_mis) du_mis`...
explicitly out of scope for version 1"), so this restriction is spec-
mandated, not a simplification introduced here.
