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

---

## Milestone 8 — simulated annealing DAG search

**`K_hat_j(P)` (spec §15/§17's estimated cross-entropy) is instantiated
directly as `-hdcd_local_fit_holdout_score(fit)`, not recomputed via a
separate Monte Carlo estimate.** Spec §15 defines `K_j(P) =
-E_c*[log c_j(U_j|U_P)]`; Milestone 7's held-out normalized score
`ell_bar_j(P) = (1/n_holdout) sum log c_j(u_ij|u_iP)` is already exactly
this quantity's empirical estimate, computed on data the fit never
trained on (spec §16's whole reason for holding data out: "raw summed
likelihoods are not directly comparable" across parent-set sizes). Since
`J_j(P) = K_hat_j(P) + lambda_E|P| + lambda_R*sum R(Theta_jk)` and M7's
`hdcd_local_fit_t` already exposes both `holdout_score` and
`roughness_penalty`, `J_j(P)` is a two-line formula over existing
Milestone 7 outputs — no new estimation machinery needed.

**Proposals respect a fixed ordering (spec §7:
`Pa(pi_j) ⊆ {pi_1,...,pi_{j-1}}`) rather than relying on `hdcd_dag_t`'s
general incremental cycle check.** This is a direct reading of spec §7
("Given ordering pi...") — the reference-DAG search operates within a
fixed topological backbone, not over arbitrary DAG space. Restricting
candidate parents to strictly-earlier-in-`ordering` nodes makes every
proposal cycle-free *by construction*, which is both more efficient
(O(1) ordering-position lookup instead of a BFS reachability check per
proposal) and more faithful to what spec §7 actually describes: the
ordering constrains the *search space*, not merely a cycle-detection
afterthought. `hdcd_dag_add_edge`'s general cycle check still runs
underneath (from Milestone 7) and would reject a cycle if one were
somehow proposed — it just never has anything to reject here. Arbitrary
non-ordering-respecting DAGs are Milestone 9's concern (`hdcd_dag_t`'s
general validator already exists for exactly that).

**SWAP is evaluated as ONE combined move (remove `k_old`, add `k_new`,
one refit), not two separate ADD/REMOVE steps.** `J_j(P_j)` depends on
the *whole* parent set `P_j`, not on individual edges — spec §14's
additive factorization is additive *across nodes*, not across a single
node's parents. Scoring a swap as two sequential single-edge moves would
require an intermediate (and never-really-existing) one-parent-fewer
state to be fit and scored, wasting a fit and misrepresenting what's
actually being compared (the *final* parent set against the *original*
one).

**The local-fit cache (`src/dag/cache.c`, spec §17.3) is internal, not
exposed in `annealing.h`.** Spec frames it as an implementation detail
("changing one local edge should require refitting only the affected
child factor unless cache reuse is possible") rather than something a
caller needs to configure or inspect. It persists across restarts within
one `hdcd_run_annealing` call (a fit for a given `(child, parent-set)`
is deterministic given the data and options, so a later restart
rediscovering an earlier restart's parent set gets a free cache hit) but
not across separate `hdcd_run_annealing` calls, since nothing outlives
one call to hold it.

**Tested "cached and uncached scores agree" (spec §31 M8) via direct
determinism of `hdcd_local_fit_node` (Milestone 7's public API), not by
reaching into the private cache.** The cache's correctness *is*
precisely this determinism property — a cache hit is only valid because
a fresh re-fit of the same `(child, parent-set)` is guaranteed to
produce a bit-identical result. Since `tests/` only sees public headers
(the cache lives in `src/dag/cache.h`, not `include/hdcd/`), the most
direct and honest test available through the public surface is calling
`hdcd_local_fit_node` twice with identical arguments and checking the
scores match exactly — which is both a valid stand-in for "the cache
behaves correctly" and independently meaningful (it's the same
reproducibility guarantee every other stochastic-but-seeded routine in
this codebase has been tested for since Milestone 1).

**`dag/proposals.c` and `dag/cache.c` (both named in spec §22) were kept
as real, separate files — unlike earlier "no empty wrapper" decisions
(Milestone 3's `dcor_fast.c`, Milestone 6's `sinkhorn/quadrature.c`).**
Both have substantial, genuinely separable logic (move generation with
three distinct kinds and ordering-aware eligibility; a keyed cache with
its own lookup/insert/free lifecycle) that would clutter
`optimize/annealing.c` if inlined. `dag/schedules.c` (also named in
spec §22) was *not* given its own file: the temperature schedule is a
single geometric-decay formula (`T_t = T_0 * cooling_rate^t`) with two
scalar parameters already in `hdcd_annealing_options_t` — a whole file,
or a "pluggable schedule object" abstraction, for one line of math would
be the over-engineering spec's own "small validation grid is sufficient
for version 1" spirit (§18) argues against.

**Automatic `lambda_E`/`lambda_R` selection via a validation grid (spec
§18) is NOT implemented.** Both remain caller-supplied, required
parameters (`lambda_R` already was, since Milestone 7). Spec §18's
requirements aren't gated by any of Milestone 8's three acceptance
criteria (spec §31: cached/uncached agreement, sparse-graph-beats-empty,
seed reproducibility) — none require automatic penalty selection to
pass. Deferred to a later pass, where it would wrap `hdcd_run_annealing`
in a small outer grid search (spec §18: "must occur outside the inner
annealing loop" — already structurally satisfied by keeping it a
separate, later concern rather than threading it through the search
loop itself).

**Verified end-to-end, not just unit-tested:** the example
(`examples/example_annealing.c`) builds a 4-node synthetic "diamond"
(`0->1, 0->2, {1,2}->3`) and the search recovers the exact true edge set
from the empty graph, dropping `J(G)` from `0` to `-0.78`, in well under
a second across 3 restarts — a genuine (if small) discovery, not a
tautology, since the search never sees the true structure, only data.

---

## Milestone 9 — alternative-DAG comparison

**Added `hdcd_dag_from_edges`, a new bulk DAG constructor, specifically
because `hdcd_dag_add_edge` makes cyclic graphs unconstructible by
design.** Spec §31 M9's acceptance criteria explicitly include "rejects
cyclic graphs" — but with only the Milestone 7 API available, there was
no way to even *build* a cyclic `hdcd_dag_t` to test rejection against,
since every edge is checked for a would-create-a-cycle condition the
moment it's added. Spec §19 itself points at the resolution: it frames
the alternative-DAG workflow as taking an arbitrary candidate graph and
validating it "independently of the reference ordering" as an explicit
first step — i.e., the input is expected to be a raw specification that
*needs* validating, not something pre-guaranteed valid by incremental
construction. `hdcd_dag_from_edges` accepts a full edge list up front
(checking only range/self-loop/duplicate/`k_max` per edge, deliberately
skipping the reachability check) and validates the *whole* resulting
graph once at the end via `hdcd_dag_topological_order` (already built in
Milestone 7 as a general, construction-method-independent validator) —
so a genuinely cyclic edge set can now actually be supplied, and is then
cleanly rejected with `HDCD_ERROR_NUMERICAL`. Both DAG constructors
converge on the same invariant (every `hdcd_dag_t` that exists is
acyclic), just enforced at different points — incrementally for
`add_edge`, once in bulk for `from_edges` — so nothing downstream
(`hdcd_dag_fit`, annealing) needs to re-validate; the invariant is
established once, at construction time, regardless of which
constructor built the graph.

**`hdcd_dag_fit` needed NO changes to support arbitrary/alternative
DAGs.** It was already fully ordering-agnostic from Milestone 7 — it
reads each node's parent set directly off the `hdcd_dag_t` and fits
nodes independently, with no reference to any topological order at all
(fitting a node's local conditional factor only needs that node's own
column and its parents' columns from the data, nothing about *other*
nodes' fit order). Spec §19 step 3 ("fit the conditional copula factors
under that graph") was therefore already satisfied; Milestone 9's actual
new work was making the *input* (arbitrary candidate DAGs, potentially
cyclic) safely constructible and validatable, not the fitting itself.

**`hdcd_dag_fit_kl_estimate`/`hdcd_dag_fit_kl_difference` are two-line
formulas over existing Milestone 7 outputs, not new estimation
machinery.** Spec §15 defines `K_j(P) = -E_c*[log c_j(U_j|U_P)]`;
Milestone 7's held-out normalized score `ell_bar_j(P)` (computed on data
the fit never trained on, specifically so it's comparable across
differently-sized parent sets — spec §16) already *is* this quantity's
empirical estimate. `kl_estimate` just negates and sums
`hdcd_local_fit_holdout_score` across nodes; `kl_difference` subtracts
two such sums. The missing shared constant (spec §15: "the entropy term
of `c*` does not depend on `G`") cancels automatically in the
difference, which is exactly the quantity spec §19 asks for and the only
one of the two functions meaningful on its own.

**`kl_difference` checks only that the two fits' dimensions match, not
that they were fit on the same underlying data.** `hdcd_dag_fit_t`
doesn't retain a pointer to its training data (each `hdcd_local_fit_t`
only keeps what it needs — the fitted Sinkhorn state and Θ — not a copy
of the original dataset), so there is no cheap way to verify "same data"
beyond dimension agreement. Documented explicitly in the header as the
caller's responsibility, rather than silently pretending to guarantee
something that isn't actually checked.

**Verified end-to-end with a three-way comparison, not just a pairwise
one:** the test builds the true chain (`0->1->2`) as the reference,
plus two deliberately different alternatives — the reversed-direction
skeleton (built via `hdcd_dag_from_edges`, exercising the
different-topological-order acceptance criterion directly) and the
fully independent empty graph — and confirms `Δ_KL(empty) >
Δ_KL(reversed) > 0`: discarding the dependency entirely loses strictly
more information than merely getting its direction wrong along the same
skeleton. This wasn't obvious in advance (a chain's *pairwise* Gaussian
dependency is symmetric in direction even though its *factorization*
isn't) and held up empirically, giving real confidence the KL-difference
machinery is measuring something sensible rather than an arbitrary
number.

**The non-causal disclaimer (spec §19's closing paragraph, echoed in
§34) is stated in three places, not left to a single docstring:** the
`hdcd_dag_fit_kl_difference` doc comment, the top-of-file umbrella
header (`hdcd.h`) comment (so it's visible to anyone who only reads the
umbrella header), and the example's printed output. Spec §31 M9's fourth
acceptance criterion ("does not label statistical comparison as causal
identification") is a documentation/communication requirement, not a
numerically-testable one — satisfying it means making the caveat hard to
miss wherever this functionality is encountered, not just present
somewhere.

---

## Milestone 10 — Python binding

**ctypes over a C extension (pybind11/Cython/cffi).** Spec §25 says "the
Python layer should be thin," and §36 rule 11 says "do not push core
numerical optimization into the language wrappers" — both point away
from a compiled extension (which would need its own build toolchain
wired into the Python packaging, duplicating what the Makefile already
does) and toward a pure-Python layer that loads the existing compiled
shared library and calls into it directly. `ctypes` is stdlib (no new
build-time dependency), and NumPy arrays already expose a raw buffer
ctypes can point at directly with zero copying, which is exactly what
spec §25 asks for ("avoid copying if input is already compatible
column-major float64").

**Added a `shared` Makefile target (`libhdcd.dylib`/`.so`) and switched
global `CFLAGS` to include `-fPIC`.** Neither existed before Milestone
10 — nothing prior needed a shared library, only the static
`libhdcd.a` used by the C tests/examples. `-fPIC` on every object file
(not a separate PIC-only object set) is the simplest way to let the
same object files back both the static archive and the shared library,
and costs nothing measurable for the static-linked test/example
binaries.

**`hdcd_marginal_logpdf` added to the C core (not just the Python
layer).** `model.logpdf(X)` needs `f_X(x) = c(F_1,...,F_d) * prod_j
f_j(x_j)` (spec §35) — the marginal-density term `f_j(x_j)`, for which
`hdcd_marginal_t` (Milestone 2) only exposed a CDF evaluator, not a
density one. This mirrors exactly how `hdcd_marginal_cdf` itself was
added: a small, natural gap-fill in the C core using machinery that
already existed (`hdcd_gaussian_mixture_logpdf`, from Milestone 1), not
new statistical machinery, and not something that belongs in the Python
layer per rule 11 above. Given its own direct C test
(`test_logpdf_matches_cdf_finite_difference`), not just indirect
exercise through the Python fixture.

**`setup.py` (not a build-backend plugin) runs `make shared` and
copies the resulting library into the package directory before
`build_py` collects files.** This is what actually makes "installable
package" (spec §31 M10) true end to end: `pip install ./python` alone
produces a working, self-contained wheel with the compiled library
bundled inside `hdcd/`, without the user needing to run `make` by hand
first. A `build_py` subclass was the simplest hook that runs early
enough (before packaging, at both `pip install` and `pip install -e`
time) without needing a full custom build-backend.

**`model.sample(...)` raises `NotImplementedError`, not a hand-rolled
Python sampler.** Spec §21 names `hdcd_sample` in its C architecture
sketch, and §25's own preferred-usage snippet shows `model.sample(1000)`
as part of the documented API — but no version-1 milestone (spec §31)
actually schedules *building* it in the C core; Milestones 1-9 build the
density-estimation/DAG pipeline, 10-12 are bindings, 13 is EVT, 14 is
performance. Implementing sampling logic in the Python layer to paper
over that gap would directly violate rule 11 ("do not push core
numerical optimization into the language wrappers") and would mean the
R and Julia bindings (Milestones 11-12) would each need to reinvent the
same missing piece independently. Raising clearly, with a docstring
explaining exactly why, is more honest than a wrapper that pretends the
capability exists.

**Struct field order in `_capi.py` is a direct transcription of each C
header, verified by re-reading the headers immediately before writing
the ctypes `Structure` subclasses** (`hdcd_bandwidth_result_t`,
`hdcd_sinkhorn_options_t`, `hdcd_local_fit_options_t` — which nests the
Sinkhorn options struct — `hdcd_annealing_options_t`, `hdcd_mst_edge_t`).
A wrong field order or type wouldn't necessarily crash; it can silently
read/write the wrong bytes. This is precisely the failure mode the
fixture-comparison tests below are designed to catch.

**"No numerical disagreement with C fixture tests" (spec §31 M10) is
tested via a fixture generated by a small dedicated C program
(`python/tests/generate_fixture.c`), not by re-deriving expected values
by hand or trusting the Python side alone.** Both the fixture generator
and the Python test call the *identical compiled shared library* — so
any disagreement between them can only be a ctypes marshaling bug
(struct layout, argtypes, memory layout), never a statistical or
algorithmic one (those are the C test suite's job, already covered).
The fixture deliberately exercises the highest-risk marshaling
surfaces: struct-by-value returns (`HdcdBandwidthResult`,
`HdcdMstEdge`), a nested struct (`HdcdLocalFitOptions` containing
`HdcdSinkhornOptions`, mixing `size_t`/`double`/`uint64_t` fields where
padding mistakes would silently corrupt values rather than crash), and
pointer-array access (`hdcd_topology_ordering`,
`hdcd_local_fit_parent_order`). The fixture uses RAW (non-copula-scale)
data specifically so the marginal-fitting path is exercised too, not
just the copula-scale machinery — unlike most C-only tests, which often
use exact-uniform Gaussian-copula constructions to isolate dependency
fitting from marginal-smoothing approximation error; here the whole
pipeline's ctypes surface is what's under test, not any one stage's
statistical accuracy. `fixture.json` is checked into the repo (not
regenerated per test run) since it's fully deterministic (spec §24's
seeded-RNG requirement) and doubles as the "same saved test fixture"
spec §29.11 asks the later R/Julia bindings to reuse for cross-language
agreement testing.

**`dependence_matrix_`/`ordering_`/`dag_` return copies set read-only
(`numpy.ndarray.setflags(write=False)`) or immutable tuples, not views
into C-owned memory.** Spec §25: "expose fitted components read-only
where reasonable." A `numpy` array built via `np.ctypeslib.as_array`
directly over C memory would be a zero-copy *view*, but the C object
backing it can be freed (by `__del__`, from a `hdcd_dag_fit_t`/
`hdcd_topology_t` going out of scope) while Python code still holds a
reference to the array — a real use-after-free hazard, not just an
immutability nicety. A defensive copy trades a small amount of memory
for genuine safety, which is the right tradeoff for a "read-only fitted
component," not a hot-path return value.

---

## Milestone 11 — R binding

**Static linking against `libhdcd.a`, not `dlopen`-ing the shared
library like the Python binding.** R packages with a `src/` directory
are compiled by `R CMD INSTALL` itself into one package-specific shared
object (`hdcd.so`); the natural way to pull in the C library is to link
`libhdcd.a` straight into that build (`PKG_LIBS` in `Makevars`), producing
one self-contained artifact with no runtime library-search-path problem
to solve at all — unlike Python's `ctypes.CDLL`, which has to *find* a
separately-built shared library at import time (`_capi.py`'s
`_candidate_library_paths` fallback chain). Different mechanism per
language because each language's own native extension-loading model
calls for it, not because of any inconsistency in approach.

**Added an `r/src/Makevars` prerequisite that runs `make -C ../.. lib`
before compiling the glue code, and a new plain `lib` Makefile target
(static library only, no tests/examples) to invoke.** Mirrors Python's
`setup.py` `build_py` override (Milestone 10): `R CMD INSTALL r` alone
should be sufficient, without the user needing to build the C library by
hand first. `lib` (rather than reusing `all`) exists specifically so this
prerequisite doesn't also rebuild the entire C test/example suite as an
unwanted side effect of installing the R package. Verified by deleting
`build/` entirely and re-running `R CMD INSTALL` from scratch: it
rebuilds everything from nothing, correctly.

**R matrices need NO column-major conversion at all** — unlike Python's
NumPy (which defaults to row-major and needs `np.asfortranarray`), R
already stores matrices column-major internally, which is *exactly*
`hdcd`'s expected core layout (spec §23). `REAL(matrix_sexp)` in the
`.Call` glue is passed straight through as `hdcd`'s `double*`, genuinely
zero-copy, with no equivalent of Python's `_as_column_major` helper
needed at all. This is spec §26's "preserve column-major memory where
possible" being close to free in R specifically, not a coincidence of
this implementation.

**`NA_real_` detection uses `ISNAN()` (catches both R's specific `NA`
*and* plain `NaN`), not `ISNA()` (which only catches `NA`).** `ISNA`
alone would silently treat a data-derived `NaN` (e.g. from `0/0`
upstream in a user's pipeline) as an ordinary observed value rather than
missing — inconsistent with the Python binding's `np.isnan`-based mask
construction, which already catches both. Same missingness semantics
across both bindings, on purpose.

**All `.Call` glue functions take positional scalar arguments (up to 18,
for `hdcd_r_run_annealing`), not a single R list unpacked field-by-field
in C.** A generic "extract named field from an R list" C helper would be
less code at the call site but more code (and more failure surface —
wrong name string, wrong type coercion) in the glue itself; explicit
positional parameters are mechanical and let a mismatched argument count
fail loudly (an R error at the `.Call` boundary) rather than silently
reading a missing list field as `NULL`/default. This is the same
"prefer explicit and boring over clever" tradeoff as the Python binding's
per-struct-field `ctypes.Structure` declarations — just adapted to R's
`.Call` convention instead of a struct-passing one, since R has no
native equivalent of passing a C struct by value across the FFI boundary.

**External-pointer finalizer safety is tested with an explicit `gc()`
mid-test, not merely by trusting R's GC to eventually run.** Spec §31
M11's "external pointer finalization" acceptance criterion is easy to
satisfy technically (register a finalizer, done) but easy to get subtly
wrong (e.g. a `hdcd_dag_t` finalized while a `hdcd_dag_fit_t` still holds
onto data derived from it, or a struct field accidentally not protected
during construction). `test-high-level-api.R` forces a collection cycle
between fitting a model and continuing to use it (`gc()` then
`hdcd_transform`/`predict`), which would surface a premature-finalization
bug as a crash or wrong values, not just as "the finalizer function
exists."

**`fit_dag`/`score_dag` are two separate R functions (matching spec
§26's own conceptual snippet), not one Python-style object carrying a
`kl_divergence_` field.** Both spec §25 (Python) and §26 (R) sketch the
SAME underlying comparison, but in each language's own idiom: Python's
snippet uses `result.kl_divergence_` (an attribute on the returned
object), R's snippet uses a second explicit `score_dag(model,
candidate_fit)` call. Followed each section's own literal pattern for
its own language rather than force one language's idiom onto the other.

**Reused `python/tests/fixture.json` verbatim for R's cross-check test,
rather than generating a separate R-only fixture.** Directly satisfies
spec §29.11's "the same saved test fixture" — one C-computed source of
truth checked by two independent language bindings, rather than two
fixtures that could quietly drift apart. The R test resolves the path
via `testthat::test_path()` (robust to the working directory at test-run
time) and skips gracefully (not a hard failure) if the fixture or
`jsonlite` isn't available, so an isolated `R CMD check` environment
without the sibling `python/` checkout degrades gracefully rather than
failing for an unrelated reason.

**Verified end-to-end exactly like the Python binding was:** deleted
`build/` entirely, ran `R CMD INSTALL r` from a clean checkout (which
rebuilt the C library from scratch via the new `Makevars` prerequisite),
then ran the full `testthat` suite — 36 assertions, 0 failures, 0
warnings, 0 skips (confirming the fixture WAS found and used, not
silently skipped) — plus a standalone smoke-test script exercising the
full `hdcd_fit` -> `transform`/`predict`/`copula_logpdf`/`fit_dag`
pipeline end to end, matching the verification rigor applied to
Milestone 10.

---

## Milestone 12 — Julia binding

**`ccall` with structs passed and returned by value directly** (no
separate glue-C layer like R's, no `ctypes.Structure` marshaling layer
like Python's). Julia's FFI can pass/return `isbits` structs across
`ccall` directly, matching C ABI layout as long as the Julia `struct`'s
field order and types mirror the C header exactly — the same discipline
as Python's `_capi.py`, just with less boilerplate, since Julia doesn't
need a separate binding-declaration step (`argtypes`/`restype`) the way
`ctypes` does. This is the most direct of the three bindings, structurally,
because `ccall` is a first-class language construct rather than a library
(`ctypes`) or a translation layer (R's `.Call`).

**No dependency on a general JSON package for the fixture-agreement
test.** `Pkg.add` needs registry/network access this sandboxed
environment may not reliably have at test time, and the ONLY thing
needed is reading one fully-controlled, self-generated fixture file —
so `test/runtests.jl` includes a ~70-line self-contained recursive-
descent JSON reader (`TinyJSON`) scoped to exactly what `fixture.json`
contains (numbers, booleans, nested arrays, unescaped string keys),
rather than pulling in JSON.jl/JSON3.jl as a real dependency for a
narrow, controlled use. Consistent with the project's general "add a
dependency only when it earns its complexity" stance (e.g. skipping
`dcor_fast.c`, `sinkhorn/quadrature.c` as empty wrapper files in earlier
milestones) — a full JSON parser would be the same kind of unjustified
weight here.

**A genuine GC-safety bug caught before it shipped, not after:** the
first draft of `run_annealing` captured `pointer(ordering0)` and
embedded it inside the `HdcdAnnealingOptions` struct, then called
`GC.@preserve ordering0` *after* the `ccall` had already returned —
which protects nothing, since the risk window is *during* the call.
Julia's `ccall` automatically roots/preserves arrays passed *directly*
as arguments (like `U`/`mask` here), but that automatic protection does
NOT extend to a raw pointer manually captured via `pointer(...)` and
stashed inside a struct value passed by value — `ordering0` needed
explicit `GC.@preserve` wrapping the actual `ccall`, not a statement
after it. Fixed before ever running the code (caught while re-reading
the draft, not via a crash), and left the reasoning as a comment at the
call site since this exact class of mistake (preserving too late, or
assuming struct-embedded pointers get the same auto-protection as
direct arguments) is the single most Julia-specific FFI hazard in this
whole binding and worth flagging for future edits.

**Julia arrays need NO column-major conversion, exactly like R
(Milestone 11) and unlike Python (Milestone 10).** Julia's native
`Matrix` layout is already column-major, matching `hdcd`'s core layout
(spec section 23) with no `asfortranarray`-style step anywhere in this
module — the same property noted for R, now confirmed true for all
three "modern statistical computing language" bindings sharing this
trait, while only the row-major-by-default `numpy` binding needed an
explicit conversion.

**`fit_dag`/`score_dag` are two separate functions returning `Delta_KL`
as a plain return value (spec section 27's own conceptual snippet:
`result = fit_dag(model, candidate)` then `score_dag(model, result)`),
matching R's pattern (Milestone 11) rather than Python's
`result.kl_divergence_` attribute-style (Milestone 10).** All three
language sections of the spec (25, 26, 27) sketch the SAME underlying
comparison in each language's own conceptual idiom; followed each one
literally for its own binding rather than picking one pattern and
forcing it onto all three.

**Verified end-to-end matching the rigor applied to Milestones 10-11:**
deleted `build/` and `julia/Manifest.toml` entirely, rebuilt the C
shared library from scratch (`make shared`), ran `Pkg.instantiate()` +
`Pkg.test()` from nothing — 32 assertions across 11 testsets, 0
failures — plus a standalone smoke-test script exercising the full
`hdcd_fit` -> `transform_copula`/`logpdf`/`copula_logpdf`/`fit_dag`/
`score_dag` pipeline end to end. `julia/Manifest.toml` itself is
gitignored (not committed): with `Libdl` (a stdlib, always bundled with
the Julia installation) as the only dependency, it pins little beyond
"which Julia version generated it," unlike a project with real external
dependencies where a lockfile earns its keep.

---

## Post-M12 validation — end-to-end vine copula recovery notebook

Requested explicitly before starting Milestones 13-14 (EVT module,
performance work): an `.Rmd` notebook (`notebooks/vine_copula_recovery.Rmd`)
that generates a complex, high-dimensional, known-ground-truth vine copula,
recovers it with the R binding's full `hdcd` pipeline, times every stage,
and produces conditional-density plots comparing the fitted result against
the true, closed-form densities.

**Ground truth is a tree-1-truncated D-vine (Markov chain), not a full
multi-tree R-vine, after a fuller vine proved numerically fragile to
generate.** A pilot chaining `copula::cCopula(..., inverse = TRUE)` calls
through a second vine tree hit `uniroot()` failures (`f.lower = f(lower) is
NA`) once an intermediate conditional value landed close to the `[0,1]`
boundary after two chained h-function inversions — a fragility in
generating the *ground truth itself*, unrelated to `hdcd`. Rather than ship
a validation notebook whose reference structure might be silently wrong,
committed to the simplest genuinely-valid vine member instead: a first-order
chain (variable *i+1* generated from a pair-copula conditioned on variable
*i* alone), mixing five different pair-copula families (Clayton, Gumbel,
Frank, Gaussian, Student-*t*) with different strengths across nine edges at
d=10. Still exercises full high-dimensional sparse-DAG recovery and diverse
conditional-density shapes; verified correct before use via KS tests
(uniform margins) and empirical-vs-true Kendall's tau per edge. A second,
related numerical issue — chained h-inverse values landing exactly on `0`
or `1` and then producing `Inf`/`NaN` one step later — was fixed by clipping
every intermediate copula-scale value (not just the final output) to
`[1e-6, 1-1e-6]` at each step of the chain.

**Extended the R binding with two new exported functions,
`hdcd_node_parents()` and `hdcd_conditional_density()`, plus score-trace
fields on the fitted model (`score_trace`, `accepted_trace`,
`acceptance_rate`), purely to support this notebook's diagnostics.** The
existing R API had no way to evaluate one node's fitted conditional copula
density on a grid of `u` values (needed to plot fitted-vs-true density
curves) or to inspect the annealing search's convergence trace (needed to
confirm the iteration budget was sufficient) — both are directly useful for
any future model diagnostics, not one-off notebook plumbing, so they were
added as first-class exported functions (`r/R/hdcd.R`, `r/src/hdcd_r.c`,
`r/NAMESPACE`) rather than notebook-local hacks. Full `testthat` suite
re-verified after the change: unchanged at 36/36 passing.

**Calibrated hyperparameters for a ~30-second, honest demonstration at
d=10, n=2000:** `max_parents=2, bernstein_degree=4, lambda_edge=0.05,
lambda_roughness=0.15, holdout_fraction=0.25, annealing_iterations=600,
annealing_restarts=3`. Measured result: total pipeline time ≈ 26 seconds
(annealing search dominates); 9/9 true skeleton edges recovered, 8 in the
exact generative direction and 1 direction-flipped; `Delta_KL(true DAG vs.
recovered DAG) ≈ -0.035` (the true DAG scores only marginally better);
mean fitted-vs-true conditional-density correlation ≈ 0.96 across all nine
edges and three conditioning values each.

**The one direction-flipped edge was kept and explained in the notebook,
not tuned away.** For a Markov chain, the pairwise statistical dependence
between adjacent variables is symmetric regardless of which is labeled
"parent" in the reference DAG — `hdcd`'s own documentation and Milestone 9
are explicit that the reference DAG is a density-factorization choice, not
a causal claim (spec §19, §34). A near-tied `Delta_KL` on the flipped edge
is direct, concrete evidence of that disclaimer rather than a defect to
hide by re-seeding until it disappears.

---

## Post-M12 feature — non-global, learned `lambda_roughness` (per-node roughness selection)

Requested explicitly after the vine-recovery notebook's "Known limitation"
section diagnosed the fixed, global `lambda_roughness` as over-smoothing
Clayton/Gumbel edges. The user asked to pursue a non-global, LEARNED
roughness penalty as an option, per spec §18 ("`lambda_R` is selected by
validation unless explicitly provided" / "a small validation grid is
sufficient for version 1").

**Implemented in the C core** (`include/hdcd/local_fit.h`,
`src/optimize/local_fit.c`), not just in a binding: two new optional
fields on `hdcd_local_fit_options_t`, `lambda_roughness_grid` (candidate
values) and `roughness_validation_fraction`. Empty/NULL grid (the
default) is a complete no-op — behavior is bit-for-bit identical to
before, verified via a dedicated regression test
(`test_roughness_grid_matches_fixed_when_singleton`) and the full
pre-existing C/R test suites passing unchanged. `hdcd_dag_fit` needed NO
changes: it already forwards one `hdcd_local_fit_options_t` per node
unchanged, so per-node selection falls out automatically once
`hdcd_local_fit_node` supports it.

**Selection happens via an INNER split of the outer TRAIN rows, never
touching the outer holdout.** `hdcd_local_fit_node`'s existing TRAIN rows
are further split into `inner_train`/`inner_val` (a fresh seeded
shuffle, `options->seed XOR`'d with a fixed constant so it's
reproducible but not identical to the outer split); every grid candidate
is fit on `inner_train` and scored on `inner_val`; the winning lambda is
then refit on the FULL outer train and scored on the untouched outer
holdout exactly as the fixed-lambda path always has. This matters
because the outer holdout feeds `hdcd_dag_fit_kl_estimate` /
`hdcd_dag_fit_kl_difference` and (when this options struct reaches
`hdcd_run_annealing`) the annealing objective itself -- letting penalty
selection see that data first would bias exactly the scores used to
compare models. Refactored the previously-inline Theta/Sinkhorn/holdout-
score logic into a shared static `fit_and_score()` helper parametrized by
which rows are "train" vs "score", used for both a grid candidate's inner
fit and the final production fit -- this is why the singleton-grid
regression test above holds exactly, not just approximately: the
production fit is byte-for-byte the same code path either way.

**Deliberately kept OUT of `hdcd_run_annealing`'s inner proposal loop.**
Spec §18 is explicit: "Penalty selection must occur outside the inner
annealing loop. Do not nest an expensive continuous penalty search inside
every graph proposal." `hdcd_run_annealing` calls `compute_node_score` ->
the local-fit cache -> `hdcd_local_fit_node` for every distinct parent set
a proposal tries; enabling the grid there would repeat an O(grid size)
search on every one of those. The C core does not forbid it (the options
struct is generic), but the R binding's `hdcd_fit()` deliberately forwards
`lambda_roughness_grid` ONLY to the final reference-DAG fit
(`.dag_fit_c`), never to `.run_annealing_c` -- the annealing search itself
keeps using the single fixed `lambda_roughness` throughout, unchanged.

**R binding**: `hdcd_r_dag_fit`'s `.Call` arity grew from 10 to 12
(two new trailing args, both defaulting to "disabled" -- `numeric(0)`
grid, `0` fraction -- so existing 10-arg-shaped call sites needed no
changes beyond the two new default parameters). `hdcd_fit()` and
`hdcd_fit_dag()` both gained `lambda_roughness_grid`/
`roughness_validation_fraction` parameters; `hdcd_fit_dag()` defaults to
reusing whatever grid `model` itself was fit with, so a candidate DAG is
compared on equal footing rather than silently reverting to a fixed
lambda. Added `hdcd_node_lambda_roughness(model, node)` (new C accessor
`hdcd_local_fit_selected_lambda_roughness`, NAN for a root node) so a
user can inspect what was actually selected per node.

**Important correction to the earlier "Known limitation" diagnosis.**
Running the new, properly-validated selection on the vine-recovery
notebook's exact data was expected to confirm that a lighter penalty
helps the Clayton/Gumbel edges. It did NOT: held-out log-likelihood --
the honest, principled criterion available in any real (non-synthetic)
application, unlike "correlation to the hidden true density" which is
only computable here because ground truth is known -- showed a clear
INTERIOR OPTIMUM near lambda approx 0.15-0.3 for every edge tested,
Clayton and Gumbel included (e.g. edge 1->2 Clayton: holdout score rises
from -0.12 at lambda=0.05 to a peak of 0.35 at lambda=0.15, then falls
off again at 0.3/0.6/1.0/2.0; edge 7->8 Gumbel peaks similarly at
lambda=0.15). The earlier notebook section's premise -- eyeballing
fitted-vs-true density correlation and concluding "a lighter penalty
fits better" -- was measuring a different thing than the estimator's own
predictive accuracy, and does not survive contact with the metric that
actually matters. The corner-flattening on sharp Archimedean
tail-dependence edges is better understood as a basis-expressivity limit
(a degree-4 centered Bernstein tensor is a smooth, bounded polynomial
family; it cannot represent a singular corner spike regardless of how
lightly it's regularized) rather than as a fixable over-smoothing bug.
See `notebooks/vine_copula_recovery.Rmd`'s revised "Known limitation"
section, which now demonstrates the auto-selection feature directly and
reports this corrected conclusion rather than the original, less rigorous
one.

**Adding fields to `hdcd_local_fit_options_t` required updating the
Python and Julia bindings too, even though neither exposes the new
feature.** Both mirror this struct's exact memory layout for FFI --
`python/hdcd/_capi.py`'s `HdcdLocalFitOptions(ctypes.Structure)` and
`julia/src/HDCD.jl`'s `struct HdcdLocalFitOptions` -- and both were
missing the three new trailing fields until updated here, which would
have silently misaligned every field of `HdcdAnnealingOptions` (which
embeds this struct) passed across either FFI boundary: not a compile
error, a silent ABI mismatch. Caught by rule, not by accident (the
Python file's own header comment: "every struct layout ... must match
the C headers EXACTLY ... or the ABI breaks silently") and confirmed by
rerunning both full test suites after the fix (Python: 9/9 passing;
Julia: 32/32 assertions passing) -- this is the check to repeat for any
future change to a struct either binding mirrors, not just this one.
Both bindings default the three new fields to "disabled" (NULL
pointer / zero size), so neither exposes per-node roughness selection
yet; only the R binding does. Extending Python/Julia to expose it is a
straightforward follow-up (the C core and R binding are the reference
implementation) but wasn't asked for here.

---

## Post-M12 direction — tail-dependence-informed `bernstein_degree` selection

The "Known limitation" investigation above landed on: the Clayton/Gumbel
corner under-fit is not a `lambda_roughness` problem (a properly validated
per-node penalty confirms 0.15 was already near-optimal); it looks like a
basis-expressivity limit of the degree-4 centered Bernstein tensor.
Discussed three ways to raise `bernstein_degree` to test that, and picked
one, logging all three here so the other two remain available to revisit
without re-deriving them:

1. **CHOSEN: tail-coefficient-gated joint (degree, lambda_roughness)
   search.** Compute an empirical tail-dependence coefficient per
   (child, parent) pair (a standard nonparametric EVT/copula estimator:
   the fraction of the top-`k` order-statistic exceedances in one
   variable that are also top-`k` exceedances in the other, à la Frahm/
   Junker/Schmidt). Use each node's strongest parent-pair coefficient to
   GATE whether that node's `bernstein_degree` is searched at all
   (edges with no real tail dependence -- Frank, Gaussian -- skip the
   search entirely, same architecture cost as today), and, when gated
   in, jointly auto-select `(bernstein_degree, lambda_roughness)` via the
   SAME held-out inner-validation split already built and proven for
   `lambda_roughness` alone. Chosen because it reuses tested
   infrastructure (`fit_and_score` already takes `m` as a parameter, so
   no further core refactor is needed to vary degree per candidate), it
   is genuinely informed by an extreme-value concept rather than a blind
   sweep, and it fixes a real unfairness in the very first (pre-
   correction) degree check earlier in this log, which raised degree
   while holding `lambda_roughness` fixed at 0.15 -- an apples-to-oranges
   comparison, since more coefficients plausibly want a different
   regularization strength, not the same one.
2. **NOT CHOSEN (bigger lift, revisit if (1) doesn't close the gap): a
   true EVT tail-splice at the copula level** -- a parametric
   extreme-value copula model grafted onto the Bernstein bulk near each
   corner, mirroring the marginal GPD-splice architecture in spec
   section 3 (Milestone 13) but applied to the bivariate copula density
   instead of a univariate marginal CDF. More principled for a genuine
   corner *singularity* a bounded polynomial can never fully reach
   regardless of degree, but a new module: threshold selection, CDF/
   density continuity at the splice boundary, its own test suite. Spec
   section 3's EVT module is explicitly about MARGINAL tails (splicing
   `F_{j,EVT}` onto each variable's own smoothed CDF) -- worth being
   precise that this would be a NEW, copula-level EVT mechanism, not the
   same thing as finishing Milestone 13, even though both are "extreme
   value" ideas.
3. **NOT CHOSEN (cheaper, revisit if tail-coefficient gating turns out to
   add complexity without adding value): plain degree/lambda grid search
   with no tail-dependence diagnostic at all** -- extend the existing
   per-node auto-selection to `bernstein_degree` exactly like
   `lambda_roughness`, letting held-out validation alone decide, on every
   node unconditionally. Simplest to implement, but does not use
   "extreme-value logic" as asked, and burns the held-out-validation
   search budget on edges (Frank, Gaussian) that gain nothing from it.

---

## Post-M12 finding — does more data resolve the degree/likelihood tradeoff? (open question)

Requested explicitly, framed by the user as possibly the core theme of
this whole exercise: "This paper might be about the genuine difficulties
of high dimensional and heavy tailed modeling." Direct follow-up to the
finding above (raising `bernstein_degree` improves shape-agreement with
the true density but costs held-out log-likelihood at $n=2000$): does
that gap close as $n$ grows?

**Experiment**: reran the identical tail-dependence-gated joint
(degree in $\{4,6,8,10\}$, lambda in $\{0.05,0.1,0.15,0.3\}$, gate
$0.05$) search against the same true DAG at $n \in \{2000, 4000, 8000,
16000\}$ (one fresh independent draw per $n$, NOT a replicated design —
see caveat below), tracking `Delta_KL` (all 9 edges) and per-edge
shape-correlation on the two worst offenders (1->2 Clayton, 7->8
Gumbel). Standalone script `notebooks/n_sweep_experiment.R`, results
saved to `notebooks/n_sweep_results.csv` and loaded (not
recomputed) by `notebooks/vine_copula_recovery.Rmd`'s "Does more data
resolve the bias-variance tradeoff?" section -- the sweep itself took
roughly 30 minutes (dominated by the $n=16000$ joint search alone taking
~880s, worse-than-linear scaling with $n$), too expensive to re-run on
every notebook render.

**Result: inconclusive, and informatively so.** `Delta_KL` fell
0.252 -> 0.203 -> 0.099 across $n=2000,4000,8000$ (each doubling
roughly halving the gap -- exactly the "more data helps" signature) and
then reversed sharply to 0.319 at $n=16000$, worse than every smaller
$n$ including 2000. Meanwhile shape-correlation to the true density
stayed in a narrow, stable band across the entire sweep (auto beating
fixed by roughly 0.02-0.04 at every single $n$, never trending toward or
away from parity), and the search selected the grid's most flexible
option (degree 10, lambda 0.05 -- the boundary of the search space)
at every $n$ tested, never backing off as more data arrived.

**Why this is being logged as an open question, not resolved either
way.** This sweep is ONE REALIZATION PER $n$: each $n$ draws an
independent fresh copula sample rather than growing one fixed dataset,
so sample-to-sample variation in exactly which points land in which
train/inner-validation/holdout split is fully confounded with the
effect of $n$ itself. A single point per $n$ cannot distinguish "the
tradeoff doesn't resolve monotonically (or needs far more data than
tested)" from "the $n=16000$ draw was unlucky." Resolving that requires
several replicate seeds per $n$ with a confidence band -- and at ~15
minutes of compute for a single $n=16000$ replicate, even 5 replicates
there costs over an hour, for one synthetic experiment's one pair of
edges.

**That gap is itself the finding worth keeping.** The question "does
more data fix this" is well-posed; affordably *answering* it rigorously
runs into the same high-dimensional, heavy-tailed cost structure the
original modeling difficulty came from (super-linear-in-$n$ fitting
cost, high-variance held-out estimates that need replication to trust).
If this becomes a paper: the honest empirical claim from this sweep is
"inconclusive, and expensive to make conclusive," which is a legitimate
and arguably more interesting result than a clean crossover would have
been. Follow-up, not done here: a properly replicated sweep (multiple
seeds per $n$, mean +/- CI) restricted to fewer $n$ points and/or a
smaller search grid to keep total compute bounded.

---

## Post-M12 feature — anisotropic (corner-relaxed) roughness penalty

Discussed several ways to raise `bernstein_degree`'s effective flexibility
near a tail-dependence corner without the generalization cost the plain
degree-grid search paid (see the two entries above); picked the cheapest,
most surgical option to try first: **relax the roughness penalty itself
near the corners**, rather than adding more coefficients everywhere.

**Design.** `hdcd_bernstein_roughness_penalty_weighted(theta, m,
corner_relief, out)` (and its gradient counterpart) weight each interior
second-difference residual, centered at grid position $(i,j)$ in the
$(m{+}1)\times(m{+}1)$ Theta grid, by

$$w(i,j) = 1 - \text{corner\_relief} \cdot \text{edge\_proximity}(i) \cdot \text{edge\_proximity}(j),$$
$$\text{edge\_proximity}(k) = 1 - \frac{\min(k,\, \dim{-}1{-}k)}{(\dim{-}1)/2} \in [0,1].$$

`edge_proximity(k)` is 1 exactly at either edge ($k=0$ or $k=\dim{-}1$)
and falls linearly to 0 at the grid's true center, so $w$ is close to 1
(full, unchanged penalty) everywhere except near the tensor's four
CORNERS — where $u$ and $z$ are both extreme simultaneously, exactly
where tail dependence concentrates — dipping toward $(1 -
\text{corner\_relief})$ right at a corner. A **linear** taper (not
Gaussian) was chosen for simplicity: one parameter (`corner_relief`,
required in $[0,1)$), no extra bandwidth to also tune, and it has two
analytically checkable properties exploited directly in
`tests/test_bernstein.c`: at $m=2$ every residual's center is forced onto
the grid's own true center, so `corner_relief` provably has NO effect at
all (a clean, theta-independent invariant); and `corner_relief=0`
reproduces the original unweighted penalty bit-for-bit. A "reimplement
the documented formula independently inside the test and compare" check
was used for the general case rather than reasoning about relative
magnitudes between different theta placements — a placement near a
boundary column/row picks up genuinely fewer overlapping difference
windows than an interior one (a real, unrelated structural asymmetry of
the plain second-difference penalty), which would otherwise confound any
attempt to compare "corner roughness" against "center roughness" by raw
magnitude.

**Threaded through as a FIXED scalar for v1, not (yet) itself
grid-searched.** Unlike `lambda_roughness_grid`/`bernstein_degree_grid`,
`corner_relief` is applied identically inside every single Theta fit
(it's a different weighting of the SAME objective, not an extra fit
call), so there is no per-candidate cost to searching it -- extending it
to a validated grid (jointly with degree/lambda, or alone) is a small,
logged follow-up, not attempted yet to keep this addition's scope
contained.

**Applied to BOTH the annealing search and the final reference-DAG fit**
(`hdcd_r_run_annealing` gained the same `corner_relief` field), unlike
the two grids, which are deliberately excluded from annealing purely for
cost reasons (spec section 18). Since `corner_relief` adds no extra fit
calls, excluding it from annealing would only buy an inconsistency for
no benefit: the reference DAG would be searched for under one roughness
measure and then re-fit under a different one. One consequence, called
out directly in the new R test (`test-high-level-api.R`): because
`corner_relief` DOES participate in the annealing objective (through
`hdcd_local_fit_roughness_penalty`, which `compute_node_score` uses), a
nonzero `corner_relief` is NOT guaranteed to leave the reference
structure unchanged the way the grids are guaranteed to -- the test
isolates the fit-quality effect via `hdcd_fit_dag()` on a fixed structure
rather than comparing two independently-annealed reference DAGs.

Same Python/Julia struct-mirror update discipline as the previous two
features (one more trailing field on `hdcd_local_fit_options_t`); both
bindings' full test suites reverified passing (Python 9/9, Julia 32/32).

## Trials queued for tonight (anisotropic roughness penalty)

Logged so the actual runs (launched right after this entry) are
traceable back to why they were run, in case they're picked up cold.
Both use the SAME ground-truth vine construction as
`notebooks/vine_copula_recovery.Rmd` (9-edge, d=10, mixed families).

**Trial A -- quick single-n sanity check (run inline, not backgrounded;
seconds, not minutes).** `corner_relief` in $\{0, 0.3, 0.6, 0.8, 0.9\}$
at $n=2000$, `bernstein_degree=4` and `lambda_roughness=0.15` both FIXED
(no grid search at all -- isolates corner_relief's own effect cleanly),
on the two most severely under-fit edges (1->2 Clayton, 7->8 Gumbel).
Tracks shape-correlation to the true density and `Delta_KL` against the
`corner_relief=0` baseline. Purpose: does relaxing the penalty near the
corner improve shape-fit the way raising degree did, WITHOUT raising
degree's held-out-likelihood cost (since no extra coefficients are being
added, just reweighted)? A go/no-go check before committing to Trial B's
overnight compute.

**Trial B -- properly replicated n x corner_relief sweep (backgrounded,
~50 minutes estimated).** This is also the direct answer to the earlier
"run a properly replicated sweep with multiple seeds per n" ask, which
was deferred as too expensive for the degree-grid intervention (~15
min/replicate at $n=16000$ alone). `corner_relief` is far cheaper to
sweep: it costs exactly one ordinary `fit_dag` call per (n,
corner_relief) point, not a multi-candidate grid search, so real
replication is affordable here in a way it wasn't for
`bernstein_degree_grid`.

- $n \in \{2000, 4000, 8000, 16000\}$ (same grid as the earlier degree
  sweep, for direct comparability).
- `corner_relief` $\in \{0, 0.3, 0.6, 0.8\}$, `bernstein_degree=4` and
  `lambda_roughness=0.15` both fixed throughout (isolating corner_relief
  exactly as Trial A does, just replicated and swept over n).
- 6 replicate seeds per $n$ (a fresh independent data draw per
  replicate, not a superset), data generation shared across the 4
  `corner_relief` values within a replicate (only the fit differs, so no
  need to regenerate data 4 times over).
- Tracks, per (n, corner_relief, replicate): shape-correlation and
  `Delta_KL` (vs. `corner_relief=0` at that same n and replicate) on the
  two worst edges, plus the fitted Theta's own diagnostics
  (`hdcd_local_fit_roughness_penalty`).
- Estimated cost: ~500s/replicate x 6 replicates ~= 50 minutes total,
  saved to `notebooks/corner_relief_sweep_results.csv` via
  `notebooks/corner_relief_sweep_experiment.R` (mirroring
  `n_sweep_experiment.R`'s pattern: standalone, reproducible, not
  re-executed on every notebook render).
- Success criterion, decided in advance rather than after seeing the
  data: `corner_relief` is worth keeping if, averaged over replicates,
  it improves shape-correlation on the worst edges WITHOUT a
  statistically clear `Delta_KL` cost (mean `Delta_KL` for the best
  `corner_relief` value should not sit outside roughly 1-2 replicate
  standard deviations above 0) -- the same standard the degree-grid
  approach failed to meet.

**Trial A result: GO.** At $n=2000$, `corner_relief=0.6` improved
shape-correlation on BOTH worst edges (1->2 Clayton: 0.781 -> 0.785;
7->8 Gumbel: 0.836 -> 0.840) AND improved held-out likelihood
(`Delta_KL = -0.0496` vs. the `corner_relief=0` baseline -- negative
means the `corner_relief=0.6` fit scored BETTER, not worse). This is the
first intervention in this whole investigation that helped both metrics
at once; plain degree escalation always paid for shape-correlation with
worse held-out likelihood. Not monotonic, though: `corner_relief=0.3`
was worse than baseline on both metrics, `0.8` and `0.9` kept improving
`Delta_KL` but lost back some of `0.6`'s shape-correlation gain on edge
1->2 specifically -- consistent with a real but non-trivial (not simply
"more relief is better") relationship, and itself a reason a single-run
sanity check isn't sufficient: a second quick spot-check at a different
seed (smoke-testing Trial B's own script before the full launch, 1
replicate at $n=2000$) already showed `corner_relief=0.3` AND `0.6` both
scoring worse than baseline in `Delta_KL` -- the opposite ranking Trial A
saw for those two values. Trial B (replicated) is what actually resolves
this, not Trial A alone; launched immediately after this entry.

**Trial B result: PASSES the pre-registered criterion, at
`corner_relief=0.6` specifically.** All 6 replicates x 4 n x 4
corner_relief completed (~48 minutes). Aggregated over replicates
(mean $\pm$ SD, $n_{\mathrm{obs}}=12$ per cell -- 6 replicates x 2 edges):

| $n$ | shape-cor, baseline | shape-cor, `relief=0.6` | `Delta_KL` mean $\pm$ SD | z-score |
|---|---|---|---|---|
| 2000  | 0.804 | **0.812** | $0.032 \pm 0.083$ | 0.39 |
| 4000  | 0.789 | **0.806** | $0.026 \pm 0.182$ | 0.14 |
| 8000  | 0.801 | **0.804** | $0.063 \pm 0.073$ | 0.86 |
| 16000 | 0.794 | **0.802** | $0.098 \pm 0.100$ | 0.98 |

`corner_relief=0.6` improves shape-correlation to the true density at
*every* $n$ tested (unlike `0.3`/`0.8`, which are mixed -- `0.8` is
even slightly worse than baseline at $n=2000$), and its `Delta_KL` cost,
while consistently positive on average, never exceeds 1 replicate SD at
any $n$ -- comfortably inside the pre-registered "should not sit outside
roughly 1-2 SD above 0" bar. This is a materially different outcome from
the `bernstein_degree_grid` sweep: there, the single-realization
`Delta_KL` swung non-monotonically across a much larger range (0.099 to
0.319, with no variance estimate at all to judge whether that range was
real or noise); here, with actual replication, the cost is both smaller
in absolute terms (0.03-0.10) and demonstrably not statistically
distinguishable from zero.

One honest caveat, not swept under the rug: the z-score rises mildly
with $n$ (0.39 -> 0.14 -> 0.86 -> 0.98), approaching but not crossing 1
SD at $n=16000$ -- consistent with either continued noise or a genuine,
slowly-growing cost that a wider $n$ range might eventually surface.
Six replicates resolves "is this a coin-flip-sized effect being
over-read from one run" (yes, that's what happened with the degree-grid
sweep); it does not resolve every possible question about the far tail
of $n$, which would need either more replicates or a purpose-built
sequential/adaptive design.

**Correction after first writing this up: the notebook's own plot
initially used $\pm$ SE (`sd/sqrt(12)`) error bars, not $\pm$ SD.** That
made the four `corner_relief` lines look visually separated in a way the
text's own z-score criterion (computed on SD, not SE) did not support --
an inconsistency between the plot and the prose judging it, caught by
re-inspecting the rendered plot rather than only the numbers. Fixed to
plot $\pm 1$ SD directly, matching the z-score criterion exactly. At the
honest scale, every `corner_relief` value's error bar overlaps every
other value's, and every `Delta_KL` error bar overlaps zero, at every
single $n$ -- no individual (n, corner_relief) point is distinguishable
from any other on its own.

**Conclusion, recalibrated accordingly: this is directional, weak
evidence, not a demonstrated effect.** The actual basis for
`corner_relief=0.6` is a *pattern in the means*, not any single
significant comparison: it has the highest (or tied-highest) mean
shape-correlation at all four independent $n$ draws, and its mean
`Delta_KL` cost never exceeds 1 SD of its own spread (z-scores
0.14-0.98) -- four-out-of-four consistent directional wins across
independent draws is not nothing, even with zero individual points
clearing significance, but it is meaningfully weaker than "the effect is
visible in the plot." It is still the first intervention in this
investigation (roughness grid, degree grid, now corner relief) to clear
its own pre-registered bar rather than being logged as inconclusive or
as a real-but-costly tradeoff -- "cleared a lenient bar via a
consistent-direction argument," not "proven to work." Still a FIXED
scalar in the current implementation (see the feature entry above) --
turning `corner_relief=0.6` into hdcd's actual new default, extending it
to a searched grid, or running a properly-powered follow-up (materially
more than 6 replicates) to actually resolve individual-point
significance are all follow-up decisions for the user, not made
unilaterally here. Full results:
`notebooks/corner_relief_sweep_results.csv` (raw, all 192 rows),
written up with plots in `notebooks/vine_copula_recovery.Rmd`'s "Does
corner relief resolve the tradeoff, properly replicated?" section.

---

## Post-M12 feature — copula-level EVT tail-splice

The user explicitly noted that none of the three interventions tried so
far (`lambda_roughness` grid, `bernstein_degree` grid, `corner_relief`)
introduce an actual parametric copula family -- all three stay purely
nonparametric, reweighting or resizing the same Bernstein tensor. Asked
to pursue the heaviest, most principled option logged as an alternative
from the very first `bernstein_degree` discussion: graft a genuine
parametric extreme-value copula onto the Bernstein bulk near each
corner.

**Family choice: Clayton (lower-tail) and Gumbel (upper-tail), the same
two families the vine-recovery notebook's ground truth already uses.**
Both are single-parameter Archimedean families with tail dependence and
overall association strength governed by the same parameter, so an
ordinary full-sample MLE already targets the tail behavior the family
exists to capture -- no need for a separate "tail-only" data subset or a
peaks-over-threshold-style estimator. New module
`src/copula/parametric_tail.c` / `include/hdcd/parametric_tail.h`
implements both densities from their standard closed forms (Gumbel's
derived and verified by reduction to the independence copula at
theta=1: `c=1` exactly, confirmed in `tests/test_parametric_tail.c`) and
MLE fitting via `hdcd_golden_section_maximize` (reusing the Milestone 1
deterministic 1D optimizer, not a new one). Tested end-to-end: exact
closed-form h-function-inverse samplers for both families (Clayton's is
closed-form; Gumbel's needs bisection on its conditional CDF) generate
known-theta synthetic data, and MLE recovers the true theta on it --
the strongest form of correctness check available short of an
independent reference implementation.

**Key architectural simplification: no hand-enforced continuity is
needed, because Sinkhorn normalization already provides it.** Spec
section 3's marginal EVT splice needs explicit CDF-continuity
enforcement at each 1D threshold because nothing else guarantees the
spliced result is a valid distribution. The copula case has a 2D
"boundary" (the edge of whatever corner region is being spliced), which
would make hand-matching continuity there real, unpleasant additional
work -- except `hdcd_sinkhorn_fit` already exists to turn ANY positive
raw kernel into a valid, copula-preserving conditional density (spec
section 11), regardless of where that kernel came from or whether it is
smooth. So instead of splicing two functions with a matched boundary,
each edge's raw kernel becomes a continuous per-(u,z) BLEND:

```
log K_blend(u,z) = (1 - w(u,z)) * g_bernstein(u,z) + w(u,z) * log c_parametric(u,z)
w(u,z) = exp(-(du^2 + dz^2) / (2 * bandwidth^2))
```

where `(du,dz)` is the distance from `(u,z)` to whichever corner the
fitted family targets ((0,0) for Clayton, (1,1) for Gumbel) -- the same
"distance-from-corner, product-of-two-1D-terms" shape as
`corner_relief`'s `edge_proximity`, just continuous in `(u,z)` instead of
discrete on the Theta coefficient grid, since this blends kernel VALUES
at arbitrary evaluation points rather than reweighting a fixed penalty
grid. `hdcd_sinkhorn_fit` is hooked up to this blended kernel completely
unmodified -- it has no idea the raw kernel it's normalizing is itself a
blend of two different model families. This is the single design choice
that turned "real new-module work" (the original scoping estimate for
this option) into a change that reuses essentially all existing
machinery.

**Theta-fitting is unaffected; only kernel EVALUATION (Sinkhorn fitting
and final density queries) sees the blend.** The Bernstein Theta
gradient-ascent objective (`<Theta,M> - lambda_R*R(Theta)`) still
optimizes exactly what it always did, ignoring the parametric piece
entirely -- `fit_theta_edge` needed no changes. The parametric family's
theta is fit ONCE per node (on that node's outer TRAIN rows), before any
`lambda_roughness_grid`/`bernstein_degree_grid` search, and reused
identically across every candidate that search tries -- refitting a
single-parameter MLE that doesn't depend on Theta's own degree/lambda,
once per grid candidate, would be pure waste.

**Gated by the same `hdcd_tail_dependence_coefficient` diagnostic as
`bernstein_degree_grid`** (computed independently for v1 even when both
options are active simultaneously -- a known, minor redundant-
computation cost, not a correctness issue; `tail_dependence_k` is now
read unconditionally in the R glue rather than only inside the degree-
grid branch, so either feature can use a custom `k` on its own). Family
choice per edge: Clayton if that edge's lower coefficient is the larger
of the two, Gumbel otherwise. A hard failure from the MLE fit
(allocation/numerical/invalid-argument) leaves that specific edge
unspliced (`HDCD_TAIL_FAMILY_NONE`) rather than aborting the whole node
fit -- the plain Bernstein kernel is always a safe fallback, matching
this codebase's established "fail one part clearly, not the whole
operation" convention.

**Deliberately excluded from `hdcd_run_annealing`, unlike
`corner_relief`.** This is the one place this feature's design diverges
from `corner_relief`'s precedent. `corner_relief` is a near-free
reweighting applied inside an already-running Theta fit -- cheap enough
to apply identically in both the annealing search and the final fit
with no real cost concern. The EVT splice is not: a per-node
tail-dependence-coefficient estimate plus a ~100-200-iteration golden-
section MLE fit is real, non-negligible cost, and annealing calls
`compute_node_score` for every distinct parent set a proposal tries.
Spec section 18's directive ("do not nest an expensive continuous
penalty search inside every graph proposal") applies here exactly as it
does to the two grids -- `hdcd_r_run_annealing` was NOT extended with
`evt_splice_gate`/`evt_splice_bandwidth`; only `hdcd_r_dag_fit` was. The
reference DAG is searched for under the plain Bernstein kernel, then the
final fit (on that already-decided structure) is where the splice
applies -- exactly the same "search under simple, final fit under rich"
split `lambda_roughness_grid`/`bernstein_degree_grid` already use, for
the same reason.

**A FIXED scalar gate/bandwidth for v1**, matching every other
intervention's v1 scope in this investigation -- not itself grid-
searched. New accessors `hdcd_local_fit_tail_family`/
`hdcd_local_fit_tail_theta` (per parent edge, since the splice is an
edge-level decision, not a node-level one like `bernstein_degree`),
exposed through the R binding as `hdcd_node_tail_family()`/
`hdcd_node_tail_theta()` (family returned as `"none"`/`"clayton"`/
`"gumbel"` strings, more R-idiomatic than raw enum codes). Full C test
coverage (default is a no-op; family selection and theta recovery on
genuine Clayton and Gumbel data; gated-off matches unspliced exactly;
root-node triviality; invalid arguments) plus a new R test exercising
the same properties through the public API. Python/Julia struct mirrors
extended again (two more trailing `double` fields) and both full suites
reverified passing (Python 9/9, Julia 32/32) -- the by-now-routine check
for any change to this struct, per the Makefile-bug entry earlier in
this log.

**Empirical validation on the real vine-copula ground truth (d=10, n=2000,
seed 20260819) surfaced a real, non-obvious tuning issue -- and a real
lesson.** The initial check spliced with `evt_splice_gate=0.1`, `bandwidth=0.15`
(the latter inherited, unjustified, from `corner_relief`'s unrelated default).
Two problems showed up:

1. `gate=0.1` was far too permissive: it spliced Clayton/Gumbel onto Frank,
   Gaussian, and t edges with no genuine tail dependence at all (e.g. a
   spurious Clayton splice on a Frank(3.5) edge dropped that edge's
   shape-correlation to the true density from 0.999 to 0.953). Raising the
   gate to `0.5` correctly suppressed every false positive -- all five
   Frank/Gaussian/t edges in the test graph now select `"none"`, unchanged
   from the unspliced fit.
2. On the four edges where the true family (Clayton or Gumbel) WAS correctly
   selected, with accurate MLE recovery (e.g. theta=2.93 vs true 3.0), the
   splice IMPROVED shape-correlation on only one of four edges and WORSENED
   it on the other three, at `bandwidth=0.15`. This was surprising: even
   splicing in the objectively correct parametric family with an accurate
   fitted parameter did not reliably help.

Root cause: the blend is a weighted GEOMETRIC mean of the raw (unnormalized)
Bernstein kernel with the already-normalized parametric copula density. A
wide bandwidth lets the parametric term dominate well outside the actual
corner -- into the bulk of the (u,z) square, where the Bernstein tensor
already fits well on its own -- and the resulting scale mismatch between an
unnormalized and a normalized term degrades the Sinkhorn-normalized fit even
on edges where family selection was correct. Confining the splice tightly
to the corner where the parametric family's advantage is real fixes this:
sweeping bandwidth at the correctly-calibrated `gate=0.5` gave shape-
correlation deltas of (bandwidth -> mean improvement across the 4 genuine
edges): 0.15 -> mixed/negative; 0.05 -> uniformly positive
(+0.025/+0.012/+0.016/+0.018); 0.08 -> uniformly positive and larger
(+0.055/+0.004/+0.013/+0.029), with `Delta_KL` moving from a near-neutral
+0.04 (bandwidth 0.15, i.e. the splice was net-neutral-to-harmful relative
to the plain fit) to a clearly favorable -0.10 (bandwidth 0.08). The default
`evt_splice_bandwidth` (used when the R-level argument is left at 0, i.e.
"unset") was changed from 0.15 to 0.08 in `src/optimize/local_fit.c`
accordingly, with a comment explaining the derivation so a future reader
does not mistake it for another `corner_relief`-inherited placeholder.

**The broader lesson, consistent with this whole investigation's "genuine
difficulties" theme: using the objectively correct parametric family with
an accurately fitted parameter is not sufficient on its own.** How that
correct piece is COMPOSED with the existing nonparametric machinery
(here: a geometric blend of mismatched-scale kernels, normalized
downstream by Sinkhorn) matters as much as getting the family and
parameter right, and the failure mode is not visible from theta-recovery
accuracy alone -- it only shows up in the composed, normalized fit.

**Correction after actually plotting the calibrated fit (bandwidth 0.08)
against the true density, not just its correlation number: the
correlation gain is real but visually small, and does NOT resolve the
under-fit.** On all four genuinely tail-dependent edges, the spliced
curve sits almost on top of the unspliced curve and nowhere near the
true corner spike's height or sharpness (see the notebook's "Does a true
EVT tail-splice..." section, `evt-splice-density-plot` chunk). The
bandwidth that avoids the earlier version's active harm and false
positives also, necessarily, confines the splice's influence to a
region small enough that Sinkhorn's global renormalization pulls the
result back toward the surrounding Bernstein-driven shape almost
everywhere -- widening the bandwidth to give the parametric piece more
influence was the first thing tried, and it made 3 of these same 4
edges worse. So the earlier framing above ("necessary but not
sufficient... matters as much as getting the family and parameter
right") undersold how large the remaining gap actually is once you look
at the curve directly rather than a summary statistic: getting the
family and parameter right, and correctly composing/calibrating the
splice, together still leave the corner visually under-fit. Of the four
interventions tried across this whole investigation, none closes that
gap in a way that would look convincing on a plot. That is the honest
final state of this line of investigation, not a caveat to a
success.

**Further correction: the "small gap" framing above was itself too
generous — the z=0.1/0.9 slice used for that check is not where the
splice acts.** The corner weight $w(u,z) = \exp(-(d_u^2+d_z^2)/(2b^2))$
at bandwidth $b=0.08$ is bounded, on a fixed-$z$ slice, by
$\exp(-d_z^2/(2b^2))$ regardless of $u$ -- at $z=0.1$ that ceiling is
only $\approx 0.46$ and the weight averages $\approx 0.15$ over the
plotted range, so the parametric term was never more than a minority
contributor on the slice used to reach that conclusion. Re-plotted at
$z=0.02$/$0.98$ (within the bandwidth's effective range, weight up to
$\approx 0.97$), the splice DOES produce a visible local effect on
every edge -- but it is a genuine shape distortion, not a step toward
the true corner spike: on edge 1->2 (Clayton, $z=0.02$), the exact
fitted values are (u, true, unspliced, evt-spliced) = (0.005, 3.01,
4.46, 1.14), (0.032, 18.61, 4.12, 3.29), (0.166, 0.04, 2.53, 2.41),
(0.193, 0.02, 2.25, 2.74), (0.246, 0.01, 1.73, 2.61) -- the spliced fit
is LOWER than the unspliced fit right where the true density peaks
(u < 0.15ish), then CROSSES OVER and becomes HIGHER than the unspliced
fit in the shoulder (u > 0.19ish) where the true density is
essentially zero. Both effects move the fit further from a clean
reproduction of the true spike, not closer -- undershooting the peak
and overshooting the shoulder simultaneously.

Leading (untested) hypothesis: near the true corner the blend weight is
close to 1, so the fitted curve there is essentially the parametric
Clayton/Gumbel density's OWN shape at the MLE-estimated theta (e.g.
2.40 vs. a true 3.0 on this edge), not the true theta. Archimedean
copula densities are known to be sensitive to theta specifically off
the $u=z$ diagonal deep in the corner (exactly the region being
plotted, since $u$ ranges over $[0.01,0.3]$ while $z$ is pinned at
$0.02$) -- a theta that looks reasonably close on an absolute scale
can still produce a genuinely different conditional-density *profile*
there, plausibly explaining both the undershoot (lower theta =
less mass concentrated exactly at the corner) and the overshoot
(lower theta = more mass spread into the region just past it). Not
isolated by experiment (e.g. re-running the splice with theta pinned
to the true value instead of its MLE estimate) -- logged here as the
leading candidate explanation, not a confirmed one.

**Net effect on the earlier "necessary but not sufficient" framing:**
it undersold the problem. This is not a case of a correct, if muted,
improvement that simply doesn't go far enough -- it is a real,
non-trivial shape distortion introduced by the splice, that happens to
still net out as a small positive on a linear-correlation summary
statistic (because the shoulder overshoot loosely tracks the true
curve's monotonic decay well enough to help the correlation number,
even though it moves the actual density further from the truth in
absolute terms). The correlation/Delta_KL numbers reported earlier in
this entry are accurate as computed, but should not be read as
evidence the splice is "working, just modestly" -- the density plot at
the correct (near-corner) slice is the honest picture, and it shows a
distortion, not a muted win.

**REMOVED from `hdcd` entirely (C core, tests, R/Python/Julia
bindings) — for a reason independent of the empirical finding above.**
Assuming a parametric copula family for the joint dependence structure
is outside this library's design: spec section 3's "Extreme-Value Tail
Extension" is a *marginal*-level provision (splicing generalized Pareto
tails onto each dimension's univariate CDF `F_j(x)`), and says nothing
about the joint copula density, which the centered-Bernstein-tensor-
plus-Sinkhorn machinery (spec section 9 onward) is deliberately
designed to estimate without assuming any family. The EVT tail-splice
never fit under that provision; it was, in retrospect, off-design from
the moment it was chosen, independent of whether it could have been
made to work well technically. Both reasons hold at once, and neither
depends on the other: this was the wrong kind of fix for this library
to carry AND, separately, it did not cleanly work even when correctly
calibrated.

Deleted: `include/hdcd/parametric_tail.h`, `src/copula/parametric_tail.c`,
`tests/test_parametric_tail.c`; `evt_splice_gate`/`evt_splice_bandwidth`
and the `tail_family`/`tail_theta` fields and accessors removed from
`hdcd_local_fit_options_t`/`hdcd_local_fit_t` (`include/hdcd/local_fit.h`,
`src/optimize/local_fit.c`); the six `test_evt_splice_*` tests removed
from `tests/test_local_fit.c`; the R glue's `evt_splice_gate`/
`evt_splice_bandwidth` SEXP args and the `hdcd_r_local_fit_tail_family`/
`hdcd_r_local_fit_tail_theta` functions removed from `r/src/hdcd_r.c`
(and `CallEntries`); `hdcd_node_tail_family()`/`hdcd_node_tail_theta()`
and the corresponding low-level wrappers removed from `r/R/hdcd.R` and
`r/NAMESPACE`; the `evt_splice_gate` R test removed from
`r/tests/testthat/test-high-level-api.R`; the two trailing fields
removed from the `HdcdLocalFitOptions` struct mirrors in
`python/hdcd/_capi.py` and `julia/src/HDCD.jl`. `hdcd_node_tail_dependence()`
(the underlying tail-dependence-coefficient diagnostic, shared with
`bernstein_degree_grid`) is unaffected and remains.

The notebook's demonstration of this finding (see "Does a true EVT
tail-splice...") was NOT deleted along with the feature -- the finding
itself (miscalibration actively harms; correct calibration produces a
correlation gain that turns out, on inspection at the right (u,z)
slice, to be a shape distortion rather than a real improvement) is a
genuine, informative result independent of whether the feature ships.
Its numbers now come from a saved run of the new
`notebooks/evt_splice_experiment.R`, executed once while the feature
still existed in the C core and committed alongside its output
(`notebooks/evt_splice_experiment_results.rds`), loaded by the notebook
rather than recomputed via the (now-removed) live API -- the same
"commit a script + its results, load rather than recompute" pattern
already used for the `n_sweep` and `corner_relief_sweep` experiments
earlier in this investigation.

**Standing design constraint, made explicit and verified: no parametric
assumptions in the copula (dependence-structure) machinery, at all.**
This generalizes beyond "don't fit Clayton/Gumbel at the corner" (the
reason the EVT splice was removed, above) to a blanket rule for
anything touching `hdcd_local_fit`/`hdcd_sinkhorn_fit`/the centered
Bernstein basis: no step in fitting c_j(u | Pa(j)) may assume a named
parametric copula family, anywhere, for any reason. This does NOT apply
to the marginal-CDF EVT extension (spec section 3, generalized Pareto
tails on each dimension's univariate `F_j(x)`) -- that is a different,
spec-sanctioned part of the pipeline (marginal fitting, not the copula
density) and is unaffected.

Verified by audit (not just assumed) immediately after the EVT splice's
removal: grepped `src/` and `include/` (excluding `src/marginal/`, which
legitimately has GPD per spec section 3) for any other named-family or
parametric-fitting code reachable from the copula-fitting path --
`hdcd_tail_dependence_coefficient` is confirmed nonparametric/empirical
(its own header says so), the dependence matrix uses distance
correlation (Szekely/Rizzo dCor, not Pearson/Gaussian), and the only
other "Gaussian"/"Frank" text in the codebase is illustrative comment
prose about which edges the tail-dependence diagnostic correctly
recognizes as NOT needing extra flexibility, not an actual fitted
model. The EVT splice was the only place this constraint was ever
violated, and it is now fully removed (see above).

**Two nonparametric candidates were discussed as possible follow-ups
for the still-unresolved sharp-corner under-fit — NO DECISION MADE, not
started, logged for later:**

- **(A) Local basis augmentation.** Instead of raising `bernstein_degree`
  globally (the `bernstein_degree_grid` finding: real shape gain, real
  held-out-likelihood cost, because most of the added coefficients are
  spent on the already-well-fit bulk), add a small number of additional
  basis functions concentrated only near the corner flagged by the same
  `hdcd_tail_dependence_coefficient` diagnostic already used to gate
  `bernstein_degree_grid`. Stays entirely inside the existing Theta
  gradient-ascent fit -- no runtime blend of two separate models, no
  Sinkhorn having to reconcile a hard handoff between two regimes (the
  specific failure mechanism diagnosed in the EVT splice). Considered
  the more promising of the two candidates: it directly targets why
  `bernstein_degree_grid` paid a real likelihood cost (wasted capacity
  in the smooth bulk) without needing any assumption about the corner's
  shape.
- **(B) Local (nonparametric) density correction.** Reuse the
  corner-splice architecture already built and validated for the (now
  removed) EVT feature -- corner weight, blend, Sinkhorn normalization
  -- but replace the parametric Clayton/Gumbel MLE piece with a
  genuinely nonparametric local density estimate (e.g. a 2D KDE fit only
  on corner-region data). Plausibly avoids the EXACT failure mode
  diagnosed for the EVT splice, because a local KDE's raw kernel is
  unnormalized, like the Bernstein term it would blend with -- no scale
  mismatch between an unnormalized and an already-normalized quantity
  (the mechanism blamed for the EVT splice's undershoot/overshoot
  distortion). The real, untested risk: the corner is exactly where
  data is sparsest, so any local nonparametric estimate there is
  inherently high-variance -- the textbook reason parametric tail models
  exist in the first place. Would need a real risk/benefit test before
  being worth building.

Neither has been attempted. Revisit if/when there is appetite to keep
pushing on the corner under-fit; otherwise "no intervention tried in
this notebook resolves it" stands as this investigation's finding (see
DECISIONS.md's EVT tail-splice entries above and the notebook's "Known
limitation" section).

## Distinguish initial fit from diagnose from tune

User feedback after the EVT splice's removal: `hdcd`'s usage pattern
should be explicit about THREE separate steps -- fit, diagnose, tune --
rather than the ad hoc "try a knob, eyeball a plot, try another knob"
loop this whole notebook investigation has been running manually. The
concrete trigger: the notebook's fitted conditional copula density
"is missing modes that occur near the edge of the support" -- confirmed
to be the same Clayton/Gumbel tail-dependence corner phenomenon this
log has been tracking all along, just stated in more general
diagnostic language.

**The gap found on inspection: the diagnostic itself (the empirical
tail-dependence coefficient, `hdcd_tail_dependence_coefficient`) was
only ever computed INSIDE the `bernstein_degree_grid`-gated code path
in `hdcd_local_fit_node`** (`src/optimize/local_fit.c`). A caller who
fit a PLAIN model (no `bernstein_degree_grid`, no `corner_relief`) could
not inspect `hdcd_local_fit_max_tail_dependence()` at all -- it stayed
NAN, because the coefficient was never computed unless the grid was
already supplied. This made "diagnose, then decide whether to tune"
structurally impossible: you had to already commit to tuning
(`bernstein_degree_grid`) before you could see the number that would
tell you whether tuning was warranted.

**Fix: decouple the diagnostic from the tuning gate.** The empirical
tail-dependence-coefficient computation in `hdcd_local_fit_node` now
runs unconditionally for every non-root node, on ANY fit -- independent
of whether `bernstein_degree_grid` is supplied. `degree_search_active`
(whether the degree grid is actually searched) is now a separate
boolean, gated on both `degree_grid_enabled` AND the coefficient
clearing `tail_dependence_gate`, computed from the now-unconditional
value rather than folding the "is this even computed" question into
the same branch as "should this be acted on." The estimator itself is
O(n) order statistics, not an optimization -- computing it
unconditionally costs essentially nothing relative to the Theta fit
that follows it. `hdcd_local_fit_max_tail_dependence()` is NAN only for
a root node (nothing to diagnose) now, not also whenever the grid was
withheld.

**New R-level `hdcd_diagnose(model)`**: takes a PLAIN `hdcd_model` (the
result of `hdcd_fit()` with no tuning options set) and returns a data
frame -- one row per non-root node, `node`/`n_parents`/
`tail_dependence`, sorted most-tail-dependent-first -- built entirely
from the now-always-available `hdcd_node_tail_dependence()` accessor
(no new C entry point needed). Deliberately does NOT bake in a
pass/fail threshold: this investigation has already seen the "right"
gate value vary by intervention and dataset (0.05 for one
`bernstein_degree_grid` demonstration, 0.5 after properly calibrating
the now-removed EVT splice) -- hard-coding a default here would repeat
that mistake. It reports evidence; the analyst (or a future,
evidence-based policy) decides the tuning gate. Only accepts
`hdcd_model`, not a bare `hdcd_fit_dag()` result, since enumerating
every node needs `model$d`, which only the former carries.

**The resulting three-step workflow**: (1) `hdcd_fit()` with tuning
options left at their defaults -- the initial fit; (2) `hdcd_diagnose()`
on that result -- which nodes actually show tail dependence, by how
much; (3) `hdcd_fit_dag()`/a re-`hdcd_fit()` call with
`bernstein_degree_grid`/`corner_relief`/a chosen `tail_dependence_gate`
-- tuning, now an informed, deliberate second step instead of a blind
default or something discoverable only after already committing to it.

Test coverage: a new C test
(`test_tail_dependence_diagnostic_available_without_any_grid`) confirms
the coefficient is populated (and correctly distinguishes tail-
dependent from independent data) on a plain fit with no grid supplied
at all; a new R test (`hdcd_diagnose reports tail-dependence on a
plain, untuned fit`) exercises the new function end-to-end and confirms
it rejects a bare `hdcd_fit_dag()` result. One existing R test
(`bernstein_degree_grid is tail-dependence-gated...`) asserted the OLD
behavior (`NA` when the grid was never supplied) and was updated to
assert the new one. Full C, R, Python, and Julia suites reverified
passing -- no ABI change (no struct fields added/removed), so no
Python/Julia struct-mirror updates were needed this time.

## Local nonparametric corner correction (Option B)

Follow-up to the "no parametric copula assumptions" and "two undecided
follow-ups" entries above: user decision was to pursue Option B (a
local nonparametric density correction) over Option A (local basis
augmentation), with an explicit requirement that tuning it be a
MANUAL, ITERATIVE process -- not another auto-searched grid -- and that
it be validated against genuinely held-out data, not the full sample.

**Design, agreed before implementation:** combine the local correction
with the raw Bernstein kernel ADDITIVELY, in RAW kernel space, after
both terms are exponentiated -- never as a geometric blend in log
space. This is the one specific, load-bearing lesson carried over from
the removed EVT splice: its failure was not "wrong family," it was
blending an already-normalized log-density against `g` (the Bernstein
bilinear form, meaningless in isolation until Sinkhorn calibrates it)
as if they lived on the same scale. Combining post-exp, additively,
avoids that specific composition risk regardless of what the local
term is. `hdcd_sinkhorn_fit` still normalizes the COMBINED raw kernel
exactly as it always has -- this is guaranteed, not new machinery --
but normalization guarantees validity, not quality, which is why this
requires held-out empirical validation rather than being assumed to
work from the architecture alone.

**Mechanism**: `corner_kde_gate`/`corner_kde_bandwidth`/`corner_kde_weight`
in `hdcd_local_fit_options_t`. Gating reuses `hdcd_tail_dependence_coefficient`
exactly like `bernstein_degree_grid`/the removed EVT splice (independently
recomputed, same accepted minor redundancy). A gated edge's corner side
(`hdcd_corner_side_t`: NONE/LOWER/UPPER -- a LOCATION, not a family) is
recorded once per node, before any grid search, and the edge's raw
kernel becomes `raw_kernel_bernstein(u,z) + corner_kde_weight *
corner_proximity(u,z) * local_kde(u,z)`, where `corner_proximity` is
the same Gaussian-bump corner-distance taper used throughout this
investigation (`corner_relief`'s edge_proximity, the removed EVT
splice's evt_corner_weight), and `local_kde` is a genuine bivariate
Gaussian-product KDE over the edge's TRAIN (u_child, z_parent) pairs.
Deliberately excluded from `hdcd_run_annealing` (real per-node
`O(n_train)`-per-raw-kernel-call cost, unlike `corner_relief`'s
near-free reweighting) -- applies only to `hdcd_dag_fit`/`hdcd_fit_dag()`
calls on an already-decided DAG, matching `bernstein_degree_grid`'s and
the removed EVT splice's precedent. NO grid-searched variant of gate/
bandwidth/weight exists, by design -- per the user's explicit "manual
and iterative" requirement, these are meant to be set explicitly and
adjusted by hand across repeated `hdcd_fit_dag()` calls, watching the
result each time, never auto-optimized by this library.

**A real bug, caught by testing on real data before shipping, not
assumed correct from the math alone.** The first implementation
deliberately dropped the KDE's volume-normalization constant
(`1/(2*pi*h^2)` for a bivariate Gaussian product kernel), reasoning
(incorrectly) that it needed to stay "raw/uncalibrated like the
Bernstein term," conflating two different things: the actual fix for
the EVT splice's failure was additive-raw-space combination, not "the
local term must be badly scaled." Without the normalizing constant,
dividing the kernel sum by `n_train` (~1500) crushed the correction to
a magnitude that measurably changed nothing -- confirmed directly on
the notebook's real vine-copula data: `Delta_KL` between corrected and
uncorrected fits was 0.0002, and even at the exact corner (z=0.02), the
two fits agreed to 3 decimal places. Restoring the normalizing constant
(`local_kde_raw()` in `src/optimize/local_fit.c`) fixed this -- the
correction now has a genuine, comparable order of magnitude to the
quantity it's added to.

**First manual-tuning-loop finding, logged as a starting point, not a
calibrated default:** even after the normalization fix, `corner_kde_weight=1`
(picked as a "neutral, equal-footing" default) still produced only a
small effect (`Delta_KL=0.0037`, correlation gains of ~0.002-0.004) --
the raw KDE and raw Bernstein kernel are evidently not naturally on
comparable absolute scales, so "weight=1" was not actually neutral in
practice. `corner_kde_weight=20` (same `corner_kde_bandwidth=0.08`)
produced a clear, consistent positive effect on all four genuinely
tail-dependent edges (correlation gains of 0.016-0.036, `Delta_KL=+0.05`)
with the gated-out smooth edges completely unaffected. A quick check of
`corner_kde_bandwidth=0.02` (tighter, to better resolve the sharp
spike) at `weight=1` did NOT help -- plausibly because the same
bandwidth parameter also controls the corner-proximity taper's width,
so tightening it shrinks the region the correction is even allowed to
act in, counteracting any sharper local resolution. This is exactly
the kind of interaction the manual loop is for; it has not been
explored further here, deliberately -- per the user's explicit
direction, further tuning is a human-driven loop, not something this
library should search on its own. `weight~20` at `bandwidth=0.08` is
logged as a reasonable STARTING point for that loop, not a validated
default (the C-level default stays `corner_kde_weight=1`, unchanged,
since baking in an untested "better" value here would repeat exactly
the mistake `hdcd_diagnose()` was built to avoid).

**Manual, iterative tuning workflow, as implemented:** [hdcd_fit_dag()]
gained `corner_kde_gate`/`corner_kde_bandwidth`/`corner_kde_weight`
override arguments (defaulting to reusing `model`'s own settings, same
pattern as every other override on that function) -- this is the
intended entry point: call it repeatedly by hand with different values,
inspecting the result after each call, rather than searching
automatically. Two new tools support inspecting a call's result without
needing the (unavailable, in a real application) true density:
- `hdcd_node_corner_side(model, node, parent)`: which corner (if any) a
  parent edge's correction targets -- `"none"`/`"lower"`/`"upper"`.
- `hdcd_node_region_score(model, node, parent, u_holdout, z_holdout,
  z_center, z_window)`: held-out log-likelihood restricted to rows
  whose `parent`-th value falls within `z_window` of `z_center`, built
  entirely from the existing `hdcd_local_fit_log_density`/its R
  wrapper (no new C entry point needed -- an R-level loop over held-out
  rows is fast enough at the row counts this diagnostic needs, roughly
  the low hundreds per corner). Exists specifically because pooled
  `Delta_KL`/`hdcd_score_dag()` is exactly what let the EVT splice's
  local distortion hide inside a favorable aggregate -- this narrows
  the same held-out evaluation to where a correction is supposed to
  act. Always reports `n` (effective sample size) alongside the score,
  since a small `n` means "can't tell," not a number to trust.

**Outer-holdout carve-out needs no new plumbing.** Because
`hdcd_fit_dag()` always uses whatever `model$U` it's given (with
`model$local_seed` driving that call's OWN internal train/holdout
split), a genuinely untouched outer holdout -- never seen by ANY
fitting or tuning call, not just excluded from scoring -- is achieved
simply by constructing a modified copy of `model` with `$U` restricted
to a "dev" row subset chosen in R before any fitting begins, and
holding the rest out entirely. All fitting/tuning iteration happens
against `dev`; the genuine outer-holdout rows are only ever passed to
`hdcd_node_region_score()`/manual histogram checks, read-only, once.

Test coverage: five new C tests (`test_corner_kde_*`, mirroring
`corner_relief`'s and `bernstein_degree_grid`'s test shapes: default is
a no-op, gating selects a side and measurably changes the fit on
tail-dependent data, gated-off matches unweighted exactly, root-node
triviality, invalid arguments) plus two new R tests (`corner_kde_gate`
selects a side and measurably changes the fit; `hdcd_node_region_score`
computes a sensible score and correctly returns `NA`/`n=0` when no rows
qualify). Python/Julia `HdcdLocalFitOptions` struct mirrors extended by
the three new trailing fields (a real ABI change this time, unlike the
diagnose/tune decoupling entry above) and both full suites reverified
passing after reinstall.

## An interface for hand tuning: `hdcd_tune_corner()`/`hdcd_plot_corner_check()`

The notebook's first manual-tuning demonstration (previous entry) took
roughly 15 lines of boilerplate per round -- fit, compute two region
scores, build a histogram data frame, plot -- repeated for each
hand-chosen parameter set. User feedback: this needs an actual
interface for hand tuning, not a pattern to retype every round.

**`hdcd_tune_corner(model_dev, candidate_edges, node, parent,
corner_kde_gate, corner_kde_bandwidth, corner_kde_weight, u_holdout,
z_holdout, z_center, z_window, baseline = NULL, ...)`**: fits one
hand-chosen configuration, scores it against an uncorrected baseline
via `hdcd_node_region_score()` (both on the SAME held-out rows), prints
a one-line comparison, and returns both fits invisibly so the caller
can keep iterating or hand the result to the plot function. Accepts a
previously-fit `baseline` (typically a prior call's own `$baseline`) so
the uncorrected fit is not needlessly recomputed every round. Explicitly
NOT a search -- one call per candidate the analyst picks, same as
calling `hdcd_fit_dag()` directly, just with the comparison built in.
This is a convenience wrapper around existing pieces
(`hdcd_fit_dag()`, `hdcd_node_region_score()`, `hdcd_node_corner_side()`),
not new fitting logic -- no C changes were needed for this entry.

**`hdcd_plot_corner_check(round, u_holdout, z_holdout, true_density =
NULL, u_grid = ..., title = NULL)`**: promotes the notebook's local
`plot_corner_check()` helper into the package proper, built from a
`hdcd_tune_corner()` result so the plot and the printed score always
agree about which rows and which corner they describe. `true_density`
is an optional, precomputed vector for synthetic-validation notebooks
like this one -- deliberately NOT a `copula`-package dependency (a real
application has no true density to plot, so the package itself should
not need to know what `copula` objects are). Requires `ggplot2`, added
as a `Suggests` (not `Imports`) dependency with a `requireNamespace()`
check at call time -- the only function in the package that needs
plotting, so the rest of `hdcd` stays free of a hard graphics
dependency.

Two new R tests (`hdcd_tune_corner` runs one round and returns
reusable fits; the plot function returns a `ggplot` object, skipped
if `ggplot2` isn't installed). No C-core or ABI changes this entry --
pure R-layer wiring around already-existing pieces -- so Python/Julia
bindings are unaffected and were not touched.
