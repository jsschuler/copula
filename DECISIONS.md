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
