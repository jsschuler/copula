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
shouldn't be built before it's needed.
