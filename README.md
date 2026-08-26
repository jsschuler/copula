# hdcd — High-Dimensional Copula Density Estimation

A nonparametric library for estimating high-dimensional joint densities by
factoring them into smoothed marginals and a sparse, DAG-structured copula.
A C numerical core with Python, R, and Julia bindings.

The full engineering and mathematical specification is
[`high_dimensional_copula_claude_code_spec.md`](high_dimensional_copula_claude_code_spec.md);
open judgment calls and deviations from it are logged in
[`DECISIONS.md`](DECISIONS.md).

## What it does

The joint density is represented as

```
f_X(x) = c(F_1(x_1), ..., F_d(x_d)) * prod_j f_j(x_j)
```

and estimated through this pipeline:

```
X
  -> smoothed marginal CDFs / densities  (Gaussian-mixture, optional EVT tails)
  -> copula transform  U_j = F_j(X_j)
  -> pairwise distance correlation
  -> persistent-topology / MST variable ordering
  -> sparse DAG structure search (simulated annealing)
  -> conditional copula factors (centered Bernstein tensor bases)
  -> Sinkhorn normalization so each factor preserves copula marginals
  -> c(U)  ->  f_X(X)
```

Composite (node-wise) likelihood handles arbitrary missingness, and any
alternative DAG — including causal orderings — can be fitted and compared by
held-out KL divergence.

### Design constraints

- **The dependence structure stays fully nonparametric.** No parametric
  copula family is ever assumed. Marginal GPD tail modeling is the one
  sanctioned parametric exception.
- **The DAG is a statistical factorization for density estimation, not a
  causal claim.** DAG comparison answers "how much dependence information
  does this factorization lose," not "which edges are causal" (spec §19).

## Repository layout

| Path | Contents |
|------|----------|
| `include/hdcd/` | Public C headers (`hdcd.h` is the umbrella) |
| `src/` | C core: `numerics`, `marginal`, `copula`, `dcor`, `topology`, `basis`, `sinkhorn`, `dag`, `optimize`, `rng` |
| `tests/` | C unit tests (one `test_*.c` per module) |
| `examples/` | Standalone C usage examples, one per milestone |
| `python/`, `r/`, `julia/` | Language bindings, each with its own README |
| `notebooks/` | R validation notebooks and tuning experiments |
| `Makefile` | Build for the C core, tests, and examples |

## Building the C core

Requires `make` and a C99 compiler. No other dependencies.

```sh
make lib        # build/libhdcd.a
make shared     # build/libhdcd.dylib (or .so) — needed by the Python/Julia bindings
make test       # build the C unit tests
make examples   # build the example programs
make all        # library + tests + examples
make clean
```

## Language bindings

Each binding compiles the C library as part of its own build — no separate
step needed. See the per-directory README for install, usage, and test
instructions.

| Language | Install (from repo root) | README |
|----------|--------------------------|--------|
| Python | `pip install ./python` | [`python/README.md`](python/README.md) |
| R | `R CMD INSTALL r` | [`r/README.md`](r/README.md) |
| Julia | `make shared`, then `Pkg.instantiate()` in `julia/` | [`julia/README.md`](julia/README.md) |

### Python quickstart

```python
import numpy as np, hdcd

X = np.random.default_rng(0).standard_normal((300, 4))
model = hdcd.fit(X, max_parents=2, bernstein_degree=3,
                 lambda_edge=0.05, lambda_roughness=0.15)

model.logpdf(X)              # joint log-density
model.copula_logpdf(model.transform(X))
model.dependence_matrix_, model.ordering_, model.dag_
model.fit_dag([(0, 1)]).kl_divergence_   # compare an alternative DAG
```

Sampling from a fitted model is not implemented — the C core has no sampling
routine yet (see `DECISIONS.md`).

## Validation

`notebooks/vine_copula_recovery.Rmd` is the end-to-end recovery check
(render with `rmarkdown::render`). The `*_experiment.R` scripts and their
`*_results.csv` are the tuning sweeps behind the current defaults for
roughness penalty, Bernstein degree, corner correction, and the EVT tail
splice.
