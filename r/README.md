# hdcd (R binding)

`.Call`-based R bindings over the `hdcd` C library (spec section 26).

## Install

From the repository root:

```sh
R CMD INSTALL r
```

This compiles the C static library via the top-level `Makefile` (`make lib`)
and links it statically into the package's shared object as part of the
build — no manual build step required. Requires a C compiler on `PATH`.

## Usage

```r
library(hdcd)

X <- matrix(rnorm(300 * 4), ncol = 4)
model <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 3L,
                   lambda_edge = 0.05, lambda_roughness = 0.15)

u <- hdcd_transform(model, X)
lp <- predict(model, X, type = "logpdf")
clp <- hdcd_copula_logpdf(model, u)

hdcd_dependence_matrix(model)
hdcd_ordering(model)
hdcd_dag(model)

candidate <- hdcd_fit_dag(model, cbind(parent = 1L, child = 2L))
hdcd_score_dag(model, candidate)
```

`hdcd_sample(...)` is not implemented: the C core has no sampling routine
yet (see its docstring and `DECISIONS.md`).

The reference DAG (and any DAG compared via `hdcd_fit_dag`/`hdcd_score_dag`)
is a statistical factorization for density estimation, not a causal model
(spec section 19).

## Tests

```sh
Rscript -e 'testthat::test_dir("r/tests/testthat", package = "hdcd")'
```

`test-fixture-agreement.R` reuses `python/tests/fixture.json` (spec section
29.11: "the same saved test fixture") to check the `.Call` binding for
numerical agreement with the C library.
