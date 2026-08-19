# HDCD.jl (Julia binding)

Direct `ccall` bindings (spec section 27) over the `hdcd` C library.

## Setup

From the repository root, build the shared library once:

```sh
make shared
```

Then, from `julia/`:

```sh
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

`HDCD.jl` locates `build/libhdcd.dylib`/`.so` relative to its own source file
by default, or via the `HDCD_LIBRARY_PATH` environment variable.

## Usage

```julia
using HDCD

X = randn(300, 4)
model = hdcd_fit(X; max_parents=2, bernstein_degree=3,
                  lambda_edge=0.05, lambda_roughness=0.15)

u = transform_copula(model, X)
lp = logpdf(model, X)
clp = copula_logpdf(model, u)

dependence_matrix(model)
ordering(model)
dag(model)

candidate = fit_dag(model, [1 2])   # an alternative candidate DAG
score_dag(model, candidate)
```

`hdcd_sample(...)` is not implemented: the C core has no sampling routine
yet (see its docstring and `DECISIONS.md`).

The reference DAG (and any DAG compared via `fit_dag`/`score_dag`) is a
statistical factorization for density estimation, not a causal model
(spec section 19).

## Tests

```sh
julia --project=. -e 'import Pkg; Pkg.test()'
```

`test/runtests.jl` reuses `python/tests/fixture.json` (spec section 29.11:
"the same saved test fixture") to check the `ccall` binding for numerical
agreement with the C library, via a small self-contained JSON reader (no
external JSON dependency).
