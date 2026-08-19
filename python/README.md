# hdcd (Python binding)

Thin ctypes binding over the `hdcd` C library (spec section 25).

## Install

From the repository root:

```sh
pip install ./python
```

This compiles the C shared library via the top-level `Makefile` (`make shared`)
as part of the build and bundles it inside the installed package — no manual
build step required. Requires a C compiler on `PATH`.

## Usage

```python
import numpy as np
import hdcd

X = np.random.default_rng(0).standard_normal((300, 4))
model = hdcd.fit(X, max_parents=2, bernstein_degree=3,
                  lambda_edge=0.05, lambda_roughness=0.15)

u = model.transform(X)
lp = model.logpdf(X)
clp = model.copula_logpdf(u)

model.dependence_matrix_
model.ordering_
model.dag_

result = model.fit_dag([(0, 1)])   # an alternative candidate DAG
result.kl_divergence_
```

`model.sample(...)` is not implemented: the C core has no sampling routine
yet (see `HdcdModel.sample`'s docstring and `DECISIONS.md`).

The reference DAG (and any DAG compared via `fit_dag`) is a statistical
factorization for density estimation, not a causal model (spec section 19).

## Tests

```sh
cd python/tests
python3 -m unittest test_python_binding -v
```

`fixture.json` holds C-computed reference values used to check the ctypes
binding for numerical agreement with the C library (spec section 31,
Milestone 10). Regenerate it after a C-side change:

```sh
cc -std=c99 -Iinclude python/tests/generate_fixture.c build/libhdcd.a -lm -o /tmp/gen
(cd python/tests && /tmp/gen)
```
