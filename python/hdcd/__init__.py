"""
hdcd: Python bindings for the hdcd high-dimensional copula density
estimation C library (spec section 25).

    import hdcd
    model = hdcd.fit(X, max_parents=2, bernstein_degree=3,
                      lambda_edge=0.05, lambda_roughness=0.15)
    u = model.transform(X)
    lp = model.logpdf(X)
    model.dependence_matrix_
    model.ordering_
    model.dag_

IMPORTANT: the DAG this library fits is a statistical factorization for
density estimation, not a causal model (spec sections 19, 34). See
HdcdModel.fit_dag's docstring.
"""

from ._capi import HdcdError
from ._model import (
    Dag,
    DagFitResult,
    DependenceMatrix,
    FitDagResult,
    HdcdModel,
    Marginal,
    Topology,
    fit,
)

__all__ = [
    "fit",
    "HdcdModel",
    "Marginal",
    "DependenceMatrix",
    "Topology",
    "Dag",
    "DagFitResult",
    "FitDagResult",
    "HdcdError",
]
