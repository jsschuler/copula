"""
High-level, Pythonic wrapper over the hdcd C library (spec section 25).

Mirrors spec section 25's preferred usage:

    model = hdcd.fit(X, max_parents=..., bernstein_degree=...,
                      lambda_edge=..., lambda_roughness=...)
    u = model.transform(X)
    lp = model.logpdf(X)
    clp = model.copula_logpdf(u)
    model.dependence_matrix_
    model.ordering_
    model.dag_
    result = model.fit_dag(candidate_edges)
    result.kl_divergence_

model.sample(...) is NOT implemented -- see its docstring and
DECISIONS.md: the C core has no sampling routine, since no version-1
milestone (spec section 31) schedules building spec section 21's
hdcd_sample, even though it's named in that section's architecture
sketch. Calling it raises NotImplementedError rather than silently
returning something wrong.
"""

import ctypes
from ctypes import POINTER, byref, c_double, c_int, c_size_t, c_uint8, c_void_p

import numpy as np

from . import _capi
from ._capi import (
    HdcdAnnealingOptions,
    HdcdLocalFitOptions,
    HdcdSinkhornOptions,
    HdcdError,
    check,
)

__all__ = ["fit", "HdcdModel", "Marginal", "DependenceMatrix", "Topology", "Dag", "DagFitResult", "FitDagResult"]


def _as_column_major(X):
    """Fortran-order float64 view (spec section 23: column-major core
    layout; section 25: "avoid copying if input is already compatible
    column-major float64" -- np.asarray/asfortranarray are no-ops when
    the input already matches)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D array (n_samples, n_features), got shape {X.shape}")
    return np.asfortranarray(X, dtype=np.float64)


def _mask_from_nan(X):
    """Explicit observed-mask, built at the wrapper boundary (spec
    section 25: "accept NaN input but convert to explicit mask at the
    wrapper boundary"; section 23: missingness must never be carried by
    NaN alone inside the C core)."""
    return np.asfortranarray(~np.isnan(X), dtype=np.uint8)


def _readonly(arr):
    arr = np.array(arr, copy=True)
    arr.setflags(write=False)
    return arr


# ---- Marginal -------------------------------------------------------------

class Marginal:
    """One fitted per-column marginal (spec section 2), hdcd_marginal_t."""

    def __init__(self, ptr, x_col, mask_col):
        self._ptr = ptr
        # Kept alive only for reference; the marginal itself owns its own
        # copy of the observed subset internally (spec section 2's O_j).
        self._x_col = x_col
        self._mask_col = mask_col

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _capi.lib.hdcd_marginal_free(ptr)
            self._ptr = None

    @classmethod
    def fit(cls, x_col, mask_col, sigma_min=-1.0, sigma_max=-1.0, tol=1e-6, max_iter=200):
        x_col = np.ascontiguousarray(x_col, dtype=np.float64)
        mask_col = np.ascontiguousarray(mask_col, dtype=np.uint8)
        out_ptr = c_void_p()
        status = _capi.lib.hdcd_marginal_fit(
            _capi._dptr(x_col), _capi._u8ptr(mask_col), c_size_t(x_col.shape[0]),
            c_double(sigma_min), c_double(sigma_max), c_double(tol), c_int(max_iter),
            byref(out_ptr),
        )
        check(status)
        return cls(out_ptr, x_col, mask_col)

    @property
    def n_observed(self):
        return _capi.lib.hdcd_marginal_n_observed(self._ptr)

    @property
    def sigma(self):
        return _capi.lib.hdcd_marginal_bandwidth_result(self._ptr).sigma

    def cdf(self, eval_points):
        eval_points = np.ascontiguousarray(eval_points, dtype=np.float64)
        out = np.empty_like(eval_points)
        status = _capi.lib.hdcd_marginal_cdf(
            self._ptr, _capi._dptr(eval_points), c_size_t(eval_points.shape[0]), _capi._dptr(out)
        )
        check(status)
        return out

    def logpdf(self, eval_points):
        eval_points = np.ascontiguousarray(eval_points, dtype=np.float64)
        out = np.empty_like(eval_points)
        status = _capi.lib.hdcd_marginal_logpdf(
            self._ptr, _capi._dptr(eval_points), c_size_t(eval_points.shape[0]), _capi._dptr(out)
        )
        check(status)
        return out


# ---- DependenceMatrix -------------------------------------------------

class DependenceMatrix:
    """d x d pairwise distance-correlation matrix (spec section 5)."""

    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _capi.lib.hdcd_dependence_matrix_free(ptr)
            self._ptr = None

    @classmethod
    def compute(cls, u, mask):
        n, d = u.shape
        out_ptr = c_void_p()
        status = _capi.lib.hdcd_compute_dependence_matrix(
            _capi._dptr(u), _capi._u8ptr(mask), c_size_t(n), c_size_t(d), byref(out_ptr)
        )
        check(status)
        return cls(out_ptr)

    @property
    def dim(self):
        return _capi.lib.hdcd_dependence_matrix_dim(self._ptr)

    def to_numpy(self):
        d = self.dim
        out = np.empty((d, d), dtype=np.float64)
        for j in range(d):
            for k in range(d):
                out[j, k] = _capi.lib.hdcd_dependence_matrix_get(self._ptr, j, k)
        return _readonly(out)

    def n_effective(self, j, k):
        return _capi.lib.hdcd_dependence_matrix_n_effective(self._ptr, j, k)


# ---- Topology ---------------------------------------------------------

class Topology:
    """MST / persistent-topology ordering result (spec section 6)."""

    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _capi.lib.hdcd_topology_free(ptr)
            self._ptr = None

    @classmethod
    def compute(cls, dependence_matrix):
        out_ptr = c_void_p()
        status = _capi.lib.hdcd_compute_topology(dependence_matrix._ptr, byref(out_ptr))
        check(status)
        return cls(out_ptr)

    @property
    def dim(self):
        return _capi.lib.hdcd_topology_dim(self._ptr)

    def ordering(self):
        d = self.dim
        ptr = _capi.lib.hdcd_topology_ordering(self._ptr)
        return _readonly(np.ctypeslib.as_array(ptr, shape=(d,)).astype(np.int64))

    def mst_edges(self):
        count = _capi.lib.hdcd_topology_mst_edge_count(self._ptr)
        edges = []
        for i in range(count):
            e = _capi.lib.hdcd_topology_mst_edge(self._ptr, i)
            edges.append((int(e.j), int(e.k), float(e.weight)))
        return tuple(edges)


# ---- Dag ----------------------------------------------------------------

class Dag:
    """A DAG handle (spec section 7), hdcd_dag_t."""

    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _capi.lib.hdcd_dag_free(ptr)
            self._ptr = None

    @classmethod
    def create(cls, d, k_max):
        out_ptr = c_void_p()
        check(_capi.lib.hdcd_dag_create(c_size_t(d), c_size_t(k_max), byref(out_ptr)))
        return cls(out_ptr)

    @classmethod
    def from_edges(cls, d, k_max, edges):
        """edges: iterable of (parent, child) pairs. Validated for
        acyclicity as a whole, so (unlike incrementally adding edges)
        this can genuinely reject a cyclic edge set (spec section 19)."""
        edges = list(edges)
        n_edges = len(edges)
        parents = np.array([e[0] for e in edges], dtype=np.uintp)
        children = np.array([e[1] for e in edges], dtype=np.uintp)
        out_ptr = c_void_p()
        status = _capi.lib.hdcd_dag_from_edges(
            c_size_t(d), c_size_t(k_max),
            _capi._sptr(parents) if n_edges else None,
            _capi._sptr(children) if n_edges else None,
            c_size_t(n_edges), byref(out_ptr),
        )
        check(status)
        return cls(out_ptr)

    @classmethod
    def clone(cls, other):
        out_ptr = c_void_p()
        check(_capi.lib.hdcd_dag_clone(other._ptr, byref(out_ptr)))
        return cls(out_ptr)

    @property
    def dim(self):
        return _capi.lib.hdcd_dag_dim(self._ptr)

    @property
    def k_max(self):
        return _capi.lib.hdcd_dag_k_max(self._ptr)

    def add_edge(self, parent, child):
        check(_capi.lib.hdcd_dag_add_edge(self._ptr, c_size_t(parent), c_size_t(child)))

    def parents(self, child):
        n = _capi.lib.hdcd_dag_n_parents(self._ptr, child)
        buf = np.empty(n, dtype=np.uintp)
        if n > 0:
            check(_capi.lib.hdcd_dag_parents(self._ptr, c_size_t(child), _capi._sptr(buf)))
        return tuple(int(p) for p in buf)

    def edges(self):
        """All (parent, child) edges, as a read-only tuple."""
        out = []
        for child in range(self.dim):
            for parent in self.parents(child):
                out.append((parent, child))
        return tuple(out)


# ---- LocalFit (borrowed) ------------------------------------------------

class LocalFit:
    """One node's fitted conditional factor (spec sections 8-11, 16).
    Borrowed from a DagFitResult -- never freed independently."""

    def __init__(self, ptr):
        self._ptr = ptr

    @property
    def n_parents(self):
        return _capi.lib.hdcd_local_fit_n_parents(self._ptr)

    @property
    def parent_order(self):
        n = self.n_parents
        if n == 0:
            return ()
        ptr = _capi.lib.hdcd_local_fit_parent_order(self._ptr)
        return tuple(int(v) for v in np.ctypeslib.as_array(ptr, shape=(n,)))

    @property
    def n_observed(self):
        return _capi.lib.hdcd_local_fit_n_observed(self._ptr)

    @property
    def n_train(self):
        return _capi.lib.hdcd_local_fit_n_train(self._ptr)

    @property
    def n_holdout(self):
        return _capi.lib.hdcd_local_fit_n_holdout(self._ptr)

    @property
    def holdout_score(self):
        return _capi.lib.hdcd_local_fit_holdout_score(self._ptr)

    @property
    def roughness_penalty(self):
        return _capi.lib.hdcd_local_fit_roughness_penalty(self._ptr)

    @property
    def converged(self):
        return bool(_capi.lib.hdcd_local_fit_theta_converged(self._ptr)) and \
            bool(_capi.lib.hdcd_local_fit_sinkhorn_converged(self._ptr))


# ---- DagFitResult ---------------------------------------------------------

class DagFitResult:
    """A fitted DAG's per-node factors and factorized joint density
    (spec sections 14, 28), hdcd_dag_fit_t."""

    def __init__(self, ptr, dim):
        self._ptr = ptr
        self._dim = dim

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _capi.lib.hdcd_dag_fit_free(ptr)
            self._ptr = None

    @classmethod
    def fit(cls, u, mask, dag, local_fit_options):
        n, d = u.shape
        out_ptr = c_void_p()
        status = _capi.lib.hdcd_dag_fit(
            _capi._dptr(u), _capi._u8ptr(mask), c_size_t(n), c_size_t(d),
            dag._ptr, byref(local_fit_options), byref(out_ptr),
        )
        check(status, allow_not_converged=True)
        return cls(out_ptr, d)

    def node(self, j):
        ptr = _capi.lib.hdcd_dag_fit_node(self._ptr, j)
        if not ptr:
            raise IndexError(j)
        return LocalFit(ptr)

    @property
    def all_converged(self):
        return bool(_capi.lib.hdcd_dag_fit_all_converged(self._ptr))

    def joint_log_density(self, u_point):
        u_point = np.ascontiguousarray(u_point, dtype=np.float64)
        out = c_double()
        status = _capi.lib.hdcd_dag_fit_joint_log_density(
            self._ptr, _capi._dptr(u_point), c_size_t(u_point.shape[0]), byref(out)
        )
        check(status)
        return out.value

    @property
    def kl_estimate_(self):
        return _capi.lib.hdcd_dag_fit_kl_estimate(self._ptr)


class FitDagResult:
    """Result of HdcdModel.fit_dag: a candidate DAG's fit plus its
    held-out KL comparison against the model's reference DAG (spec
    section 19).

    IMPORTANT: this is a purely statistical, observational comparison
    of distributional fit. It does not establish causal direction and
    does not distinguish Markov-equivalent causal DAGs (spec section
    19's closing paragraph).
    """

    def __init__(self, dag, dag_fit, kl_divergence):
        self.dag = dag
        self.dag_fit = dag_fit
        self.kl_divergence_ = kl_divergence


# ---- HdcdModel --------------------------------------------------------

class HdcdModel:
    def __init__(self, marginals, dag, dag_fit, dependence_matrix, topology, k_max, local_fit_options):
        self._marginals = marginals
        self._dag = dag
        self._dag_fit = dag_fit
        self._dependence_matrix = dependence_matrix
        self._topology = topology
        self._k_max = k_max
        self._local_fit_options = local_fit_options

    @property
    def d(self):
        return len(self._marginals)

    @property
    def dependence_matrix_(self):
        return self._dependence_matrix.to_numpy()

    @property
    def ordering_(self):
        return self._topology.ordering()

    @property
    def dag_(self):
        return self._dag.edges()

    @property
    def all_converged_(self):
        return self._dag_fit.all_converged

    def transform(self, X):
        """x -> u = clip(F_hat_j(x_j), epsilon) per column, using the
        marginals fitted by `fit` (spec section 4)."""
        X = _as_column_major(X)
        mask = _mask_from_nan(X)
        n, d = X.shape
        if d != self.d:
            raise ValueError(f"expected {self.d} columns, got {d}")
        U = np.empty((n, d), dtype=np.float64, order="F")
        for j in range(d):
            x_col = np.ascontiguousarray(X[:, j])
            mask_col = np.ascontiguousarray(mask[:, j])
            u_col = np.empty(n, dtype=np.float64)
            status = _capi.lib.hdcd_transform_to_copula(
                self._marginals[j]._ptr, _capi._dptr(x_col), _capi._u8ptr(mask_col),
                c_size_t(n), c_double(_capi.HDCD_DEFAULT_COPULA_EPSILON), _capi._dptr(u_col),
            )
            check(status)
            U[:, j] = u_col
        return U

    def copula_logpdf(self, u):
        """log c_G(u) row-wise (spec section 14). Rows with any NaN are
        reported as NaN (spec section 16: full likelihood under missing
        coordinates is out of scope for v1)."""
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            u = u[np.newaxis, :]
        n, d = u.shape
        if d != self.d:
            raise ValueError(f"expected {self.d} columns, got {d}")
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            row = u[i]
            if np.any(np.isnan(row)):
                out[i] = np.nan
                continue
            out[i] = self._dag_fit.joint_log_density(row)
        return out

    def logpdf(self, X):
        """log f_X(x) = log c_G(u) + sum_j log f_j(x_j) (spec section 35)."""
        X = _as_column_major(X)
        n, d = X.shape
        if d != self.d:
            raise ValueError(f"expected {self.d} columns, got {d}")

        marginal_term = np.zeros(n, dtype=np.float64)
        any_missing = np.zeros(n, dtype=bool)
        for j in range(d):
            col = X[:, j]
            missing = np.isnan(col)
            any_missing |= missing
            observed = ~missing
            if np.any(observed):
                marginal_term[observed] += self._marginals[j].logpdf(col[observed])

        u = self.transform(X)
        copula_term = self.copula_logpdf(u)

        result = copula_term + marginal_term
        result[any_missing] = np.nan
        return result

    def sample(self, n_samples):
        """NOT IMPLEMENTED: the C core has no sampling routine.

        Spec section 21 names hdcd_sample in its architecture sketch,
        and spec section 25 shows model.sample(...) as preferred Python
        usage, but no version-1 milestone (spec section 31) actually
        schedules implementing it in the C core -- Milestones 1-9 build
        the density-estimation pipeline, Milestones 10-12 are language
        bindings, 13 is the EVT module, 14 is performance. Rather than
        fabricate a sampler in the Python layer (which would violate
        spec section 36 rule 11: "Do not push core numerical
        optimization into the language wrappers"), this raises clearly.
        See DECISIONS.md.
        """
        raise NotImplementedError(
            "hdcd_sample is not implemented in the C core (see HdcdModel.sample's "
            "docstring and DECISIONS.md for why)."
        )

    def fit_dag(self, edges):
        """Fit an alternative candidate DAG (given as an iterable of
        (parent, child) pairs) over the SAME training data this model
        was fit on, and compare it to the reference DAG via held-out KL
        (spec section 19).

        IMPORTANT: this is a purely statistical comparison of
        distributional fit. It does not establish causal direction.
        """
        dag = Dag.from_edges(self.d, self._k_max, edges)
        dag_fit = DagFitResult.fit(self._u, self._mask, dag, self._local_fit_options)
        kl_div = _capi.lib.hdcd_dag_fit_kl_difference(dag_fit._ptr, self._dag_fit._ptr)
        return FitDagResult(dag, dag_fit, kl_div)


def fit(
    X,
    max_parents=2,
    bernstein_degree=3,
    lambda_edge=0.05,
    lambda_roughness=0.15,
    holdout_fraction=0.25,
    seed=0,
    initial_temperature=0.5,
    cooling_rate=0.95,
    annealing_iterations=150,
    annealing_restarts=1,
    p_add=1.0,
    p_remove=1.0,
    p_swap=1.0,
):
    """Fit the full pipeline (spec section 1's pipeline diagram, driven
    end to end): marginals -> copula transform -> dependence matrix ->
    persistent-topology ordering -> simulated-annealing DAG search ->
    fixed-DAG fitting. Returns a fitted HdcdModel.
    """
    X = _as_column_major(X)
    n, d = X.shape
    mask = _mask_from_nan(X)

    marginals = [Marginal.fit(X[:, j], mask[:, j]) for j in range(d)]

    U = np.empty((n, d), dtype=np.float64, order="F")
    for j in range(d):
        x_col = np.ascontiguousarray(X[:, j])
        mask_col = np.ascontiguousarray(mask[:, j])
        u_col = np.empty(n, dtype=np.float64)
        status = _capi.lib.hdcd_transform_to_copula(
            marginals[j]._ptr, _capi._dptr(x_col), _capi._u8ptr(mask_col),
            c_size_t(n), c_double(_capi.HDCD_DEFAULT_COPULA_EPSILON), _capi._dptr(u_col),
        )
        check(status)
        U[:, j] = u_col

    dependence_matrix = DependenceMatrix.compute(U, mask)
    topology = Topology.compute(dependence_matrix)
    ordering = np.ascontiguousarray(topology.ordering(), dtype=np.uintp)

    sinkhorn_options = HdcdSinkhornOptions(n_quadrature_nodes=0, tol=0.0, max_iterations=0)
    local_fit_options = HdcdLocalFitOptions(
        bernstein_degree=bernstein_degree,
        lambda_roughness=lambda_roughness,
        holdout_fraction=holdout_fraction,
        seed=seed + 1,
        theta_max_iterations=0,
        theta_tol=0.0,
        sinkhorn_options=sinkhorn_options,
    )

    annealing_options = HdcdAnnealingOptions(
        k_max=max_parents,
        lambda_edge=lambda_edge,
        ordering=_capi._sptr(ordering),
        local_fit_options=local_fit_options,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        max_iterations=annealing_iterations,
        restarts=annealing_restarts,
        p_add=p_add,
        p_remove=p_remove,
        p_swap=p_swap,
        seed=seed,
        initial_dag=None,
    )

    result_ptr = c_void_p()
    status = _capi.lib.hdcd_run_annealing(
        _capi._dptr(U), _capi._u8ptr(mask), c_size_t(n), c_size_t(d),
        byref(annealing_options), byref(result_ptr),
    )
    check(status, allow_not_converged=True)

    best_dag_borrowed = _capi.lib.hdcd_annealing_best_dag(result_ptr)
    best_dag_clone_ptr = c_void_p()
    check(_capi.lib.hdcd_dag_clone(best_dag_borrowed, byref(best_dag_clone_ptr)))
    _capi.lib.hdcd_annealing_result_free(result_ptr)
    reference_dag = Dag(best_dag_clone_ptr)

    dag_fit = DagFitResult.fit(U, mask, reference_dag, local_fit_options)

    model = HdcdModel(marginals, reference_dag, dag_fit, dependence_matrix, topology, max_parents, local_fit_options)
    # Retained for fit_dag()'s alternative-DAG comparisons against the
    # SAME training data (spec section 19).
    model._u = U
    model._mask = mask
    return model
