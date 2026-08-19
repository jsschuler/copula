"""
Low-level ctypes bindings to the hdcd C shared library.

This module is deliberately as thin and mechanical as possible (spec
section 25: "The Python layer should be thin"; section 36 rule 11: "Do
not push core numerical optimization into the language wrappers") --
every struct layout and function signature here must match the C
headers under include/hdcd/ EXACTLY (field order, including padding
rules), or the ABI breaks silently. It binds only what the high-level
hdcd/_model.py wrapper actually needs to drive the documented Python
API (spec section 25's example usage), not the library's entire public
C surface.
"""

import ctypes
import os
import platform
import sys
from ctypes import (
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_uint8,
    c_uint64,
    c_void_p,
    POINTER,
    Structure,
)


# ---- library loading ------------------------------------------------

def _candidate_library_paths():
    lib_name = "libhdcd.dylib" if platform.system() == "Darwin" else "libhdcd.so"
    here = os.path.dirname(os.path.abspath(__file__))

    env_override = os.environ.get("HDCD_LIBRARY_PATH")
    if env_override:
        yield env_override

    # Bundled inside the installed package (normal `pip install` path).
    yield os.path.join(here, lib_name)

    # Development fallback: repo checkout with the C library built via
    # `make shared` but the Python package not (yet) installed.
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    yield os.path.join(repo_root, "build", lib_name)


def _load_library():
    tried = []
    for path in _candidate_library_paths():
        tried.append(path)
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise OSError(
        "could not find the hdcd shared library. Tried:\n  "
        + "\n  ".join(tried)
        + "\nBuild it with `make shared` in the repository root, or set "
        "the HDCD_LIBRARY_PATH environment variable to point at "
        "libhdcd.dylib/libhdcd.so directly."
    )


_lib = _load_library()


# ---- status codes (hdcd/status.h) ------------------------------------

HDCD_OK = 0
HDCD_ERROR_INVALID_ARGUMENT = 1
HDCD_ERROR_NOT_CONVERGED = 2
HDCD_ERROR_NUMERICAL = 3
HDCD_ERROR_ALLOCATION = 4
HDCD_ERROR_UNSUPPORTED = 5

_STATUS_NAMES = {
    HDCD_OK: "ok",
    HDCD_ERROR_INVALID_ARGUMENT: "invalid argument",
    HDCD_ERROR_NOT_CONVERGED: "not converged",
    HDCD_ERROR_NUMERICAL: "numerical error",
    HDCD_ERROR_ALLOCATION: "allocation failure",
    HDCD_ERROR_UNSUPPORTED: "unsupported",
}


class HdcdError(RuntimeError):
    """Raised when a hdcd C call returns a hard-failure status code.

    HDCD_ERROR_NOT_CONVERGED is usually NOT raised as an HdcdError by
    the high-level wrapper (spec section 24: the C API returns a
    best-effort populated result on non-convergence, not a null
    handle) -- callers check convergence via the relevant `*_converged`
    accessor/attribute instead. This exception is for HARD failures.
    """


def status_message(status):
    return _lib.hdcd_status_message(c_int(status)).decode("utf-8")


def check(status, *, allow_not_converged=False):
    if status == HDCD_OK:
        return
    if allow_not_converged and status == HDCD_ERROR_NOT_CONVERGED:
        return
    raise HdcdError(f"{_STATUS_NAMES.get(status, 'unknown status')} ({status}): {status_message(status)}")


_lib.hdcd_status_message.argtypes = [c_int]
_lib.hdcd_status_message.restype = c_char_p


# ---- structs ------------------------------------------------------------

class HdcdBandwidthResult(Structure):
    _fields_ = [
        ("sigma", c_double),
        ("eta", c_double),
        ("loglik", c_double),
        ("lower", c_double),
        ("upper", c_double),
        ("iterations", c_int),
        ("converged", c_int),
        ("status", c_int),
    ]


class HdcdSinkhornOptions(Structure):
    _fields_ = [
        ("n_quadrature_nodes", c_size_t),
        ("tol", c_double),
        ("max_iterations", c_int),
    ]


class HdcdLocalFitOptions(Structure):
    _fields_ = [
        ("bernstein_degree", c_size_t),
        ("lambda_roughness", c_double),
        ("holdout_fraction", c_double),
        ("seed", c_uint64),
        ("theta_max_iterations", c_size_t),
        ("theta_tol", c_double),
        ("sinkhorn_options", HdcdSinkhornOptions),
    ]


class HdcdAnnealingOptions(Structure):
    _fields_ = [
        ("k_max", c_size_t),
        ("lambda_edge", c_double),
        ("ordering", POINTER(c_size_t)),
        ("local_fit_options", HdcdLocalFitOptions),
        ("initial_temperature", c_double),
        ("cooling_rate", c_double),
        ("max_iterations", c_size_t),
        ("restarts", c_size_t),
        ("p_add", c_double),
        ("p_remove", c_double),
        ("p_swap", c_double),
        ("seed", c_uint64),
        ("initial_dag", c_void_p),
    ]


class HdcdMstEdge(Structure):
    _fields_ = [
        ("j", c_size_t),
        ("k", c_size_t),
        ("weight", c_double),
    ]


def _dptr(arr):
    """A ctypes double* view of a contiguous float64 numpy array."""
    return arr.ctypes.data_as(POINTER(c_double))


def _sptr(arr):
    """A ctypes size_t* view of a contiguous size_t (or compatible int) numpy array."""
    return arr.ctypes.data_as(POINTER(c_size_t))


def _u8ptr(arr):
    return arr.ctypes.data_as(POINTER(c_uint8))


# ---- marginal (hdcd/marginal.h) ------------------------------------------

_lib.hdcd_marginal_fit.argtypes = [
    POINTER(c_double), POINTER(c_uint8), c_size_t,
    c_double, c_double, c_double, c_int,
    POINTER(c_void_p),
]
_lib.hdcd_marginal_fit.restype = c_int

_lib.hdcd_marginal_free.argtypes = [c_void_p]
_lib.hdcd_marginal_free.restype = None

_lib.hdcd_marginal_n_observed.argtypes = [c_void_p]
_lib.hdcd_marginal_n_observed.restype = c_size_t

_lib.hdcd_marginal_bandwidth_result.argtypes = [c_void_p]
_lib.hdcd_marginal_bandwidth_result.restype = HdcdBandwidthResult

_lib.hdcd_marginal_cdf.argtypes = [c_void_p, POINTER(c_double), c_size_t, POINTER(c_double)]
_lib.hdcd_marginal_cdf.restype = c_int

_lib.hdcd_marginal_logpdf.argtypes = [c_void_p, POINTER(c_double), c_size_t, POINTER(c_double)]
_lib.hdcd_marginal_logpdf.restype = c_int


# ---- copula transform (hdcd/copula.h) ------------------------------------

_lib.hdcd_transform_to_copula.argtypes = [
    c_void_p, POINTER(c_double), POINTER(c_uint8), c_size_t, c_double, POINTER(c_double),
]
_lib.hdcd_transform_to_copula.restype = c_int

HDCD_DEFAULT_COPULA_EPSILON = 1e-9


# ---- dependence matrix (hdcd/dcor.h) -------------------------------------

_lib.hdcd_compute_dependence_matrix.argtypes = [
    POINTER(c_double), POINTER(c_uint8), c_size_t, c_size_t, POINTER(c_void_p),
]
_lib.hdcd_compute_dependence_matrix.restype = c_int

_lib.hdcd_dependence_matrix_free.argtypes = [c_void_p]
_lib.hdcd_dependence_matrix_free.restype = None

_lib.hdcd_dependence_matrix_dim.argtypes = [c_void_p]
_lib.hdcd_dependence_matrix_dim.restype = c_size_t

_lib.hdcd_dependence_matrix_get.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_dependence_matrix_get.restype = c_double

_lib.hdcd_dependence_matrix_n_effective.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_dependence_matrix_n_effective.restype = c_size_t


# ---- topology (hdcd/topology.h) ------------------------------------------

_lib.hdcd_compute_topology.argtypes = [c_void_p, POINTER(c_void_p)]
_lib.hdcd_compute_topology.restype = c_int

_lib.hdcd_topology_free.argtypes = [c_void_p]
_lib.hdcd_topology_free.restype = None

_lib.hdcd_topology_dim.argtypes = [c_void_p]
_lib.hdcd_topology_dim.restype = c_size_t

_lib.hdcd_topology_mst_edge_count.argtypes = [c_void_p]
_lib.hdcd_topology_mst_edge_count.restype = c_size_t

_lib.hdcd_topology_mst_edge.argtypes = [c_void_p, c_size_t]
_lib.hdcd_topology_mst_edge.restype = HdcdMstEdge

_lib.hdcd_topology_merge_level.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_topology_merge_level.restype = c_double

_lib.hdcd_topology_affinity.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_topology_affinity.restype = c_double

_lib.hdcd_topology_ordering.argtypes = [c_void_p]
_lib.hdcd_topology_ordering.restype = POINTER(c_size_t)


# ---- DAG (hdcd/dag.h) -----------------------------------------------------

_lib.hdcd_dag_create.argtypes = [c_size_t, c_size_t, POINTER(c_void_p)]
_lib.hdcd_dag_create.restype = c_int

_lib.hdcd_dag_free.argtypes = [c_void_p]
_lib.hdcd_dag_free.restype = None

_lib.hdcd_dag_dim.argtypes = [c_void_p]
_lib.hdcd_dag_dim.restype = c_size_t

_lib.hdcd_dag_k_max.argtypes = [c_void_p]
_lib.hdcd_dag_k_max.restype = c_size_t

_lib.hdcd_dag_add_edge.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_dag_add_edge.restype = c_int

_lib.hdcd_dag_remove_edge.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_dag_remove_edge.restype = c_int

_lib.hdcd_dag_has_edge.argtypes = [c_void_p, c_size_t, c_size_t]
_lib.hdcd_dag_has_edge.restype = c_int

_lib.hdcd_dag_n_parents.argtypes = [c_void_p, c_size_t]
_lib.hdcd_dag_n_parents.restype = c_size_t

_lib.hdcd_dag_parents.argtypes = [c_void_p, c_size_t, POINTER(c_size_t)]
_lib.hdcd_dag_parents.restype = c_int

_lib.hdcd_dag_topological_order.argtypes = [c_void_p, POINTER(c_size_t)]
_lib.hdcd_dag_topological_order.restype = c_int

_lib.hdcd_dag_clone.argtypes = [c_void_p, POINTER(c_void_p)]
_lib.hdcd_dag_clone.restype = c_int

_lib.hdcd_dag_from_edges.argtypes = [
    c_size_t, c_size_t, POINTER(c_size_t), POINTER(c_size_t), c_size_t, POINTER(c_void_p),
]
_lib.hdcd_dag_from_edges.restype = c_int


# ---- local fit (hdcd/local_fit.h) -----------------------------------------

_lib.hdcd_local_fit_n_parents.argtypes = [c_void_p]
_lib.hdcd_local_fit_n_parents.restype = c_size_t

_lib.hdcd_local_fit_parent_order.argtypes = [c_void_p]
_lib.hdcd_local_fit_parent_order.restype = POINTER(c_size_t)

_lib.hdcd_local_fit_n_observed.argtypes = [c_void_p]
_lib.hdcd_local_fit_n_observed.restype = c_size_t

_lib.hdcd_local_fit_n_train.argtypes = [c_void_p]
_lib.hdcd_local_fit_n_train.restype = c_size_t

_lib.hdcd_local_fit_n_holdout.argtypes = [c_void_p]
_lib.hdcd_local_fit_n_holdout.restype = c_size_t

_lib.hdcd_local_fit_holdout_score.argtypes = [c_void_p]
_lib.hdcd_local_fit_holdout_score.restype = c_double

_lib.hdcd_local_fit_roughness_penalty.argtypes = [c_void_p]
_lib.hdcd_local_fit_roughness_penalty.restype = c_double

_lib.hdcd_local_fit_theta_converged.argtypes = [c_void_p]
_lib.hdcd_local_fit_theta_converged.restype = c_int

_lib.hdcd_local_fit_sinkhorn_converged.argtypes = [c_void_p]
_lib.hdcd_local_fit_sinkhorn_converged.restype = c_int

_lib.hdcd_local_fit_log_density.argtypes = [c_void_p, c_double, POINTER(c_double), c_size_t, POINTER(c_double)]
_lib.hdcd_local_fit_log_density.restype = c_int


# ---- dag fit (hdcd/dag_fit.h) ---------------------------------------------

_lib.hdcd_dag_fit.argtypes = [
    POINTER(c_double), POINTER(c_uint8), c_size_t, c_size_t,
    c_void_p, POINTER(HdcdLocalFitOptions), POINTER(c_void_p),
]
_lib.hdcd_dag_fit.restype = c_int

_lib.hdcd_dag_fit_free.argtypes = [c_void_p]
_lib.hdcd_dag_fit_free.restype = None

_lib.hdcd_dag_fit_dim.argtypes = [c_void_p]
_lib.hdcd_dag_fit_dim.restype = c_size_t

_lib.hdcd_dag_fit_node.argtypes = [c_void_p, c_size_t]
_lib.hdcd_dag_fit_node.restype = c_void_p

_lib.hdcd_dag_fit_node_converged.argtypes = [c_void_p, c_size_t]
_lib.hdcd_dag_fit_node_converged.restype = c_int

_lib.hdcd_dag_fit_all_converged.argtypes = [c_void_p]
_lib.hdcd_dag_fit_all_converged.restype = c_int

_lib.hdcd_dag_fit_joint_log_density.argtypes = [c_void_p, POINTER(c_double), c_size_t, POINTER(c_double)]
_lib.hdcd_dag_fit_joint_log_density.restype = c_int

_lib.hdcd_dag_fit_kl_estimate.argtypes = [c_void_p]
_lib.hdcd_dag_fit_kl_estimate.restype = c_double

_lib.hdcd_dag_fit_kl_difference.argtypes = [c_void_p, c_void_p]
_lib.hdcd_dag_fit_kl_difference.restype = c_double


# ---- annealing (hdcd/annealing.h) -----------------------------------------

_lib.hdcd_run_annealing.argtypes = [
    POINTER(c_double), POINTER(c_uint8), c_size_t, c_size_t,
    POINTER(HdcdAnnealingOptions), POINTER(c_void_p),
]
_lib.hdcd_run_annealing.restype = c_int

_lib.hdcd_annealing_result_free.argtypes = [c_void_p]
_lib.hdcd_annealing_result_free.restype = None

_lib.hdcd_annealing_best_dag.argtypes = [c_void_p]
_lib.hdcd_annealing_best_dag.restype = c_void_p

_lib.hdcd_annealing_best_score.argtypes = [c_void_p]
_lib.hdcd_annealing_best_score.restype = c_double

_lib.hdcd_annealing_current_dag.argtypes = [c_void_p]
_lib.hdcd_annealing_current_dag.restype = c_void_p

_lib.hdcd_annealing_current_score.argtypes = [c_void_p]
_lib.hdcd_annealing_current_score.restype = c_double

_lib.hdcd_annealing_n_iterations.argtypes = [c_void_p]
_lib.hdcd_annealing_n_iterations.restype = c_size_t

_lib.hdcd_annealing_score_trace.argtypes = [c_void_p, c_size_t]
_lib.hdcd_annealing_score_trace.restype = c_double

_lib.hdcd_annealing_accepted_trace.argtypes = [c_void_p, c_size_t]
_lib.hdcd_annealing_accepted_trace.restype = c_int

_lib.hdcd_annealing_acceptance_rate.argtypes = [c_void_p]
_lib.hdcd_annealing_acceptance_rate.restype = c_double


lib = _lib
