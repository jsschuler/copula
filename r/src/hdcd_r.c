/*
 * .Call glue between R and the hdcd C library (spec section 26: "Use
 * .Call, not .C"). Deliberately mechanical -- each function converts
 * SEXPs to/from plain C types and calls straight into the hdcd public
 * headers; no new statistical logic lives here (spec section 36 rule 11).
 *
 * Conventions used throughout:
 *   - Opaque hdcd handles are wrapped as R external pointers with a
 *     registered C finalizer (spec section 26: "use external pointers
 *     with finalizers for C model handles"), so R's garbage collector
 *     frees the underlying C object automatically.
 *   - NA_real_ (and any other NaN) is converted to an explicit uint8_t
 *     observed mask at this wrapper boundary (spec section 26), never
 *     passed through as a sentinel into the C core (spec section 23).
 *   - R matrices are already stored column-major (R's native layout),
 *     so REAL(matrix_sexp) is passed straight through as hdcd's
 *     expected column-major double* with no transposition or copy
 *     (spec section 26: "preserve column-major memory where possible").
 *   - hdcd uses 0-indexed dimensions/nodes; R is 1-indexed. Conversion
 *     happens at this boundary, consistently, in both directions.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "hdcd/hdcd.h"

#include <stdlib.h>
#include <string.h>

/* ---- external pointer / finalizer plumbing ---------------------------- */

static void marginal_finalizer(SEXP ext) {
    hdcd_marginal_t *ptr = (hdcd_marginal_t *)R_ExternalPtrAddr(ext);
    if (ptr != NULL) {
        hdcd_marginal_free(ptr);
        R_ClearExternalPtr(ext);
    }
}

static void dependence_matrix_finalizer(SEXP ext) {
    hdcd_dependence_matrix_t *ptr = (hdcd_dependence_matrix_t *)R_ExternalPtrAddr(ext);
    if (ptr != NULL) {
        hdcd_dependence_matrix_free(ptr);
        R_ClearExternalPtr(ext);
    }
}

static void topology_finalizer(SEXP ext) {
    hdcd_topology_t *ptr = (hdcd_topology_t *)R_ExternalPtrAddr(ext);
    if (ptr != NULL) {
        hdcd_topology_free(ptr);
        R_ClearExternalPtr(ext);
    }
}

static void dag_finalizer(SEXP ext) {
    hdcd_dag_t *ptr = (hdcd_dag_t *)R_ExternalPtrAddr(ext);
    if (ptr != NULL) {
        hdcd_dag_free(ptr);
        R_ClearExternalPtr(ext);
    }
}

static void dag_fit_finalizer(SEXP ext) {
    hdcd_dag_fit_t *ptr = (hdcd_dag_fit_t *)R_ExternalPtrAddr(ext);
    if (ptr != NULL) {
        hdcd_dag_fit_free(ptr);
        R_ClearExternalPtr(ext);
    }
}

static SEXP wrap_pointer(void *ptr, R_CFinalizer_t finalizer, const char *tag) {
    SEXP ext = PROTECT(R_MakeExternalPtr(ptr, Rf_install(tag), R_NilValue));
    R_RegisterCFinalizerEx(ext, finalizer, TRUE);
    UNPROTECT(1);
    return ext;
}

static void *unwrap_pointer(SEXP ext, const char *what) {
    void *ptr = R_ExternalPtrAddr(ext);
    if (ptr == NULL) {
        Rf_error("%s pointer is NULL (already freed, or invalid handle)", what);
    }
    return ptr;
}

static void check_status(hdcd_status_t status, int allow_not_converged) {
    if (status == HDCD_OK) {
        return;
    }
    if (allow_not_converged && status == HDCD_ERROR_NOT_CONVERGED) {
        return;
    }
    Rf_error("hdcd error: %s", hdcd_status_message(status));
}

/* NA_real_ and any other NaN both mean "missing" (spec section 26). */
static uint8_t *mask_from_na(const double *x, size_t n) {
    uint8_t *mask = (uint8_t *)malloc(n * sizeof(uint8_t));
    if (mask == NULL) {
        Rf_error("allocation failure building observed mask");
    }
    for (size_t i = 0; i < n; i++) {
        mask[i] = ISNAN(x[i]) ? 0 : 1;
    }
    return mask;
}

/* ---- marginal (hdcd/marginal.h) --------------------------------------- */

SEXP hdcd_r_marginal_fit(SEXP x, SEXP sigma_min, SEXP sigma_max, SEXP tol, SEXP max_iter) {
    size_t n = (size_t)Rf_xlength(x);
    const double *xp = REAL(x);
    uint8_t *mask = mask_from_na(xp, n);

    hdcd_marginal_t *marginal = NULL;
    hdcd_status_t status = hdcd_marginal_fit(
        xp, mask, n,
        Rf_asReal(sigma_min), Rf_asReal(sigma_max), Rf_asReal(tol), Rf_asInteger(max_iter),
        &marginal
    );
    free(mask);
    check_status(status, 0);

    return wrap_pointer(marginal, marginal_finalizer, "hdcd_marginal");
}

SEXP hdcd_r_marginal_cdf(SEXP marginal_ext, SEXP eval_points) {
    hdcd_marginal_t *marginal = (hdcd_marginal_t *)unwrap_pointer(marginal_ext, "marginal");
    size_t m = (size_t)Rf_xlength(eval_points);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)m));
    hdcd_status_t status = hdcd_marginal_cdf(marginal, REAL(eval_points), m, REAL(out));
    UNPROTECT(1);
    check_status(status, 0);
    return out;
}

SEXP hdcd_r_marginal_logpdf(SEXP marginal_ext, SEXP eval_points) {
    hdcd_marginal_t *marginal = (hdcd_marginal_t *)unwrap_pointer(marginal_ext, "marginal");
    size_t m = (size_t)Rf_xlength(eval_points);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)m));
    hdcd_status_t status = hdcd_marginal_logpdf(marginal, REAL(eval_points), m, REAL(out));
    UNPROTECT(1);
    check_status(status, 0);
    return out;
}

SEXP hdcd_r_marginal_sigma(SEXP marginal_ext) {
    hdcd_marginal_t *marginal = (hdcd_marginal_t *)unwrap_pointer(marginal_ext, "marginal");
    hdcd_bandwidth_result_t bw = hdcd_marginal_bandwidth_result(marginal);
    return Rf_ScalarReal(bw.sigma);
}

/* ---- copula transform (hdcd/copula.h) --------------------------------- */

SEXP hdcd_r_transform_to_copula(SEXP marginal_ext, SEXP x, SEXP epsilon) {
    hdcd_marginal_t *marginal = (hdcd_marginal_t *)unwrap_pointer(marginal_ext, "marginal");
    size_t n = (size_t)Rf_xlength(x);
    const double *xp = REAL(x);
    uint8_t *mask = mask_from_na(xp, n);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n));
    hdcd_status_t status = hdcd_transform_to_copula(marginal, xp, mask, n, Rf_asReal(epsilon), REAL(out));
    free(mask);
    UNPROTECT(1);
    check_status(status, 0);
    return out;
}

/* ---- dependence matrix / topology (hdcd/dcor.h, hdcd/topology.h) ------ */

SEXP hdcd_r_compute_dependence_matrix(SEXP u, SEXP n, SEXP d) {
    size_t nn = (size_t)Rf_asInteger(n);
    size_t dd = (size_t)Rf_asInteger(d);
    const double *up = REAL(u); /* R matrix: already column-major */
    uint8_t *mask = mask_from_na(up, nn * dd);

    hdcd_dependence_matrix_t *dm = NULL;
    hdcd_status_t status = hdcd_compute_dependence_matrix(up, mask, nn, dd, &dm);
    free(mask);
    check_status(status, 0);

    return wrap_pointer(dm, dependence_matrix_finalizer, "hdcd_dependence_matrix");
}

SEXP hdcd_r_dependence_matrix_dense(SEXP dm_ext, SEXP d) {
    hdcd_dependence_matrix_t *dm = (hdcd_dependence_matrix_t *)unwrap_pointer(dm_ext, "dependence_matrix");
    size_t dd = (size_t)Rf_asInteger(d);
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int)dd, (int)dd));
    double *outp = REAL(out);
    for (size_t j = 0; j < dd; j++) {
        for (size_t k = 0; k < dd; k++) {
            outp[j + k * dd] = hdcd_dependence_matrix_get(dm, j, k); /* column-major */
        }
    }
    UNPROTECT(1);
    return out;
}

SEXP hdcd_r_compute_topology(SEXP dm_ext) {
    hdcd_dependence_matrix_t *dm = (hdcd_dependence_matrix_t *)unwrap_pointer(dm_ext, "dependence_matrix");
    hdcd_topology_t *topo = NULL;
    hdcd_status_t status = hdcd_compute_topology(dm, &topo);
    check_status(status, 0);
    return wrap_pointer(topo, topology_finalizer, "hdcd_topology");
}

SEXP hdcd_r_topology_ordering(SEXP topo_ext) {
    hdcd_topology_t *topo = (hdcd_topology_t *)unwrap_pointer(topo_ext, "topology");
    size_t d = hdcd_topology_dim(topo);
    const size_t *ordering = hdcd_topology_ordering(topo);
    SEXP out = PROTECT(Rf_allocVector(INTSXP, (int)d));
    int *outp = INTEGER(out);
    for (size_t i = 0; i < d; i++) {
        outp[i] = (int)ordering[i] + 1; /* 1-indexed */
    }
    UNPROTECT(1);
    return out;
}

/* ---- DAG (hdcd/dag.h) --------------------------------------------------- */

SEXP hdcd_r_dag_create(SEXP d, SEXP k_max) {
    hdcd_dag_t *dag = NULL;
    hdcd_status_t status = hdcd_dag_create((size_t)Rf_asInteger(d), (size_t)Rf_asInteger(k_max), &dag);
    check_status(status, 0);
    return wrap_pointer(dag, dag_finalizer, "hdcd_dag");
}

SEXP hdcd_r_dag_from_edges(SEXP d, SEXP k_max, SEXP parents_1idx, SEXP children_1idx) {
    size_t n_edges = (size_t)Rf_xlength(parents_1idx);
    size_t *parents = NULL;
    size_t *children = NULL;
    if (n_edges > 0) {
        parents = (size_t *)malloc(n_edges * sizeof(size_t));
        children = (size_t *)malloc(n_edges * sizeof(size_t));
        int *pp = INTEGER(parents_1idx);
        int *cp = INTEGER(children_1idx);
        for (size_t i = 0; i < n_edges; i++) {
            parents[i] = (size_t)(pp[i] - 1);
            children[i] = (size_t)(cp[i] - 1);
        }
    }
    hdcd_dag_t *dag = NULL;
    hdcd_status_t status = hdcd_dag_from_edges(
        (size_t)Rf_asInteger(d), (size_t)Rf_asInteger(k_max), parents, children, n_edges, &dag
    );
    free(parents);
    free(children);
    check_status(status, 0);
    return wrap_pointer(dag, dag_finalizer, "hdcd_dag");
}

SEXP hdcd_r_dag_add_edge(SEXP dag_ext, SEXP parent_1idx, SEXP child_1idx) {
    hdcd_dag_t *dag = (hdcd_dag_t *)unwrap_pointer(dag_ext, "dag");
    hdcd_status_t status = hdcd_dag_add_edge(
        dag, (size_t)(Rf_asInteger(parent_1idx) - 1), (size_t)(Rf_asInteger(child_1idx) - 1)
    );
    check_status(status, 0);
    return R_NilValue;
}

SEXP hdcd_r_dag_clone(SEXP dag_ext) {
    hdcd_dag_t *dag = (hdcd_dag_t *)unwrap_pointer(dag_ext, "dag");
    hdcd_dag_t *clone = NULL;
    hdcd_status_t status = hdcd_dag_clone(dag, &clone);
    check_status(status, 0);
    return wrap_pointer(clone, dag_finalizer, "hdcd_dag");
}

SEXP hdcd_r_dag_edges(SEXP dag_ext, SEXP d) {
    hdcd_dag_t *dag = (hdcd_dag_t *)unwrap_pointer(dag_ext, "dag");
    size_t dd = (size_t)Rf_asInteger(d);

    size_t total = 0;
    for (size_t c = 0; c < dd; c++) {
        total += hdcd_dag_n_parents(dag, c);
    }

    SEXP out = PROTECT(Rf_allocMatrix(INTSXP, (int)total, 2));
    int *outp = INTEGER(out);
    size_t idx = 0;
    for (size_t c = 0; c < dd; c++) {
        size_t np = hdcd_dag_n_parents(dag, c);
        if (np == 0) {
            continue;
        }
        size_t *parents = (size_t *)malloc(np * sizeof(size_t));
        hdcd_dag_parents(dag, c, parents);
        for (size_t i = 0; i < np; i++) {
            outp[idx] = (int)parents[i] + 1;         /* column 1: parent */
            outp[idx + total] = (int)c + 1;            /* column 2: child */
            idx++;
        }
        free(parents);
    }
    UNPROTECT(1);
    return out;
}

/* ---- DAG fit (hdcd/dag_fit.h) ------------------------------------------- */

SEXP hdcd_r_dag_fit(
    SEXP u, SEXP n, SEXP d, SEXP dag_ext,
    SEXP bernstein_degree, SEXP lambda_roughness, SEXP holdout_fraction, SEXP seed,
    SEXP theta_max_iterations, SEXP theta_tol
) {
    hdcd_dag_t *dag = (hdcd_dag_t *)unwrap_pointer(dag_ext, "dag");
    size_t nn = (size_t)Rf_asInteger(n);
    size_t dd = (size_t)Rf_asInteger(d);
    const double *up = REAL(u);
    uint8_t *mask = mask_from_na(up, nn * dd);

    hdcd_local_fit_options_t options;
    memset(&options, 0, sizeof(options));
    options.bernstein_degree = (size_t)Rf_asInteger(bernstein_degree);
    options.lambda_roughness = Rf_asReal(lambda_roughness);
    options.holdout_fraction = Rf_asReal(holdout_fraction);
    options.seed = (uint64_t)Rf_asInteger(seed);
    options.theta_max_iterations = (size_t)Rf_asInteger(theta_max_iterations);
    options.theta_tol = Rf_asReal(theta_tol);
    /* sinkhorn_options left zeroed: library defaults. */

    hdcd_dag_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_dag_fit(up, mask, nn, dd, dag, &options, &fit);
    free(mask);
    check_status(status, 1); /* non-convergence still returns a usable, populated fit */

    return wrap_pointer(fit, dag_fit_finalizer, "hdcd_dag_fit");
}

SEXP hdcd_r_dag_fit_joint_log_density(SEXP dag_fit_ext, SEXP u_point) {
    hdcd_dag_fit_t *fit = (hdcd_dag_fit_t *)unwrap_pointer(dag_fit_ext, "dag_fit");
    size_t d = (size_t)Rf_xlength(u_point);
    double out;
    hdcd_status_t status = hdcd_dag_fit_joint_log_density(fit, REAL(u_point), d, &out);
    check_status(status, 0);
    return Rf_ScalarReal(out);
}

SEXP hdcd_r_dag_fit_holdout_scores(SEXP dag_fit_ext, SEXP d) {
    hdcd_dag_fit_t *fit = (hdcd_dag_fit_t *)unwrap_pointer(dag_fit_ext, "dag_fit");
    size_t dd = (size_t)Rf_asInteger(d);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, (int)dd));
    double *outp = REAL(out);
    for (size_t j = 0; j < dd; j++) {
        const hdcd_local_fit_t *node = hdcd_dag_fit_node(fit, j);
        outp[j] = hdcd_local_fit_holdout_score(node);
    }
    UNPROTECT(1);
    return out;
}

SEXP hdcd_r_dag_fit_all_converged(SEXP dag_fit_ext) {
    hdcd_dag_fit_t *fit = (hdcd_dag_fit_t *)unwrap_pointer(dag_fit_ext, "dag_fit");
    return Rf_ScalarLogical(hdcd_dag_fit_all_converged(fit));
}

SEXP hdcd_r_dag_fit_kl_estimate(SEXP dag_fit_ext) {
    hdcd_dag_fit_t *fit = (hdcd_dag_fit_t *)unwrap_pointer(dag_fit_ext, "dag_fit");
    return Rf_ScalarReal(hdcd_dag_fit_kl_estimate(fit));
}

SEXP hdcd_r_dag_fit_kl_difference(SEXP candidate_ext, SEXP reference_ext) {
    hdcd_dag_fit_t *candidate = (hdcd_dag_fit_t *)unwrap_pointer(candidate_ext, "dag_fit (candidate)");
    hdcd_dag_fit_t *reference = (hdcd_dag_fit_t *)unwrap_pointer(reference_ext, "dag_fit (reference)");
    return Rf_ScalarReal(hdcd_dag_fit_kl_difference(candidate, reference));
}

/* ---- annealing (hdcd/annealing.h) ---------------------------------------- */

SEXP hdcd_r_run_annealing(
    SEXP u, SEXP n, SEXP d, SEXP ordering_1idx,
    SEXP k_max, SEXP lambda_edge,
    SEXP bernstein_degree, SEXP lambda_roughness, SEXP holdout_fraction, SEXP local_seed,
    SEXP initial_temperature, SEXP cooling_rate, SEXP max_iterations, SEXP restarts,
    SEXP p_add, SEXP p_remove, SEXP p_swap, SEXP anneal_seed
) {
    size_t nn = (size_t)Rf_asInteger(n);
    size_t dd = (size_t)Rf_asInteger(d);
    const double *up = REAL(u);
    uint8_t *mask = mask_from_na(up, nn * dd);

    size_t *ordering = (size_t *)malloc(dd * sizeof(size_t));
    int *ord_p = INTEGER(ordering_1idx);
    for (size_t i = 0; i < dd; i++) {
        ordering[i] = (size_t)(ord_p[i] - 1);
    }

    hdcd_local_fit_options_t local_fit_options;
    memset(&local_fit_options, 0, sizeof(local_fit_options));
    local_fit_options.bernstein_degree = (size_t)Rf_asInteger(bernstein_degree);
    local_fit_options.lambda_roughness = Rf_asReal(lambda_roughness);
    local_fit_options.holdout_fraction = Rf_asReal(holdout_fraction);
    local_fit_options.seed = (uint64_t)Rf_asInteger(local_seed);

    hdcd_annealing_options_t options;
    memset(&options, 0, sizeof(options));
    options.k_max = (size_t)Rf_asInteger(k_max);
    options.lambda_edge = Rf_asReal(lambda_edge);
    options.ordering = ordering;
    options.local_fit_options = local_fit_options;
    options.initial_temperature = Rf_asReal(initial_temperature);
    options.cooling_rate = Rf_asReal(cooling_rate);
    options.max_iterations = (size_t)Rf_asInteger(max_iterations);
    options.restarts = (size_t)Rf_asInteger(restarts);
    options.p_add = Rf_asReal(p_add);
    options.p_remove = Rf_asReal(p_remove);
    options.p_swap = Rf_asReal(p_swap);
    options.seed = (uint64_t)Rf_asInteger(anneal_seed);
    options.initial_dag = NULL;

    hdcd_annealing_result_t *result = NULL;
    hdcd_status_t status = hdcd_run_annealing(up, mask, nn, dd, &options, &result);
    free(mask);
    free(ordering);
    check_status(status, 1); /* non-convergence still returns a usable result */

    const hdcd_dag_t *best = hdcd_annealing_best_dag(result); /* borrowed */
    hdcd_dag_t *best_clone = NULL;
    hdcd_status_t clone_status = hdcd_dag_clone(best, &best_clone);
    double best_score = hdcd_annealing_best_score(result);
    hdcd_annealing_result_free(result);
    check_status(clone_status, 0);

    SEXP dag_sexp = PROTECT(wrap_pointer(best_clone, dag_finalizer, "hdcd_dag"));
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, dag_sexp);
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal(best_score));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("dag"));
    SET_STRING_ELT(names, 1, Rf_mkChar("score"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(3);
    return out;
}

/* ---- registration --------------------------------------------------------- */

static const R_CallMethodDef CallEntries[] = {
    {"hdcd_r_marginal_fit", (DL_FUNC)&hdcd_r_marginal_fit, 5},
    {"hdcd_r_marginal_cdf", (DL_FUNC)&hdcd_r_marginal_cdf, 2},
    {"hdcd_r_marginal_logpdf", (DL_FUNC)&hdcd_r_marginal_logpdf, 2},
    {"hdcd_r_marginal_sigma", (DL_FUNC)&hdcd_r_marginal_sigma, 1},
    {"hdcd_r_transform_to_copula", (DL_FUNC)&hdcd_r_transform_to_copula, 3},
    {"hdcd_r_compute_dependence_matrix", (DL_FUNC)&hdcd_r_compute_dependence_matrix, 3},
    {"hdcd_r_dependence_matrix_dense", (DL_FUNC)&hdcd_r_dependence_matrix_dense, 2},
    {"hdcd_r_compute_topology", (DL_FUNC)&hdcd_r_compute_topology, 1},
    {"hdcd_r_topology_ordering", (DL_FUNC)&hdcd_r_topology_ordering, 1},
    {"hdcd_r_dag_create", (DL_FUNC)&hdcd_r_dag_create, 2},
    {"hdcd_r_dag_from_edges", (DL_FUNC)&hdcd_r_dag_from_edges, 4},
    {"hdcd_r_dag_add_edge", (DL_FUNC)&hdcd_r_dag_add_edge, 3},
    {"hdcd_r_dag_clone", (DL_FUNC)&hdcd_r_dag_clone, 1},
    {"hdcd_r_dag_edges", (DL_FUNC)&hdcd_r_dag_edges, 2},
    {"hdcd_r_dag_fit", (DL_FUNC)&hdcd_r_dag_fit, 10},
    {"hdcd_r_dag_fit_joint_log_density", (DL_FUNC)&hdcd_r_dag_fit_joint_log_density, 2},
    {"hdcd_r_dag_fit_holdout_scores", (DL_FUNC)&hdcd_r_dag_fit_holdout_scores, 2},
    {"hdcd_r_dag_fit_all_converged", (DL_FUNC)&hdcd_r_dag_fit_all_converged, 1},
    {"hdcd_r_dag_fit_kl_estimate", (DL_FUNC)&hdcd_r_dag_fit_kl_estimate, 1},
    {"hdcd_r_dag_fit_kl_difference", (DL_FUNC)&hdcd_r_dag_fit_kl_difference, 2},
    {"hdcd_r_run_annealing", (DL_FUNC)&hdcd_r_run_annealing, 18},
    {NULL, NULL, 0}
};

void R_init_hdcd(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
