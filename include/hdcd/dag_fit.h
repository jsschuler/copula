#ifndef HDCD_DAG_FIT_H
#define HDCD_DAG_FIT_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/dag.h"
#include "hdcd/local_fit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every node's local conditional copula factor for one fitted DAG, plus
 * the resulting factorized joint copula density. */
typedef struct hdcd_dag_fit hdcd_dag_fit_t;

/*
 * Fit every node's local conditional copula factor for a validated DAG
 * (spec section 28) and expose the resulting factorized joint copula
 * density (spec section 14):
 *   c_G(u) = prod_j c_j(u_j | u_Pa_G(j))
 *   log c_G(u) = sum_j log c_j(u_j | u_Pa_G(j))
 * an additive factorization exploited directly by
 * hdcd_dag_fit_joint_log_density.
 *
 * `dag`'s dimension must match `d`. The SAME hdcd_local_fit_options_t is
 * used for every node (a v1 simplification -- spec section 9 permits
 * node-wise Bernstein-degree tuning, not implemented here; see
 * DECISIONS.md), except `options->seed` is perturbed deterministically
 * per node so every node's train/holdout split is reproducible yet not
 * identical across nodes.
 *
 * If every node's local fit fully converges, returns HDCD_OK. If any
 * node's Theta or Sinkhorn fit did not converge, returns
 * HDCD_ERROR_NOT_CONVERGED but *out is still fully populated -- check
 * hdcd_dag_fit_node_converged per node (spec section 24: fail clearly,
 * not silently). A HARD failure on any single node (allocation,
 * insufficient data, numerical error) aborts the whole fit.
 */
hdcd_status_t hdcd_dag_fit(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    const hdcd_dag_t *dag,
    const hdcd_local_fit_options_t *options,
    hdcd_dag_fit_t **out
);

void hdcd_dag_fit_free(hdcd_dag_fit_t *fit);

size_t hdcd_dag_fit_dim(const hdcd_dag_fit_t *fit);

/* Borrowed pointer to node j's local fit; valid for fit's lifetime. NULL for out-of-range j. */
const hdcd_local_fit_t *hdcd_dag_fit_node(const hdcd_dag_fit_t *fit, size_t j);

int hdcd_dag_fit_node_converged(const hdcd_dag_fit_t *fit, size_t j);
int hdcd_dag_fit_all_converged(const hdcd_dag_fit_t *fit);

/*
 * Full factorized joint LOG copula density (spec section 14) at a
 * single, FULLY OBSERVED point u_point[0..d-1]. v1 does not integrate
 * out missing dimensions here -- spec section 16 explicitly puts that
 * out of scope ("Full observed-data likelihood ... is explicitly out of
 * scope for version 1"), so every entry of u_point must be observed.
 */
hdcd_status_t hdcd_dag_fit_joint_log_density(
    const hdcd_dag_fit_t *fit,
    const double *u_point, size_t d,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_DAG_FIT_H */
