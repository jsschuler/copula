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
 * `dag` may be ANY valid DAG over d nodes -- the reference DAG built by
 * annealing (Milestone 8), or an arbitrary candidate with a completely
 * different topological order supplied via hdcd_dag_from_edges (spec
 * section 19: "the public API must accept an arbitrary DAG G*"). This
 * function reads each node's parent set directly from `dag` and fits
 * nodes independently, so it has no notion of -- and imposes no
 * requirement on -- any particular ordering; the acyclicity spec
 * section 19 step 1 asks for is validated once, at DAG-construction
 * time (hdcd_dag_add_edge incrementally, or hdcd_dag_from_edges in
 * bulk), not here.
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

/*
 * Estimated KL divergence of `fit`'s factorization from the reference
 * dependence distribution, up to a shared additive constant (spec
 * section 15): sum_j K_hat_j(Pa(j)) = sum_j -(held-out normalized local
 * score). The missing constant (the entropy of c*) is identical for
 * any two fits over the SAME data, so it cancels in
 * hdcd_dag_fit_kl_difference -- this quantity alone is only meaningful
 * as a component of that difference, not as a standalone divergence
 * value. NaN if `fit` is NULL.
 */
double hdcd_dag_fit_kl_estimate(const hdcd_dag_fit_t *fit);

/*
 * Delta_KL = kl_estimate(candidate) - kl_estimate(reference) (spec
 * section 19): how much dependence information is lost when the joint
 * distribution is constrained to factor according to `candidate`
 * instead of `reference`. Both must be hdcd_dag_fit results over the
 * SAME dataset (same n, d, u, mask) for the comparison to be
 * meaningful; this function only checks that their dimensions agree
 * (returns NaN if they don't), since it does not retain the training
 * data to check more than that.
 *
 * Positive means candidate fits the data WORSE than reference; negative
 * means candidate fits BETTER. This is a purely observational,
 * distributional-fit comparison (spec section 19): it does NOT by
 * itself establish causal direction, and does NOT distinguish between
 * Markov-equivalent causal DAGs without additional assumptions or
 * interventions neither `candidate` nor `reference` encode. Treat a
 * favorable Delta_KL as evidence that `candidate`'s factorization is a
 * good statistical approximation, not as evidence that its edges are
 * causal.
 */
double hdcd_dag_fit_kl_difference(const hdcd_dag_fit_t *candidate, const hdcd_dag_fit_t *reference);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_DAG_FIT_H */
