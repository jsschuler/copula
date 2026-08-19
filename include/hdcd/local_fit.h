#ifndef HDCD_LOCAL_FIT_H
#define HDCD_LOCAL_FIT_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"
#include "hdcd/sinkhorn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hdcd_local_fit_options {
    size_t bernstein_degree;   /* m: Bernstein degree for every edge into this node (spec section 9) */
    double lambda_roughness;    /* lambda_R; must be > 0 when lambda_roughness_grid is NOT
                                  * supplied -- see DECISIONS.md for why this implementation
                                  * requires strict positivity */
    double holdout_fraction;     /* in (0,1): fraction of usable rows held out for scoring (spec section 16) */
    uint64_t seed;                 /* seeds the deterministic train/holdout split */
    size_t theta_max_iterations;    /* inner gradient-ascent budget per edge; 0 selects a default */
    double theta_tol;                /* relative per-iteration objective-improvement tolerance
                                       * for the inner Theta fit (see DECISIONS.md for why this,
                                       * not a raw gradient-norm threshold); 0 selects a default */
    hdcd_sinkhorn_options_t sinkhorn_options; /* forwarded to hdcd_sinkhorn_fit */

    /* Optional: a non-global, PER-NODE learned roughness penalty (spec
     * section 18: "lambda_R is selected by validation unless explicitly
     * provided" / "a small validation grid is sufficient for version
     * 1"), as an alternative to the single fixed `lambda_roughness`
     * above applied identically everywhere.
     *
     * NULL, or lambda_roughness_grid_size == 0 (the default), disables
     * this entirely: behavior is exactly as if this field did not
     * exist, using `lambda_roughness` verbatim. This is the ONLY way
     * these two fields interact with existing callers -- fully
     * backward compatible.
     *
     * When non-empty: hdcd_local_fit_node further splits its TRAIN rows
     * (never the outer holdout) into an inner train/validation pair
     * (see roughness_validation_fraction below), fits every candidate
     * in the grid on the inner-train rows, scores each on the inner-
     * validation rows, and only then refits the winning lambda on the
     * FULL train rows before scoring on the untouched outer holdout
     * exactly as usual. The outer holdout -- the score used for
     * hdcd_dag_fit_kl_estimate/hdcd_dag_fit_kl_difference and, when this
     * options struct is reused there, for the annealing objective -- is
     * therefore never used to pick lambda, avoiding any lookahead bias
     * from penalty selection into model comparison.
     *
     * Per spec section 18 ("Penalty selection must occur outside the
     * inner annealing loop. Do not nest an expensive continuous penalty
     * search inside every graph proposal."): hdcd_run_annealing reuses
     * one hdcd_local_fit_options_t verbatim on every proposal, so
     * enabling the grid there would repeat this search on every
     * distinct parent set the search tries. This is intentionally left
     * to the caller to avoid, not forbidden by this struct -- restrict
     * grid search to hdcd_dag_fit calls made on an already-decided DAG
     * (the reference DAG the annealing search converged on, or an
     * explicit candidate DAG), where it runs once per node. */
    const double *lambda_roughness_grid;
    size_t lambda_roughness_grid_size;
    double roughness_validation_fraction; /* in (0,1): fraction of TRAIN rows further
                                            * reserved for inner grid validation; only
                                            * used when the grid is non-empty; 0 selects
                                            * a default (0.3) */
} hdcd_local_fit_options_t;

/* One node's fitted conditional copula factor c_j(u_j | u_Pa(j)). */
typedef struct hdcd_local_fit hdcd_local_fit_t;

/*
 * Fit node `child`'s conditional copula factor given its parent set
 * (spec sections 8, 9, 11, 16, 28):
 *   1. collect O_j(P_j): rows where child and every parent are observed;
 *   2. deterministically (seeded) split O_j(P_j) into train/holdout;
 *   3. fit one (m+1)x(m+1) Theta per parent by gradient ascent on the
 *      roughness-penalized raw-kernel objective on the TRAIN rows
 *      (see DECISIONS.md for why Theta-fitting and Sinkhorn
 *      normalization are two separate sequential steps, not one joint
 *      optimization -- this mirrors spec section 28's literal step
 *      ordering: "optimize edge coefficient matrices" then "apply
 *      Sinkhorn normalization");
 *   4. Sinkhorn-normalize the resulting raw kernel using the TRAIN
 *      rows' own parent values as the Monte Carlo z-samples (spec
 *      section 11.2 -- for this node, q_j(z) IS exactly the empirical
 *      distribution of its training parent values, so no separate
 *      sampler is needed or built here);
 *   5. score the fitted, normalized c_j on the HOLDOUT rows only (spec
 *      section 16's normalized held-out score bar-ell_j(P_j)).
 *
 * `u`/`mask` are the full copula-scale dataset, n x d, COLUMN-MAJOR
 * (spec section 23): u[col*n + row]. `parents` (length n_parents) need
 * not be sorted and are not reordered; child must not appear in it and
 * it must not contain duplicates.
 *
 * n_parents == 0 (a root node) is handled trivially per spec section
 * 12's base case p_1(u_1) = 1: c_j(u) = 1 identically, no Theta or
 * Sinkhorn fitting is performed, and the holdout score is exactly 0.
 *
 * On non-convergence of either the Theta fit or the Sinkhorn fit,
 * returns HDCD_ERROR_NOT_CONVERGED but *out is still populated with the
 * best-effort fitted state (spec section 24: fail clearly, not silently).
 */
hdcd_status_t hdcd_local_fit_node(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_local_fit_options_t *options,
    hdcd_local_fit_t **out
);

void hdcd_local_fit_free(hdcd_local_fit_t *fit);

size_t hdcd_local_fit_n_parents(const hdcd_local_fit_t *fit);

/* Parent indices, in the exact order hdcd_local_fit_log_density expects
 * its z[] argument. Pointer valid for the lifetime of `fit`. */
const size_t *hdcd_local_fit_parent_order(const hdcd_local_fit_t *fit);

size_t hdcd_local_fit_n_observed(const hdcd_local_fit_t *fit); /* |O_j(P_j)| */
size_t hdcd_local_fit_n_train(const hdcd_local_fit_t *fit);
size_t hdcd_local_fit_n_holdout(const hdcd_local_fit_t *fit);   /* effective sample size for the held-out score */

double hdcd_local_fit_holdout_score(const hdcd_local_fit_t *fit);      /* bar-ell_j(P_j) */
double hdcd_local_fit_roughness_penalty(const hdcd_local_fit_t *fit);   /* sum_k R(Theta_jk) */

/* The lambda_roughness actually used for this node's fitted Theta:
 * options->lambda_roughness verbatim when options->lambda_roughness_grid
 * was empty/NULL, or the grid candidate selected by inner-validation
 * when it was not. NAN for a root node (nothing to select) or a NULL
 * fit. */
double hdcd_local_fit_selected_lambda_roughness(const hdcd_local_fit_t *fit);

int hdcd_local_fit_theta_converged(const hdcd_local_fit_t *fit);
int hdcd_local_fit_sinkhorn_converged(const hdcd_local_fit_t *fit); /* 1 for a root node (nothing to converge) */

/*
 * log c_j(u | z). `z` must be given in hdcd_local_fit_parent_order()'s
 * order, length hdcd_local_fit_n_parents(fit). For a root node, z_dim
 * must be 0 and the result is always 0 (c_j(u) = 1).
 */
hdcd_status_t hdcd_local_fit_log_density(
    const hdcd_local_fit_t *fit,
    double u, const double *z, size_t z_dim,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_LOCAL_FIT_H */
