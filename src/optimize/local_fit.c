#include "hdcd/local_fit.h"
#include "hdcd/bernstein.h"
#include "hdcd/rng.h"
#include "hdcd/tail_dependence.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HDCD_LOCAL_FIT_DEFAULT_THETA_MAX_ITER 2000
#define HDCD_LOCAL_FIT_DEFAULT_THETA_TOL 1e-6
#define HDCD_LOCAL_FIT_ARMIJO_C1 1e-4
#define HDCD_LOCAL_FIT_MIN_STEP 1e-14
#define HDCD_LOCAL_FIT_LOG_KERNEL_CLIP 50.0

/*
 * R(Theta) (spec section 10) has a non-trivial null space: any surface
 * of the form Theta[r][s] = a*r*s + b*r + c*s + d has zero second
 * difference along both axes (confirmed and exploited deliberately in
 * examples/example_bernstein.c, Milestone 5), so R contributes NOTHING
 * to <Theta,M> - lambda_R*R(Theta) along those ~4 directions. Whenever
 * M has a nonzero projection onto that null space (generic for real
 * data), the raw objective is unbounded along it: gradient ascent never
 * converges and Theta drifts to overfit the training data (discovered
 * empirically -- held-out score got WORSE with more iterations, and the
 * gradient norm never fell below tolerance no matter the budget). A
 * small fixed-fraction L2 backstop makes the objective globally
 * strictly concave (bounded optimum) without materially changing the
 * roughness penalty's intended smoothing behavior for typical
 * lambda_roughness values. See DECISIONS.md.
 */
#define HDCD_LOCAL_FIT_RIDGE_FRACTION 0.02

typedef struct {
    size_t m;
    size_t n_parents;
    const double *theta; /* n_parents * (m+1)*(m+1); points at the owning hdcd_local_fit_t's theta */
} kernel_userdata_t;

struct hdcd_local_fit {
    size_t n_parents;
    size_t *parent_order;      /* copy of caller's parents[], unmodified order */
    size_t m;                  /* Bernstein degree */
    double *theta;               /* n_parents * (m+1)*(m+1), concatenated per parent, row-major each */
    /* Owned so it outlives this function: `sinkhorn` stores a pointer to
     * it internally and dereferences it on every future evaluation call,
     * not just during fitting -- it must live as long as `sinkhorn` does,
     * not just until hdcd_local_fit_node() returns. */
    kernel_userdata_t *kernel_userdata;
    hdcd_sinkhorn_t *sinkhorn;   /* NULL for a root node */
    size_t n_observed;
    size_t n_train;
    size_t n_holdout;
    double holdout_score;
    double roughness_penalty;
    double selected_lambda_roughness; /* NAN for a root node */
    double max_tail_dependence;        /* NAN unless bernstein_degree_grid was non-empty */
    int theta_converged;
};

static double raw_kernel_callback(double u, const double *z, size_t z_dim, void *userdata) {
    const kernel_userdata_t *kd = (const kernel_userdata_t *)userdata;
    (void)z_dim;
    size_t dim = kd->m + 1;
    double log_k = 0.0;
    for (size_t k = 0; k < kd->n_parents; k++) {
        double g;
        hdcd_bernstein_tensor_interaction(u, z[k], kd->m, &kd->theta[k * dim * dim], &g);
        log_k += g;
    }
    if (log_k > HDCD_LOCAL_FIT_LOG_KERNEL_CLIP) log_k = HDCD_LOCAL_FIT_LOG_KERNEL_CLIP;
    if (log_k < -HDCD_LOCAL_FIT_LOG_KERNEL_CLIP) log_k = -HDCD_LOCAL_FIT_LOG_KERNEL_CLIP;
    return exp(log_k);
}

/* Objective for one edge's Theta: <Theta, M> - lambda_R * R(Theta). */
static double theta_objective(const double *theta, const double *m_stat, size_t m, double lambda_r) {
    size_t dim = m + 1;
    double dot = 0.0;
    double ridge = 0.0;
    for (size_t i = 0; i < dim * dim; i++) {
        dot += theta[i] * m_stat[i];
        ridge += theta[i] * theta[i];
    }
    double roughness;
    hdcd_bernstein_roughness_penalty(theta, m, &roughness);
    double lambda_ridge = HDCD_LOCAL_FIT_RIDGE_FRACTION * lambda_r;
    return dot - lambda_r * roughness - lambda_ridge * ridge;
}

/* Gradient-ascend one edge's Theta on <Theta,M> - lambda_R*R(Theta), a
 * strictly concave objective for lambda_R > 0 (M is linear-in-Theta,
 * R is a positive-semidefinite quadratic form), via backtracking
 * (Armijo) line search. Returns 1 if the gradient norm fell below tol. */
static int fit_theta_edge(
    double *theta, const double *m_stat, size_t m, double lambda_r,
    size_t max_iter, double tol
) {
    size_t dim = m + 1;
    size_t n_entries = dim * dim;
    double *grad = (double *)malloc(n_entries * sizeof(double));
    double *rough_grad = (double *)malloc(n_entries * sizeof(double));
    double *candidate = (double *)malloc(n_entries * sizeof(double));
    if (grad == NULL || rough_grad == NULL || candidate == NULL) {
        free(grad);
        free(rough_grad);
        free(candidate);
        return 0;
    }

    for (size_t i = 0; i < n_entries; i++) {
        theta[i] = 0.0;
    }

    double step = 1.0;
    int converged = 0;
    double lambda_ridge = HDCD_LOCAL_FIT_RIDGE_FRACTION * lambda_r;
    double prev_obj = theta_objective(theta, m_stat, m, lambda_r);

    /*
     * Convergence is judged by RELATIVE OBJECTIVE IMPROVEMENT, not raw
     * gradient norm. The ridge backstop above is deliberately tiny
     * (HDCD_LOCAL_FIT_RIDGE_FRACTION * lambda_r) so it barely perturbs
     * R's intended smoothing behavior -- but that also means its
     * curvature is tiny, so the objective is genuinely ill-conditioned
     * (curvature ratio ~1/HDCD_LOCAL_FIT_RIDGE_FRACTION between the
     * roughness-penalized and ridge-only directions). Plain gradient
     * ascent crawls along the flat ridge-only direction almost
     * indefinitely even after the objective value itself has clearly
     * plateaued -- confirmed empirically (gradient-norm convergence
     * never triggered even at 2000 iterations, while the objective
     * stopped moving in a few dozen). Relative objective improvement is
     * the criterion that actually reflects "is this fit still changing
     * in any way that matters," and is standard practice for exactly
     * this ill-conditioned-but-bounded situation. See DECISIONS.md.
     */
    for (size_t iter = 0; iter < max_iter; iter++) {
        hdcd_bernstein_roughness_gradient(theta, m, rough_grad);

        double grad_norm_sq = 0.0;
        for (size_t i = 0; i < n_entries; i++) {
            grad[i] = m_stat[i] - lambda_r * rough_grad[i] - 2.0 * lambda_ridge * theta[i];
            grad_norm_sq += grad[i] * grad[i];
        }

        double t = step;
        double candidate_obj;
        while (1) {
            for (size_t i = 0; i < n_entries; i++) {
                candidate[i] = theta[i] + t * grad[i];
            }
            candidate_obj = theta_objective(candidate, m_stat, m, lambda_r);
            if (candidate_obj >= prev_obj + HDCD_LOCAL_FIT_ARMIJO_C1 * t * grad_norm_sq) {
                break;
            }
            t *= 0.5;
            if (t < HDCD_LOCAL_FIT_MIN_STEP) {
                break;
            }
        }

        memcpy(theta, candidate, n_entries * sizeof(double));

        double improvement = fabs(candidate_obj - prev_obj);
        prev_obj = candidate_obj;
        if (improvement < tol * (1.0 + fabs(prev_obj))) {
            converged = 1;
            break;
        }

        step = t * 1.5;
        if (step > 4.0) step = 4.0;
        if (step < HDCD_LOCAL_FIT_MIN_STEP) step = HDCD_LOCAL_FIT_MIN_STEP;
    }

    free(grad);
    free(rough_grad);
    free(candidate);
    return converged;
}

typedef struct {
    double *theta;                     /* owned; n_parents*(m+1)*(m+1) */
    kernel_userdata_t *kernel_userdata; /* owned; ->theta aliases the field above */
    hdcd_sinkhorn_t *sinkhorn;         /* owned */
    double score;                       /* mean held-out log-density on the score rows */
    double roughness_penalty;
    int theta_converged;
} local_fit_candidate_t;

static void local_fit_candidate_release(local_fit_candidate_t *c) {
    if (c == NULL) {
        return;
    }
    hdcd_sinkhorn_free(c->sinkhorn); /* must be freed before kernel_userdata, which it references */
    free(c->kernel_userdata);
    free(c->theta);
    c->sinkhorn = NULL;
    c->kernel_userdata = NULL;
    c->theta = NULL;
}

/*
 * Fits every parent edge's Theta (gradient ascent) plus the Sinkhorn
 * normalization on `train_rows`, then scores the resulting c_j on
 * `score_rows` -- the shared core of hdcd_local_fit_node's steps 2-5,
 * parametrized so it can serve both the normal single-lambda fit (train
 * = the outer TRAIN split, score = the outer HOLDOUT split) and one
 * roughness-grid candidate's inner-validation fit (train/score are an
 * inner split of the outer TRAIN rows only -- the outer holdout is
 * never touched by grid search).
 *
 * HDCD_OK on success (the candidate may still be non-convergent; see
 * out->theta_converged). Any other status means *out was never
 * populated (nothing to release).
 */
static hdcd_status_t fit_and_score(
    const double *u, size_t n, size_t child, const size_t *parents, size_t n_parents,
    const size_t *train_rows, size_t n_train_rows,
    const size_t *score_rows, size_t n_score_rows,
    size_t m, double lambda_r,
    size_t theta_max_iter, double theta_tol,
    const hdcd_sinkhorn_options_t *sinkhorn_options,
    local_fit_candidate_t *out
) {
    size_t dim = m + 1;

    double *m_stat = (double *)calloc(n_parents * dim * dim, sizeof(double));
    double *z_row = (double *)malloc(n_parents * sizeof(double));
    double *z_samples = (double *)malloc((size_t)n_train_rows * n_parents * sizeof(double));
    double *grad_scratch = (double *)malloc(dim * dim * sizeof(double));
    double *theta = (double *)calloc(n_parents * dim * dim, sizeof(double));
    if (m_stat == NULL || z_row == NULL || z_samples == NULL || grad_scratch == NULL || theta == NULL) {
        free(m_stat); free(z_row); free(z_samples); free(grad_scratch); free(theta);
        return HDCD_ERROR_ALLOCATION;
    }

    /* Sufficient statistics M_jk, MEAN (not sum) over train_rows -- see
     * the comment on the analogous loop below for why. */
    for (size_t i = 0; i < n_train_rows; i++) {
        size_t row = train_rows[i];
        double u_val = u[child * n + row];
        for (size_t k = 0; k < n_parents; k++) {
            double z_val = u[parents[k] * n + row];
            z_samples[i * n_parents + k] = z_val;
            hdcd_bernstein_tensor_gradient(u_val, z_val, m, grad_scratch);
            double *m_k = &m_stat[k * dim * dim];
            for (size_t e = 0; e < dim * dim; e++) {
                m_k[e] += grad_scratch[e];
            }
        }
    }
    for (size_t e = 0; e < n_parents * dim * dim; e++) {
        m_stat[e] /= (double)n_train_rows;
    }
    free(grad_scratch);

    int all_theta_converged = 1;
    for (size_t k = 0; k < n_parents; k++) {
        int converged = fit_theta_edge(&theta[k * dim * dim], &m_stat[k * dim * dim], m, lambda_r, theta_max_iter, theta_tol);
        if (!converged) {
            all_theta_converged = 0;
        }
    }
    free(m_stat);

    double roughness_penalty = 0.0;
    for (size_t k = 0; k < n_parents; k++) {
        double r;
        hdcd_bernstein_roughness_penalty(&theta[k * dim * dim], m, &r);
        roughness_penalty += r;
    }

    kernel_userdata_t *kd = (kernel_userdata_t *)malloc(sizeof(kernel_userdata_t));
    if (kd == NULL) {
        free(z_row);
        free(z_samples);
        free(theta);
        return HDCD_ERROR_ALLOCATION;
    }
    kd->m = m;
    kd->n_parents = n_parents;
    kd->theta = theta;

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t sinkhorn_status = hdcd_sinkhorn_fit(
        raw_kernel_callback, kd, z_samples, n_train_rows, n_parents, sinkhorn_options, &sk
    );
    free(z_samples);
    if (sk == NULL) {
        free(z_row);
        free(kd);
        free(theta);
        return sinkhorn_status;
    }

    double sum_loglik = 0.0;
    hdcd_status_t score_status = HDCD_OK;
    for (size_t i = 0; i < n_score_rows && score_status == HDCD_OK; i++) {
        size_t row = score_rows[i];
        double u_val = u[child * n + row];
        for (size_t k = 0; k < n_parents; k++) {
            z_row[k] = u[parents[k] * n + row];
        }
        double c;
        score_status = hdcd_sinkhorn_conditional_density(sk, u_val, z_row, n_parents, &c);
        if (score_status == HDCD_OK) {
            if (!(c > 0.0) || isnan(c)) {
                score_status = HDCD_ERROR_NUMERICAL;
            } else {
                sum_loglik += log(c);
            }
        }
    }
    free(z_row);

    if (score_status != HDCD_OK) {
        hdcd_sinkhorn_free(sk);
        free(kd);
        free(theta);
        return score_status;
    }

    out->theta = theta;
    out->kernel_userdata = kd;
    out->sinkhorn = sk;
    out->score = sum_loglik / (double)n_score_rows;
    out->roughness_penalty = roughness_penalty;
    out->theta_converged = all_theta_converged;
    return HDCD_OK;
}

hdcd_status_t hdcd_local_fit_node(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_local_fit_options_t *options,
    hdcd_local_fit_t **out
) {
    if (u == NULL || mask == NULL || out == NULL || options == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0 || d == 0 || child >= d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n_parents > 0 && parents == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    for (size_t k = 0; k < n_parents; k++) {
        if (parents[k] >= d || parents[k] == child) {
            return HDCD_ERROR_INVALID_ARGUMENT;
        }
        for (size_t k2 = k + 1; k2 < n_parents; k2++) {
            if (parents[k] == parents[k2]) {
                return HDCD_ERROR_INVALID_ARGUMENT;
            }
        }
    }
    if (!(options->holdout_fraction > 0.0) || !(options->holdout_fraction < 1.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    int lambda_grid_enabled = (options->lambda_roughness_grid != NULL && options->lambda_roughness_grid_size > 0);
    int degree_grid_enabled = (options->bernstein_degree_grid != NULL && options->bernstein_degree_grid_size > 0);
    double roughness_validation_fraction = 0.3;
    if (n_parents > 0) {
        if (lambda_grid_enabled) {
            for (size_t i = 0; i < options->lambda_roughness_grid_size; i++) {
                if (!(options->lambda_roughness_grid[i] > 0.0)) {
                    return HDCD_ERROR_INVALID_ARGUMENT;
                }
            }
        } else if (!(options->lambda_roughness > 0.0)) {
            return HDCD_ERROR_INVALID_ARGUMENT;
        }
        if (degree_grid_enabled) {
            for (size_t i = 0; i < options->bernstein_degree_grid_size; i++) {
                if (options->bernstein_degree_grid[i] < 1) {
                    return HDCD_ERROR_INVALID_ARGUMENT;
                }
            }
            if (!(options->tail_dependence_gate >= 0.0) || options->tail_dependence_gate > 1.0) {
                return HDCD_ERROR_INVALID_ARGUMENT;
            }
        }
        if ((lambda_grid_enabled || degree_grid_enabled) && options->roughness_validation_fraction > 0.0) {
            roughness_validation_fraction = options->roughness_validation_fraction;
        }
        if ((lambda_grid_enabled || degree_grid_enabled)
            && (!(roughness_validation_fraction > 0.0) || !(roughness_validation_fraction < 1.0))) {
            return HDCD_ERROR_INVALID_ARGUMENT;
        }
    }
    *out = NULL;

    /* Step 1: collect O_j(P_j). */
    size_t *usable_rows = (size_t *)malloc(n * sizeof(size_t));
    if (usable_rows == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    size_t n_usable = 0;
    for (size_t i = 0; i < n; i++) {
        if (!mask[child * n + i]) {
            continue;
        }
        int all_parents_observed = 1;
        for (size_t k = 0; k < n_parents; k++) {
            if (!mask[parents[k] * n + i]) {
                all_parents_observed = 0;
                break;
            }
        }
        if (all_parents_observed) {
            usable_rows[n_usable++] = i;
        }
    }

    if (n_usable == 0) {
        free(usable_rows);
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    hdcd_local_fit_t *fit = (hdcd_local_fit_t *)calloc(1, sizeof(hdcd_local_fit_t));
    if (fit == NULL) {
        free(usable_rows);
        return HDCD_ERROR_ALLOCATION;
    }
    fit->n_parents = n_parents;
    fit->n_observed = n_usable;
    if (n_parents > 0) {
        fit->parent_order = (size_t *)malloc(n_parents * sizeof(size_t));
        if (fit->parent_order == NULL) {
            free(usable_rows);
            free(fit);
            return HDCD_ERROR_ALLOCATION;
        }
        memcpy(fit->parent_order, parents, n_parents * sizeof(size_t));
    }

    /* Root node: c_j(u) = 1 identically (spec section 12 base case). */
    if (n_parents == 0) {
        free(usable_rows);
        fit->m = 0;
        fit->theta = NULL;
        fit->sinkhorn = NULL;
        fit->n_train = 0;
        fit->n_holdout = n_usable;
        fit->holdout_score = 0.0;
        fit->roughness_penalty = 0.0;
        fit->selected_lambda_roughness = NAN;
        fit->max_tail_dependence = NAN;
        fit->theta_converged = 1;
        *out = fit;
        return HDCD_OK;
    }

    size_t n_train = (size_t)((double)n_usable * (1.0 - options->holdout_fraction));
    size_t n_holdout = n_usable - n_train;
    if (n_train < 2 || n_holdout < 1) {
        free(usable_rows);
        hdcd_local_fit_free(fit);
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    hdcd_rng_t rng;
    hdcd_rng_seed(&rng, options->seed);
    hdcd_rng_shuffle_indices(&rng, usable_rows, n_usable);

    fit->n_train = n_train;
    fit->n_holdout = n_holdout;
    fit->max_tail_dependence = NAN;

    size_t theta_max_iter = (options->theta_max_iterations != 0)
                             ? options->theta_max_iterations : HDCD_LOCAL_FIT_DEFAULT_THETA_MAX_ITER;
    double theta_tol = (options->theta_tol > 0.0) ? options->theta_tol : HDCD_LOCAL_FIT_DEFAULT_THETA_TOL;

    /* Tail-dependence gate for bernstein_degree_grid (spec section 18-
     * style penalty-selection philosophy, applied to degree instead of
     * lambda; see DECISIONS.md's "tail-dependence-informed bernstein_degree
     * selection"). A pure descriptive diagnostic over ALL usable rows
     * (train + holdout): it decides SEARCH STRATEGY, not a fitted
     * parameter, so there is no leakage concern in using the full set. */
    int degree_search_active = 0;
    if (degree_grid_enabled) {
        double max_tail_dep = 0.0;
        double *col_child = (double *)malloc(n_usable * sizeof(double));
        double *col_parent = (double *)malloc(n_usable * sizeof(double));
        if (col_child == NULL || col_parent == NULL) {
            free(col_child);
            free(col_parent);
            free(usable_rows);
            hdcd_local_fit_free(fit);
            return HDCD_ERROR_ALLOCATION;
        }
        for (size_t i = 0; i < n_usable; i++) {
            col_child[i] = u[child * n + usable_rows[i]];
        }
        for (size_t k = 0; k < n_parents; k++) {
            for (size_t i = 0; i < n_usable; i++) {
                col_parent[i] = u[parents[k] * n + usable_rows[i]];
            }
            double lam_upper, lam_lower;
            hdcd_status_t td_status = hdcd_tail_dependence_coefficient(
                col_child, col_parent, n_usable, options->tail_dependence_k, &lam_upper, &lam_lower
            );
            if (td_status == HDCD_OK) {
                if (lam_upper > max_tail_dep) max_tail_dep = lam_upper;
                if (lam_lower > max_tail_dep) max_tail_dep = lam_lower;
            }
            /* Not enough usable rows for even the estimator's own
             * minimum (n_usable < 8): treated as "no tail-dependence
             * evidence" rather than a hard failure of the whole fit. */
        }
        free(col_child);
        free(col_parent);
        fit->max_tail_dependence = max_tail_dep;
        degree_search_active = (options->tail_dependence_gate <= 0.0) || (max_tail_dep >= options->tail_dependence_gate);
    }

    size_t degree_candidates_buf[1];
    const size_t *degree_candidates;
    size_t n_degree_candidates;
    if (degree_search_active) {
        degree_candidates = options->bernstein_degree_grid;
        n_degree_candidates = options->bernstein_degree_grid_size;
    } else {
        degree_candidates_buf[0] = options->bernstein_degree;
        degree_candidates = degree_candidates_buf;
        n_degree_candidates = 1;
    }

    double lambda_candidates_buf[1];
    const double *lambda_candidates;
    size_t n_lambda_candidates;
    if (lambda_grid_enabled) {
        lambda_candidates = options->lambda_roughness_grid;
        n_lambda_candidates = options->lambda_roughness_grid_size;
    } else {
        lambda_candidates_buf[0] = options->lambda_roughness;
        lambda_candidates = lambda_candidates_buf;
        n_lambda_candidates = 1;
    }

    size_t m_to_use;
    double lambda_to_use;
    if (n_degree_candidates == 1 && n_lambda_candidates == 1) {
        m_to_use = degree_candidates[0];
        lambda_to_use = lambda_candidates[0];
    } else {
        /* Select (bernstein_degree, lambda_roughness) PER NODE via a
         * small joint validation grid, using an INNER split of the
         * outer TRAIN rows only -- the outer holdout (used for
         * hdcd_dag_fit_kl_estimate and, if this options struct reaches
         * hdcd_run_annealing, for the annealing objective) is never
         * touched here, so it stays an unbiased score for whatever
         * (degree, lambda) pair ends up selected. */
        size_t n_inner_val = (size_t)((double)n_train * roughness_validation_fraction);
        size_t n_inner_train = n_train - n_inner_val;
        if (n_inner_train < 2 || n_inner_val < 1) {
            free(usable_rows);
            hdcd_local_fit_free(fit);
            return HDCD_ERROR_INVALID_ARGUMENT;
        }

        size_t *inner_rows = (size_t *)malloc(n_train * sizeof(size_t));
        if (inner_rows == NULL) {
            free(usable_rows);
            hdcd_local_fit_free(fit);
            return HDCD_ERROR_ALLOCATION;
        }
        memcpy(inner_rows, usable_rows, n_train * sizeof(size_t));
        hdcd_rng_t inner_rng;
        hdcd_rng_seed(&inner_rng, options->seed ^ 0xD1B54A32D192ED03ULL);
        hdcd_rng_shuffle_indices(&inner_rng, inner_rows, n_train);

        double best_score = -INFINITY;
        size_t best_m = degree_candidates[0];
        double best_lambda = lambda_candidates[0];
        int any_ok = 0;
        for (size_t di = 0; di < n_degree_candidates; di++) {
            for (size_t li = 0; li < n_lambda_candidates; li++) {
                local_fit_candidate_t cand;
                hdcd_status_t cstatus = fit_and_score(
                    u, n, child, parents, n_parents,
                    inner_rows, n_inner_train,
                    inner_rows + n_inner_train, n_inner_val,
                    degree_candidates[di], lambda_candidates[li], theta_max_iter, theta_tol,
                    &options->sinkhorn_options, &cand
                );
                if (cstatus == HDCD_ERROR_ALLOCATION) {
                    free(inner_rows);
                    free(usable_rows);
                    hdcd_local_fit_free(fit);
                    return HDCD_ERROR_ALLOCATION;
                }
                if (cstatus == HDCD_OK) {
                    any_ok = 1;
                    if (cand.score > best_score) {
                        best_score = cand.score;
                        best_m = degree_candidates[di];
                        best_lambda = lambda_candidates[li];
                    }
                    local_fit_candidate_release(&cand);
                }
                /* cstatus == HDCD_ERROR_NUMERICAL / HDCD_ERROR_INVALID_ARGUMENT:
                 * this candidate (degree, lambda) pair produced a
                 * degenerate fit (e.g. the overfitting failure mode a
                 * too-small lambda can hit -- see
                 * notebooks/vine_copula_recovery.Rmd's roughness
                 * sensitivity check). Skip it, nothing was allocated to
                 * release, keep searching the rest of the grid. */
            }
        }
        free(inner_rows);
        if (!any_ok) {
            free(usable_rows);
            hdcd_local_fit_free(fit);
            return HDCD_ERROR_NUMERICAL;
        }
        m_to_use = best_m;
        lambda_to_use = best_lambda;
    }
    fit->m = m_to_use;
    fit->selected_lambda_roughness = lambda_to_use;

    /* Final fit: Theta + Sinkhorn on the FULL train rows at the selected
     * (degree, lambda), scored on the (untouched) outer holdout -- spec
     * section 28 steps 3-4 and section 16's held-out score, exactly as
     * if the grid(s) had never run. */
    local_fit_candidate_t production;
    hdcd_status_t status = fit_and_score(
        u, n, child, parents, n_parents,
        usable_rows, n_train,
        usable_rows + n_train, n_holdout,
        fit->m, lambda_to_use, theta_max_iter, theta_tol,
        &options->sinkhorn_options, &production
    );
    free(usable_rows);

    if (status != HDCD_OK) {
        /* Hard failure (invalid arguments, numerical error) rather than
         * mere non-convergence: nothing usable to keep. */
        hdcd_local_fit_free(fit);
        return status;
    }

    fit->theta = production.theta;
    fit->kernel_userdata = production.kernel_userdata;
    fit->sinkhorn = production.sinkhorn;
    fit->holdout_score = production.score;
    fit->roughness_penalty = production.roughness_penalty;
    fit->theta_converged = production.theta_converged;

    *out = fit;

    if (!fit->theta_converged || !hdcd_sinkhorn_converged(fit->sinkhorn)) {
        return HDCD_ERROR_NOT_CONVERGED;
    }
    return HDCD_OK;
}

void hdcd_local_fit_free(hdcd_local_fit_t *fit) {
    if (fit == NULL) {
        return;
    }
    free(fit->parent_order);
    hdcd_sinkhorn_free(fit->sinkhorn); /* must be freed before kernel_userdata, which it references */
    free(fit->kernel_userdata); /* does not own ->theta, so no double free with the line below */
    free(fit->theta);
    free(fit);
}

size_t hdcd_local_fit_n_parents(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->n_parents : 0;
}

const size_t *hdcd_local_fit_parent_order(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->parent_order : NULL;
}

size_t hdcd_local_fit_n_observed(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->n_observed : 0;
}

size_t hdcd_local_fit_n_train(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->n_train : 0;
}

size_t hdcd_local_fit_n_holdout(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->n_holdout : 0;
}

double hdcd_local_fit_holdout_score(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->holdout_score : NAN;
}

double hdcd_local_fit_roughness_penalty(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->roughness_penalty : NAN;
}

double hdcd_local_fit_selected_lambda_roughness(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->selected_lambda_roughness : NAN;
}

size_t hdcd_local_fit_selected_bernstein_degree(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->m : 0;
}

double hdcd_local_fit_max_tail_dependence(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->max_tail_dependence : NAN;
}

int hdcd_local_fit_theta_converged(const hdcd_local_fit_t *fit) {
    return (fit != NULL) ? fit->theta_converged : 0;
}

int hdcd_local_fit_sinkhorn_converged(const hdcd_local_fit_t *fit) {
    if (fit == NULL) {
        return 0;
    }
    if (fit->sinkhorn == NULL) {
        return 1; /* root node: nothing to converge */
    }
    return hdcd_sinkhorn_converged(fit->sinkhorn);
}

hdcd_status_t hdcd_local_fit_log_density(
    const hdcd_local_fit_t *fit,
    double u, const double *z, size_t z_dim,
    double *out
) {
    if (fit == NULL || out == NULL || z_dim != fit->n_parents) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (fit->n_parents == 0) {
        *out = 0.0;
        return HDCD_OK;
    }
    if (z == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    double c;
    hdcd_status_t status = hdcd_sinkhorn_conditional_density(fit->sinkhorn, u, z, z_dim, &c);
    if (status != HDCD_OK) {
        return status;
    }
    if (!(c > 0.0) || isnan(c)) {
        return HDCD_ERROR_NUMERICAL;
    }
    *out = log(c);
    return HDCD_OK;
}
