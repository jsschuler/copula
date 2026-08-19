#include "hdcd/local_fit.h"
#include "hdcd/bernstein.h"
#include "hdcd/rng.h"

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
    if (n_parents > 0 && !(options->lambda_roughness > 0.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
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

    fit->m = options->bernstein_degree;
    fit->n_train = n_train;
    fit->n_holdout = n_holdout;
    size_t dim = fit->m + 1;

    fit->theta = (double *)calloc(n_parents * dim * dim, sizeof(double));
    double *m_stat = (double *)calloc(n_parents * dim * dim, sizeof(double));
    double *z_row = (double *)malloc(n_parents * sizeof(double));
    double *z_samples = (double *)malloc((size_t)n_train * n_parents * sizeof(double));
    double *grad_scratch = (double *)malloc(dim * dim * sizeof(double));

    if (fit->theta == NULL || m_stat == NULL || z_row == NULL || z_samples == NULL || grad_scratch == NULL) {
        free(usable_rows);
        free(m_stat);
        free(z_row);
        free(z_samples);
        free(grad_scratch);
        hdcd_local_fit_free(fit);
        return HDCD_ERROR_ALLOCATION;
    }

    /* Step 2 (data prep): sufficient statistics M_jk and Sinkhorn z_samples from the TRAIN rows.
     * M_jk is the MEAN (not sum) of the per-row outer products: averaging
     * keeps lambda_roughness's effective strength independent of
     * n_train. A raw sum would make the "reward" term grow linearly
     * with dataset size while the penalty term does not, so the same
     * lambda_roughness would regularize less and less as n_train grows
     * -- discovered via divergence (Theta and Sinkhorn both failing to
     * converge, with an exploding roughness penalty) while testing
     * against real dependent data; see DECISIONS.md. */
    for (size_t i = 0; i < n_train; i++) {
        size_t row = usable_rows[i];
        double u_val = u[child * n + row];
        for (size_t k = 0; k < n_parents; k++) {
            double z_val = u[parents[k] * n + row];
            z_row[k] = z_val;
            z_samples[i * n_parents + k] = z_val;
            hdcd_bernstein_tensor_gradient(u_val, z_val, fit->m, grad_scratch);
            double *m_k = &m_stat[k * dim * dim];
            for (size_t e = 0; e < dim * dim; e++) {
                m_k[e] += grad_scratch[e];
            }
        }
    }
    for (size_t e = 0; e < n_parents * dim * dim; e++) {
        m_stat[e] /= (double)n_train;
    }
    free(grad_scratch);

    /* Step 3: fit each parent's Theta via gradient ascent (spec section 28 step 3). */
    size_t theta_max_iter = (options->theta_max_iterations != 0)
                             ? options->theta_max_iterations : HDCD_LOCAL_FIT_DEFAULT_THETA_MAX_ITER;
    double theta_tol = (options->theta_tol > 0.0) ? options->theta_tol : HDCD_LOCAL_FIT_DEFAULT_THETA_TOL;

    int all_theta_converged = 1;
    for (size_t k = 0; k < n_parents; k++) {
        int converged = fit_theta_edge(
            &fit->theta[k * dim * dim], &m_stat[k * dim * dim], fit->m,
            options->lambda_roughness, theta_max_iter, theta_tol
        );
        if (!converged) {
            all_theta_converged = 0;
        }
    }
    fit->theta_converged = all_theta_converged;
    free(m_stat);

    fit->roughness_penalty = 0.0;
    for (size_t k = 0; k < n_parents; k++) {
        double r;
        hdcd_bernstein_roughness_penalty(&fit->theta[k * dim * dim], fit->m, &r);
        fit->roughness_penalty += r;
    }

    /* Step 4: Sinkhorn-normalize (spec section 28 step 4), using the
     * TRAIN rows' own parent values as the Monte Carlo z-samples (spec
     * section 11.2; see DECISIONS.md).
     *
     * `fit->kernel_userdata` is owned by `fit`, not freed at the end of
     * this function: hdcd_sinkhorn_t stores the raw kernel callback's
     * userdata pointer internally and calls back through it on every
     * future hdcd_sinkhorn_conditional_density() evaluation, not only
     * during this fit -- freeing it here would leave `fit->sinkhorn`
     * holding a dangling pointer for the rest of `fit`'s lifetime. */
    fit->kernel_userdata = (kernel_userdata_t *)malloc(sizeof(kernel_userdata_t));
    if (fit->kernel_userdata == NULL) {
        free(usable_rows);
        free(z_row);
        free(z_samples);
        hdcd_local_fit_free(fit);
        return HDCD_ERROR_ALLOCATION;
    }
    fit->kernel_userdata->m = fit->m;
    fit->kernel_userdata->n_parents = n_parents;
    fit->kernel_userdata->theta = fit->theta;

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t sinkhorn_status = hdcd_sinkhorn_fit(
        raw_kernel_callback, fit->kernel_userdata, z_samples, n_train, n_parents,
        &options->sinkhorn_options, &sk
    );
    free(z_samples);

    if (sk == NULL) {
        /* Hard failure (invalid arguments, numerical error) rather than
         * mere non-convergence: nothing usable to keep. */
        free(usable_rows);
        free(z_row);
        hdcd_local_fit_free(fit);
        return sinkhorn_status;
    }
    fit->sinkhorn = sk;

    /* Step 5: score on the HOLDOUT rows only (spec section 16). */
    double sum_loglik = 0.0;
    hdcd_status_t score_status = HDCD_OK;
    for (size_t i = 0; i < n_holdout && score_status == HDCD_OK; i++) {
        size_t row = usable_rows[n_train + i];
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

    free(usable_rows);
    free(z_row);

    if (score_status != HDCD_OK) {
        hdcd_local_fit_free(fit);
        return score_status;
    }

    fit->holdout_score = sum_loglik / (double)n_holdout;

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
