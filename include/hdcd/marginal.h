#ifndef HDCD_MARGINAL_H
#define HDCD_MARGINAL_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Gaussian-smoothed empirical marginal (spec section 2).
 *
 *   F_hat(x; sigma) = (1/n) sum_i Phi((x - x_i) / sigma)
 *   f_hat(x; sigma) = (1/(n sigma)) sum_i phi((x - x_i) / sigma)
 *
 * All functions evaluate at `eval_points[0..m-1]` against the fitted
 * sample `data[0..n-1]` and write `m` outputs to `out`. `sigma` must be
 * strictly positive. `data`/`eval_points` may alias; both must be
 * non-NULL with n >= 1, m >= 1.
 */

hdcd_status_t hdcd_gaussian_mixture_cdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
);

hdcd_status_t hdcd_gaussian_mixture_pdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
);

/* Numerically stable log-density via log-sum-exp (spec section 24). */
hdcd_status_t hdcd_gaussian_mixture_logpdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
);

/*
 * f_hat'(x; sigma) = -(1/(n sigma^3)) sum_i (x - x_i) phi((x - x_i) / sigma)
 * (spec section 2, density derivative).
 */
hdcd_status_t hdcd_gaussian_mixture_dpdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
);

/* ---- bandwidth selection (spec section 2.1) --------------------------- */

typedef struct hdcd_bandwidth_result {
    double sigma;       /* selected bandwidth */
    double eta;          /* log(sigma) at the optimum */
    double loglik;        /* leave-one-out log-likelihood at the optimum */
    double lower;          /* lower bound used for sigma (post-derivation) */
    double upper;            /* upper bound used for sigma */
    int iterations;
    int converged;
    hdcd_status_t status;
} hdcd_bandwidth_result_t;

/*
 * Select the bandwidth by leave-one-out log-likelihood cross-validation,
 * optimizing over eta = log(sigma) with a deterministic bounded 1D
 * optimizer (spec section 2.1).
 *
 * If sigma_min <= 0 or sigma_max <= 0, default bounds are derived from a
 * robust scale estimate of `data` as
 *   sigma_min = 0.05 * robust_scale, sigma_max = 3.0 * robust_scale.
 * Otherwise the caller-supplied bounds are used as-is (requires
 * 0 < sigma_min < sigma_max).
 *
 * Requires n >= 2 (leave-one-out needs at least one remaining point).
 */
hdcd_status_t hdcd_select_bandwidth_loo(
    const double *data, size_t n,
    double sigma_min, double sigma_max,
    double tol, int max_iter,
    hdcd_bandwidth_result_t *out
);

/* ---- fitted marginal object (spec section 31, Milestone 2) ----------- */

/*
 * Opaque handle for a marginal fitted to dimension j of the input data.
 * Owns a copy of the observed training values (data[i] such that
 * observed_mask[i] != 0), per O_j in spec section 2, plus the bandwidth
 * selected for it.
 */
typedef struct hdcd_marginal hdcd_marginal_t;

/*
 * Fit a Gaussian-smoothed marginal to dimension j via leave-one-out
 * bandwidth cross-validation (spec section 2.1), using only the entries
 * where observed_mask[i] != 0.
 *
 * `x` and `observed_mask` have the same shape (spec section 23):
 * missingness is carried by the mask, not by NaN sentinels in `x`.
 * sigma_min/sigma_max/tol/max_iter are forwarded to
 * hdcd_select_bandwidth_loo (sigma_min <= 0 or sigma_max <= 0 selects
 * the default robust-scale-derived bounds).
 *
 * Requires at least 2 observed entries. On success, *out is a newly
 * allocated handle that must be released with hdcd_marginal_free.
 */
hdcd_status_t hdcd_marginal_fit(
    const double *x, const uint8_t *observed_mask, size_t n,
    double sigma_min, double sigma_max,
    double tol, int max_iter,
    hdcd_marginal_t **out
);

void hdcd_marginal_free(hdcd_marginal_t *marginal);

/* Number of observed training entries used to fit this marginal (n_j). */
size_t hdcd_marginal_n_observed(const hdcd_marginal_t *marginal);

/* Bandwidth-selection diagnostics recorded at fit time. */
hdcd_bandwidth_result_t hdcd_marginal_bandwidth_result(const hdcd_marginal_t *marginal);

/*
 * Evaluate the fitted marginal CDF F_hat_j at eval_points[0..m-1].
 * Equivalent to calling hdcd_gaussian_mixture_cdf with this marginal's
 * stored training data and selected sigma.
 */
hdcd_status_t hdcd_marginal_cdf(
    const hdcd_marginal_t *marginal,
    const double *eval_points, size_t m,
    double *out
);

/*
 * Evaluate the fitted marginal log-density log f_hat_j at
 * eval_points[0..m-1] (spec section 35: f_X(x) = c(F_1,...,F_d) *
 * prod_j f_j(x_j) needs this term). Equivalent to calling
 * hdcd_gaussian_mixture_logpdf with this marginal's stored training
 * data and selected sigma.
 */
hdcd_status_t hdcd_marginal_logpdf(
    const hdcd_marginal_t *marginal,
    const double *eval_points, size_t m,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_MARGINAL_H */
