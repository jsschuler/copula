#ifndef HDCD_MARGINAL_H
#define HDCD_MARGINAL_H

#include <stddef.h>
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

#ifdef __cplusplus
}
#endif

#endif /* HDCD_MARGINAL_H */
