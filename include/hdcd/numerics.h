#ifndef HDCD_NUMERICS_H
#define HDCD_NUMERICS_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- log-sum-exp ---------------------------------------------------- */

/*
 * Numerically stable log(sum(exp(log_terms[i]))) for i = 0..n-1.
 * Required for stable mixture log-densities (spec section 24).
 * n == 0 is invalid (there is no meaningful sum of an empty set of
 * log-terms in this library's use cases); returns -INFINITY via *out
 * and HDCD_ERROR_INVALID_ARGUMENT.
 */
hdcd_status_t hdcd_logsumexp(const double *log_terms, size_t n, double *out);

/*
 * log(mean(exp(log_terms))) = logsumexp(log_terms) - log(n).
 * Used for mixture log-densities where each component has equal weight 1/n.
 */
hdcd_status_t hdcd_log_mean_exp(const double *log_terms, size_t n, double *out);

/* ---- robust scale ----------------------------------------------------- */

/*
 * Median Absolute Deviation, scaled by 1.4826 so it is a consistent
 * estimator of the standard deviation under normality.
 * Modifies a temporary copy internally; does not alter `x`.
 * Requires n >= 1. Returns HDCD_ERROR_INVALID_ARGUMENT for n == 0 or
 * NULL arguments, HDCD_ERROR_ALLOCATION if the internal copy fails.
 */
hdcd_status_t hdcd_mad(const double *x, size_t n, double *out);

/*
 * Interquartile range (Type-7 / linear-interpolation quantiles, the
 * common default), i.e. Q3 - Q1. Requires n >= 2.
 */
hdcd_status_t hdcd_iqr(const double *x, size_t n, double *out);

/*
 * Robust scale estimate used to derive default bandwidth search bounds
 * (spec section 2.1): min(IQR / 1.349, MAD). Falls back to MAD alone if
 * IQR is degenerate (== 0), and to the sample standard deviation if both
 * are degenerate. Requires n >= 2.
 */
hdcd_status_t hdcd_robust_scale(const double *x, size_t n, double *out);

/* ---- deterministic bounded 1D optimizer ------------------------------- */

typedef struct hdcd_optimizer_1d_result {
    double x_opt;      /* argmax found */
    double f_opt;      /* objective value at x_opt */
    int iterations;    /* number of iterations performed */
    int converged;     /* 1 if the bracket shrank below tolerance, else 0 */
    hdcd_status_t status;
} hdcd_optimizer_1d_result_t;

typedef double (*hdcd_objective_1d_fn)(double x, void *userdata);

/*
 * Deterministic bounded maximization by golden-section search.
 * No randomness is used anywhere in this routine, so results are exactly
 * reproducible for identical inputs (spec section 24, "use deterministic
 * seeded RNG" / section 2.1 "use a deterministic bounded one-dimensional
 * optimizer").
 *
 * Requires lower < upper, tol > 0, max_iter > 0. If the objective
 * evaluates to NaN anywhere, returns HDCD_ERROR_NUMERICAL immediately
 * rather than silently continuing.
 */
hdcd_optimizer_1d_result_t hdcd_golden_section_maximize(
    hdcd_objective_1d_fn f,
    void *userdata,
    double lower,
    double upper,
    double tol,
    int max_iter
);

/* ---- quadrature ------------------------------------------------------- */

/*
 * Composite Simpson's rule quadrature nodes and weights on [0,1].
 * Requires n_nodes odd and >= 3 (n_nodes - 1 even subintervals). Both
 * output arrays must have length n_nodes; the weights sum to 1 (up to
 * floating-point rounding). Used as the deterministic default for the
 * u-integral in Sinkhorn normalization (spec section 11.2).
 */
hdcd_status_t hdcd_simpson_nodes_weights(size_t n_nodes, double *nodes_out, double *weights_out);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_NUMERICS_H */
