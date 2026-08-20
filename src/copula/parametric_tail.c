#include "hdcd/parametric_tail.h"
#include "hdcd/numerics.h"

#include <math.h>

#define HDCD_CLAYTON_THETA_MIN 1e-3
#define HDCD_CLAYTON_THETA_MAX 50.0
#define HDCD_GUMBEL_THETA_MIN (1.0 + 1e-3)
#define HDCD_GUMBEL_THETA_MAX 50.0
#define HDCD_TAIL_MLE_TOL 1e-6
#define HDCD_TAIL_MLE_MAX_ITER 200

hdcd_status_t hdcd_clayton_density(double u, double v, double theta, double *out) {
    if (out == NULL || !(u > 0.0) || !(u < 1.0) || !(v > 0.0) || !(v < 1.0) || !(theta > 0.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    double up = pow(u, -theta);
    double vp = pow(v, -theta);
    double base = up + vp - 1.0;
    if (!(base > 0.0)) {
        return HDCD_ERROR_NUMERICAL;
    }
    double log_density = log1p(theta)
                          - (theta + 1.0) * (log(u) + log(v))
                          - (1.0 / theta + 2.0) * log(base);
    double density = exp(log_density);
    if (isnan(density) || isinf(density)) {
        return HDCD_ERROR_NUMERICAL;
    }
    *out = density;
    return HDCD_OK;
}

hdcd_status_t hdcd_gumbel_density(double u, double v, double theta, double *out) {
    if (out == NULL || !(u > 0.0) || !(u < 1.0) || !(v > 0.0) || !(v < 1.0) || !(theta >= 1.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    double x = -log(u);
    double y = -log(v);
    double log_A = log(pow(x, theta) + pow(y, theta)) / theta;
    double A = exp(log_A);
    double log_C = -A; /* log of the Gumbel CDF */

    double log_density = log_C
                          + (theta - 1.0) * (log(x) + log(y))
                          - (log(u) + log(v))
                          + (1.0 - 2.0 * theta) * log_A
                          + log(A + theta - 1.0);
    if (!(A + theta - 1.0 > 0.0)) {
        return HDCD_ERROR_NUMERICAL;
    }
    double density = exp(log_density);
    if (isnan(density) || isinf(density)) {
        return HDCD_ERROR_NUMERICAL;
    }
    *out = density;
    return HDCD_OK;
}

hdcd_status_t hdcd_tail_family_density(hdcd_tail_family_t family, double u, double v, double theta, double *out) {
    switch (family) {
        case HDCD_TAIL_FAMILY_CLAYTON:
            return hdcd_clayton_density(u, v, theta, out);
        case HDCD_TAIL_FAMILY_GUMBEL:
            return hdcd_gumbel_density(u, v, theta, out);
        default:
            return HDCD_ERROR_INVALID_ARGUMENT;
    }
}

typedef struct {
    hdcd_tail_family_t family;
    const double *u;
    const double *v;
    size_t n;
} mle_userdata_t;

static double loglik_objective(double theta, void *userdata) {
    const mle_userdata_t *ud = (const mle_userdata_t *)userdata;
    double total = 0.0;
    for (size_t i = 0; i < ud->n; i++) {
        double density;
        hdcd_status_t status = hdcd_tail_family_density(ud->family, ud->u[i], ud->v[i], theta, &density);
        if (status != HDCD_OK || !(density > 0.0)) {
            return -1e300; /* a very poor (but finite, comparable) score for a degenerate theta */
        }
        total += log(density);
    }
    return total;
}

hdcd_status_t hdcd_tail_family_fit(
    hdcd_tail_family_t family,
    const double *u, const double *v, size_t n,
    double *theta_out
) {
    if (u == NULL || v == NULL || theta_out == NULL || n < 2) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    double lower, upper;
    switch (family) {
        case HDCD_TAIL_FAMILY_CLAYTON:
            lower = HDCD_CLAYTON_THETA_MIN;
            upper = HDCD_CLAYTON_THETA_MAX;
            break;
        case HDCD_TAIL_FAMILY_GUMBEL:
            lower = HDCD_GUMBEL_THETA_MIN;
            upper = HDCD_GUMBEL_THETA_MAX;
            break;
        default:
            return HDCD_ERROR_INVALID_ARGUMENT;
    }

    mle_userdata_t ud;
    ud.family = family;
    ud.u = u;
    ud.v = v;
    ud.n = n;

    hdcd_optimizer_1d_result_t result = hdcd_golden_section_maximize(
        loglik_objective, &ud, lower, upper, HDCD_TAIL_MLE_TOL, HDCD_TAIL_MLE_MAX_ITER
    );
    if (result.status != HDCD_OK) {
        return result.status;
    }
    *theta_out = result.x_opt;
    /* Non-convergence is reported but still usable (spec section 24:
     * fail clearly, not silently) -- mirrors hdcd_local_fit_node's own
     * convention of returning a best-effort fitted state alongside
     * HDCD_ERROR_NOT_CONVERGED. */
    return result.converged ? HDCD_OK : HDCD_ERROR_NOT_CONVERGED;
}
