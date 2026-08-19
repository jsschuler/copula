#include "hdcd/marginal.h"

#include <math.h>

hdcd_status_t hdcd_gaussian_mixture_cdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
) {
    if (data == NULL || eval_points == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0 || m == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(sigma > 0.0) || isnan(sigma) || isinf(sigma)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    const double inv_sqrt2 = 0.7071067811865476;

    for (size_t k = 0; k < m; k++) {
        double x = eval_points[k];
        double accum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double z = (x - data[i]) / sigma;
            /* Phi(z) = 0.5 * erfc(-z / sqrt(2)) */
            accum += 0.5 * erfc(-z * inv_sqrt2);
        }
        double value = accum / (double)n;
        if (isnan(value)) {
            return HDCD_ERROR_NUMERICAL;
        }
        out[k] = value;
    }

    return HDCD_OK;
}
