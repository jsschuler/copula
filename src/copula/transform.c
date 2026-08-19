#include "hdcd/copula.h"

#include <math.h>

hdcd_status_t hdcd_transform_to_copula(
    const hdcd_marginal_t *marginal,
    const double *x,
    const uint8_t *observed_mask,
    size_t n,
    double epsilon,
    double *u_out
) {
    if (marginal == NULL || x == NULL || observed_mask == NULL || u_out == NULL || n == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    if (epsilon <= 0.0) {
        epsilon = HDCD_DEFAULT_COPULA_EPSILON;
    }
    if (!(epsilon < 0.5) || isnan(epsilon)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < n; i++) {
        if (!observed_mask[i]) {
            u_out[i] = NAN;
            continue;
        }

        double raw;
        hdcd_status_t status = hdcd_marginal_cdf(marginal, &x[i], 1, &raw);
        if (status != HDCD_OK) {
            return status;
        }

        double clipped = raw;
        if (clipped < epsilon) {
            clipped = epsilon;
        }
        if (clipped > 1.0 - epsilon) {
            clipped = 1.0 - epsilon;
        }
        u_out[i] = clipped;
    }

    return HDCD_OK;
}
