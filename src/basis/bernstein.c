#include "hdcd/bernstein.h"

#include <math.h>
#include <stdlib.h>

hdcd_status_t hdcd_bernstein_basis(double u, size_t m, double *out) {
    if (out == NULL || isnan(u) || u < 0.0 || u > 1.0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double binom = 1.0; /* C(m,0) */
    for (size_t r = 0; r <= m; r++) {
        double value = binom * pow(u, (double)r) * pow(1.0 - u, (double)(m - r));
        if (isnan(value)) {
            return HDCD_ERROR_NUMERICAL;
        }
        out[r] = value;
        if (r < m) {
            binom *= (double)(m - r) / (double)(r + 1);
        }
    }
    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_basis_centered(double u, size_t m, double *out) {
    hdcd_status_t status = hdcd_bernstein_basis(u, m, out);
    if (status != HDCD_OK) {
        return status;
    }
    double shift = 1.0 / (double)(m + 1);
    for (size_t r = 0; r <= m; r++) {
        out[r] -= shift;
    }
    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_basis_derivative(double u, size_t m, double *out) {
    if (out == NULL || isnan(u) || u < 0.0 || u > 1.0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    if (m == 0) {
        out[0] = 0.0;
        return HDCD_OK;
    }

    double *lower = (double *)malloc(m * sizeof(double)); /* degree m-1, indices 0..m-1 */
    if (lower == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    hdcd_status_t status = hdcd_bernstein_basis(u, m - 1, lower);
    if (status != HDCD_OK) {
        free(lower);
        return status;
    }

    for (size_t r = 0; r <= m; r++) {
        double left = (r >= 1) ? lower[r - 1] : 0.0;
        double right = (r <= m - 1) ? lower[r] : 0.0;
        out[r] = (double)m * (left - right);
    }

    free(lower);
    return HDCD_OK;
}
