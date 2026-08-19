#include "hdcd/bernstein.h"

hdcd_status_t hdcd_bernstein_roughness_penalty(
    const double *theta, size_t m,
    double *out
) {
    if (theta == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    double penalty = 0.0;

    /* ||Delta_u^2 theta||_F^2: second differences down each column. */
    for (size_t s = 0; s < dim; s++) {
        for (size_t r = 0; r + 2 < dim; r++) {
            double d = theta[r * dim + s] - 2.0 * theta[(r + 1) * dim + s] + theta[(r + 2) * dim + s];
            penalty += d * d;
        }
    }

    /* ||theta (Delta_z^2)^T||_F^2: second differences across each row. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s + 2 < dim; s++) {
            double d = theta[r * dim + s] - 2.0 * theta[r * dim + (s + 1)] + theta[r * dim + (s + 2)];
            penalty += d * d;
        }
    }

    *out = penalty;
    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_roughness_gradient(
    const double *theta, size_t m,
    double *grad_out
) {
    if (theta == NULL || grad_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    for (size_t i = 0; i < dim * dim; i++) {
        grad_out[i] = 0.0;
    }

    /* d/dtheta of ||Delta_u^2 theta||_F^2: scatter 2*d*[1,-2,1] onto the
     * three rows contributing to each second-difference residual d. */
    for (size_t s = 0; s < dim; s++) {
        for (size_t r = 0; r + 2 < dim; r++) {
            double d = theta[r * dim + s] - 2.0 * theta[(r + 1) * dim + s] + theta[(r + 2) * dim + s];
            grad_out[r * dim + s] += 2.0 * d;
            grad_out[(r + 1) * dim + s] += -4.0 * d;
            grad_out[(r + 2) * dim + s] += 2.0 * d;
        }
    }

    /* d/dtheta of ||theta (Delta_z^2)^T||_F^2, analogous along rows. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s + 2 < dim; s++) {
            double d = theta[r * dim + s] - 2.0 * theta[r * dim + (s + 1)] + theta[r * dim + (s + 2)];
            grad_out[r * dim + s] += 2.0 * d;
            grad_out[r * dim + (s + 1)] += -4.0 * d;
            grad_out[r * dim + (s + 2)] += 2.0 * d;
        }
    }

    return HDCD_OK;
}
