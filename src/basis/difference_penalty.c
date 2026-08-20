#include "hdcd/bernstein.h"

/* edge_proximity(k) in [0,1]: 1 exactly at either edge (k=0 or
 * k=dim-1), falling linearly to 0 at the grid's center. See the
 * corner-relaxed penalty's derivation in hdcd/bernstein.h. */
static double edge_proximity(size_t k, size_t dim) {
    if (dim <= 1) {
        return 0.0;
    }
    double half = (double)(dim - 1) / 2.0;
    size_t dist_to_nearest_edge = (k < dim - 1 - k) ? k : (dim - 1 - k);
    return 1.0 - (double)dist_to_nearest_edge / half;
}

static double corner_weight(size_t i, size_t j, size_t dim, double corner_relief) {
    return 1.0 - corner_relief * edge_proximity(i, dim) * edge_proximity(j, dim);
}

hdcd_status_t hdcd_bernstein_roughness_penalty_weighted(
    const double *theta, size_t m, double corner_relief,
    double *out
) {
    if (theta == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(corner_relief >= 0.0) || !(corner_relief < 1.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    double penalty = 0.0;

    /* ||Delta_u^2 theta||_F^2: second differences down each column.
     * The residual for triple (r, r+1, r+2) is centered at row r+1. */
    for (size_t s = 0; s < dim; s++) {
        for (size_t r = 0; r + 2 < dim; r++) {
            double d = theta[r * dim + s] - 2.0 * theta[(r + 1) * dim + s] + theta[(r + 2) * dim + s];
            double w = corner_weight(r + 1, s, dim, corner_relief);
            penalty += w * d * d;
        }
    }

    /* ||theta (Delta_z^2)^T||_F^2: second differences across each row.
     * The residual for triple (s, s+1, s+2) is centered at column s+1. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s + 2 < dim; s++) {
            double d = theta[r * dim + s] - 2.0 * theta[r * dim + (s + 1)] + theta[r * dim + (s + 2)];
            double w = corner_weight(r, s + 1, dim, corner_relief);
            penalty += w * d * d;
        }
    }

    *out = penalty;
    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_roughness_penalty(
    const double *theta, size_t m,
    double *out
) {
    return hdcd_bernstein_roughness_penalty_weighted(theta, m, 0.0, out);
}

hdcd_status_t hdcd_bernstein_roughness_gradient_weighted(
    const double *theta, size_t m, double corner_relief,
    double *grad_out
) {
    if (theta == NULL || grad_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(corner_relief >= 0.0) || !(corner_relief < 1.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    for (size_t i = 0; i < dim * dim; i++) {
        grad_out[i] = 0.0;
    }

    /* d/dtheta of the weighted column-difference term: scatter
     * 2*w*d*[1,-2,1] onto the three rows contributing to each residual. */
    for (size_t s = 0; s < dim; s++) {
        for (size_t r = 0; r + 2 < dim; r++) {
            double d = theta[r * dim + s] - 2.0 * theta[(r + 1) * dim + s] + theta[(r + 2) * dim + s];
            double w = corner_weight(r + 1, s, dim, corner_relief);
            grad_out[r * dim + s] += 2.0 * w * d;
            grad_out[(r + 1) * dim + s] += -4.0 * w * d;
            grad_out[(r + 2) * dim + s] += 2.0 * w * d;
        }
    }

    /* d/dtheta of the weighted row-difference term, analogous along rows. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s + 2 < dim; s++) {
            double d = theta[r * dim + s] - 2.0 * theta[r * dim + (s + 1)] + theta[r * dim + (s + 2)];
            double w = corner_weight(r, s + 1, dim, corner_relief);
            grad_out[r * dim + s] += 2.0 * w * d;
            grad_out[r * dim + (s + 1)] += -4.0 * w * d;
            grad_out[r * dim + (s + 2)] += 2.0 * w * d;
        }
    }

    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_roughness_gradient(
    const double *theta, size_t m,
    double *grad_out
) {
    return hdcd_bernstein_roughness_gradient_weighted(theta, m, 0.0, grad_out);
}
