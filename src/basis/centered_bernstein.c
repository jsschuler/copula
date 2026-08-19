#include "hdcd/bernstein.h"

#include <stdlib.h>

hdcd_status_t hdcd_bernstein_tensor_interaction(
    double u, double z, size_t m,
    const double *theta,
    double *out
) {
    if (theta == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    double *bu = (double *)malloc(dim * sizeof(double));
    double *bz = (double *)malloc(dim * sizeof(double));
    if (bu == NULL || bz == NULL) {
        free(bu);
        free(bz);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_status_t status = hdcd_bernstein_basis_centered(u, m, bu);
    if (status == HDCD_OK) {
        status = hdcd_bernstein_basis_centered(z, m, bz);
    }
    if (status != HDCD_OK) {
        free(bu);
        free(bz);
        return status;
    }

    double total = 0.0;
    for (size_t r = 0; r < dim; r++) {
        double row_sum = 0.0;
        for (size_t s = 0; s < dim; s++) {
            row_sum += theta[r * dim + s] * bz[s];
        }
        total += bu[r] * row_sum;
    }

    free(bu);
    free(bz);

    *out = total;
    return HDCD_OK;
}

hdcd_status_t hdcd_bernstein_tensor_gradient(
    double u, double z, size_t m,
    double *grad_out
) {
    if (grad_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t dim = m + 1;
    double *bu = (double *)malloc(dim * sizeof(double));
    double *bz = (double *)malloc(dim * sizeof(double));
    if (bu == NULL || bz == NULL) {
        free(bu);
        free(bz);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_status_t status = hdcd_bernstein_basis_centered(u, m, bu);
    if (status == HDCD_OK) {
        status = hdcd_bernstein_basis_centered(z, m, bz);
    }
    if (status != HDCD_OK) {
        free(bu);
        free(bz);
        return status;
    }

    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            grad_out[r * dim + s] = bu[r] * bz[s];
        }
    }

    free(bu);
    free(bz);
    return HDCD_OK;
}
