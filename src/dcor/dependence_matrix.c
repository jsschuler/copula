#include "hdcd/dcor.h"

#include <math.h>
#include <stdlib.h>

struct hdcd_dependence_matrix {
    size_t d;
    double *D;       /* d x d, row-major */
    size_t *n_eff;   /* d x d, row-major */
};

hdcd_status_t hdcd_compute_dependence_matrix(
    const double *u,
    const uint8_t *observed_mask,
    size_t n, size_t d,
    hdcd_dependence_matrix_t **out
) {
    if (u == NULL || observed_mask == NULL || out == NULL || n == 0 || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    double *D = (double *)malloc(d * d * sizeof(double));
    size_t *n_eff = (size_t *)malloc(d * d * sizeof(size_t));
    double *xbuf = (double *)malloc(n * sizeof(double));
    double *ybuf = (double *)malloc(n * sizeof(double));
    if (D == NULL || n_eff == NULL || xbuf == NULL || ybuf == NULL) {
        free(D);
        free(n_eff);
        free(xbuf);
        free(ybuf);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_status_t status = HDCD_OK;

    for (size_t j = 0; j < d && status == HDCD_OK; j++) {
        size_t n_j = 0;
        for (size_t i = 0; i < n; i++) {
            if (observed_mask[j * n + i]) {
                n_j++;
            }
        }
        D[j * d + j] = 1.0;
        n_eff[j * d + j] = n_j;

        for (size_t k = j + 1; k < d; k++) {
            size_t m = 0;
            for (size_t i = 0; i < n; i++) {
                if (observed_mask[j * n + i] && observed_mask[k * n + i]) {
                    xbuf[m] = u[j * n + i];
                    ybuf[m] = u[k * n + i];
                    m++;
                }
            }

            double dcor_val;
            if (m < 2) {
                /* Pairwise-complete sample too small for dCor to be
                 * defined; do not fabricate a value (spec section 5:
                 * store the effective sample size, don't hide it). */
                dcor_val = NAN;
            } else {
                status = hdcd_dcor_exact(xbuf, ybuf, m, &dcor_val);
                if (status != HDCD_OK) {
                    break;
                }
            }

            D[j * d + k] = dcor_val;
            D[k * d + j] = dcor_val;
            n_eff[j * d + k] = m;
            n_eff[k * d + j] = m;
        }
    }

    free(xbuf);
    free(ybuf);

    if (status != HDCD_OK) {
        free(D);
        free(n_eff);
        return status;
    }

    hdcd_dependence_matrix_t *dm = (hdcd_dependence_matrix_t *)malloc(sizeof(hdcd_dependence_matrix_t));
    if (dm == NULL) {
        free(D);
        free(n_eff);
        return HDCD_ERROR_ALLOCATION;
    }
    dm->d = d;
    dm->D = D;
    dm->n_eff = n_eff;

    *out = dm;
    return HDCD_OK;
}

void hdcd_dependence_matrix_free(hdcd_dependence_matrix_t *dm) {
    if (dm == NULL) {
        return;
    }
    free(dm->D);
    free(dm->n_eff);
    free(dm);
}

size_t hdcd_dependence_matrix_dim(const hdcd_dependence_matrix_t *dm) {
    return (dm != NULL) ? dm->d : 0;
}

double hdcd_dependence_matrix_get(const hdcd_dependence_matrix_t *dm, size_t j, size_t k) {
    if (dm == NULL || j >= dm->d || k >= dm->d) {
        return NAN;
    }
    return dm->D[j * dm->d + k];
}

size_t hdcd_dependence_matrix_n_effective(const hdcd_dependence_matrix_t *dm, size_t j, size_t k) {
    if (dm == NULL || j >= dm->d || k >= dm->d) {
        return 0;
    }
    return dm->n_eff[j * dm->d + k];
}
