#include "hdcd/dcor.h"

#include <math.h>
#include <stdlib.h>

hdcd_status_t hdcd_dcor_exact(const double *x, const double *y, size_t n, double *out) {
    if (x == NULL || y == NULL || out == NULL || n < 2) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double *row_sum_a = (double *)malloc(n * sizeof(double));
    double *row_sum_b = (double *)malloc(n * sizeof(double));
    if (row_sum_a == NULL || row_sum_b == NULL) {
        free(row_sum_a);
        free(row_sum_b);
        return HDCD_ERROR_ALLOCATION;
    }

    for (size_t i = 0; i < n; i++) {
        row_sum_a[i] = 0.0;
        row_sum_b[i] = 0.0;
    }

    double total_a = 0.0, total_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double aij = fabs(x[i] - x[j]);
            double bij = fabs(y[i] - y[j]);
            row_sum_a[i] += aij;
            row_sum_b[i] += bij;
            total_a += aij;
            total_b += bij;
        }
    }

    double n2 = (double)n * (double)n;
    double grand_a = total_a / n2;
    double grand_b = total_b / n2;

    double dcov = 0.0, dvar_x = 0.0, dvar_y = 0.0;
    for (size_t i = 0; i < n; i++) {
        double rmi_a = row_sum_a[i] / (double)n;
        double rmi_b = row_sum_b[i] / (double)n;
        for (size_t j = 0; j < n; j++) {
            double rmj_a = row_sum_a[j] / (double)n;
            double rmj_b = row_sum_b[j] / (double)n;

            double aij = fabs(x[i] - x[j]);
            double bij = fabs(y[i] - y[j]);

            double Aij = aij - rmi_a - rmj_a + grand_a;
            double Bij = bij - rmi_b - rmj_b + grand_b;

            dcov += Aij * Bij;
            dvar_x += Aij * Aij;
            dvar_y += Bij * Bij;
        }
    }

    free(row_sum_a);
    free(row_sum_b);

    dcov /= n2;
    dvar_x /= n2;
    dvar_y /= n2;

    if (isnan(dcov) || isnan(dvar_x) || isnan(dvar_y)) {
        return HDCD_ERROR_NUMERICAL;
    }

    /* Constant series (zero distance variance): define dCor = 0 rather
     * than the 0/0 the raw formula would produce. */
    if (dvar_x <= 0.0 || dvar_y <= 0.0) {
        *out = 0.0;
        return HDCD_OK;
    }

    double denom = sqrt(dvar_x * dvar_y);
    double dcor_sq = dcov / denom;
    /* dCov^2(X,Y) is mathematically non-negative; guard tiny negative
     * floating-point noise rather than feeding sqrt() a negative value. */
    if (dcor_sq < 0.0) {
        dcor_sq = 0.0;
    }

    double dcor = sqrt(dcor_sq);
    if (dcor > 1.0) {
        dcor = 1.0; /* guard tiny numerical overshoot past the theoretical bound */
    }

    *out = dcor;
    return HDCD_OK;
}
