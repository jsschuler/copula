#include "hdcd/numerics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Consistency constant so MAD estimates the standard deviation under
 * normality: 1 / Phi^{-1}(3/4). */
#define HDCD_MAD_CONSISTENCY_CONSTANT 1.4826

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* Linear-interpolation ("Type 7") quantile on a already-sorted array. */
static double sorted_quantile(const double *sorted, size_t n, double p) {
    if (n == 1) {
        return sorted[0];
    }
    double h = (double)(n - 1) * p;
    size_t lo = (size_t)floor(h);
    size_t hi = (size_t)ceil(h);
    if (hi >= n) {
        hi = n - 1;
    }
    double frac = h - (double)lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

static double median_sorted(const double *sorted, size_t n) {
    return sorted_quantile(sorted, n, 0.5);
}

hdcd_status_t hdcd_mad(const double *x, size_t n, double *out) {
    if (x == NULL || out == NULL || n == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double *sorted = (double *)malloc(n * sizeof(double));
    if (sorted == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    memcpy(sorted, x, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);
    double med = median_sorted(sorted, n);

    double *abs_dev = sorted; /* reuse buffer */
    for (size_t i = 0; i < n; i++) {
        abs_dev[i] = fabs(x[i] - med);
    }
    qsort(abs_dev, n, sizeof(double), compare_doubles);
    double mad_raw = median_sorted(abs_dev, n);

    free(sorted);

    *out = HDCD_MAD_CONSISTENCY_CONSTANT * mad_raw;
    return HDCD_OK;
}

hdcd_status_t hdcd_iqr(const double *x, size_t n, double *out) {
    if (x == NULL || out == NULL || n < 2) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double *sorted = (double *)malloc(n * sizeof(double));
    if (sorted == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    memcpy(sorted, x, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_doubles);

    double q1 = sorted_quantile(sorted, n, 0.25);
    double q3 = sorted_quantile(sorted, n, 0.75);
    free(sorted);

    *out = q3 - q1;
    return HDCD_OK;
}

hdcd_status_t hdcd_robust_scale(const double *x, size_t n, double *out) {
    if (x == NULL || out == NULL || n < 2) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double iqr_scale = 0.0;
    hdcd_status_t status = hdcd_iqr(x, n, &iqr_scale);
    if (status != HDCD_OK) {
        return status;
    }
    iqr_scale /= 1.349;

    double mad_scale = 0.0;
    status = hdcd_mad(x, n, &mad_scale);
    if (status != HDCD_OK) {
        return status;
    }

    double scale;
    if (iqr_scale > 0.0 && mad_scale > 0.0) {
        scale = (iqr_scale < mad_scale) ? iqr_scale : mad_scale;
    } else if (iqr_scale > 0.0) {
        scale = iqr_scale;
    } else if (mad_scale > 0.0) {
        scale = mad_scale;
    } else {
        /* Both robust measures are degenerate (heavily tied data);
         * fall back to the sample standard deviation. */
        double mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            mean += x[i];
        }
        mean /= (double)n;

        double var = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = x[i] - mean;
            var += d * d;
        }
        var /= (double)(n - 1);
        scale = sqrt(var);
    }

    *out = scale;
    return HDCD_OK;
}
