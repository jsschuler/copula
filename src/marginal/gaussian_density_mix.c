#include "hdcd/marginal.h"
#include "hdcd/numerics.h"

#include <math.h>
#include <stdlib.h>

#define HDCD_LOG_SQRT_2PI 0.9189385332046727 /* log(sqrt(2*pi)) */

static hdcd_status_t validate_inputs(
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
    return HDCD_OK;
}

hdcd_status_t hdcd_gaussian_mixture_logpdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
) {
    hdcd_status_t status = validate_inputs(data, n, sigma, eval_points, m, out);
    if (status != HDCD_OK) {
        return status;
    }

    double *log_terms = (double *)malloc(n * sizeof(double));
    if (log_terms == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }

    double log_sigma = log(sigma);

    for (size_t k = 0; k < m; k++) {
        double x = eval_points[k];
        for (size_t i = 0; i < n; i++) {
            double z = (x - data[i]) / sigma;
            log_terms[i] = -0.5 * z * z;
        }

        double lme;
        status = hdcd_log_mean_exp(log_terms, n, &lme);
        if (status != HDCD_OK) {
            free(log_terms);
            return status;
        }

        out[k] = lme - log_sigma - HDCD_LOG_SQRT_2PI;
    }

    free(log_terms);
    return HDCD_OK;
}

hdcd_status_t hdcd_gaussian_mixture_pdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
) {
    hdcd_status_t status = hdcd_gaussian_mixture_logpdf(data, n, sigma, eval_points, m, out);
    if (status != HDCD_OK) {
        return status;
    }
    for (size_t k = 0; k < m; k++) {
        out[k] = exp(out[k]);
    }
    return HDCD_OK;
}

hdcd_status_t hdcd_gaussian_mixture_dpdf(
    const double *data, size_t n,
    double sigma,
    const double *eval_points, size_t m,
    double *out
) {
    hdcd_status_t status = validate_inputs(data, n, sigma, eval_points, m, out);
    if (status != HDCD_OK) {
        return status;
    }

    double sigma3 = sigma * sigma * sigma;
    double norm_const = 0.3989422804014327; /* 1 / sqrt(2*pi) */

    for (size_t k = 0; k < m; k++) {
        double x = eval_points[k];
        double accum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = x - data[i];
            double z = diff / sigma;
            double phi = norm_const * exp(-0.5 * z * z);
            accum += diff * phi;
        }
        double value = -accum / ((double)n * sigma3);
        if (isnan(value)) {
            return HDCD_ERROR_NUMERICAL;
        }
        out[k] = value;
    }

    return HDCD_OK;
}
