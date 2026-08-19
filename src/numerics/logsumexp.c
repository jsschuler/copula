#include "hdcd/numerics.h"

#include <math.h>
#include <stddef.h>

hdcd_status_t hdcd_logsumexp(const double *log_terms, size_t n, double *out) {
    if (log_terms == NULL || out == NULL || n == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double max_val = log_terms[0];
    for (size_t i = 1; i < n; i++) {
        if (isnan(log_terms[i])) {
            return HDCD_ERROR_NUMERICAL;
        }
        if (log_terms[i] > max_val) {
            max_val = log_terms[i];
        }
    }
    if (isnan(max_val)) {
        return HDCD_ERROR_NUMERICAL;
    }

    /* All terms are -inf: sum is 0, log is -inf. Avoid NaN from (-inf - -inf). */
    if (isinf(max_val) && max_val < 0.0) {
        *out = -INFINITY;
        return HDCD_OK;
    }

    double accum = 0.0;
    for (size_t i = 0; i < n; i++) {
        accum += exp(log_terms[i] - max_val);
    }

    double result = max_val + log(accum);
    if (isnan(result)) {
        return HDCD_ERROR_NUMERICAL;
    }

    *out = result;
    return HDCD_OK;
}

hdcd_status_t hdcd_log_mean_exp(const double *log_terms, size_t n, double *out) {
    if (out == NULL || n == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double lse;
    hdcd_status_t status = hdcd_logsumexp(log_terms, n, &lse);
    if (status != HDCD_OK) {
        return status;
    }

    *out = lse - log((double)n);
    return HDCD_OK;
}
