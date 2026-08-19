#include "hdcd/marginal.h"

#include <stdlib.h>
#include <string.h>

struct hdcd_marginal {
    double *data;       /* owned copy of observed training values, size n_obs */
    size_t n_obs;
    hdcd_bandwidth_result_t bandwidth;
};

hdcd_status_t hdcd_marginal_fit(
    const double *x, const uint8_t *observed_mask, size_t n,
    double sigma_min, double sigma_max,
    double tol, int max_iter,
    hdcd_marginal_t **out
) {
    if (x == NULL || observed_mask == NULL || out == NULL || n == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    size_t n_obs = 0;
    for (size_t i = 0; i < n; i++) {
        if (observed_mask[i]) {
            n_obs++;
        }
    }
    if (n_obs == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double *data = (double *)malloc(n_obs * sizeof(double));
    if (data == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (observed_mask[i]) {
            data[k++] = x[i];
        }
    }

    hdcd_bandwidth_result_t bandwidth;
    hdcd_status_t status = hdcd_select_bandwidth_loo(
        data, n_obs, sigma_min, sigma_max, tol, max_iter, &bandwidth
    );
    if (status != HDCD_OK) {
        free(data);
        return status;
    }

    hdcd_marginal_t *marginal = (hdcd_marginal_t *)malloc(sizeof(hdcd_marginal_t));
    if (marginal == NULL) {
        free(data);
        return HDCD_ERROR_ALLOCATION;
    }
    marginal->data = data;
    marginal->n_obs = n_obs;
    marginal->bandwidth = bandwidth;

    *out = marginal;
    return HDCD_OK;
}

void hdcd_marginal_free(hdcd_marginal_t *marginal) {
    if (marginal == NULL) {
        return;
    }
    free(marginal->data);
    free(marginal);
}

size_t hdcd_marginal_n_observed(const hdcd_marginal_t *marginal) {
    return (marginal != NULL) ? marginal->n_obs : 0;
}

hdcd_bandwidth_result_t hdcd_marginal_bandwidth_result(const hdcd_marginal_t *marginal) {
    hdcd_bandwidth_result_t empty;
    memset(&empty, 0, sizeof(empty));
    if (marginal == NULL) {
        empty.status = HDCD_ERROR_INVALID_ARGUMENT;
        return empty;
    }
    return marginal->bandwidth;
}

hdcd_status_t hdcd_marginal_cdf(
    const hdcd_marginal_t *marginal,
    const double *eval_points, size_t m,
    double *out
) {
    if (marginal == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    return hdcd_gaussian_mixture_cdf(
        marginal->data, marginal->n_obs, marginal->bandwidth.sigma, eval_points, m, out
    );
}
