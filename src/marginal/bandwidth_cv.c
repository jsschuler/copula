#include "hdcd/marginal.h"
#include "hdcd/numerics.h"

#include <math.h>
#include <stdlib.h>

#define HDCD_LOG_SQRT_2PI 0.9189385332046727 /* log(sqrt(2*pi)) */
#define HDCD_DEFAULT_LOWER_MULT 0.05
#define HDCD_DEFAULT_UPPER_MULT 3.0

typedef struct {
    const double *data;
    size_t n;
    double *log_terms_scratch; /* length n - 1 */
    hdcd_status_t status;      /* sticky: set on first numerical failure */
} hdcd_loo_context_t;

/* Leave-one-out log-likelihood at a given sigma, summed over all i. */
static double loo_loglik(double sigma, const hdcd_loo_context_t *ctx) {
    const double *data = ctx->data;
    size_t n = ctx->n;
    double log_sigma = log(sigma);
    double total = 0.0;

    for (size_t i = 0; i < n; i++) {
        double xi = data[i];
        size_t count = 0;
        for (size_t r = 0; r < n; r++) {
            if (r == i) {
                continue;
            }
            double z = (xi - data[r]) / sigma;
            ctx->log_terms_scratch[count++] = -0.5 * z * z;
        }

        double lse;
        hdcd_status_t status = hdcd_logsumexp(ctx->log_terms_scratch, count, &lse);
        if (status != HDCD_OK) {
            return NAN;
        }

        double log_fi = lse - log((double)count) - log_sigma - HDCD_LOG_SQRT_2PI;
        if (isnan(log_fi) || isinf(log_fi)) {
            return NAN;
        }
        total += log_fi;
    }

    return total;
}

static double objective_eta(double eta, void *userdata) {
    hdcd_loo_context_t *ctx = (hdcd_loo_context_t *)userdata;
    double sigma = exp(eta);
    double value = loo_loglik(sigma, ctx);
    if (isnan(value)) {
        ctx->status = HDCD_ERROR_NUMERICAL;
    }
    return value;
}

hdcd_status_t hdcd_select_bandwidth_loo(
    const double *data, size_t n,
    double sigma_min, double sigma_max,
    double tol, int max_iter,
    hdcd_bandwidth_result_t *out
) {
    if (data == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n < 2) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(tol > 0.0) || max_iter <= 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    out->sigma = NAN;
    out->eta = NAN;
    out->loglik = NAN;
    out->iterations = 0;
    out->converged = 0;
    out->status = HDCD_OK;

    if (sigma_min <= 0.0 || sigma_max <= 0.0) {
        double scale;
        hdcd_status_t status = hdcd_robust_scale(data, n, &scale);
        if (status != HDCD_OK) {
            out->status = status;
            return status;
        }
        if (!(scale > 0.0)) {
            out->status = HDCD_ERROR_NUMERICAL;
            return HDCD_ERROR_NUMERICAL;
        }
        sigma_min = HDCD_DEFAULT_LOWER_MULT * scale;
        sigma_max = HDCD_DEFAULT_UPPER_MULT * scale;
    }

    if (!(sigma_min < sigma_max) || !(sigma_min > 0.0)) {
        out->status = HDCD_ERROR_INVALID_ARGUMENT;
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    out->lower = sigma_min;
    out->upper = sigma_max;

    double *scratch = (double *)malloc((n - 1) * sizeof(double));
    if (scratch == NULL) {
        out->status = HDCD_ERROR_ALLOCATION;
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_loo_context_t ctx;
    ctx.data = data;
    ctx.n = n;
    ctx.log_terms_scratch = scratch;
    ctx.status = HDCD_OK;

    double eta_min = log(sigma_min);
    double eta_max = log(sigma_max);

    hdcd_optimizer_1d_result_t opt = hdcd_golden_section_maximize(
        objective_eta, &ctx, eta_min, eta_max, tol, max_iter
    );

    free(scratch);

    if (ctx.status != HDCD_OK) {
        out->status = ctx.status;
        return ctx.status;
    }
    if (opt.status != HDCD_OK && opt.status != HDCD_ERROR_NOT_CONVERGED) {
        out->status = opt.status;
        return opt.status;
    }

    out->eta = opt.x_opt;
    out->sigma = exp(opt.x_opt);
    out->loglik = opt.f_opt;
    out->iterations = opt.iterations;
    out->converged = opt.converged;
    out->status = opt.status;

    return opt.status;
}
