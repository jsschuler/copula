#include "hdcd/marginal.h"
#include "test_utils.h"

#include <stdlib.h>

/* Deterministic dataset (no RNG), roughly standard-normal-shaped. */
static double sample_data[20] = {
    -1.83, -1.42, -1.10, -0.95, -0.71, -0.58, -0.33, -0.21, -0.05, 0.02,
     0.14,  0.27,  0.39,  0.52,  0.68,  0.81,  1.03,  1.29,  1.61,  2.05
};

/* Black-box leave-one-out log-likelihood computed only through the
 * public logpdf API, independent of the internal optimizer/objective,
 * used to check local optimality of the selected bandwidth. */
static double loo_loglik_reference(const double *data, size_t n, double sigma) {
    double *subset = (double *)malloc((n - 1) * sizeof(double));
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        size_t count = 0;
        for (size_t r = 0; r < n; r++) {
            if (r != i) {
                subset[count++] = data[r];
            }
        }
        double log_fi;
        hdcd_status_t status = hdcd_gaussian_mixture_logpdf(subset, count, sigma, &data[i], 1, &log_fi);
        HDCD_CHECK(status == HDCD_OK);
        total += log_fi;
    }
    free(subset);
    return total;
}

static void test_selected_bandwidth_in_bounds(void) {
    hdcd_bandwidth_result_t r;
    hdcd_status_t status = hdcd_select_bandwidth_loo(sample_data, 20, -1.0, -1.0, 1e-6, 200, &r);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(r.converged);
    HDCD_CHECK(r.sigma > r.lower && r.sigma < r.upper);
    HDCD_CHECK(r.sigma > 0.0);
    HDCD_PASS("selected bandwidth lies within derived bounds");
}

static void test_reproducibility(void) {
    hdcd_bandwidth_result_t r1, r2;
    HDCD_CHECK(hdcd_select_bandwidth_loo(sample_data, 20, -1.0, -1.0, 1e-6, 200, &r1) == HDCD_OK);
    HDCD_CHECK(hdcd_select_bandwidth_loo(sample_data, 20, -1.0, -1.0, 1e-6, 200, &r2) == HDCD_OK);
    HDCD_CHECK(r1.sigma == r2.sigma);
    HDCD_CHECK(r1.iterations == r2.iterations);
    HDCD_PASS("bandwidth selection is exactly reproducible");
}

static void test_local_optimality(void) {
    hdcd_bandwidth_result_t r;
    HDCD_CHECK(hdcd_select_bandwidth_loo(sample_data, 20, -1.0, -1.0, 1e-6, 200, &r) == HDCD_OK);

    double ll_at_opt = loo_loglik_reference(sample_data, 20, r.sigma);
    double ll_smaller = loo_loglik_reference(sample_data, 20, r.sigma * 0.5);
    double ll_larger = loo_loglik_reference(sample_data, 20, r.sigma * 2.0);

    HDCD_CHECK(ll_at_opt >= ll_smaller);
    HDCD_CHECK(ll_at_opt >= ll_larger);
    HDCD_PASS("selected bandwidth is locally optimal vs half/double sigma");
}

static void test_explicit_bounds_respected(void) {
    hdcd_bandwidth_result_t r;
    hdcd_status_t status = hdcd_select_bandwidth_loo(sample_data, 20, 0.2, 0.3, 1e-6, 200, &r);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(r.sigma >= 0.2 && r.sigma <= 0.3);
    HDCD_CHECK_NEAR(r.lower, 0.2, 1e-12);
    HDCD_CHECK_NEAR(r.upper, 0.3, 1e-12);
    HDCD_PASS("explicit caller-supplied bounds are respected");
}

static void test_invalid_arguments(void) {
    hdcd_bandwidth_result_t r;
    HDCD_CHECK(hdcd_select_bandwidth_loo(NULL, 20, -1.0, -1.0, 1e-6, 200, &r) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_select_bandwidth_loo(sample_data, 1, -1.0, -1.0, 1e-6, 200, &r) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_select_bandwidth_loo(sample_data, 20, 0.5, 0.1, 1e-6, 200, &r) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("bandwidth selection rejects invalid arguments");
}

int main(void) {
    test_selected_bandwidth_in_bounds();
    test_reproducibility();
    test_local_optimality();
    test_explicit_bounds_respected();
    test_invalid_arguments();
    printf("All bandwidth_cv tests passed.\n");
    return 0;
}
