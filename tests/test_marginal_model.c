#include "hdcd/marginal.h"
#include "test_utils.h"

static double sample_data[20] = {
    -1.83, -1.42, -1.10, -0.95, -0.71, -0.58, -0.33, -0.21, -0.05, 0.02,
     0.14,  0.27,  0.39,  0.52,  0.68,  0.81,  1.03,  1.29,  1.61,  2.05
};

static void test_fit_all_observed(void) {
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;

    hdcd_marginal_t *m = NULL;
    hdcd_status_t status = hdcd_marginal_fit(sample_data, mask, 20, -1.0, -1.0, 1e-6, 200, &m);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(m != NULL);
    HDCD_CHECK(hdcd_marginal_n_observed(m) == 20);

    hdcd_bandwidth_result_t bw = hdcd_marginal_bandwidth_result(m);
    HDCD_CHECK(bw.status == HDCD_OK);
    HDCD_CHECK(bw.sigma > 0.0);

    hdcd_marginal_free(m);
    HDCD_PASS("fit with fully observed data");
}

static void test_fit_with_missingness(void) {
    /* Mark every third entry missing; n_observed must reflect only the
     * mask, per O_j in spec section 2 -- fitting must not silently use
     * the whole array (spec section 36 rule 7: no complete-case deletion
     * of the whole dataset, but also no ignoring the mask). */
    uint8_t mask[20];
    size_t expected_n_obs = 0;
    for (int i = 0; i < 20; i++) {
        mask[i] = (i % 3 != 0) ? 1 : 0;
        if (mask[i]) expected_n_obs++;
    }

    hdcd_marginal_t *m = NULL;
    hdcd_status_t status = hdcd_marginal_fit(sample_data, mask, 20, -1.0, -1.0, 1e-6, 200, &m);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(hdcd_marginal_n_observed(m) == expected_n_obs);

    hdcd_marginal_free(m);
    HDCD_PASS("fit respects the observed mask");
}

static void test_cdf_monotonic_and_bounded(void) {
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;
    hdcd_marginal_t *m = NULL;
    HDCD_CHECK(hdcd_marginal_fit(sample_data, mask, 20, -1.0, -1.0, 1e-6, 200, &m) == HDCD_OK);

    double points[7] = {-5, -2, -0.5, 0, 0.5, 2, 5};
    double cdf[7];
    HDCD_CHECK(hdcd_marginal_cdf(m, points, 7, cdf) == HDCD_OK);

    for (int i = 0; i < 7; i++) {
        HDCD_CHECK(cdf[i] >= 0.0 && cdf[i] <= 1.0);
        if (i > 0) {
            HDCD_CHECK(cdf[i] >= cdf[i - 1]);
        }
    }
    HDCD_CHECK(cdf[0] < 0.01);
    HDCD_CHECK(cdf[6] > 0.99);

    hdcd_marginal_free(m);
    HDCD_PASS("fitted marginal CDF is monotonic and bounded in [0,1]");
}

static void test_invalid_arguments(void) {
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;
    hdcd_marginal_t *m = NULL;

    HDCD_CHECK(hdcd_marginal_fit(NULL, mask, 20, -1.0, -1.0, 1e-6, 200, &m) == HDCD_ERROR_INVALID_ARGUMENT);

    uint8_t all_missing[20] = {0};
    HDCD_CHECK(hdcd_marginal_fit(sample_data, all_missing, 20, -1.0, -1.0, 1e-6, 200, &m)
               == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_marginal_cdf(NULL, sample_data, 1, sample_data) == HDCD_ERROR_INVALID_ARGUMENT);

    /* hdcd_marginal_free(NULL) must not crash. */
    hdcd_marginal_free(NULL);

    HDCD_PASS("marginal fit/cdf reject invalid arguments");
}

int main(void) {
    test_fit_all_observed();
    test_fit_with_missingness();
    test_cdf_monotonic_and_bounded();
    test_invalid_arguments();
    printf("All marginal_model tests passed.\n");
    return 0;
}
