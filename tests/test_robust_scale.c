#include "hdcd/numerics.h"
#include "test_utils.h"

static void test_mad_symmetric_data(void) {
    /* Symmetric around 0: {-2,-1,0,1,2}. Median = 0.
     * Absolute deviations: {2,1,0,1,2} -> median = 1.
     * MAD = 1.4826 * 1. */
    double x[5] = {-2, -1, 0, 1, 2};
    double out;
    HDCD_CHECK(hdcd_mad(x, 5, &out) == HDCD_OK);
    HDCD_CHECK_NEAR(out, 1.4826, 1e-9);
    HDCD_PASS("MAD on symmetric integer data");
}

static void test_iqr_known(void) {
    /* Sorted 0..9 (n=10). Type-7 quantiles:
     * h = (n-1)*p. Q1: h=2.25 -> 2 + 0.25*(3-2) = 2.25.
     * Q3: h=6.75 -> 6 + 0.75*(7-6) = 6.75. IQR = 4.5. */
    double x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double out;
    HDCD_CHECK(hdcd_iqr(x, 10, &out) == HDCD_OK);
    HDCD_CHECK_NEAR(out, 4.5, 1e-9);
    HDCD_PASS("IQR on 0..9 matches Type-7 quantile formula");
}

static void test_robust_scale_positive(void) {
    double x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double out;
    HDCD_CHECK(hdcd_robust_scale(x, 10, &out) == HDCD_OK);
    HDCD_CHECK(out > 0.0);
    HDCD_PASS("robust_scale positive on spread-out data");
}

static void test_robust_scale_degenerate_falls_back(void) {
    /* Heavily tied data: IQR == 0 and MAD == 0, must fall back to sd. */
    double x[6] = {5, 5, 5, 5, 5, 6};
    double out;
    HDCD_CHECK(hdcd_robust_scale(x, 6, &out) == HDCD_OK);
    HDCD_CHECK(out > 0.0);
    HDCD_PASS("robust_scale falls back to sample sd when degenerate");
}

static void test_invalid_arguments(void) {
    double out;
    double x[1] = {1.0};
    HDCD_CHECK(hdcd_mad(NULL, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_mad(x, 0, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_iqr(x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_robust_scale(x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("robust scale functions reject invalid arguments");
}

int main(void) {
    test_mad_symmetric_data();
    test_iqr_known();
    test_robust_scale_positive();
    test_robust_scale_degenerate_falls_back();
    test_invalid_arguments();
    printf("All robust_scale tests passed.\n");
    return 0;
}
