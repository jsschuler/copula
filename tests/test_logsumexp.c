#include "hdcd/numerics.h"
#include "test_utils.h"

static void test_known_values(void) {
    double terms[2] = {0.0, 0.0};
    double out;
    HDCD_CHECK(hdcd_logsumexp(terms, 2, &out) == HDCD_OK);
    HDCD_CHECK_NEAR(out, log(2.0), 1e-12);
    HDCD_PASS("logsumexp known value log(2)");
}

static void test_single_term(void) {
    double terms[1] = {3.5};
    double out;
    HDCD_CHECK(hdcd_logsumexp(terms, 1, &out) == HDCD_OK);
    HDCD_CHECK_NEAR(out, 3.5, 1e-12);
    HDCD_PASS("logsumexp single term is identity");
}

static void test_stability_large_values(void) {
    /* Naive sum(exp(x)) would overflow to inf for x this large;
     * logsumexp must stay finite and accurate. */
    double terms[3] = {1000.0, 1000.0, 999.0};
    double out;
    HDCD_CHECK(hdcd_logsumexp(terms, 3, &out) == HDCD_OK);
    HDCD_CHECK(isfinite(out));
    /* Reference: 1000 + log(1 + 1 + exp(-1)) computed by hand. */
    double expected = 1000.0 + log(2.0 + exp(-1.0));
    HDCD_CHECK_NEAR(out, expected, 1e-9);
    HDCD_PASS("logsumexp stable for large magnitude terms");
}

static void test_all_neg_infinity(void) {
    double terms[2] = {-INFINITY, -INFINITY};
    double out;
    HDCD_CHECK(hdcd_logsumexp(terms, 2, &out) == HDCD_OK);
    HDCD_CHECK(isinf(out) && out < 0.0);
    HDCD_PASS("logsumexp handles all -inf without NaN");
}

static void test_log_mean_exp(void) {
    double terms[4] = {0.0, 0.0, 0.0, 0.0};
    double out;
    HDCD_CHECK(hdcd_log_mean_exp(terms, 4, &out) == HDCD_OK);
    HDCD_CHECK_NEAR(out, 0.0, 1e-12); /* mean(exp(0)) = 1, log(1) = 0 */
    HDCD_PASS("log_mean_exp of constant terms");
}

static void test_invalid_arguments(void) {
    double out;
    HDCD_CHECK(hdcd_logsumexp(NULL, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    double terms[1] = {0.0};
    HDCD_CHECK(hdcd_logsumexp(terms, 0, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_logsumexp(terms, 1, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("logsumexp rejects invalid arguments");
}

static void test_nan_propagation_is_rejected(void) {
    double terms[2] = {0.0, NAN};
    double out;
    HDCD_CHECK(hdcd_logsumexp(terms, 2, &out) == HDCD_ERROR_NUMERICAL);
    HDCD_PASS("logsumexp reports NaN input as numerical error");
}

int main(void) {
    test_known_values();
    test_single_term();
    test_stability_large_values();
    test_all_neg_infinity();
    test_log_mean_exp();
    test_invalid_arguments();
    test_nan_propagation_is_rejected();
    printf("All logsumexp tests passed.\n");
    return 0;
}
