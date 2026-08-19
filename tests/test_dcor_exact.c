#include "hdcd/dcor.h"
#include "test_utils.h"

#include <stdint.h>
#include <stdlib.h>

/* Self-contained deterministic PRNG, test-only (see the note in
 * tests/test_copula_transform.c -- this is unrelated to the library's
 * future seeded rng/ module for sampling and annealing). */
static uint64_t rng_state;

static void rng_seed(uint64_t seed) {
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t result = rng_state * 0x2545F4914F6CDD1DULL;
    double u = (double)(result >> 11) * (1.0 / 9007199254740992.0);
    if (u <= 0.0) u = 1e-12;
    if (u >= 1.0) u = 1.0 - 1e-12;
    return u;
}

static double rng_normal(void) {
    const double two_pi = 6.283185307179586;
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}

static void test_symmetry(void) {
    double x[6] = {0.1, -0.4, 0.9, 1.3, -1.1, 0.5};
    double y[6] = {0.3, 0.2, -0.8, 1.1, 0.0, -0.6};

    double d_xy, d_yx;
    HDCD_CHECK(hdcd_dcor_exact(x, y, 6, &d_xy) == HDCD_OK);
    HDCD_CHECK(hdcd_dcor_exact(y, x, 6, &d_yx) == HDCD_OK);
    HDCD_CHECK_NEAR(d_xy, d_yx, 1e-12);
    HDCD_PASS("dCor(x,y) == dCor(y,x)");
}

static void test_self_correlation_near_one(void) {
    double x[8] = {0.1, -0.4, 0.9, 1.3, -1.1, 0.5, 2.0, -0.7};
    double d;
    HDCD_CHECK(hdcd_dcor_exact(x, x, 8, &d) == HDCD_OK);
    HDCD_CHECK_NEAR(d, 1.0, 1e-9);
    HDCD_PASS("dCor(x,x) is 1 for non-constant x");
}

static void test_constant_series_is_zero(void) {
    double x[5] = {3.0, 3.0, 3.0, 3.0, 3.0};
    double y[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double d;
    HDCD_CHECK(hdcd_dcor_exact(x, y, 5, &d) == HDCD_OK);
    HDCD_CHECK_NEAR(d, 0.0, 1e-12);
    HDCD_PASS("dCor is 0 for a constant series, not NaN");
}

static void test_independence_benchmark_near_zero(void) {
    const size_t n = 300;
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));
    HDCD_CHECK(x != NULL && y != NULL);

    rng_seed(7);
    for (size_t i = 0; i < n; i++) {
        x[i] = rng_normal();
        y[i] = rng_normal();
    }

    double d;
    HDCD_CHECK(hdcd_dcor_exact(x, y, n, &d) == HDCD_OK);
    HDCD_CHECK(d >= 0.0);
    HDCD_CHECK(d < 0.15);

    free(x);
    free(y);
    HDCD_PASS("dCor near 0 for independent samples");
}

static void test_nonlinear_dependence_benchmark_positive(void) {
    /* Y = X^2 with X symmetric about 0: Pearson correlation ~= 0, but
     * distance correlation must detect the dependence (spec section
     * 29.8: "nonlinear relationships where Pearson correlation is near
     * zero but distance correlation is positive"). */
    const size_t n = 300;
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));
    HDCD_CHECK(x != NULL && y != NULL);

    rng_seed(11);
    double sum_x = 0.0, sum_y = 0.0;
    for (size_t i = 0; i < n; i++) {
        x[i] = rng_normal();
        y[i] = x[i] * x[i];
        sum_x += x[i];
        sum_y += y[i];
    }
    double mean_x = sum_x / (double)n;
    double mean_y = sum_y / (double)n;

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    double pearson = cov / sqrt(var_x * var_y);
    HDCD_CHECK(fabs(pearson) < 0.15); /* confirms this is a fair nonlinear benchmark */

    double d;
    HDCD_CHECK(hdcd_dcor_exact(x, y, n, &d) == HDCD_OK);
    HDCD_CHECK(d > 0.3);

    free(x);
    free(y);
    HDCD_PASS("dCor positive for Y=X^2 despite near-zero Pearson correlation");
}

static void test_invalid_arguments(void) {
    double x[2] = {0.0, 1.0};
    double out;
    HDCD_CHECK(hdcd_dcor_exact(NULL, x, 2, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dcor_exact(x, x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dcor_exact(x, x, 2, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("dCor rejects invalid arguments");
}

int main(void) {
    test_symmetry();
    test_self_correlation_near_one();
    test_constant_series_is_zero();
    test_independence_benchmark_near_zero();
    test_nonlinear_dependence_benchmark_positive();
    test_invalid_arguments();
    printf("All dcor_exact tests passed.\n");
    return 0;
}
