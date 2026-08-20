#include "hdcd/tail_dependence.h"
#include "test_utils.h"

#include <stdint.h>
#include <stdlib.h>

/* Self-contained deterministic PRNG, test-only (see the note in
 * tests/test_copula_transform.c). */
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

/* A "comonotonic-mixture" construction with an EXACTLY known tail-
 * dependence coefficient p, symmetric in both tails: with probability p
 * v=u (comonotonic -- every extreme of u is the identical extreme of
 * v), else v is an independent fresh uniform. As q -> 1,
 * P(V>q | U>q) = p*1 + (1-p)*P(V>q | U>q, independent) -> p, and
 * symmetrically for the lower tail -- this is the standard textbook
 * device for constructing a copula with a prescribed tail-dependence
 * coefficient without needing a full parametric family. */
static void make_comonotonic_mixture(size_t n, double p, uint64_t seed, double *u, double *v) {
    rng_seed(seed);
    for (size_t i = 0; i < n; i++) {
        double w = rng_uniform();
        u[i] = w;
        v[i] = (rng_uniform() < p) ? w : rng_uniform();
    }
}

static void test_comonotonic_gives_lambda_near_one(void) {
    size_t n = 500;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    rng_seed(1);
    for (size_t i = 0; i < n; i++) {
        double w = rng_uniform();
        u[i] = w;
        v[i] = w; /* fully comonotonic: v is an exact copy of u */
    }

    double lambda_upper, lambda_lower;
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, n, 0, &lambda_upper, &lambda_lower) == HDCD_OK);
    HDCD_CHECK_NEAR(lambda_upper, 1.0, 1e-12);
    HDCD_CHECK_NEAR(lambda_lower, 1.0, 1e-12);

    free(u); free(v);
    HDCD_PASS("fully comonotonic data gives tail-dependence coefficient exactly 1 in both tails");
}

static void test_independent_gives_small_coefficient(void) {
    size_t n = 2000;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    rng_seed(2);
    for (size_t i = 0; i < n; i++) {
        u[i] = rng_uniform();
        v[i] = rng_uniform();
    }

    double lambda_upper, lambda_lower;
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, n, 0, &lambda_upper, &lambda_lower) == HDCD_OK);
    HDCD_CHECK(lambda_upper >= 0.0 && lambda_upper < 0.3);
    HDCD_CHECK(lambda_lower >= 0.0 && lambda_lower < 0.3);

    free(u); free(v);
    HDCD_PASS("independent data gives a small (near-chance-level) tail-dependence coefficient");
}

static void test_mixture_recovers_known_coefficient(void) {
    size_t n = 4000;
    double p = 0.5;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    make_comonotonic_mixture(n, p, 3, u, v);

    double lambda_upper, lambda_lower;
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, n, 0, &lambda_upper, &lambda_lower) == HDCD_OK);
    HDCD_CHECK(fabs(lambda_upper - p) < 0.15);
    HDCD_CHECK(fabs(lambda_lower - p) < 0.15);

    free(u); free(v);
    HDCD_PASS("comonotonic-mixture data recovers its known tail-dependence coefficient within sampling tolerance");
}

static void test_symmetric_in_its_two_arguments(void) {
    size_t n = 300;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    make_comonotonic_mixture(n, 0.4, 4, u, v);

    double lambda_upper_uv, lambda_lower_uv, lambda_upper_vu, lambda_lower_vu;
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, n, 20, &lambda_upper_uv, &lambda_lower_uv) == HDCD_OK);
    HDCD_CHECK(hdcd_tail_dependence_coefficient(v, u, n, 20, &lambda_upper_vu, &lambda_lower_vu) == HDCD_OK);
    HDCD_CHECK_NEAR(lambda_upper_uv, lambda_upper_vu, 1e-12);
    HDCD_CHECK_NEAR(lambda_lower_uv, lambda_lower_vu, 1e-12);

    free(u); free(v);
    HDCD_PASS("tail-dependence coefficient is symmetric in its two arguments");
}

static void test_explicit_k_is_respected(void) {
    size_t n = 200;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    rng_seed(5);
    for (size_t i = 0; i < n; i++) {
        u[i] = rng_uniform();
        v[i] = rng_uniform();
    }

    double lambda_upper, lambda_lower;
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, n, 5, &lambda_upper, &lambda_lower) == HDCD_OK);
    /* With only k=5 extreme points per tail, the coefficient must still
     * land on an exact multiple of 1/5. */
    double scaled_upper = lambda_upper * 5.0;
    HDCD_CHECK_NEAR(scaled_upper, round(scaled_upper), 1e-9);

    free(u); free(v);
    HDCD_PASS("an explicit k is honored (coefficient is an exact multiple of 1/k)");
}

static void test_invalid_arguments(void) {
    double u[10], v[10];
    for (int i = 0; i < 10; i++) { u[i] = (double)i / 10.0; v[i] = (double)i / 10.0; }
    double lambda_upper, lambda_lower;

    HDCD_CHECK(hdcd_tail_dependence_coefficient(NULL, v, 10, 0, &lambda_upper, &lambda_lower) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, 5, 0, &lambda_upper, &lambda_lower) == HDCD_ERROR_INVALID_ARGUMENT); /* n < 8 */
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, 10, 10, &lambda_upper, &lambda_lower) == HDCD_ERROR_INVALID_ARGUMENT); /* k >= n */
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u, v, 10, 0, NULL, &lambda_lower) == HDCD_ERROR_INVALID_ARGUMENT);

    double u_nan[8] = {0, 1, 2, 3, 4, 5, 6, 0.0 / 0.0};
    double v_ok[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    HDCD_CHECK(hdcd_tail_dependence_coefficient(u_nan, v_ok, 8, 0, &lambda_upper, &lambda_lower) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_PASS("tail-dependence coefficient rejects invalid arguments");
}

int main(void) {
    test_comonotonic_gives_lambda_near_one();
    test_independent_gives_small_coefficient();
    test_mixture_recovers_known_coefficient();
    test_symmetric_in_its_two_arguments();
    test_explicit_k_is_respected();
    test_invalid_arguments();
    printf("All tail_dependence tests passed.\n");
    return 0;
}
