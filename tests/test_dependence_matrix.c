#include "hdcd/dcor.h"
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

static double rng_normal(void) {
    const double two_pi = 6.283185307179586;
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}

/*
 * Synthetic 4-column dataset, column-major (spec section 23):
 *   col 0: independent normal, missing where i % 7 == 0
 *   col 1: independent normal, fully observed
 *   col 2: col0^2 (nonlinear dependence on col 0), fully observed
 *   col 3: independent normal, missing where i % 5 == 0
 */
static void build_dataset(size_t n, double *u, uint8_t *mask) {
    rng_seed(99);
    for (size_t i = 0; i < n; i++) {
        double x0 = rng_normal();
        double x1 = rng_normal();
        double x3 = rng_normal();

        u[0 * n + i] = x0;
        u[1 * n + i] = x1;
        u[2 * n + i] = x0 * x0;
        u[3 * n + i] = x3;

        mask[0 * n + i] = (i % 7 != 0) ? 1 : 0;
        mask[1 * n + i] = 1;
        mask[2 * n + i] = 1;
        mask[3 * n + i] = (i % 5 != 0) ? 1 : 0;
    }
}

static void test_symmetry_and_diagonal(void) {
    const size_t n = 250, d = 4;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(u != NULL && mask != NULL);
    build_dataset(n, u, mask);

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);
    HDCD_CHECK(hdcd_dependence_matrix_dim(dm) == d);

    for (size_t j = 0; j < d; j++) {
        HDCD_CHECK_NEAR(hdcd_dependence_matrix_get(dm, j, j), 1.0, 1e-9);
        for (size_t k = 0; k < d; k++) {
            double djk = hdcd_dependence_matrix_get(dm, j, k);
            double dkj = hdcd_dependence_matrix_get(dm, k, j);
            HDCD_CHECK_NEAR(djk, dkj, 1e-12);
        }
    }

    hdcd_dependence_matrix_free(dm);
    free(u);
    free(mask);
    HDCD_PASS("dependence matrix is symmetric with unit diagonal");
}

static void test_independence_and_nonlinear_benchmarks(void) {
    const size_t n = 250, d = 4;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(u != NULL && mask != NULL);
    build_dataset(n, u, mask);

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    /* col0 and col1 are independent. */
    HDCD_CHECK(hdcd_dependence_matrix_get(dm, 0, 1) < 0.2);
    /* col2 = col0^2: nonlinear dependence must be detected. */
    HDCD_CHECK(hdcd_dependence_matrix_get(dm, 0, 2) > 0.3);

    hdcd_dependence_matrix_free(dm);
    free(u);
    free(mask);
    HDCD_PASS("dependence matrix recovers independence and nonlinear benchmarks");
}

static void test_effective_sample_sizes(void) {
    const size_t n = 250, d = 4;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(u != NULL && mask != NULL);
    build_dataset(n, u, mask);

    size_t expected_n0 = 0, expected_n03 = 0;
    for (size_t i = 0; i < n; i++) {
        int obs0 = (i % 7 != 0);
        int obs3 = (i % 5 != 0);
        if (obs0) expected_n0++;
        if (obs0 && obs3) expected_n03++;
    }

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    HDCD_CHECK(hdcd_dependence_matrix_n_effective(dm, 0, 0) == expected_n0);
    HDCD_CHECK(hdcd_dependence_matrix_n_effective(dm, 0, 3) == expected_n03);
    HDCD_CHECK(hdcd_dependence_matrix_n_effective(dm, 3, 0) == expected_n03);
    HDCD_CHECK(hdcd_dependence_matrix_n_effective(dm, 1, 1) == n); /* fully observed */

    hdcd_dependence_matrix_free(dm);
    free(u);
    free(mask);
    HDCD_PASS("pairwise-complete effective sample sizes are stored correctly");
}

static void test_insufficient_overlap_gives_nan(void) {
    /* Two columns whose observed sets barely overlap: fewer than 2
     * pairwise-complete rows means dCor is undefined, not fabricated. */
    const size_t n = 4, d = 2;
    double u[8] = {0.1, 0.2, 0.3, 0.4,   1.1, 1.2, 1.3, 1.4};
    uint8_t mask[8] = {
        1, 1, 0, 0,  /* col 0 observed at rows 0,1 */
        0, 1, 1, 1   /* col 1 observed at rows 1,2,3 */
    };
    /* O_01 = {1}: only one pairwise-complete row. */

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);
    HDCD_CHECK(hdcd_dependence_matrix_n_effective(dm, 0, 1) == 1);
    HDCD_CHECK(isnan(hdcd_dependence_matrix_get(dm, 0, 1)));

    hdcd_dependence_matrix_free(dm);
    HDCD_PASS("dependence matrix reports NaN when pairwise-complete overlap < 2");
}

static void test_invalid_arguments(void) {
    double u[4] = {0.0, 1.0, 2.0, 3.0};
    uint8_t mask[4] = {1, 1, 1, 1};
    hdcd_dependence_matrix_t *dm = NULL;

    HDCD_CHECK(hdcd_compute_dependence_matrix(NULL, mask, 2, 2, &dm) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, 0, 2, &dm) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, 2, 0, &dm) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(isnan(hdcd_dependence_matrix_get(NULL, 0, 0)));
    HDCD_CHECK(hdcd_dependence_matrix_n_effective(NULL, 0, 0) == 0);
    HDCD_CHECK(hdcd_dependence_matrix_dim(NULL) == 0);

    hdcd_dependence_matrix_free(NULL); /* must not crash */

    HDCD_PASS("dependence matrix rejects invalid arguments");
}

int main(void) {
    test_symmetry_and_diagonal();
    test_independence_and_nonlinear_benchmarks();
    test_effective_sample_sizes();
    test_insufficient_overlap_gives_nan();
    test_invalid_arguments();
    printf("All dependence_matrix tests passed.\n");
    return 0;
}
