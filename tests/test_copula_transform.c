#include "hdcd/copula.h"
#include "hdcd/marginal.h"
#include "test_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static double sample_data[20] = {
    -1.83, -1.42, -1.10, -0.95, -0.71, -0.58, -0.33, -0.21, -0.05, 0.02,
     0.14,  0.27,  0.39,  0.52,  0.68,  0.81,  1.03,  1.29,  1.61,  2.05
};

/* ---- self-contained deterministic PRNG, test-only -------------------
 * Not part of the public API; the library's own stochastic routines
 * (annealing, sampling) will get their own seeded RNG module in a later
 * milestone (spec section 24: "use deterministic seeded RNG"). This is
 * only to generate a reproducible synthetic sample for the uniformity
 * check below.
 */
static uint64_t rng_state;

static void rng_seed(uint64_t seed) {
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static double rng_uniform(void) {
    /* xorshift64* */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t result = rng_state * 0x2545F4914F6CDD1DULL;
    double u = (double)(result >> 11) * (1.0 / 9007199254740992.0); /* / 2^53 */
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

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static void test_values_in_allowed_interval(void) {
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;
    hdcd_marginal_t *m = NULL;
    HDCD_CHECK(hdcd_marginal_fit(sample_data, mask, 20, -1.0, -1.0, 1e-6, 200, &m) == HDCD_OK);

    double u[20];
    double epsilon = 1e-6;
    HDCD_CHECK(hdcd_transform_to_copula(m, sample_data, mask, 20, epsilon, u) == HDCD_OK);

    for (int i = 0; i < 20; i++) {
        HDCD_CHECK(u[i] >= epsilon);
        HDCD_CHECK(u[i] <= 1.0 - epsilon);
    }

    hdcd_marginal_free(m);
    HDCD_PASS("transformed values lie in [epsilon, 1-epsilon]");
}

static void test_default_epsilon_clips(void) {
    /* Extreme evaluation points, far outside the training range, should
     * clip to the default epsilon bounds rather than saturating at
     * exactly 0 or 1. */
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;
    hdcd_marginal_t *m = NULL;
    HDCD_CHECK(hdcd_marginal_fit(sample_data, mask, 20, -1.0, -1.0, 1e-6, 200, &m) == HDCD_OK);

    double extreme_x[2] = {-1000.0, 1000.0};
    uint8_t extreme_mask[2] = {1, 1};
    double u[2];
    HDCD_CHECK(hdcd_transform_to_copula(m, extreme_x, extreme_mask, 2, 0.0, u) == HDCD_OK);

    HDCD_CHECK_NEAR(u[0], HDCD_DEFAULT_COPULA_EPSILON, 1e-15);
    HDCD_CHECK_NEAR(u[1], 1.0 - HDCD_DEFAULT_COPULA_EPSILON, 1e-15);

    hdcd_marginal_free(m);
    HDCD_PASS("extreme values clip to the default epsilon bounds");
}

static void test_missingness_propagates_unchanged(void) {
    uint8_t mask[20];
    for (int i = 0; i < 20; i++) mask[i] = 1;
    /* Mark a few entries missing in the *evaluation* array (independent
     * of which entries were used to fit the marginal). */
    mask[2] = 0;
    mask[9] = 0;
    mask[15] = 0;

    uint8_t mask_before[20];
    memcpy(mask_before, mask, sizeof(mask));

    hdcd_marginal_t *m = NULL;
    uint8_t all_observed[20];
    for (int i = 0; i < 20; i++) all_observed[i] = 1;
    HDCD_CHECK(hdcd_marginal_fit(sample_data, all_observed, 20, -1.0, -1.0, 1e-6, 200, &m) == HDCD_OK);

    double u[20];
    HDCD_CHECK(hdcd_transform_to_copula(m, sample_data, mask, 20, 0.0, u) == HDCD_OK);

    /* The mask itself must be untouched by the call. */
    HDCD_CHECK(memcmp(mask, mask_before, sizeof(mask)) == 0);

    for (int i = 0; i < 20; i++) {
        if (!mask[i]) {
            HDCD_CHECK(isnan(u[i]));
        } else {
            HDCD_CHECK(!isnan(u[i]));
        }
    }

    hdcd_marginal_free(m);
    HDCD_PASS("missingness propagates unchanged to the transformed output");
}

static void test_simulated_data_approximately_uniform(void) {
    /* Spec section 29.2: for simulated data from a known continuous
     * marginal model, transformed coordinates should be approximately
     * uniform under a correctly specified fitted marginal. */
    const size_t n = 300;
    double *data = (double *)malloc(n * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n);
    HDCD_CHECK(data != NULL && mask != NULL);

    rng_seed(42);
    for (size_t i = 0; i < n; i++) {
        data[i] = rng_normal();
        mask[i] = 1;
    }

    hdcd_marginal_t *m = NULL;
    HDCD_CHECK(hdcd_marginal_fit(data, mask, n, -1.0, -1.0, 1e-3, 60, &m) == HDCD_OK);

    double *u = (double *)malloc(n * sizeof(double));
    HDCD_CHECK(hdcd_transform_to_copula(m, data, mask, n, 0.0, u) == HDCD_OK);

    /* Mean and variance close to Uniform(0,1)'s 0.5 and 1/12. */
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) mean += u[i];
    mean /= (double)n;
    HDCD_CHECK(fabs(mean - 0.5) < 0.05);

    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = u[i] - mean;
        var += d * d;
    }
    var /= (double)(n - 1);
    HDCD_CHECK(fabs(var - (1.0 / 12.0)) < 0.03);

    /* Kolmogorov-Smirnov-style max deviation from the uniform CDF:
     * sorted u[i] should track the expected order statistic (i+0.5)/n. */
    qsort(u, n, sizeof(double), compare_doubles);
    double max_dev = 0.0;
    for (size_t i = 0; i < n; i++) {
        double expected = ((double)i + 0.5) / (double)n;
        double dev = fabs(u[i] - expected);
        if (dev > max_dev) max_dev = dev;
    }
    HDCD_CHECK(max_dev < 0.08);

    hdcd_marginal_free(m);
    free(data);
    free(mask);
    free(u);
    HDCD_PASS("transform of simulated normal data is approximately uniform");
}

int main(void) {
    test_values_in_allowed_interval();
    test_default_epsilon_clips();
    test_missingness_propagates_unchanged();
    test_simulated_data_approximately_uniform();
    printf("All copula_transform tests passed.\n");
    return 0;
}
