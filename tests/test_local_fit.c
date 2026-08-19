#include "hdcd/local_fit.h"
#include "test_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static double std_normal_cdf(double x) {
    return 0.5 * erfc(-x * 0.7071067811865476);
}

/* Exact Gaussian-copula sample: Phi is the TRUE CDF of a standard
 * normal, so U0=Phi(Z0), U1=Phi(rho*Z0+sqrt(1-rho^2)*Z1) are EXACTLY
 * Uniform(0,1) marginally (probability integral transform), with
 * Gaussian dependency of strength rho -- this isolates the DAG-fitting
 * pipeline from the (approximate) marginal-smoothing machinery tested
 * in Milestones 1-2. */
static void make_gaussian_copula_data(size_t n, double rho, uint64_t seed, double *u0, double *u1) {
    rng_seed(seed);
    for (size_t i = 0; i < n; i++) {
        double z0 = rng_normal();
        double z1 = rho * z0 + sqrt(1.0 - rho * rho) * rng_normal();
        u0[i] = std_normal_cdf(z0);
        u1[i] = std_normal_cdf(z1);
    }
}

static hdcd_local_fit_options_t default_options(uint64_t seed) {
    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 4;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = seed;
    /* theta_max_iterations, theta_tol, sinkhorn_options all zeroed -> defaults */
    return opt;
}

static void test_root_node_is_trivial(void) {
    size_t n = 20, d = 2;
    double u[40];
    uint8_t mask[40];
    for (size_t i = 0; i < 40; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_local_fit_options_t opt = default_options(1);
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 0, NULL, 0, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_n_parents(fit) == 0);
    HDCD_CHECK(hdcd_local_fit_n_observed(fit) == n);
    HDCD_CHECK(hdcd_local_fit_n_train(fit) == 0);
    HDCD_CHECK(hdcd_local_fit_n_holdout(fit) == n);
    HDCD_CHECK_NEAR(hdcd_local_fit_holdout_score(fit), 0.0, 1e-15);
    HDCD_CHECK(hdcd_local_fit_theta_converged(fit));
    HDCD_CHECK(hdcd_local_fit_sinkhorn_converged(fit));

    double log_c;
    HDCD_CHECK(hdcd_local_fit_log_density(fit, 0.37, NULL, 0, &log_c) == HDCD_OK);
    HDCD_CHECK_NEAR(log_c, 0.0, 1e-15);

    hdcd_local_fit_free(fit);
    HDCD_PASS("root node (no parents) gives c_j(u)=1 trivially");
}

static void test_dependent_data_beats_independence_baseline(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.8, 42, u0, u1);

    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(7);
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_theta_converged(fit));
    HDCD_CHECK(hdcd_local_fit_sinkhorn_converged(fit));

    /* A model that correctly captures rho=0.8 dependency should beat
     * the independence baseline c_j=1 (log-score 0) out of sample. */
    HDCD_CHECK(hdcd_local_fit_holdout_score(fit) > 0.05);

    HDCD_CHECK(hdcd_local_fit_n_parents(fit) == 1);
    HDCD_CHECK(hdcd_local_fit_parent_order(fit)[0] == 0);
    HDCD_CHECK(hdcd_local_fit_n_observed(fit) == n);
    HDCD_CHECK(hdcd_local_fit_n_train(fit) + hdcd_local_fit_n_holdout(fit) == n);

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("fitted dependent-data model beats the independence baseline out of sample");
}

static void test_independent_data_stays_near_baseline(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.0, 43, u0, u1); /* rho=0: truly independent */

    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(8);
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit) == HDCD_OK);

    /* Regularization should keep a spurious-dependency claim modest for
     * genuinely independent data: the held-out score should not blow up
     * positive, unlike the strongly-dependent case above. */
    HDCD_CHECK(hdcd_local_fit_holdout_score(fit) < 0.05);

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("independent data does not spuriously beat the independence baseline");
}

static void test_missing_data_row_counts(void) {
    size_t n = 100, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);

    rng_seed(99);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            u[j * n + i] = rng_uniform();
        }
    }
    /* child = column 2, parents = {0,1}. Mark every 4th row missing in
     * parent 0 and every 5th row missing in child; only rows where ALL
     * THREE are observed should count. */
    for (size_t i = 0; i < n; i++) {
        mask[0 * n + i] = (i % 4 != 0) ? 1 : 0;
        mask[1 * n + i] = 1;
        mask[2 * n + i] = (i % 5 != 0) ? 1 : 0;
    }
    size_t expected_usable = 0;
    for (size_t i = 0; i < n; i++) {
        if ((i % 4 != 0) && (i % 5 != 0)) expected_usable++;
    }

    hdcd_local_fit_options_t opt = default_options(3);
    opt.bernstein_degree = 3;
    size_t parents[2] = {0, 1};
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 2, parents, 2, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_n_observed(fit) == expected_usable);
    HDCD_CHECK(hdcd_local_fit_n_train(fit) + hdcd_local_fit_n_holdout(fit) == expected_usable);

    hdcd_local_fit_free(fit);
    free(u); free(mask);
    HDCD_PASS("multi-parent usable-row count respects joint missingness");
}

static void test_invalid_arguments(void) {
    size_t n = 50, d = 2;
    double u[100];
    uint8_t mask[100];
    for (size_t i = 0; i < 100; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_local_fit_options_t opt = default_options(1);
    hdcd_local_fit_t *fit = NULL;

    size_t bad_parent_self[1] = {1};
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, bad_parent_self, 1, &opt, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    size_t dup_parents[2] = {0, 0};
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, dup_parents, 2, &opt, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_lambda = opt;
    bad_lambda.lambda_roughness = 0.0;
    size_t p0 = 0;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &p0, 1, &bad_lambda, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_holdout = opt;
    bad_holdout.holdout_fraction = 1.0;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &p0, 1, &bad_holdout, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_local_fit_node(NULL, mask, n, d, 1, &p0, 1, &opt, &fit) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &p0, 1, NULL, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_local_fit_n_parents(NULL) == 0);
    HDCD_CHECK(hdcd_local_fit_parent_order(NULL) == NULL);
    HDCD_CHECK(isnan(hdcd_local_fit_holdout_score(NULL)));
    hdcd_local_fit_free(NULL); /* must not crash */

    HDCD_PASS("local_fit API rejects invalid arguments");
}

int main(void) {
    test_root_node_is_trivial();
    test_dependent_data_beats_independence_baseline();
    test_independent_data_stays_near_baseline();
    test_missing_data_row_counts();
    test_invalid_arguments();
    printf("All local_fit tests passed.\n");
    return 0;
}
