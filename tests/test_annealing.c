#include "hdcd/annealing.h"
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

/* Exact Gaussian-copula chain 0 -> 1 -> 2 (see the note in
 * tests/test_dag_fit.c: Phi is the true normal CDF, so every column is
 * exactly Uniform(0,1) marginally). */
static void build_chain_data(size_t n, double rho, uint64_t seed, double *u, uint8_t *mask) {
    rng_seed(seed);
    for (size_t i = 0; i < n; i++) {
        double z0 = rng_normal();
        double z1 = rho * z0 + sqrt(1.0 - rho * rho) * rng_normal();
        double z2 = rho * z1 + sqrt(1.0 - rho * rho) * rng_normal();
        u[0 * n + i] = std_normal_cdf(z0);
        u[1 * n + i] = std_normal_cdf(z1);
        u[2 * n + i] = std_normal_cdf(z2);
    }
    for (size_t i = 0; i < 3 * n; i++) mask[i] = 1;
}

static hdcd_local_fit_options_t default_local_fit_options(uint64_t seed) {
    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 3;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = seed;
    return opt;
}

static hdcd_annealing_options_t default_annealing_options(const size_t *ordering, uint64_t seed) {
    hdcd_annealing_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.k_max = 2;
    opt.lambda_edge = 0.05;
    opt.ordering = ordering;
    opt.local_fit_options = default_local_fit_options(seed + 1);
    opt.initial_temperature = 0.5;
    opt.cooling_rate = 0.95;
    opt.max_iterations = 80;
    opt.restarts = 1;
    opt.p_add = 1.0;
    opt.p_remove = 1.0;
    opt.p_swap = 1.0;
    opt.seed = seed;
    return opt;
}

static void test_cached_and_uncached_scores_agree(void) {
    /* The annealing cache (spec section 17.3) is only correct if
     * fitting the same (child, parent set) twice gives IDENTICAL
     * scores -- that determinism is exactly what makes cache reuse
     * valid instead of a correctness bug. Verify it directly via the
     * public Milestone 7 API: two independent hdcd_local_fit_node
     * calls with identical arguments (as a fresh fit and a cache hit
     * would each effectively perform) must agree bit-for-bit. */
    size_t n = 400, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.7, 17, u, mask);

    hdcd_local_fit_options_t opt = default_local_fit_options(5);
    size_t parent = 0;

    hdcd_local_fit_t *fit1 = NULL;
    hdcd_local_fit_t *fit2 = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit1) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit2) == HDCD_OK);

    HDCD_CHECK_NEAR(hdcd_local_fit_holdout_score(fit1), hdcd_local_fit_holdout_score(fit2), 1e-15);
    HDCD_CHECK_NEAR(hdcd_local_fit_roughness_penalty(fit1), hdcd_local_fit_roughness_penalty(fit2), 1e-15);
    HDCD_CHECK(hdcd_local_fit_n_train(fit1) == hdcd_local_fit_n_train(fit2));
    HDCD_CHECK(hdcd_local_fit_n_holdout(fit1) == hdcd_local_fit_n_holdout(fit2));

    hdcd_local_fit_free(fit1);
    hdcd_local_fit_free(fit2);
    free(u);
    free(mask);
    HDCD_PASS("repeated fits of the same (child, parent set) agree exactly -- the property the cache relies on");
}

static void test_known_sparse_graph_improves_over_empty(void) {
    size_t n = 500, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.75, 31, u, mask);

    size_t ordering[3] = {0, 1, 2}; /* consistent with the true chain 0 -> 1 -> 2 */
    hdcd_annealing_options_t opt = default_annealing_options(ordering, 41);

    hdcd_annealing_result_t *result = NULL;
    hdcd_status_t status = hdcd_run_annealing(u, mask, n, d, &opt, &result);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(result != NULL);

    /* The empty graph (every node a root) has J(G) = 0 exactly (every
     * root's K_hat, edge penalty, and roughness are all 0 -- spec
     * section 12's base case, established in Milestone 7). A search
     * that discovers the real chain dependency must do strictly better. */
    HDCD_CHECK(hdcd_annealing_best_score(result) < -0.1);

    const hdcd_dag_t *best = hdcd_annealing_best_dag(result);
    HDCD_CHECK(best != NULL);
    /* The search should have found (or nearly found) the true edges. */
    HDCD_CHECK(hdcd_dag_has_edge(best, 0, 1) || hdcd_dag_has_edge(best, 1, 2));
    HDCD_CHECK(hdcd_annealing_n_iterations(result) > 0);
    HDCD_CHECK(hdcd_annealing_acceptance_rate(result) >= 0.0 && hdcd_annealing_acceptance_rate(result) <= 1.0);

    hdcd_annealing_result_free(result);
    free(u);
    free(mask);
    HDCD_PASS("annealing search on a known sparse dependent graph improves over the empty graph");
}

static void test_reproducibility(void) {
    size_t n = 300, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.6, 71, u, mask);

    size_t ordering[3] = {0, 1, 2};
    hdcd_annealing_options_t opt = default_annealing_options(ordering, 99);
    opt.max_iterations = 20; /* small budget: this test only needs identical traces, not a thorough search */

    hdcd_annealing_result_t *r1 = NULL;
    hdcd_annealing_result_t *r2 = NULL;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &opt, &r1) == HDCD_OK);
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &opt, &r2) == HDCD_OK);

    HDCD_CHECK(hdcd_annealing_n_iterations(r1) == hdcd_annealing_n_iterations(r2));
    HDCD_CHECK_NEAR(hdcd_annealing_best_score(r1), hdcd_annealing_best_score(r2), 1e-15);
    HDCD_CHECK_NEAR(hdcd_annealing_current_score(r1), hdcd_annealing_current_score(r2), 1e-15);

    size_t n_iter = hdcd_annealing_n_iterations(r1);
    for (size_t i = 0; i < n_iter; i++) {
        HDCD_CHECK_NEAR(hdcd_annealing_score_trace(r1, i), hdcd_annealing_score_trace(r2, i), 1e-15);
        HDCD_CHECK(hdcd_annealing_accepted_trace(r1, i) == hdcd_annealing_accepted_trace(r2, i));
    }

    for (size_t j = 0; j < d; j++) {
        HDCD_CHECK(hdcd_dag_n_parents(hdcd_annealing_best_dag(r1), j) == hdcd_dag_n_parents(hdcd_annealing_best_dag(r2), j));
        HDCD_CHECK(hdcd_dag_n_parents(hdcd_annealing_current_dag(r1), j) == hdcd_dag_n_parents(hdcd_annealing_current_dag(r2), j));
    }

    hdcd_annealing_result_free(r1);
    hdcd_annealing_result_free(r2);
    free(u);
    free(mask);
    HDCD_PASS("fixed seed reproduces an identical search trace");
}

static void test_k_max_respected_after_search(void) {
    size_t n = 300, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.8, 5, u, mask);

    size_t ordering[3] = {0, 1, 2};
    hdcd_annealing_options_t opt = default_annealing_options(ordering, 3);
    opt.k_max = 1; /* only one parent allowed per node */
    opt.max_iterations = 60;

    hdcd_annealing_result_t *result = NULL;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &opt, &result) == HDCD_OK);

    const hdcd_dag_t *best = hdcd_annealing_best_dag(result);
    for (size_t j = 0; j < d; j++) {
        HDCD_CHECK(hdcd_dag_n_parents(best, j) <= 1);
    }

    hdcd_annealing_result_free(result);
    free(u);
    free(mask);
    HDCD_PASS("k_max hard limit is respected throughout the search");
}

static void test_invalid_arguments(void) {
    size_t n = 50, d = 3;
    double u[150];
    uint8_t mask[150];
    for (size_t i = 0; i < 150; i++) { u[i] = 0.5; mask[i] = 1; }

    size_t ordering[3] = {0, 1, 2};
    hdcd_annealing_options_t opt = default_annealing_options(ordering, 1);
    hdcd_annealing_result_t *result = NULL;

    HDCD_CHECK(hdcd_run_annealing(NULL, mask, n, d, &opt, &result) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, NULL, &result) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_annealing_options_t bad_ordering = opt;
    size_t dup_ordering[3] = {0, 0, 2};
    bad_ordering.ordering = dup_ordering;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &bad_ordering, &result) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_annealing_options_t bad_temp = opt;
    bad_temp.initial_temperature = 0.0;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &bad_temp, &result) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_annealing_options_t bad_cooling = opt;
    bad_cooling.cooling_rate = 1.0;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &bad_cooling, &result) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_annealing_options_t bad_probs = opt;
    bad_probs.p_add = bad_probs.p_remove = bad_probs.p_swap = 0.0;
    HDCD_CHECK(hdcd_run_annealing(u, mask, n, d, &bad_probs, &result) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_annealing_best_dag(NULL) == NULL);
    HDCD_CHECK(isnan(hdcd_annealing_best_score(NULL)));
    HDCD_CHECK(hdcd_annealing_n_iterations(NULL) == 0);
    hdcd_annealing_result_free(NULL); /* must not crash */

    HDCD_PASS("annealing API rejects invalid arguments");
}

int main(void) {
    test_cached_and_uncached_scores_agree();
    test_known_sparse_graph_improves_over_empty();
    test_reproducibility();
    test_k_max_respected_after_search();
    test_invalid_arguments();
    printf("All annealing tests passed.\n");
    return 0;
}
