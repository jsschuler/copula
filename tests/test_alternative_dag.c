#include "hdcd/dag.h"
#include "hdcd/dag_fit.h"
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
 * tests/test_dag_fit.c). */
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

static void test_accepts_arbitrary_dag_and_produces_kl_comparison(void) {
    /* Spec section 19's whole workflow, end to end: fit a reference DAG
     * (the TRUE chain 0->1->2, built incrementally) and an alternative
     * candidate DAG (built via hdcd_dag_from_edges, with a
     * deliberately different/misspecified structure -- the reverse
     * chain 2->1->0, which has a topological order [2,1,0] instead of
     * the reference's [0,1,2]), then compare via held-out KL. */
    size_t n = 600, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.8, 123, u, mask);

    hdcd_dag_t *reference_dag = NULL;
    HDCD_CHECK(hdcd_dag_create(d, 2, &reference_dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(reference_dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(reference_dag, 1, 2) == HDCD_OK);

    /* Alternative: same UNDIRECTED skeleton but built as an
     * independently-supplied edge list with the opposite direction and
     * a different topological order -- exercises "accepts valid DAGs
     * with different topological orders" (spec section 31 M9). */
    size_t alt_parents[2] = {2, 1};
    size_t alt_children[2] = {1, 0};
    hdcd_dag_t *reversed_dag = NULL;
    HDCD_CHECK(hdcd_dag_from_edges(d, 2, alt_parents, alt_children, 2, &reversed_dag) == HDCD_OK);

    /* Misspecified: an empty DAG (claims full independence, discarding
     * the real chain dependency entirely) -- should lose the most
     * information relative to the reference. */
    hdcd_dag_t *empty_dag = NULL;
    HDCD_CHECK(hdcd_dag_create(d, 2, &empty_dag) == HDCD_OK);

    hdcd_local_fit_options_t opt = default_local_fit_options(9);

    hdcd_dag_fit_t *reference_fit = NULL;
    hdcd_dag_fit_t *reversed_fit = NULL;
    hdcd_dag_fit_t *empty_fit = NULL;
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, reference_dag, &opt, &reference_fit) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, reversed_dag, &opt, &reversed_fit) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, empty_dag, &opt, &empty_fit) == HDCD_OK);

    /* Held-out KL comparison (spec section 19). */
    double delta_reversed = hdcd_dag_fit_kl_difference(reversed_fit, reference_fit);
    double delta_empty = hdcd_dag_fit_kl_difference(empty_fit, reference_fit);

    HDCD_CHECK(!isnan(delta_reversed));
    HDCD_CHECK(!isnan(delta_empty));
    HDCD_CHECK_NEAR(hdcd_dag_fit_kl_difference(reference_fit, reference_fit), 0.0, 1e-12);

    /* Discarding the dependency entirely (empty) must lose strictly
     * more information than merely reversing edge direction along the
     * same skeleton (reversed still captures SOME of the same-pair
     * dependency, since a Gaussian-copula-like chain's pairwise
     * dependency is symmetric in direction, unlike its factorization). */
    HDCD_CHECK(delta_empty > delta_reversed);
    HDCD_CHECK(delta_empty > 0.0); /* empty strictly loses information vs. the true chain */

    hdcd_dag_fit_free(reference_fit);
    hdcd_dag_fit_free(reversed_fit);
    hdcd_dag_fit_free(empty_fit);
    hdcd_dag_free(reference_dag);
    hdcd_dag_free(reversed_dag);
    hdcd_dag_free(empty_dag);
    free(u);
    free(mask);
    HDCD_PASS("accepts an arbitrary DAG and produces a meaningful held-out KL comparison");
}

static void test_kl_difference_dimension_mismatch_is_nan(void) {
    size_t n = 200, d2 = 2, d3 = 3;
    double *u2 = (double *)malloc(n * d2 * sizeof(double));
    uint8_t *mask2 = (uint8_t *)malloc(n * d2);
    double *u3 = (double *)malloc(n * d3 * sizeof(double));
    uint8_t *mask3 = (uint8_t *)malloc(n * d3);
    rng_seed(4);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d3; j++) {
            double v = rng_uniform();
            if (j < d2) u2[j * n + i] = v;
            u3[j * n + i] = v;
        }
        mask2[0 * n + i] = mask2[1 * n + i] = 1;
        mask3[0 * n + i] = mask3[1 * n + i] = mask3[2 * n + i] = 1;
    }

    hdcd_dag_t *dag2 = NULL, *dag3 = NULL;
    HDCD_CHECK(hdcd_dag_create(d2, 2, &dag2) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_create(d3, 2, &dag3) == HDCD_OK);

    hdcd_local_fit_options_t opt = default_local_fit_options(2);
    hdcd_dag_fit_t *fit2 = NULL, *fit3 = NULL;
    HDCD_CHECK(hdcd_dag_fit(u2, mask2, n, d2, dag2, &opt, &fit2) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_fit(u3, mask3, n, d3, dag3, &opt, &fit3) == HDCD_OK);

    HDCD_CHECK(isnan(hdcd_dag_fit_kl_difference(fit2, fit3)));
    HDCD_CHECK(isnan(hdcd_dag_fit_kl_difference(NULL, fit3)));
    HDCD_CHECK(isnan(hdcd_dag_fit_kl_estimate(NULL)));

    hdcd_dag_fit_free(fit2);
    hdcd_dag_fit_free(fit3);
    hdcd_dag_free(dag2);
    hdcd_dag_free(dag3);
    free(u2); free(mask2); free(u3); free(mask3);
    HDCD_PASS("KL difference between mismatched-dimension fits is NaN, not a fabricated number");
}

int main(void) {
    test_accepts_arbitrary_dag_and_produces_kl_comparison();
    test_kl_difference_dimension_mismatch_is_nan();
    printf("All alternative_dag tests passed.\n");
    return 0;
}
