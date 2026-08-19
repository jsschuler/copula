#include "hdcd/dag_fit.h"
#include "hdcd/dag.h"
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

static hdcd_local_fit_options_t default_options(uint64_t seed) {
    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 3;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = seed;
    return opt;
}

/*
 * Three-node chain DAG 0 -> 1 -> 2, built via an exact Gaussian-copula
 * chain (Phi is the true normal CDF, so every column is exactly
 * Uniform(0,1) marginally): Z0 -> Z1 = rho*Z0+... -> Z2 = rho*Z1+...
 */
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

static void test_valid_known_dag_fits(void) {
    size_t n = 600, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.75, 21, u, mask);

    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(d, 2, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 2) == HDCD_OK);

    hdcd_local_fit_options_t opt = default_options(11);
    hdcd_dag_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_dag_fit(u, mask, n, d, dag, &opt, &fit);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(fit != NULL);
    HDCD_CHECK(hdcd_dag_fit_dim(fit) == d);
    HDCD_CHECK(hdcd_dag_fit_all_converged(fit));

    /* Node 0 is a root: trivial c_0(u)=1. Nodes 1,2 have one parent each. */
    HDCD_CHECK(hdcd_local_fit_n_parents(hdcd_dag_fit_node(fit, 0)) == 0);
    HDCD_CHECK(hdcd_local_fit_n_parents(hdcd_dag_fit_node(fit, 1)) == 1);
    HDCD_CHECK(hdcd_local_fit_parent_order(hdcd_dag_fit_node(fit, 1))[0] == 0);
    HDCD_CHECK(hdcd_local_fit_n_parents(hdcd_dag_fit_node(fit, 2)) == 1);
    HDCD_CHECK(hdcd_local_fit_parent_order(hdcd_dag_fit_node(fit, 2))[0] == 1);

    /* Both dependent edges should beat the independence baseline (log-score 0). */
    HDCD_CHECK(hdcd_local_fit_holdout_score(hdcd_dag_fit_node(fit, 1)) > 0.05);
    HDCD_CHECK(hdcd_local_fit_holdout_score(hdcd_dag_fit_node(fit, 2)) > 0.05);

    hdcd_dag_fit_free(fit);
    hdcd_dag_free(dag);
    free(u);
    free(mask);
    HDCD_PASS("a valid known DAG (chain with real dependency) fits successfully");
}

static void test_factorized_log_density_matches_manual_sum(void) {
    size_t n = 400, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    build_chain_data(n, 0.6, 55, u, mask);

    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(d, 2, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 2) == HDCD_OK);

    hdcd_local_fit_options_t opt = default_options(6);
    hdcd_dag_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, dag, &opt, &fit) == HDCD_OK);

    double u_point[3] = {0.3, 0.6, 0.4};
    double joint_log_density;
    HDCD_CHECK(hdcd_dag_fit_joint_log_density(fit, u_point, d, &joint_log_density) == HDCD_OK);

    /* spec section 14: log c_G(u) = sum_j log c_j(u_j | u_Pa(j)) --
     * verify the aggregate matches manually summing each node's local
     * log-density via the public per-node API (additive factorization
     * exploited, not just claimed). */
    double manual_sum = 0.0;
    double z0[1] = {u_point[0]};
    double z1[1] = {u_point[1]};
    double log_c;
    HDCD_CHECK(hdcd_local_fit_log_density(hdcd_dag_fit_node(fit, 0), u_point[0], NULL, 0, &log_c) == HDCD_OK);
    manual_sum += log_c;
    HDCD_CHECK(hdcd_local_fit_log_density(hdcd_dag_fit_node(fit, 1), u_point[1], z0, 1, &log_c) == HDCD_OK);
    manual_sum += log_c;
    HDCD_CHECK(hdcd_local_fit_log_density(hdcd_dag_fit_node(fit, 2), u_point[2], z1, 1, &log_c) == HDCD_OK);
    manual_sum += log_c;

    HDCD_CHECK_NEAR(joint_log_density, manual_sum, 1e-12);

    hdcd_dag_fit_free(fit);
    hdcd_dag_free(dag);
    free(u);
    free(mask);
    HDCD_PASS("factorized joint log density matches the manual per-node sum exactly");
}

static void test_independence_dag_stays_near_uniform(void) {
    /* Normalization invariant sanity check: for genuinely independent
     * columns, an empty-DAG (all roots) fit should give a joint log
     * density close to 0 (c_G(u) close to 1) at generic points, since
     * every factor is close to 1. */
    size_t n = 400, d = 3;
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    rng_seed(303);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            u[j * n + i] = rng_uniform();
        }
        mask[0 * n + i] = mask[1 * n + i] = mask[2 * n + i] = 1;
    }

    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(d, 2, &dag) == HDCD_OK); /* no edges: all roots */

    hdcd_local_fit_options_t opt = default_options(2);
    hdcd_dag_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, dag, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_fit_all_converged(fit));

    double u_point[3] = {0.5, 0.5, 0.5};
    double joint_log_density;
    HDCD_CHECK(hdcd_dag_fit_joint_log_density(fit, u_point, d, &joint_log_density) == HDCD_OK);
    HDCD_CHECK_NEAR(joint_log_density, 0.0, 1e-12); /* every root factor is exactly 1 */

    hdcd_dag_fit_free(fit);
    hdcd_dag_free(dag);
    free(u);
    free(mask);
    HDCD_PASS("all-roots DAG gives joint log density exactly 0 (c_G=1)");
}

static void test_invalid_arguments(void) {
    size_t n = 50, d = 2;
    double u[100];
    uint8_t mask[100];
    for (size_t i = 0; i < 100; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_dag_t *dag3 = NULL;
    HDCD_CHECK(hdcd_dag_create(3, 2, &dag3) == HDCD_OK); /* dimension mismatch vs d=2 */

    hdcd_local_fit_options_t opt = default_options(1);
    hdcd_dag_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_dag_fit(u, mask, n, d, dag3, &opt, &fit) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_fit(NULL, mask, n, d, dag3, &opt, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_dag_fit_dim(NULL) == 0);
    HDCD_CHECK(hdcd_dag_fit_node(NULL, 0) == NULL);
    HDCD_CHECK(hdcd_dag_fit_node_converged(NULL, 0) == 0);
    HDCD_CHECK(hdcd_dag_fit_all_converged(NULL) == 0);

    double out;
    HDCD_CHECK(hdcd_dag_fit_joint_log_density(NULL, u, d, &out) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_dag_fit_free(NULL); /* must not crash */
    hdcd_dag_free(dag3);
    HDCD_PASS("dag_fit API rejects invalid arguments");
}

int main(void) {
    test_valid_known_dag_fits();
    test_factorized_log_density_matches_manual_sum();
    test_independence_dag_stays_near_uniform();
    test_invalid_arguments();
    printf("All dag_fit tests passed.\n");
    return 0;
}
