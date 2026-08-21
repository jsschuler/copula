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

/* A "comonotonic-mixture" copula-scale pair with an EXACTLY known,
 * symmetric tail-dependence coefficient p (see the identical helper and
 * its derivation comment in tests/test_tail_dependence.c): with
 * probability p, v = u (comonotonic); otherwise v is an independent
 * fresh uniform. */
static void make_tail_dependent_data(size_t n, double p, uint64_t seed, double *u0, double *u1) {
    rng_seed(seed);
    for (size_t i = 0; i < n; i++) {
        double w = rng_uniform();
        u0[i] = w;
        u1[i] = (rng_uniform() < p) ? w : rng_uniform();
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

static void test_roughness_grid_matches_fixed_when_singleton(void) {
    /* A grid containing exactly the fixed lambda must reproduce the
     * fixed-lambda fit exactly: the inner-validation search only
     * SELECTS lambda_to_use, it never feeds any state into the final
     * production fit, which is recomputed from the full train/holdout
     * split either way. */
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.8, 42, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t fixed_opt = default_options(7);
    size_t parent = 0;
    hdcd_local_fit_t *fixed_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &fixed_opt, &fixed_fit) == HDCD_OK);

    hdcd_local_fit_options_t grid_opt = fixed_opt;
    double grid[1] = {0.15};
    grid_opt.lambda_roughness_grid = grid;
    grid_opt.lambda_roughness_grid_size = 1;
    hdcd_local_fit_t *grid_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &grid_opt, &grid_fit) == HDCD_OK);

    HDCD_CHECK_NEAR(hdcd_local_fit_selected_lambda_roughness(grid_fit), 0.15, 1e-15);
    HDCD_CHECK_NEAR(hdcd_local_fit_holdout_score(grid_fit), hdcd_local_fit_holdout_score(fixed_fit), 1e-9);
    HDCD_CHECK_NEAR(hdcd_local_fit_roughness_penalty(grid_fit), hdcd_local_fit_roughness_penalty(fixed_fit), 1e-9);

    hdcd_local_fit_free(fixed_fit);
    hdcd_local_fit_free(grid_fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("a singleton roughness grid reproduces the fixed-lambda fit exactly");
}

static void test_roughness_grid_picks_a_lighter_penalty(void) {
    /* Deliberately mis-set the "default" to a much-too-strong penalty
     * for a strong (rho=0.85) Gaussian dependency, then give the grid a
     * range spanning that bad default down to much lighter values. The
     * grid should both select something lighter than the bad default
     * and score at least as well out of sample -- exactly the
     * over-smoothing failure mode notebooks/vine_copula_recovery.Rmd
     * diagnosed for a single global lambda_roughness. */
    size_t n = 800, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.85, 123, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    double bad_lambda = 5.0;
    hdcd_local_fit_options_t bad_fixed = default_options(21);
    bad_fixed.lambda_roughness = bad_lambda;
    size_t parent = 0;
    hdcd_local_fit_t *bad_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_fixed, &bad_fit) == HDCD_OK);

    hdcd_local_fit_options_t grid_opt = default_options(21);
    double grid[5] = {5.0, 1.0, 0.3, 0.1, 0.03};
    grid_opt.lambda_roughness_grid = grid;
    grid_opt.lambda_roughness_grid_size = 5;
    grid_opt.roughness_validation_fraction = 0.3;
    hdcd_local_fit_t *grid_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &grid_opt, &grid_fit) == HDCD_OK);

    double selected = hdcd_local_fit_selected_lambda_roughness(grid_fit);
    int is_grid_member = 0;
    for (size_t i = 0; i < 5; i++) {
        if (selected == grid[i]) is_grid_member = 1;
    }
    HDCD_CHECK(is_grid_member);
    HDCD_CHECK(selected < bad_lambda);
    HDCD_CHECK(hdcd_local_fit_holdout_score(grid_fit) >= hdcd_local_fit_holdout_score(bad_fit));

    hdcd_local_fit_free(bad_fit);
    hdcd_local_fit_free(grid_fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("per-node roughness grid selects a lighter penalty than a badly-fixed default and scores at least as well");
}

static void test_root_node_selected_lambda_is_nan(void) {
    size_t n = 20, d = 2;
    double u[40];
    uint8_t mask[40];
    for (size_t i = 0; i < 40; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_local_fit_options_t opt = default_options(1);
    double grid[2] = {0.5, 0.1};
    opt.lambda_roughness_grid = grid;
    opt.lambda_roughness_grid_size = 2;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 0, NULL, 0, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(isnan(hdcd_local_fit_selected_lambda_roughness(fit)));

    hdcd_local_fit_free(fit);
    HDCD_PASS("root node's selected lambda_roughness is NAN (nothing to select)");
}

static void test_roughness_grid_invalid_arguments(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.5, 55, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;

    hdcd_local_fit_options_t bad_grid_value = default_options(1);
    double grid_with_zero[2] = {0.2, 0.0};
    bad_grid_value.lambda_roughness_grid = grid_with_zero;
    bad_grid_value.lambda_roughness_grid_size = 2;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_grid_value, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_fraction = default_options(1);
    double grid_ok[2] = {0.2, 0.05};
    bad_fraction.lambda_roughness_grid = grid_ok;
    bad_fraction.lambda_roughness_grid_size = 2;
    bad_fraction.roughness_validation_fraction = 1.5;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_fraction, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    /* Too little data left for the inner split once holdout_fraction and
     * roughness_validation_fraction are both applied. */
    hdcd_local_fit_options_t tiny_data = default_options(1);
    tiny_data.lambda_roughness_grid = grid_ok;
    tiny_data.lambda_roughness_grid_size = 2;
    tiny_data.holdout_fraction = 0.5;
    tiny_data.roughness_validation_fraction = 0.999;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &tiny_data, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("roughness grid rejects non-positive candidates, an out-of-range validation fraction, and insufficient data");
}

static void test_tail_dependence_diagnostic_available_without_any_grid(void) {
    /* The whole point of decoupling this diagnostic from
     * bernstein_degree_grid (see DECISIONS.md's "distinguish initial fit
     * from diagnose from tune" entry): it must be inspectable on a
     * PLAIN fit -- no bernstein_degree_grid, no corner_relief, nothing
     * -- so a caller can decide whether tuning is warranted before
     * opting into it, not only after. */
    size_t n = 1500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.6, 71, u0, u1); /* genuinely tail-dependent */
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t plain_opt = default_options(12); /* no grids, no corner_relief */
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &plain_opt, &fit) == HDCD_OK);

    double max_td = hdcd_local_fit_max_tail_dependence(fit);
    HDCD_CHECK(!isnan(max_td));
    HDCD_CHECK(max_td > 0.1); /* real tail dependence, correctly measured even though nothing was tuned */
    HDCD_CHECK(hdcd_local_fit_selected_bernstein_degree(fit) == plain_opt.bernstein_degree); /* untouched: no grid supplied */

    /* An independent-data control on the same plain (no-grid) options:
     * the diagnostic should stay low, confirming it is not just always
     * reporting a large number regardless of the data. */
    double *v0 = (double *)malloc(n * sizeof(double));
    double *v1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.0, 72, v0, v1);
    double *v = (double *)malloc(n * d * sizeof(double));
    memcpy(&v[0 * n], v0, n * sizeof(double));
    memcpy(&v[1 * n], v1, n * sizeof(double));
    hdcd_local_fit_t *indep_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(v, mask, n, d, 1, &parent, 1, &plain_opt, &indep_fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_max_tail_dependence(indep_fit) < max_td);

    hdcd_local_fit_free(fit);
    hdcd_local_fit_free(indep_fit);
    free(u0); free(u1); free(u); free(mask);
    free(v0); free(v1); free(v);
    HDCD_PASS("tail-dependence diagnostic is populated on a plain fit with no grid, correctly distinguishing tail-dependent from independent data");
}

static void test_degree_grid_gated_off_matches_fixed_degree(void) {
    /* A high gate (0.9) that essentially no ordinary dependency clears:
     * the degree search must be skipped even though a grid was supplied,
     * and the result must exactly match a plain fixed-degree fit. */
    size_t n = 600, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.6, 61, u0, u1); /* Gaussian: no tail dependence at all */
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t fixed_opt = default_options(9);
    size_t parent = 0;
    hdcd_local_fit_t *fixed_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &fixed_opt, &fixed_fit) == HDCD_OK);

    hdcd_local_fit_options_t gated_opt = fixed_opt;
    size_t degree_grid[3] = {4, 6, 8};
    gated_opt.bernstein_degree_grid = degree_grid;
    gated_opt.bernstein_degree_grid_size = 3;
    gated_opt.tail_dependence_gate = 0.9;
    hdcd_local_fit_t *gated_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &gated_opt, &gated_fit) == HDCD_OK);

    HDCD_CHECK(hdcd_local_fit_selected_bernstein_degree(gated_fit) == 4); /* unchanged: gate not met */
    HDCD_CHECK(hdcd_local_fit_max_tail_dependence(gated_fit) < 0.9);
    HDCD_CHECK_NEAR(hdcd_local_fit_holdout_score(gated_fit), hdcd_local_fit_holdout_score(fixed_fit), 1e-9);

    hdcd_local_fit_free(fixed_fit);
    hdcd_local_fit_free(gated_fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("bernstein_degree_grid is skipped (result matches the fixed-degree fit) when the tail-dependence gate is not met");
}

static void test_degree_grid_activates_for_tail_dependent_data(void) {
    /* Strong (p=0.6), genuinely tail-dependent data with a low gate
     * (0.1, easily cleared): the degree search must actually run, and
     * the diagnostic must reflect real tail dependence. */
    size_t n = 1500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.6, 62, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(10);
    size_t degree_grid[3] = {4, 6, 8};
    opt.bernstein_degree_grid = degree_grid;
    opt.bernstein_degree_grid_size = 3;
    opt.tail_dependence_gate = 0.1;
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit) == HDCD_OK);

    double max_td = hdcd_local_fit_max_tail_dependence(fit);
    HDCD_CHECK(max_td >= 0.1); /* gate cleared -- confirms the search actually ran, not just a default fallback */
    size_t selected_degree = hdcd_local_fit_selected_bernstein_degree(fit);
    int is_grid_member = (selected_degree == 4 || selected_degree == 6 || selected_degree == 8);
    HDCD_CHECK(is_grid_member);

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("bernstein_degree_grid activates and selects a grid member when the tail-dependence gate is cleared");
}

static void test_joint_degree_lambda_search(void) {
    /* Both grids supplied together: the winner must be a member of
     * EACH respective grid, exercising the actual cross-product search. */
    size_t n = 1500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.5, 63, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(11);
    size_t degree_grid[2] = {4, 6};
    double lambda_grid[2] = {0.15, 0.3};
    opt.bernstein_degree_grid = degree_grid;
    opt.bernstein_degree_grid_size = 2;
    opt.tail_dependence_gate = 0.05;
    opt.lambda_roughness_grid = lambda_grid;
    opt.lambda_roughness_grid_size = 2;
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit) == HDCD_OK);

    size_t selected_degree = hdcd_local_fit_selected_bernstein_degree(fit);
    double selected_lambda = hdcd_local_fit_selected_lambda_roughness(fit);
    HDCD_CHECK(selected_degree == 4 || selected_degree == 6);
    HDCD_CHECK(selected_lambda == 0.15 || selected_lambda == 0.3);

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("bernstein_degree_grid and lambda_roughness_grid together run a joint cross-product search");
}

static void test_root_node_degree_diagnostics_are_trivial(void) {
    size_t n = 20, d = 2;
    double u[40];
    uint8_t mask[40];
    for (size_t i = 0; i < 40; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_local_fit_options_t opt = default_options(1);
    size_t degree_grid[2] = {4, 8};
    opt.bernstein_degree_grid = degree_grid;
    opt.bernstein_degree_grid_size = 2;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 0, NULL, 0, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_selected_bernstein_degree(fit) == 0);
    HDCD_CHECK(isnan(hdcd_local_fit_max_tail_dependence(fit)));

    hdcd_local_fit_free(fit);
    HDCD_PASS("root node's degree/tail-dependence diagnostics are trivial (nothing to select or measure)");
}

static void test_degree_grid_invalid_arguments(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.5, 64, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;

    hdcd_local_fit_options_t bad_degree_value = default_options(1);
    size_t degree_grid_with_zero[2] = {4, 0};
    bad_degree_value.bernstein_degree_grid = degree_grid_with_zero;
    bad_degree_value.bernstein_degree_grid_size = 2;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_degree_value, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_gate_low = default_options(1);
    size_t degree_grid_ok[2] = {4, 6};
    bad_gate_low.bernstein_degree_grid = degree_grid_ok;
    bad_gate_low.bernstein_degree_grid_size = 2;
    bad_gate_low.tail_dependence_gate = -0.1;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_gate_low, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_gate_high = default_options(1);
    bad_gate_high.bernstein_degree_grid = degree_grid_ok;
    bad_gate_high.bernstein_degree_grid_size = 2;
    bad_gate_high.tail_dependence_gate = 1.5;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_gate_high, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("bernstein_degree_grid rejects a zero-degree candidate and an out-of-range tail_dependence_gate");
}

static void test_corner_relief_default_matches_unweighted(void) {
    /* corner_relief = 0 (the memset/default_options() default) must
     * reproduce exactly the same fit as before this option existed. */
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.7, 71, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(31);
    HDCD_CHECK_NEAR(opt.corner_relief, 0.0, 1e-15); /* confirms memset zeroed it, not an explicit default_options() line */
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_theta_converged(fit));

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_relief=0 (the default) fits and converges exactly as before this option existed");
}

static void test_corner_relief_changes_the_fit(void) {
    /* A nonzero corner_relief must actually change the fitted Theta
     * (and therefore the holdout score / roughness penalty) on data
     * with real tail dependence -- otherwise the option would be a
     * silent no-op. */
    size_t n = 1500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.6, 72, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt_flat = default_options(32);
    hdcd_local_fit_options_t opt_relief = default_options(32);
    opt_relief.corner_relief = 0.8;
    size_t parent = 0;

    hdcd_local_fit_t *fit_flat = NULL;
    hdcd_local_fit_t *fit_relief = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt_flat, &fit_flat) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt_relief, &fit_relief) == HDCD_OK);

    HDCD_CHECK(fabs(hdcd_local_fit_holdout_score(fit_flat) - hdcd_local_fit_holdout_score(fit_relief)) > 1e-9
               || fabs(hdcd_local_fit_roughness_penalty(fit_flat) - hdcd_local_fit_roughness_penalty(fit_relief)) > 1e-9);

    hdcd_local_fit_free(fit_flat);
    hdcd_local_fit_free(fit_relief);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("a nonzero corner_relief measurably changes the fit on tail-dependent data");
}

static void test_corner_relief_invalid_arguments(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.5, 73, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;

    hdcd_local_fit_options_t bad_negative = default_options(1);
    bad_negative.corner_relief = -0.1;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_negative, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_one = default_options(1);
    bad_one.corner_relief = 1.0;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_one, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_relief outside [0,1) is rejected");
}

static void test_corner_kde_default_matches_unweighted(void) {
    /* corner_kde_gate = 0 (the memset/default_options() default) must
     * reproduce exactly the same fit as before this option existed. */
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.5, 81, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt = default_options(41);
    HDCD_CHECK_NEAR(opt.corner_kde_gate, 0.0, 1e-15); /* memset default, not an explicit default_options() line */
    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_corner_side(fit, 0) == HDCD_CORNER_NONE);

    hdcd_local_fit_free(fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_kde_gate=0 (the default) leaves every edge uncorrected, exactly as before this option existed");
}

static void test_corner_kde_selects_side_and_changes_fit_on_tail_dependent_data(void) {
    /* Strong, genuinely tail-dependent data with an easily-cleared gate:
     * a corner side must be selected, and the fit must measurably
     * differ from the uncorrected one -- otherwise the option would be
     * a silent no-op. */
    size_t n = 1500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_tail_dependent_data(n, 0.6, 82, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt_flat = default_options(42);
    hdcd_local_fit_options_t opt_kde = default_options(42);
    opt_kde.corner_kde_gate = 0.1;
    opt_kde.corner_kde_bandwidth = 0.1;
    opt_kde.corner_kde_weight = 1.0;
    size_t parent = 0;

    hdcd_local_fit_t *fit_flat = NULL;
    hdcd_local_fit_t *fit_kde = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt_flat, &fit_flat) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt_kde, &fit_kde) == HDCD_OK);

    HDCD_CHECK(hdcd_local_fit_corner_side(fit_kde, 0) != HDCD_CORNER_NONE);
    HDCD_CHECK(hdcd_local_fit_corner_side(fit_flat, 0) == HDCD_CORNER_NONE); /* gate=0: never computed */

    double log_flat, log_kde;
    double u_val = 0.05, z_val = 0.05; /* near the corner, where the correction is most active */
    HDCD_CHECK(hdcd_local_fit_log_density(fit_flat, u_val, &z_val, 1, &log_flat) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_log_density(fit_kde, u_val, &z_val, 1, &log_kde) == HDCD_OK);
    HDCD_CHECK(fabs(log_flat - log_kde) > 1e-6);

    hdcd_local_fit_free(fit_flat);
    hdcd_local_fit_free(fit_kde);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_kde_gate selects a corner side and measurably changes the fit near the corner on tail-dependent data");
}

static void test_corner_kde_gated_off_matches_unweighted(void) {
    /* A gate (0.9) essentially nothing clears: the corrected fit must
     * exactly match a plain (corner_kde_gate=0) fit. */
    size_t n = 600, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.6, 83, u0, u1); /* Gaussian: no tail dependence at all */
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    hdcd_local_fit_options_t opt_flat = default_options(43);
    hdcd_local_fit_options_t gated_opt = default_options(43);
    gated_opt.corner_kde_gate = 0.9;
    size_t parent = 0;

    hdcd_local_fit_t *fit_flat = NULL;
    hdcd_local_fit_t *gated_fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &opt_flat, &fit_flat) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &gated_opt, &gated_fit) == HDCD_OK);

    HDCD_CHECK(hdcd_local_fit_corner_side(gated_fit, 0) == HDCD_CORNER_NONE);
    HDCD_CHECK_NEAR(hdcd_local_fit_holdout_score(gated_fit), hdcd_local_fit_holdout_score(fit_flat), 1e-9);

    hdcd_local_fit_free(fit_flat);
    hdcd_local_fit_free(gated_fit);
    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_kde_gate is skipped (result matches the uncorrected fit) when no edge clears it");
}

static void test_corner_kde_root_node_trivial(void) {
    size_t n = 20, d = 2;
    double u[40];
    uint8_t mask[40];
    for (size_t i = 0; i < 40; i++) { u[i] = 0.5; mask[i] = 1; }

    hdcd_local_fit_options_t opt = default_options(44);
    opt.corner_kde_gate = 0.1;
    hdcd_local_fit_t *fit = NULL;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 0, NULL, 0, &opt, &fit) == HDCD_OK);
    HDCD_CHECK(hdcd_local_fit_corner_side(fit, 0) == HDCD_CORNER_NONE);

    hdcd_local_fit_free(fit);
    HDCD_PASS("root node's corner_kde diagnostics are trivial (nothing to correct)");
}

static void test_corner_kde_invalid_arguments(void) {
    size_t n = 500, d = 2;
    double *u0 = (double *)malloc(n * sizeof(double));
    double *u1 = (double *)malloc(n * sizeof(double));
    make_gaussian_copula_data(n, 0.5, 84, u0, u1);
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    memcpy(&u[0 * n], u0, n * sizeof(double));
    memcpy(&u[1 * n], u1, n * sizeof(double));
    for (size_t i = 0; i < n * d; i++) mask[i] = 1;

    size_t parent = 0;
    hdcd_local_fit_t *fit = NULL;

    hdcd_local_fit_options_t bad_negative_gate = default_options(1);
    bad_negative_gate.corner_kde_gate = -0.1;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_negative_gate, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_high_gate = default_options(1);
    bad_high_gate.corner_kde_gate = 1.5;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_high_gate, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_bandwidth = default_options(1);
    bad_bandwidth.corner_kde_gate = 0.1;
    bad_bandwidth.corner_kde_bandwidth = -0.05;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_bandwidth, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_local_fit_options_t bad_weight = default_options(1);
    bad_weight.corner_kde_gate = 0.1;
    bad_weight.corner_kde_weight = -1.0;
    HDCD_CHECK(hdcd_local_fit_node(u, mask, n, d, 1, &parent, 1, &bad_weight, &fit) == HDCD_ERROR_INVALID_ARGUMENT);

    free(u0); free(u1); free(u); free(mask);
    HDCD_PASS("corner_kde_gate outside [0,1], a negative bandwidth, and a negative weight are all rejected");
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
    test_roughness_grid_matches_fixed_when_singleton();
    test_roughness_grid_picks_a_lighter_penalty();
    test_root_node_selected_lambda_is_nan();
    test_roughness_grid_invalid_arguments();
    test_tail_dependence_diagnostic_available_without_any_grid();
    test_degree_grid_gated_off_matches_fixed_degree();
    test_degree_grid_activates_for_tail_dependent_data();
    test_joint_degree_lambda_search();
    test_root_node_degree_diagnostics_are_trivial();
    test_degree_grid_invalid_arguments();
    test_corner_relief_default_matches_unweighted();
    test_corner_relief_changes_the_fit();
    test_corner_relief_invalid_arguments();
    test_corner_kde_default_matches_unweighted();
    test_corner_kde_selects_side_and_changes_fit_on_tail_dependent_data();
    test_corner_kde_gated_off_matches_unweighted();
    test_corner_kde_root_node_trivial();
    test_corner_kde_invalid_arguments();
    test_invalid_arguments();
    printf("All local_fit tests passed.\n");
    return 0;
}
