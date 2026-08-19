#include "hdcd/sinkhorn.h"
#include "hdcd/bernstein.h"
#include "hdcd/numerics.h"
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

/* ---- benchmark kernels ------------------------------------------------ */

static double kernel_constant(double u, const double *z, size_t z_dim, void *userdata) {
    (void)u; (void)z; (void)z_dim; (void)userdata;
    return 1.0;
}

typedef struct { double rho; } tilt_params_t;

static double kernel_exponential_tilt(double u, const double *z, size_t z_dim, void *userdata) {
    (void)z_dim;
    const tilt_params_t *p = (const tilt_params_t *)userdata;
    return exp(p->rho * (u - 0.5) * (z[0] - 0.5));
}

typedef struct { size_t m; const double *theta; } bernstein_params_t;

static double kernel_bernstein(double u, const double *z, size_t z_dim, void *userdata) {
    (void)z_dim;
    const bernstein_params_t *p = (const bernstein_params_t *)userdata;
    double g;
    hdcd_bernstein_tensor_interaction(u, z[0], p->m, p->theta, &g);
    return exp(g);
}

/* Independent (test-only) Simpson integral of a fixed-z slice of a
 * fitted hdcd_sinkhorn_t, used to cross-check hdcd_sinkhorn_conditional_
 * integral_error via the PUBLIC evaluation API only, with its own fresh
 * quadrature grid (not reusing the internal one). */
static double integrate_conditional_over_u(const hdcd_sinkhorn_t *sk, double z_val) {
    size_t n = 129;
    double *nodes = (double *)malloc(n * sizeof(double));
    double *weights = (double *)malloc(n * sizeof(double));
    hdcd_simpson_nodes_weights(n, nodes, weights);

    double integral = 0.0;
    for (size_t i = 0; i < n; i++) {
        double c;
        hdcd_status_t status = hdcd_sinkhorn_conditional_density(sk, nodes[i], &z_val, 1, &c);
        HDCD_CHECK(status == HDCD_OK);
        integral += weights[i] * c;
    }
    free(nodes);
    free(weights);
    return integral;
}

static double *make_uniform_samples(size_t n, uint64_t seed) {
    double *z = (double *)malloc(n * sizeof(double));
    rng_seed(seed);
    for (size_t i = 0; i < n; i++) {
        z[i] = rng_uniform();
    }
    return z;
}

static void test_constant_kernel_converges_trivially(void) {
    size_t n_z = 300;
    double *z_samples = make_uniform_samples(n_z, 1);

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t status = hdcd_sinkhorn_fit(kernel_constant, NULL, z_samples, n_z, 1, NULL, &sk);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(hdcd_sinkhorn_converged(sk));
    HDCD_CHECK(hdcd_sinkhorn_iterations(sk) <= 2);
    HDCD_CHECK(hdcd_sinkhorn_conditional_integral_error(sk) < 1e-6);
    HDCD_CHECK(hdcd_sinkhorn_marginal_preservation_error(sk) < 1e-6);

    double c;
    HDCD_CHECK(hdcd_sinkhorn_conditional_density(sk, 0.5, &(double){0.5}, 1, &c) == HDCD_OK);
    HDCD_CHECK_NEAR(c, 1.0, 1e-6);

    hdcd_sinkhorn_free(sk);
    free(z_samples);
    HDCD_PASS("constant kernel converges trivially to c(u|z)=1");
}

static void test_conditional_integral_error_below_tolerance(void) {
    /* Independently re-integrate over u via the PUBLIC API, for both a
     * z used in fitting and a fresh z, and check both are within a
     * tight tolerance of 1 -- spec section 31 M6's first acceptance
     * criterion, verified end-to-end rather than by trusting the
     * internal diagnostic alone. */
    size_t n_z = 300;
    double *z_samples = make_uniform_samples(n_z, 2);
    tilt_params_t params = {3.0};

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t status = hdcd_sinkhorn_fit(kernel_exponential_tilt, &params, z_samples, n_z, 1, NULL, &sk);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(hdcd_sinkhorn_converged(sk));

    double training_z = z_samples[17];
    double fresh_z = 0.6789; /* not in z_samples */

    HDCD_CHECK_NEAR(integrate_conditional_over_u(sk, training_z), 1.0, 1e-4);
    HDCD_CHECK_NEAR(integrate_conditional_over_u(sk, fresh_z), 1.0, 1e-4);
    HDCD_CHECK(hdcd_sinkhorn_conditional_integral_error(sk) < 1e-6);

    hdcd_sinkhorn_free(sk);
    free(z_samples);
    HDCD_PASS("conditional integral error is below tolerance (internal diagnostic and independent re-integration)");
}

static void test_marginal_preservation_error_below_tolerance(void) {
    size_t n_z = 400;
    double *z_samples = make_uniform_samples(n_z, 3);
    tilt_params_t params = {2.5};

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t status = hdcd_sinkhorn_fit(kernel_exponential_tilt, &params, z_samples, n_z, 1, NULL, &sk);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(hdcd_sinkhorn_marginal_preservation_error(sk) < 1e-6);

    /* Looser generalization check: average c(u_fixed|z) over a FRESH,
     * independent z sample (not used in fitting) for a few u values --
     * this has genuine Monte Carlo noise (~1/sqrt(n)), unlike the tight
     * internal diagnostic above which is self-consistent against the
     * training samples by construction. */
    size_t n_fresh = 4000;
    double *fresh_z = make_uniform_samples(n_fresh, 999);
    double test_u[3] = {0.2, 0.5, 0.85};
    for (int i = 0; i < 3; i++) {
        double sum = 0.0;
        for (size_t m = 0; m < n_fresh; m++) {
            double c;
            HDCD_CHECK(hdcd_sinkhorn_conditional_density(sk, test_u[i], &fresh_z[m], 1, &c) == HDCD_OK);
            sum += c;
        }
        double mean = sum / (double)n_fresh;
        HDCD_CHECK(fabs(mean - 1.0) < 0.08);
    }

    free(fresh_z);
    hdcd_sinkhorn_free(sk);
    free(z_samples);
    HDCD_PASS("marginal preservation error is below tolerance (internal diagnostic and fresh-sample generalization)");
}

static void test_robust_convergence_over_benchmark_kernels(void) {
    size_t n_z = 300;
    double *z_samples = make_uniform_samples(n_z, 4);

    /* Benchmark 1: constant (independence) kernel. */
    {
        hdcd_sinkhorn_t *sk = NULL;
        HDCD_CHECK(hdcd_sinkhorn_fit(kernel_constant, NULL, z_samples, n_z, 1, NULL, &sk) == HDCD_OK);
        HDCD_CHECK(hdcd_sinkhorn_converged(sk));
        hdcd_sinkhorn_free(sk);
    }

    /* Benchmark 2: exponential-tilt kernel, moderate coupling. */
    {
        tilt_params_t params = {4.0};
        hdcd_sinkhorn_t *sk = NULL;
        HDCD_CHECK(hdcd_sinkhorn_fit(kernel_exponential_tilt, &params, z_samples, n_z, 1, NULL, &sk) == HDCD_OK);
        HDCD_CHECK(hdcd_sinkhorn_converged(sk));
        hdcd_sinkhorn_free(sk);
    }

    /* Benchmark 3: Bernstein tensor kernel with a non-separable theta
     * (spec section 9 kernel, as it will actually be used downstream). */
    {
        size_t m = 3, dim = m + 1;
        double *theta = (double *)malloc(dim * dim * sizeof(double));
        rng_seed(55);
        for (size_t i = 0; i < dim * dim; i++) {
            theta[i] = (rng_uniform() - 0.5) * 1.5;
        }
        bernstein_params_t params = {m, theta};
        hdcd_sinkhorn_t *sk = NULL;
        HDCD_CHECK(hdcd_sinkhorn_fit(kernel_bernstein, &params, z_samples, n_z, 1, NULL, &sk) == HDCD_OK);
        HDCD_CHECK(hdcd_sinkhorn_converged(sk));
        HDCD_CHECK(hdcd_sinkhorn_conditional_integral_error(sk) < 1e-6);
        HDCD_CHECK(hdcd_sinkhorn_marginal_preservation_error(sk) < 1e-6);
        hdcd_sinkhorn_free(sk);
        free(theta);
    }

    free(z_samples);
    HDCD_PASS("Sinkhorn converges robustly across constant, exponential-tilt, and Bernstein benchmark kernels");
}

static void test_nonconvergence_diagnostics_are_clear(void) {
    size_t n_z = 300;
    double *z_samples = make_uniform_samples(n_z, 5);
    tilt_params_t params = {6.0}; /* strong coupling: needs several iterations */

    hdcd_sinkhorn_options_t options;
    options.n_quadrature_nodes = 0;
    options.tol = 1e-12; /* unreachable in the iteration budget below */
    options.max_iterations = 1;

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t status = hdcd_sinkhorn_fit(kernel_exponential_tilt, &params, z_samples, n_z, 1, &options, &sk);

    HDCD_CHECK(status == HDCD_ERROR_NOT_CONVERGED);
    HDCD_CHECK(sk != NULL); /* best-effort state still populated, not silently dropped */
    HDCD_CHECK(!hdcd_sinkhorn_converged(sk));
    HDCD_CHECK(hdcd_sinkhorn_iterations(sk) == 1);
    HDCD_CHECK(!isnan(hdcd_sinkhorn_conditional_integral_error(sk)));
    HDCD_CHECK(!isnan(hdcd_sinkhorn_marginal_preservation_error(sk)));

    /* The half-fitted state must still be usable, not garbage. */
    double c;
    HDCD_CHECK(hdcd_sinkhorn_conditional_density(sk, 0.5, &(double){0.5}, 1, &c) == HDCD_OK);
    HDCD_CHECK(isfinite(c) && c > 0.0);

    hdcd_sinkhorn_free(sk);
    free(z_samples);
    HDCD_PASS("non-convergence returns HDCD_ERROR_NOT_CONVERGED with usable diagnostics, not a silent failure");
}

static void test_invalid_arguments(void) {
    double z_samples[4] = {0.1, 0.2, 0.3, 0.4};
    hdcd_sinkhorn_t *sk = NULL;

    HDCD_CHECK(hdcd_sinkhorn_fit(NULL, NULL, z_samples, 4, 1, NULL, &sk) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_sinkhorn_fit(kernel_constant, NULL, NULL, 4, 1, NULL, &sk) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_sinkhorn_fit(kernel_constant, NULL, z_samples, 0, 1, NULL, &sk) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_sinkhorn_fit(kernel_constant, NULL, z_samples, 4, 0, NULL, &sk) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_sinkhorn_options_t bad_options;
    bad_options.n_quadrature_nodes = 4; /* even: invalid for Simpson */
    bad_options.tol = 0.0;
    bad_options.max_iterations = 0;
    HDCD_CHECK(hdcd_sinkhorn_fit(kernel_constant, NULL, z_samples, 4, 1, &bad_options, &sk) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_sinkhorn_converged(NULL) == 0);
    HDCD_CHECK(hdcd_sinkhorn_iterations(NULL) == 0);
    HDCD_CHECK(isnan(hdcd_sinkhorn_conditional_integral_error(NULL)));
    HDCD_CHECK(isnan(hdcd_sinkhorn_marginal_preservation_error(NULL)));

    double out;
    HDCD_CHECK(hdcd_sinkhorn_conditional_density(NULL, 0.5, z_samples, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);

    hdcd_sinkhorn_free(NULL); /* must not crash */

    HDCD_PASS("Sinkhorn API rejects invalid arguments");
}

int main(void) {
    test_constant_kernel_converges_trivially();
    test_conditional_integral_error_below_tolerance();
    test_marginal_preservation_error_below_tolerance();
    test_robust_convergence_over_benchmark_kernels();
    test_nonconvergence_diagnostics_are_clear();
    test_invalid_arguments();
    printf("All sinkhorn tests passed.\n");
    return 0;
}
