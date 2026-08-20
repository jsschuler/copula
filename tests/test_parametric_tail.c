#include "hdcd/parametric_tail.h"
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

static void test_clayton_density_known_value(void) {
    /* c(0.5,0.5;theta=2) hand-derived from the closed form:
     * (1+theta)*(uv)^{-theta-1}*(u^{-theta}+v^{-theta}-1)^{-1/theta-2}
     * = 3 * 0.25^{-3} * (4+4-1)^{-2.5} = 3 * 64 * 7^{-2.5}. */
    double expected = 3.0 * 64.0 * pow(7.0, -2.5);
    double actual;
    HDCD_CHECK(hdcd_clayton_density(0.5, 0.5, 2.0, &actual) == HDCD_OK);
    HDCD_CHECK_NEAR(actual, expected, 1e-9);
    HDCD_PASS("Clayton density matches a hand-derived value at (0.5,0.5,theta=2)");
}

static void test_clayton_density_integrates_to_one(void) {
    /* 2D composite Simpson's rule over [eps,1-eps]^2 for a few
     * moderate theta, confirming the density formula itself is a
     * legitimate pdf. Kept to moderate theta (Kendall's tau up to
     * ~0.56) -- stronger dependence sharpens the corner singularity
     * enough that plain Simpson quadrature error swamps a check this
     * cheap; the formula itself is validated exactly, at any theta, by
     * the hand-derived point-value check above. */
    size_t n = 400; /* even */
    double eps = 1e-4;
    double h = (1.0 - 2.0 * eps) / (double)n;
    for (double theta = 0.5; theta <= 2.5; theta += 1.0) {
        double total = 0.0;
        for (size_t i = 0; i <= n; i++) {
            double u = eps + i * h;
            double wu = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            for (size_t j = 0; j <= n; j++) {
                double v = eps + j * h;
                double wv = (j == 0 || j == n) ? 1.0 : (j % 2 == 1 ? 4.0 : 2.0);
                double dens;
                HDCD_CHECK(hdcd_clayton_density(u, v, theta, &dens) == HDCD_OK);
                total += wu * wv * dens;
            }
        }
        total *= (h / 3.0) * (h / 3.0);
        /* Composite Simpson assumes smoothness; Clayton's density has a
         * genuine corner singularity as (u,v) -> (0,0), so quadrature
         * error here is real (not a formula bug -- confirmed exact at a
         * hand-derived point above) and grows with theta. 3% is a loose
         * sanity bound, not a precision claim. */
        HDCD_CHECK_NEAR(total, 1.0, 0.03);
    }
    HDCD_PASS("Clayton density integrates to ~1 over the unit square for several theta");
}

static void test_gumbel_theta_one_is_independence(void) {
    double points[5][2] = {{0.1, 0.1}, {0.3, 0.7}, {0.5, 0.5}, {0.9, 0.2}, {0.99, 0.99}};
    for (int i = 0; i < 5; i++) {
        double dens;
        HDCD_CHECK(hdcd_gumbel_density(points[i][0], points[i][1], 1.0, &dens) == HDCD_OK);
        HDCD_CHECK_NEAR(dens, 1.0, 1e-8);
    }
    HDCD_PASS("Gumbel density at theta=1 reduces to the independence copula (c=1) everywhere");
}

static void test_gumbel_density_integrates_to_one(void) {
    size_t n = 200;
    double eps = 1e-4;
    double h = (1.0 - 2.0 * eps) / (double)n;
    for (double theta = 1.2; theta <= 4.0; theta += 1.0) {
        double total = 0.0;
        for (size_t i = 0; i <= n; i++) {
            double u = eps + i * h;
            double wu = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            for (size_t j = 0; j <= n; j++) {
                double v = eps + j * h;
                double wv = (j == 0 || j == n) ? 1.0 : (j % 2 == 1 ? 4.0 : 2.0);
                double dens;
                HDCD_CHECK(hdcd_gumbel_density(u, v, theta, &dens) == HDCD_OK);
                total += wu * wv * dens;
            }
        }
        total *= (h / 3.0) * (h / 3.0);
        HDCD_CHECK_NEAR(total, 1.0, 0.02);
    }
    HDCD_PASS("Gumbel density integrates to ~1 over the unit square for several theta");
}

/* Closed-form Clayton conditional-inverse (h-function inverse), used to
 * generate EXACT Clayton samples for the MLE-recovery test below --
 * derived from C(v|u) = u^{-theta-1}(u^{-theta}+v^{-theta}-1)^{-1/theta-1}. */
static double clayton_hinv(double u, double p, double theta) {
    double a = pow(p * pow(u, theta + 1.0), -theta / (theta + 1.0));
    double base = a - pow(u, -theta) + 1.0;
    return pow(base, -1.0 / theta);
}

static void test_clayton_mle_recovers_known_theta(void) {
    size_t n = 3000;
    double true_theta = 2.5;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    rng_seed(11);
    for (size_t i = 0; i < n; i++) {
        double uu = rng_uniform();
        double p = rng_uniform();
        double vv = clayton_hinv(uu, p, true_theta);
        u[i] = fmin(fmax(uu, 1e-6), 1.0 - 1e-6);
        v[i] = fmin(fmax(vv, 1e-6), 1.0 - 1e-6);
    }

    double fitted_theta;
    hdcd_status_t status = hdcd_tail_family_fit(HDCD_TAIL_FAMILY_CLAYTON, u, v, n, &fitted_theta);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(fabs(fitted_theta - true_theta) < 0.2);

    free(u); free(v);
    HDCD_PASS("Clayton MLE recovers the true theta from exactly-simulated Clayton data");
}

/* Closed-form Gumbel conditional CDF h(v|u), monotone increasing in v --
 * inverted here via bisection to generate EXACT Gumbel samples for the
 * MLE-recovery test below. */
static double gumbel_h(double u, double v, double theta) {
    double x = -log(u);
    double y = -log(v);
    double log_A = log(pow(x, theta) + pow(y, theta)) / theta;
    double A = exp(log_A);
    double log_C = -A;
    return exp(log_C - log(u) + (1.0 - theta) * log_A + (theta - 1.0) * log(x));
}

static double gumbel_hinv(double u, double p, double theta) {
    double lo = 1e-9, hi = 1.0 - 1e-9;
    for (int iter = 0; iter < 100; iter++) {
        double mid = 0.5 * (lo + hi);
        double val = gumbel_h(u, mid, theta);
        if (val < p) {
            lo = mid; /* h(v|u) increasing in v */
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

static void test_gumbel_mle_recovers_known_theta(void) {
    size_t n = 3000;
    double true_theta = 2.0;
    double *u = (double *)malloc(n * sizeof(double));
    double *v = (double *)malloc(n * sizeof(double));
    rng_seed(22);
    for (size_t i = 0; i < n; i++) {
        double uu = rng_uniform();
        double p = rng_uniform();
        double vv = gumbel_hinv(uu, p, true_theta);
        u[i] = fmin(fmax(uu, 1e-6), 1.0 - 1e-6);
        v[i] = fmin(fmax(vv, 1e-6), 1.0 - 1e-6);
    }

    double fitted_theta;
    hdcd_status_t status = hdcd_tail_family_fit(HDCD_TAIL_FAMILY_GUMBEL, u, v, n, &fitted_theta);
    HDCD_CHECK(status == HDCD_OK);
    HDCD_CHECK(fabs(fitted_theta - true_theta) < 0.2);

    free(u); free(v);
    HDCD_PASS("Gumbel MLE recovers the true theta from exactly-simulated Gumbel data");
}

static void test_invalid_arguments(void) {
    double out;
    HDCD_CHECK(hdcd_clayton_density(0.5, 0.5, -1.0, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_clayton_density(0.0, 0.5, 1.0, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_clayton_density(0.5, 0.5, 1.0, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_gumbel_density(0.5, 0.5, 0.5, &out) == HDCD_ERROR_INVALID_ARGUMENT); /* theta < 1 */
    HDCD_CHECK(hdcd_gumbel_density(1.0, 0.5, 2.0, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_tail_family_density(HDCD_TAIL_FAMILY_NONE, 0.5, 0.5, 1.0, &out) == HDCD_ERROR_INVALID_ARGUMENT);

    double u[4] = {0.1, 0.2, 0.3, 0.4}, v[4] = {0.1, 0.2, 0.3, 0.4};
    double theta_out;
    HDCD_CHECK(hdcd_tail_family_fit(HDCD_TAIL_FAMILY_CLAYTON, NULL, v, 4, &theta_out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_tail_family_fit(HDCD_TAIL_FAMILY_CLAYTON, u, v, 1, &theta_out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_tail_family_fit(HDCD_TAIL_FAMILY_NONE, u, v, 4, &theta_out) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_PASS("parametric_tail API rejects invalid arguments");
}

int main(void) {
    test_clayton_density_known_value();
    test_clayton_density_integrates_to_one();
    test_gumbel_theta_one_is_independence();
    test_gumbel_density_integrates_to_one();
    test_clayton_mle_recovers_known_theta();
    test_gumbel_mle_recovers_known_theta();
    test_invalid_arguments();
    printf("All parametric_tail tests passed.\n");
    return 0;
}
