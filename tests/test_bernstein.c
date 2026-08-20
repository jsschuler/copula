#include "hdcd/bernstein.h"
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

/* Independent (test-only) Beta(a,b) density via lgamma, used only to
 * cross-check the (m+1)*B_{r,m}(u) = BetaPDF(u; r+1, m-r+1) identity
 * through a numerically different code path than hdcd_bernstein_basis. */
static double beta_pdf_reference(double u, double a, double b) {
    double log_norm = lgamma(a) + lgamma(b) - lgamma(a + b);
    double log_val = (a - 1.0) * log(u) + (b - 1.0) * log(1.0 - u) - log_norm;
    return exp(log_val);
}

static void test_partition_of_unity(void) {
    size_t degrees[3] = {1, 4, 10};
    double points[5] = {0.0, 0.1, 0.5, 0.83, 1.0};

    for (int di = 0; di < 3; di++) {
        size_t m = degrees[di];
        double *basis = (double *)malloc((m + 1) * sizeof(double));
        for (int pi = 0; pi < 5; pi++) {
            double u = points[pi];
            HDCD_CHECK(hdcd_bernstein_basis(u, m, basis) == HDCD_OK);
            double sum = 0.0;
            for (size_t r = 0; r <= m; r++) {
                HDCD_CHECK(basis[r] >= -1e-12); /* non-negativity on [0,1] */
                sum += basis[r];
            }
            HDCD_CHECK_NEAR(sum, 1.0, 1e-10);
        }
        free(basis);
    }
    HDCD_PASS("Bernstein basis is a partition of unity and non-negative on [0,1]");
}

static void test_boundary_values(void) {
    size_t m = 6;
    double *basis = (double *)malloc((m + 1) * sizeof(double));

    HDCD_CHECK(hdcd_bernstein_basis(0.0, m, basis) == HDCD_OK);
    HDCD_CHECK_NEAR(basis[0], 1.0, 1e-12);
    for (size_t r = 1; r <= m; r++) HDCD_CHECK_NEAR(basis[r], 0.0, 1e-12);

    HDCD_CHECK(hdcd_bernstein_basis(1.0, m, basis) == HDCD_OK);
    HDCD_CHECK_NEAR(basis[m], 1.0, 1e-12);
    for (size_t r = 0; r < m; r++) HDCD_CHECK_NEAR(basis[r], 0.0, 1e-12);

    free(basis);
    HDCD_PASS("Bernstein basis boundary values match B_0(0)=1, B_m(1)=1");
}

static void test_beta_pdf_identity(void) {
    size_t m = 7;
    double *basis = (double *)malloc((m + 1) * sizeof(double));
    double points[3] = {0.2, 0.5, 0.9};

    for (int pi = 0; pi < 3; pi++) {
        double u = points[pi];
        HDCD_CHECK(hdcd_bernstein_basis(u, m, basis) == HDCD_OK);
        for (size_t r = 0; r <= m; r++) {
            double lhs = (double)(m + 1) * basis[r];
            double rhs = beta_pdf_reference(u, (double)(r + 1), (double)(m - r + 1));
            HDCD_CHECK_NEAR(lhs, rhs, 1e-8);
        }
    }
    free(basis);
    HDCD_PASS("(m+1)*B_{r,m}(u) matches an independently computed Beta(r+1,m-r+1) density");
}

static void test_centering_integral_near_zero(void) {
    /* integral_0^1 B~_{r,m}(u) du = 0 exactly (since integral of the raw
     * basis is 1/(m+1)); verify via Simpson's rule quadrature. */
    size_t degrees[2] = {3, 8};
    for (int di = 0; di < 2; di++) {
        size_t m = degrees[di];
        for (size_t r = 0; r <= m; r++) {
            int steps = 4000; /* even, for Simpson */
            double h = 1.0 / (double)steps;
            double integral = 0.0;
            double *basis = (double *)malloc((m + 1) * sizeof(double));
            for (int i = 0; i <= steps; i++) {
                double u = h * (double)i;
                HDCD_CHECK(hdcd_bernstein_basis_centered(u, m, basis) == HDCD_OK);
                double weight = (i == 0 || i == steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
                integral += weight * basis[r];
            }
            integral *= h / 3.0;
            free(basis);
            HDCD_CHECK(fabs(integral) < 1e-6);
        }
    }
    HDCD_PASS("centered basis integrates to ~0 over [0,1] for every r,m tested");
}

static void test_derivative_matches_finite_difference(void) {
    size_t degrees[2] = {3, 6};
    double points[3] = {0.15, 0.5, 0.82};
    double eps = 1e-6;

    for (int di = 0; di < 2; di++) {
        size_t m = degrees[di];
        double *b_lo = (double *)malloc((m + 1) * sizeof(double));
        double *b_hi = (double *)malloc((m + 1) * sizeof(double));
        double *deriv = (double *)malloc((m + 1) * sizeof(double));

        for (int pi = 0; pi < 3; pi++) {
            double u = points[pi];
            HDCD_CHECK(hdcd_bernstein_basis(u - eps, m, b_lo) == HDCD_OK);
            HDCD_CHECK(hdcd_bernstein_basis(u + eps, m, b_hi) == HDCD_OK);
            HDCD_CHECK(hdcd_bernstein_basis_derivative(u, m, deriv) == HDCD_OK);
            for (size_t r = 0; r <= m; r++) {
                double fd = (b_hi[r] - b_lo[r]) / (2.0 * eps);
                HDCD_CHECK_NEAR(fd, deriv[r], 1e-5);
            }
        }
        free(b_lo);
        free(b_hi);
        free(deriv);
    }
    HDCD_PASS("Bernstein basis derivative matches finite differences");
}

static void test_tensor_gradient_matches_finite_difference(void) {
    size_t m = 4;
    size_t dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    double *grad = (double *)malloc(dim * dim * sizeof(double));

    rng_seed(1234);
    for (size_t i = 0; i < dim * dim; i++) {
        theta[i] = rng_uniform() * 4.0 - 2.0; /* in [-2, 2] */
    }

    double u = 0.37, z = 0.64;
    HDCD_CHECK(hdcd_bernstein_tensor_gradient(u, z, m, grad) == HDCD_OK);

    double eps = 1e-6;
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            double original = theta[r * dim + s];

            theta[r * dim + s] = original + eps;
            double g_hi;
            HDCD_CHECK(hdcd_bernstein_tensor_interaction(u, z, m, theta, &g_hi) == HDCD_OK);

            theta[r * dim + s] = original - eps;
            double g_lo;
            HDCD_CHECK(hdcd_bernstein_tensor_interaction(u, z, m, theta, &g_lo) == HDCD_OK);

            theta[r * dim + s] = original;

            double fd = (g_hi - g_lo) / (2.0 * eps);
            HDCD_CHECK_NEAR(fd, grad[r * dim + s], 1e-5);
        }
    }

    free(theta);
    free(grad);
    HDCD_PASS("tensor interaction gradient matches finite differences on theta");
}

static void test_roughness_gradient_matches_finite_difference(void) {
    size_t m = 5;
    size_t dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    double *grad = (double *)malloc(dim * dim * sizeof(double));

    rng_seed(999);
    for (size_t i = 0; i < dim * dim; i++) {
        theta[i] = rng_uniform() * 6.0 - 3.0;
    }

    HDCD_CHECK(hdcd_bernstein_roughness_gradient(theta, m, grad) == HDCD_OK);

    double eps = 1e-6;
    for (size_t a = 0; a < dim; a++) {
        for (size_t b = 0; b < dim; b++) {
            double original = theta[a * dim + b];

            theta[a * dim + b] = original + eps;
            double r_hi;
            HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &r_hi) == HDCD_OK);

            theta[a * dim + b] = original - eps;
            double r_lo;
            HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &r_lo) == HDCD_OK);

            theta[a * dim + b] = original;

            double fd = (r_hi - r_lo) / (2.0 * eps);
            HDCD_CHECK_NEAR(fd, grad[a * dim + b], 1e-4);
        }
    }

    free(theta);
    free(grad);
    HDCD_PASS("roughness penalty gradient matches finite differences on theta");
}

static void test_roughness_penalty_small_degrees_is_zero(void) {
    /* m < 2 means dim < 3: no interior second-difference triple exists
     * along either axis, so the penalty must be exactly 0. */
    for (size_t m = 0; m <= 1; m++) {
        size_t dim = m + 1;
        double *theta = (double *)malloc(dim * dim * sizeof(double));
        for (size_t i = 0; i < dim * dim; i++) theta[i] = (double)(i + 1) * 3.7;
        double penalty;
        HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &penalty) == HDCD_OK);
        HDCD_CHECK_NEAR(penalty, 0.0, 1e-15);
        free(theta);
    }
    HDCD_PASS("roughness penalty is exactly 0 for m < 2");
}

static void test_roughness_penalty_zero_for_linear_theta(void) {
    /* A second difference vanishes for any linear (affine-in-index)
     * sequence; theta[r][s] = a*r + b*s + c has zero second difference
     * along both axes, so R(theta) must be 0 regardless of degree. */
    size_t m = 6, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            theta[r * dim + s] = 2.0 * (double)r - 1.5 * (double)s + 7.0;
        }
    }
    double penalty;
    HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &penalty) == HDCD_OK);
    HDCD_CHECK(penalty < 1e-8);
    free(theta);
    HDCD_PASS("roughness penalty vanishes for an affine-in-index theta");
}

static void test_weighted_penalty_matches_unweighted_at_zero_relief(void) {
    size_t m = 5, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    rng_seed(42);
    for (size_t i = 0; i < dim * dim; i++) theta[i] = rng_uniform() * 4.0 - 2.0;

    double unweighted, weighted;
    HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &unweighted) == HDCD_OK);
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, 0.0, &weighted) == HDCD_OK);
    HDCD_CHECK_NEAR(unweighted, weighted, 1e-15);

    double *grad_unweighted = (double *)malloc(dim * dim * sizeof(double));
    double *grad_weighted = (double *)malloc(dim * dim * sizeof(double));
    HDCD_CHECK(hdcd_bernstein_roughness_gradient(theta, m, grad_unweighted) == HDCD_OK);
    HDCD_CHECK(hdcd_bernstein_roughness_gradient_weighted(theta, m, 0.0, grad_weighted) == HDCD_OK);
    for (size_t i = 0; i < dim * dim; i++) {
        HDCD_CHECK_NEAR(grad_unweighted[i], grad_weighted[i], 1e-15);
    }

    free(theta); free(grad_unweighted); free(grad_weighted);
    HDCD_PASS("corner_relief=0 reproduces the unweighted penalty and gradient exactly");
}

static void test_weighted_gradient_matches_finite_difference(void) {
    size_t m = 6, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    double *grad = (double *)malloc(dim * dim * sizeof(double));
    double corner_relief = 0.7;

    rng_seed(123);
    for (size_t i = 0; i < dim * dim; i++) theta[i] = rng_uniform() * 6.0 - 3.0;

    HDCD_CHECK(hdcd_bernstein_roughness_gradient_weighted(theta, m, corner_relief, grad) == HDCD_OK);

    double eps = 1e-6;
    for (size_t a = 0; a < dim; a++) {
        for (size_t b = 0; b < dim; b++) {
            double original = theta[a * dim + b];

            theta[a * dim + b] = original + eps;
            double r_hi;
            HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, corner_relief, &r_hi) == HDCD_OK);

            theta[a * dim + b] = original - eps;
            double r_lo;
            HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, corner_relief, &r_lo) == HDCD_OK);

            theta[a * dim + b] = original;

            double fd = (r_hi - r_lo) / (2.0 * eps);
            HDCD_CHECK_NEAR(fd, grad[a * dim + b], 1e-4);
        }
    }

    free(theta); free(grad);
    HDCD_PASS("weighted roughness penalty gradient matches finite differences on theta");
}

static double reference_edge_proximity(size_t k, size_t dim) {
    double half = (double)(dim - 1) / 2.0;
    size_t dist = (k < dim - 1 - k) ? k : (dim - 1 - k);
    return 1.0 - (double)dist / half;
}

static double reference_weighted_penalty(const double *theta, size_t m, double corner_relief) {
    size_t dim = m + 1;
    double penalty = 0.0;
    for (size_t s = 0; s < dim; s++) {
        for (size_t r = 0; r + 2 < dim; r++) {
            double d = theta[r * dim + s] - 2.0 * theta[(r + 1) * dim + s] + theta[(r + 2) * dim + s];
            double w = 1.0 - corner_relief * reference_edge_proximity(r + 1, dim) * reference_edge_proximity(s, dim);
            penalty += w * d * d;
        }
    }
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s + 2 < dim; s++) {
            double d = theta[r * dim + s] - 2.0 * theta[r * dim + (s + 1)] + theta[r * dim + (s + 2)];
            double w = 1.0 - corner_relief * reference_edge_proximity(r, dim) * reference_edge_proximity(s + 1, dim);
            penalty += w * d * d;
        }
    }
    return penalty;
}

static void test_weighted_penalty_matches_independent_reimplementation(void) {
    /* Independently reimplement the documented weight formula from
     * hdcd/bernstein.h in the test itself and check the library's
     * output against it directly -- more robust than reasoning about
     * relative magnitudes between different theta placements, since the
     * standard second-difference penalty already has a real, unrelated
     * boundary-vs-interior asymmetry (fewer overlapping difference
     * windows touch a boundary column/row than an interior one) that
     * would otherwise confound a magnitude comparison. */
    size_t m = 8, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    rng_seed(77);
    for (size_t i = 0; i < dim * dim; i++) theta[i] = rng_uniform() * 5.0 - 2.5;

    for (double corner_relief = 0.0; corner_relief < 0.95; corner_relief += 0.3) {
        double actual, expected = reference_weighted_penalty(theta, m, corner_relief);
        HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, corner_relief, &actual) == HDCD_OK);
        HDCD_CHECK_NEAR(actual, expected, 1e-9);
    }

    free(theta);
    HDCD_PASS("weighted roughness penalty matches an independent reimplementation of the documented weight formula");
}

static void test_corner_relief_relaxes_penalty_monotonically(void) {
    /* For a generic (nonzero-everywhere) theta, raising corner_relief
     * can only ever hold weights fixed or shrink them further towards
     * the corners -- never raise them -- so the total penalty must be
     * monotonically non-increasing in corner_relief. */
    size_t m = 7, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    rng_seed(2024);
    for (size_t i = 0; i < dim * dim; i++) theta[i] = rng_uniform() * 3.0 - 1.5;

    double prev = -1.0;
    for (double corner_relief = 0.0; corner_relief < 0.99; corner_relief += 0.2) {
        double penalty;
        HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, corner_relief, &penalty) == HDCD_OK);
        if (prev >= 0.0) {
            HDCD_CHECK(penalty <= prev + 1e-9);
        }
        prev = penalty;
    }
    /* And it should be a STRICT decrease overall for generic random theta. */
    double p_zero, p_high;
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, 0.0, &p_zero) == HDCD_OK);
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, 0.9, &p_high) == HDCD_OK);
    HDCD_CHECK(p_high < p_zero);

    free(theta);
    HDCD_PASS("raising corner_relief monotonically (and strictly, for generic theta) shrinks the total penalty");
}

static void test_corner_relief_has_no_effect_at_m_equals_2(void) {
    /* At m=2 (dim=3), EVERY column-difference residual's row-center is
     * forced to be exactly row index 1 (the only valid r+1, since r=0
     * is the only row with r+2<3) -- the grid's own vertical center --
     * and likewise every row-difference residual's column-center is
     * forced to column 1, the horizontal center. edge_proximity() is
     * exactly 0 at a true center index, so corner_weight() is exactly 1
     * for every single residual regardless of corner_relief: an exact,
     * theta-independent invariant, not an approximate one. */
    size_t m = 2, dim = m + 1;
    double *theta = (double *)malloc(dim * dim * sizeof(double));
    rng_seed(55);
    for (size_t i = 0; i < dim * dim; i++) theta[i] = rng_uniform() * 4.0 - 2.0;

    double unweighted;
    HDCD_CHECK(hdcd_bernstein_roughness_penalty(theta, m, &unweighted) == HDCD_OK);
    for (double corner_relief = 0.0; corner_relief < 0.99; corner_relief += 0.3) {
        double weighted;
        HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, m, corner_relief, &weighted) == HDCD_OK);
        HDCD_CHECK_NEAR(weighted, unweighted, 1e-12);
    }

    free(theta);
    HDCD_PASS("at m=2 every residual is centered on the grid's own center, so corner_relief has provably no effect");
}

static void test_corner_relief_invalid_arguments(void) {
    double theta[16] = {0};
    double scalar, grad[16];
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, 3, -0.1, &scalar) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(theta, 3, 1.0, &scalar) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_gradient_weighted(theta, 3, -0.1, grad) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_gradient_weighted(theta, 3, 1.0, grad) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_penalty_weighted(NULL, 3, 0.5, &scalar) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("corner_relief outside [0,1) is rejected");
}

static void test_invalid_arguments(void) {
    double out[4];
    HDCD_CHECK(hdcd_bernstein_basis(-0.1, 3, out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_basis(1.1, 3, out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_basis(0.5, 3, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_basis_centered(-0.1, 3, out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_basis_derivative(1.1, 3, out) == HDCD_ERROR_INVALID_ARGUMENT);

    double theta[16] = {0};
    double grad[16];
    double scalar;
    HDCD_CHECK(hdcd_bernstein_tensor_interaction(0.5, 0.5, 3, NULL, &scalar) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_tensor_gradient(0.5, 0.5, 3, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_penalty(NULL, 3, &scalar) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_bernstein_roughness_gradient(theta, 3, NULL) == HDCD_ERROR_INVALID_ARGUMENT);
    (void)grad;

    HDCD_PASS("Bernstein API rejects invalid arguments");
}

int main(void) {
    test_partition_of_unity();
    test_boundary_values();
    test_beta_pdf_identity();
    test_centering_integral_near_zero();
    test_derivative_matches_finite_difference();
    test_tensor_gradient_matches_finite_difference();
    test_roughness_gradient_matches_finite_difference();
    test_roughness_penalty_small_degrees_is_zero();
    test_roughness_penalty_zero_for_linear_theta();
    test_weighted_penalty_matches_unweighted_at_zero_relief();
    test_weighted_gradient_matches_finite_difference();
    test_weighted_penalty_matches_independent_reimplementation();
    test_corner_relief_relaxes_penalty_monotonically();
    test_corner_relief_has_no_effect_at_m_equals_2();
    test_corner_relief_invalid_arguments();
    test_invalid_arguments();
    printf("All bernstein tests passed.\n");
    return 0;
}
