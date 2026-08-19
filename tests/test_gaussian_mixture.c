#include "hdcd/marginal.h"
#include "test_utils.h"

static double sample_data[6] = {-1.5, -0.5, 0.0, 0.3, 1.2, 2.0};

static void test_pdf_integrates_to_one(void) {
    double sigma = 0.4;
    /* Range wide enough to capture all mass given sigma and data spread. */
    double lo = -6.0, hi = 8.0;
    int steps = 200000;
    double h = (hi - lo) / (double)steps;

    double integral = 0.0;
    for (int i = 0; i <= steps; i++) {
        double x = lo + h * (double)i;
        double density;
        HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, &x, 1, &density) == HDCD_OK);
        double weight = (i == 0 || i == steps) ? 0.5 : 1.0;
        integral += weight * density;
    }
    integral *= h;

    HDCD_CHECK_NEAR(integral, 1.0, 1e-4);
    HDCD_PASS("Gaussian mixture pdf integrates to 1 (trapezoid)");
}

static void test_cdf_derivative_matches_pdf(void) {
    double sigma = 0.4;
    double eps = 1e-5;
    double points[5] = {-2.0, -0.5, 0.1, 1.0, 3.0};

    for (int i = 0; i < 5; i++) {
        double x = points[i];
        double x_lo = x - eps, x_hi = x + eps;
        double f_lo, f_hi, density;
        HDCD_CHECK(hdcd_gaussian_mixture_cdf(sample_data, 6, sigma, &x_lo, 1, &f_lo) == HDCD_OK);
        HDCD_CHECK(hdcd_gaussian_mixture_cdf(sample_data, 6, sigma, &x_hi, 1, &f_hi) == HDCD_OK);
        HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, &x, 1, &density) == HDCD_OK);

        double finite_diff = (f_hi - f_lo) / (2.0 * eps);
        HDCD_CHECK_NEAR(finite_diff, density, 1e-5);
    }
    HDCD_PASS("Gaussian mixture CDF derivative matches pdf");
}

static void test_logpdf_matches_log_pdf(void) {
    double sigma = 0.4;
    double points[4] = {-3.0, 0.0, 0.5, 5.0};
    double log_density[4], density[4];
    HDCD_CHECK(hdcd_gaussian_mixture_logpdf(sample_data, 6, sigma, points, 4, log_density) == HDCD_OK);
    HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, points, 4, density) == HDCD_OK);

    for (int i = 0; i < 4; i++) {
        HDCD_CHECK_NEAR(log_density[i], log(density[i]), 1e-8);
    }
    HDCD_PASS("logpdf matches log(pdf)");
}

static void test_dpdf_matches_finite_difference(void) {
    double sigma = 0.4;
    double eps = 1e-6;
    double points[4] = {-2.0, -0.5, 0.3, 2.5};

    for (int i = 0; i < 4; i++) {
        double x = points[i];
        double x_lo = x - eps, x_hi = x + eps;
        double f_lo, f_hi, deriv;
        HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, &x_lo, 1, &f_lo) == HDCD_OK);
        HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, &x_hi, 1, &f_hi) == HDCD_OK);
        HDCD_CHECK(hdcd_gaussian_mixture_dpdf(sample_data, 6, sigma, &x, 1, &deriv) == HDCD_OK);

        double finite_diff = (f_hi - f_lo) / (2.0 * eps);
        HDCD_CHECK_NEAR(finite_diff, deriv, 1e-4);
    }
    HDCD_PASS("density derivative matches finite difference");
}

static void test_cdf_bounds(void) {
    double sigma = 0.4;
    double lo = -100.0, hi = 100.0;
    double f_lo, f_hi;
    HDCD_CHECK(hdcd_gaussian_mixture_cdf(sample_data, 6, sigma, &lo, 1, &f_lo) == HDCD_OK);
    HDCD_CHECK(hdcd_gaussian_mixture_cdf(sample_data, 6, sigma, &hi, 1, &f_hi) == HDCD_OK);
    HDCD_CHECK_NEAR(f_lo, 0.0, 1e-9);
    HDCD_CHECK_NEAR(f_hi, 1.0, 1e-9);
    HDCD_PASS("CDF approaches 0 and 1 in the tails");
}

static void test_pdf_nonnegative(void) {
    double sigma = 0.4;
    double points[7] = {-10, -3, -1, 0, 1, 3, 10};
    double density[7];
    HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, sigma, points, 7, density) == HDCD_OK);
    for (int i = 0; i < 7; i++) {
        HDCD_CHECK(density[i] >= 0.0);
    }
    HDCD_PASS("pdf is non-negative everywhere tested");
}

static void test_invalid_arguments(void) {
    double x = 0.0, out;
    HDCD_CHECK(hdcd_gaussian_mixture_pdf(NULL, 6, 0.4, &x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 0, 0.4, &x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_gaussian_mixture_pdf(sample_data, 6, -1.0, &x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_gaussian_mixture_cdf(sample_data, 6, 0.0, &x, 1, &out) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_PASS("Gaussian mixture functions reject invalid arguments");
}

int main(void) {
    test_pdf_integrates_to_one();
    test_cdf_derivative_matches_pdf();
    test_logpdf_matches_log_pdf();
    test_dpdf_matches_finite_difference();
    test_cdf_bounds();
    test_pdf_nonnegative();
    test_invalid_arguments();
    printf("All gaussian_mixture tests passed.\n");
    return 0;
}
