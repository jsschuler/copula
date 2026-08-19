#include "hdcd/numerics.h"
#include "test_utils.h"

static double neg_parabola(double x, void *userdata) {
    (void)userdata;
    double c = 2.0;
    return -(x - c) * (x - c);
}

static double asymmetric_concave(double x, void *userdata) {
    (void)userdata;
    /* f'(x) = 1/(x+1) - 3 = 0 => x = -2/3; maximized there on any
     * interval containing it. */
    return log(x + 1.0) - 3.0 * x;
}

static void test_recovers_known_maximum(void) {
    hdcd_optimizer_1d_result_t r = hdcd_golden_section_maximize(
        neg_parabola, NULL, -10.0, 10.0, 1e-8, 200
    );
    HDCD_CHECK(r.status == HDCD_OK);
    HDCD_CHECK(r.converged);
    HDCD_CHECK_NEAR(r.x_opt, 2.0, 1e-5);
    HDCD_CHECK_NEAR(r.f_opt, 0.0, 1e-8);
    HDCD_PASS("golden section recovers known parabola maximum");
}

static void test_asymmetric_objective(void) {
    hdcd_optimizer_1d_result_t r = hdcd_golden_section_maximize(
        asymmetric_concave, NULL, -0.9, 5.0, 1e-8, 200
    );
    HDCD_CHECK(r.status == HDCD_OK);
    HDCD_CHECK_NEAR(r.x_opt, -2.0 / 3.0, 1e-5);
    HDCD_PASS("golden section recovers known asymmetric maximum");
}

static void test_deterministic_reproducibility(void) {
    hdcd_optimizer_1d_result_t r1 = hdcd_golden_section_maximize(
        neg_parabola, NULL, -10.0, 10.0, 1e-8, 200
    );
    hdcd_optimizer_1d_result_t r2 = hdcd_golden_section_maximize(
        neg_parabola, NULL, -10.0, 10.0, 1e-8, 200
    );
    HDCD_CHECK(r1.x_opt == r2.x_opt);
    HDCD_CHECK(r1.f_opt == r2.f_opt);
    HDCD_CHECK(r1.iterations == r2.iterations);
    HDCD_PASS("golden section is exactly reproducible");
}

static void test_invalid_arguments(void) {
    hdcd_optimizer_1d_result_t r = hdcd_golden_section_maximize(
        neg_parabola, NULL, 5.0, 5.0, 1e-8, 200
    );
    HDCD_CHECK(r.status == HDCD_ERROR_INVALID_ARGUMENT);

    r = hdcd_golden_section_maximize(NULL, NULL, 0.0, 1.0, 1e-8, 200);
    HDCD_CHECK(r.status == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_PASS("golden section rejects invalid arguments");
}

int main(void) {
    test_recovers_known_maximum();
    test_asymmetric_objective();
    test_deterministic_reproducibility();
    test_invalid_arguments();
    printf("All optimizer_1d tests passed.\n");
    return 0;
}
