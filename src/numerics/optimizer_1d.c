#include "hdcd/numerics.h"

#include <math.h>

/* 1 / golden ratio */
#define HDCD_INV_GOLDEN 0.6180339887498949

hdcd_optimizer_1d_result_t hdcd_golden_section_maximize(
    hdcd_objective_1d_fn f,
    void *userdata,
    double lower,
    double upper,
    double tol,
    int max_iter
) {
    hdcd_optimizer_1d_result_t result;
    result.x_opt = NAN;
    result.f_opt = NAN;
    result.iterations = 0;
    result.converged = 0;
    result.status = HDCD_OK;

    if (f == NULL || !(lower < upper) || !(tol > 0.0) || max_iter <= 0) {
        result.status = HDCD_ERROR_INVALID_ARGUMENT;
        return result;
    }

    double a = lower;
    double b = upper;
    double span = b - a;

    /* Interior points, placed at the golden-ratio interior positions. */
    double c = b - HDCD_INV_GOLDEN * span;
    double d = a + HDCD_INV_GOLDEN * span;

    double fc = f(c, userdata);
    double fd = f(d, userdata);
    if (isnan(fc) || isnan(fd)) {
        result.status = HDCD_ERROR_NUMERICAL;
        return result;
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        if ((b - a) < tol) {
            break;
        }

        if (fc > fd) {
            /* Maximum is in [a, d]. */
            b = d;
            d = c;
            fd = fc;
            c = b - HDCD_INV_GOLDEN * (b - a);
            fc = f(c, userdata);
            if (isnan(fc)) {
                result.status = HDCD_ERROR_NUMERICAL;
                return result;
            }
        } else {
            /* Maximum is in [c, b]. */
            a = c;
            c = d;
            fc = fd;
            d = a + HDCD_INV_GOLDEN * (b - a);
            fd = f(d, userdata);
            if (isnan(fd)) {
                result.status = HDCD_ERROR_NUMERICAL;
                return result;
            }
        }
    }

    result.iterations = iter;
    result.converged = ((b - a) < tol) ? 1 : 0;

    double x_opt = 0.5 * (a + b);
    double f_opt = f(x_opt, userdata);
    if (isnan(f_opt)) {
        result.status = HDCD_ERROR_NUMERICAL;
        return result;
    }

    /* Report whichever of the bracket's known evaluations is best, in case
     * the midpoint is (numerically) worse than an already-evaluated point. */
    if (fc >= f_opt && fc >= fd) {
        x_opt = c;
        f_opt = fc;
    } else if (fd >= f_opt && fd >= fc) {
        x_opt = d;
        f_opt = fd;
    }

    result.x_opt = x_opt;
    result.f_opt = f_opt;

    if (!result.converged) {
        result.status = HDCD_ERROR_NOT_CONVERGED;
    }

    return result;
}
