#include "hdcd/hdcd.h"

#include <stdio.h>

/*
 * Milestone 1 example: fit a Gaussian-smoothed marginal to a small
 * synthetic sample via leave-one-out bandwidth selection, then report
 * the fitted density/CDF at a few evaluation points.
 */

static double sample_data[16] = {
    -2.10, -1.55, -1.20, -0.88, -0.60, -0.31, -0.10, 0.05,
     0.22,  0.41,  0.63,  0.90,  1.18,  1.52,  1.95,  2.40
};

int main(void) {
    printf("hdcd Milestone 1 example: marginal Gaussian-mixture fit\n");
    printf("n = %zu observations\n\n", sizeof(sample_data) / sizeof(sample_data[0]));

    hdcd_bandwidth_result_t bw;
    hdcd_status_t status = hdcd_select_bandwidth_loo(
        sample_data, 16,
        -1.0, -1.0,   /* derive default bounds from robust scale */
        1e-6, 200,
        &bw
    );

    if (status != HDCD_OK) {
        fprintf(stderr, "bandwidth selection failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("selected sigma      = %.6f\n", bw.sigma);
    printf("search bounds        = [%.6f, %.6f]\n", bw.lower, bw.upper);
    printf("LOO log-likelihood  = %.6f\n", bw.loglik);
    printf("optimizer iterations = %d\n", bw.iterations);
    printf("converged            = %s\n\n", bw.converged ? "yes" : "no");

    double eval_points[5] = {-2.5, -0.5, 0.0, 0.5, 2.5};
    double density[5];
    double cdf[5];

    status = hdcd_gaussian_mixture_pdf(sample_data, 16, bw.sigma, eval_points, 5, density);
    if (status != HDCD_OK) {
        fprintf(stderr, "pdf evaluation failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    status = hdcd_gaussian_mixture_cdf(sample_data, 16, bw.sigma, eval_points, 5, cdf);
    if (status != HDCD_OK) {
        fprintf(stderr, "cdf evaluation failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("%10s %14s %14s\n", "x", "f_hat(x)", "F_hat(x)");
    for (int i = 0; i < 5; i++) {
        printf("%10.4f %14.6f %14.6f\n", eval_points[i], density[i], cdf[i]);
    }

    return 0;
}
