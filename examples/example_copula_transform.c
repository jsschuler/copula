#include "hdcd/hdcd.h"

#include <stdio.h>

/*
 * Milestone 2 example: fit a marginal with missing data and transform
 * observed values to the copula scale U_j = F_hat_j(X_j) (spec section
 * 4). Missing entries propagate through the mask, not through NaN.
 */

static double sample_data[16] = {
    -2.10, -1.55, -1.20, -0.88, -0.60, -0.31, -0.10, 0.05,
     0.22,  0.41,  0.63,  0.90,  1.18,  1.52,  1.95,  2.40
};

int main(void) {
    /* Every fourth entry is missing. */
    uint8_t observed_mask[16];
    for (int i = 0; i < 16; i++) {
        observed_mask[i] = (i % 4 != 3) ? 1 : 0;
    }

    hdcd_marginal_t *marginal = NULL;
    hdcd_status_t status = hdcd_marginal_fit(
        sample_data, observed_mask, 16,
        -1.0, -1.0, 1e-6, 200,
        &marginal
    );
    if (status != HDCD_OK) {
        fprintf(stderr, "marginal fit failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("hdcd Milestone 2 example: copula transform with missing data\n");
    printf("n_observed = %zu / 16\n\n", hdcd_marginal_n_observed(marginal));

    double u[16];
    status = hdcd_transform_to_copula(marginal, sample_data, observed_mask, 16, 0.0, u);
    if (status != HDCD_OK) {
        fprintf(stderr, "transform failed: %s\n", hdcd_status_message(status));
        hdcd_marginal_free(marginal);
        return 1;
    }

    printf("%6s %10s %8s %10s\n", "i", "x", "observed", "u");
    for (int i = 0; i < 16; i++) {
        if (observed_mask[i]) {
            printf("%6d %10.4f %8s %10.6f\n", i, sample_data[i], "yes", u[i]);
        } else {
            printf("%6d %10.4f %8s %10s\n", i, sample_data[i], "no", "NaN");
        }
    }

    hdcd_marginal_free(marginal);
    return 0;
}
