#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Milestone 3 example: transform two dimensions to the copula scale and
 * compute their pairwise distance-correlation dependence matrix (spec
 * section 5), including a nonlinear pair that a Pearson correlation
 * matrix would miss.
 */

/* Minimal deterministic PRNG for synthetic example data only. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

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

int main(void) {
    const size_t n = 200, d = 3;
    /* col0: normal; col1: independent normal; col2 = col0^2 (nonlinear). */
    double *x = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);

    for (size_t i = 0; i < n; i++) {
        double x0 = rng_normal();
        x[0 * n + i] = x0;
        x[1 * n + i] = rng_normal();
        x[2 * n + i] = x0 * x0;
        mask[0 * n + i] = mask[1 * n + i] = mask[2 * n + i] = 1;
    }

    double *u = (double *)malloc(n * d * sizeof(double));
    for (size_t j = 0; j < d; j++) {
        hdcd_marginal_t *marginal = NULL;
        hdcd_status_t status = hdcd_marginal_fit(
            &x[j * n], &mask[j * n], n, -1.0, -1.0, 1e-3, 60, &marginal
        );
        if (status != HDCD_OK) {
            fprintf(stderr, "marginal fit failed for column %zu: %s\n", j, hdcd_status_message(status));
            return 1;
        }
        status = hdcd_transform_to_copula(marginal, &x[j * n], &mask[j * n], n, 0.0, &u[j * n]);
        hdcd_marginal_free(marginal);
        if (status != HDCD_OK) {
            fprintf(stderr, "transform failed for column %zu: %s\n", j, hdcd_status_message(status));
            return 1;
        }
    }

    hdcd_dependence_matrix_t *dm = NULL;
    hdcd_status_t status = hdcd_compute_dependence_matrix(u, mask, n, d, &dm);
    if (status != HDCD_OK) {
        fprintf(stderr, "dependence matrix failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("hdcd Milestone 3 example: pairwise distance-correlation matrix\n");
    printf("columns: 0=normal, 1=independent normal, 2=col0^2 (nonlinear)\n\n");
    printf("%12s", "");
    for (size_t k = 0; k < d; k++) printf("%10s%zu", "col", k);
    printf("\n");
    for (size_t j = 0; j < d; j++) {
        printf("%10s%zu", "col", j);
        for (size_t k = 0; k < d; k++) {
            printf("%11.4f", hdcd_dependence_matrix_get(dm, j, k));
        }
        printf("\n");
    }

    hdcd_dependence_matrix_free(dm);
    free(x);
    free(u);
    free(mask);
    return 0;
}
