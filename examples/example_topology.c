#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Milestone 4 example: three independent latent-factor clusters (2
 * observed variables each) should produce a persistent-topology
 * ordering that keeps each pair of siblings contiguous, in an order
 * driven by within-cluster tightness rather than column index.
 */

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
    const size_t n = 250, d = 6;
    /* Cluster tightness increases 0 -> 2 -> 4 (noise std shrinks). */
    const double noise_sd[3] = {0.9, 0.4, 0.1};

    double *x = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    for (size_t i = 0; i < n; i++) {
        for (size_t c = 0; c < 3; c++) {
            double latent = rng_normal();
            x[(2 * c) * n + i] = latent + noise_sd[c] * rng_normal();
            x[(2 * c + 1) * n + i] = latent + noise_sd[c] * rng_normal();
        }
        for (size_t j = 0; j < d; j++) mask[j * n + i] = 1;
    }

    double *u = (double *)malloc(n * d * sizeof(double));
    for (size_t j = 0; j < d; j++) {
        hdcd_marginal_t *marginal = NULL;
        hdcd_status_t status = hdcd_marginal_fit(&x[j * n], &mask[j * n], n, -1.0, -1.0, 1e-3, 60, &marginal);
        if (status != HDCD_OK) { fprintf(stderr, "fit failed: %s\n", hdcd_status_message(status)); return 1; }
        status = hdcd_transform_to_copula(marginal, &x[j * n], &mask[j * n], n, 0.0, &u[j * n]);
        hdcd_marginal_free(marginal);
        if (status != HDCD_OK) { fprintf(stderr, "transform failed: %s\n", hdcd_status_message(status)); return 1; }
    }

    hdcd_dependence_matrix_t *dm = NULL;
    hdcd_status_t status = hdcd_compute_dependence_matrix(u, mask, n, d, &dm);
    if (status != HDCD_OK) { fprintf(stderr, "dependence matrix failed: %s\n", hdcd_status_message(status)); return 1; }

    hdcd_topology_t *topo = NULL;
    status = hdcd_compute_topology(dm, &topo);
    if (status != HDCD_OK) { fprintf(stderr, "topology failed: %s\n", hdcd_status_message(status)); return 1; }

    printf("hdcd Milestone 4 example: persistent-topology ordering\n");
    printf("clusters: {0,1} loosest, {2,3} medium, {4,5} tightest\n\n");

    printf("MST edges (ascending weight):\n");
    for (size_t e = 0; e < hdcd_topology_mst_edge_count(topo); e++) {
        hdcd_mst_edge_t edge = hdcd_topology_mst_edge(topo, e);
        printf("  (%zu, %zu)  weight=%.4f\n", edge.j, edge.k, edge.weight);
    }

    const size_t *ordering = hdcd_topology_ordering(topo);
    printf("\nfinal ordering: [");
    for (size_t p = 0; p < d; p++) {
        printf("%zu%s", ordering[p], (p + 1 < d) ? ", " : "");
    }
    printf("]\n");
    printf(
        "\nNote: S(C) (spec section 6.3) sums affinity over ALL pairs in a\n"
        "component -- it is an extensive quantity, not a per-pair average.\n"
        "A larger supercluster formed by merging two looser clusters can\n"
        "outscore -- and so be visited before -- a smaller, individually\n"
        "tighter cluster. That is expected, not a bug: see DECISIONS.md.\n"
    );

    hdcd_topology_free(topo);
    hdcd_dependence_matrix_free(dm);
    free(x);
    free(u);
    free(mask);
    return 0;
}
