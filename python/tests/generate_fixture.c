#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Generates python/tests/fixture.json: a small, fully deterministic
 * dataset plus C-computed reference outputs for every pipeline stage
 * (marginal fit/cdf/logpdf, copula transform, dependence matrix,
 * topology, DAG fit, joint log density). The Python binding test suite
 * (test_python_binding.py) loads this fixture, drives the SAME inputs
 * through ctypes, and checks its results against these values to tight
 * tolerance -- spec section 31 Milestone 10's "no numerical
 * disagreement with C fixture tests" acceptance criterion. Since both
 * sides call the identical compiled hdcd shared library, any
 * disagreement can only come from a ctypes marshaling bug (wrong
 * struct layout, wrong argtypes, wrong memory layout) -- exactly what
 * this is meant to catch.
 *
 * Run manually to regenerate: build/examples-style compile, then run
 * with cwd = python/tests/ so fixture.json lands next to the test.
 */

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

static double rng_normal(void) {
    const double two_pi = 6.283185307179586;
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}

int main(void) {
    const size_t n = 150, d = 3;
    const double rho = 0.7;

    /* Raw (non-copula-scale) data: exercises the full marginal-fit ->
     * transform -> dependence -> topology -> dag_fit pipeline, not just
     * the copula-scale machinery. */
    double *X = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    rng_seed(20260819);
    for (size_t i = 0; i < n; i++) {
        double z0 = 5.0 + 2.0 * rng_normal();
        double z1 = rho * (z0 - 5.0) / 2.0 + sqrt(1.0 - rho * rho) * rng_normal();
        double z2 = rho * z1 + sqrt(1.0 - rho * rho) * rng_normal();
        X[0 * n + i] = z0;
        X[1 * n + i] = z1;
        X[2 * n + i] = z2;
        mask[0 * n + i] = mask[1 * n + i] = mask[2 * n + i] = 1;
    }

    FILE *f = fopen("fixture.json", "w");
    if (f == NULL) {
        fprintf(stderr, "could not open fixture.json for writing\n");
        return 1;
    }

    fprintf(f, "{\n");

    fprintf(f, "  \"n\": %zu,\n  \"d\": %zu,\n", n, d);

    fprintf(f, "  \"X\": [\n");
    for (size_t i = 0; i < n; i++) {
        fprintf(f, "    [%.17g, %.17g, %.17g]%s\n",
                X[0 * n + i], X[1 * n + i], X[2 * n + i], (i + 1 < n) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Marginal fit + cdf/logpdf at fixed points. */
    hdcd_marginal_t *marginals[3];
    for (size_t j = 0; j < d; j++) {
        hdcd_status_t status = hdcd_marginal_fit(&X[j * n], &mask[j * n], n, -1.0, -1.0, 1e-6, 200, &marginals[j]);
        if (status != HDCD_OK) { fprintf(stderr, "marginal fit failed\n"); return 1; }
    }

    double eval_points[4] = {2.0, 5.0, 8.0, 11.0};
    fprintf(f, "  \"marginal_eval_points\": [2.0, 5.0, 8.0, 11.0],\n");
    fprintf(f, "  \"marginal_sigma\": [");
    for (size_t j = 0; j < d; j++) {
        hdcd_bandwidth_result_t bw = hdcd_marginal_bandwidth_result(marginals[j]);
        fprintf(f, "%.17g%s", bw.sigma, (j + 1 < d) ? ", " : "");
    }
    fprintf(f, "],\n");

    fprintf(f, "  \"marginal_cdf\": [\n");
    for (size_t j = 0; j < d; j++) {
        double cdf[4];
        hdcd_marginal_cdf(marginals[j], eval_points, 4, cdf);
        fprintf(f, "    [%.17g, %.17g, %.17g, %.17g]%s\n", cdf[0], cdf[1], cdf[2], cdf[3], (j + 1 < d) ? "," : "");
    }
    fprintf(f, "  ],\n");

    fprintf(f, "  \"marginal_logpdf\": [\n");
    for (size_t j = 0; j < d; j++) {
        double lp[4];
        hdcd_marginal_logpdf(marginals[j], eval_points, 4, lp);
        fprintf(f, "    [%.17g, %.17g, %.17g, %.17g]%s\n", lp[0], lp[1], lp[2], lp[3], (j + 1 < d) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Copula transform. */
    double *U = (double *)malloc(n * d * sizeof(double));
    for (size_t j = 0; j < d; j++) {
        hdcd_transform_to_copula(marginals[j], &X[j * n], &mask[j * n], n, 0.0, &U[j * n]);
    }
    fprintf(f, "  \"U_first_5_rows\": [\n");
    for (size_t i = 0; i < 5; i++) {
        fprintf(f, "    [%.17g, %.17g, %.17g]%s\n", U[0 * n + i], U[1 * n + i], U[2 * n + i], (i < 4) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Dependence matrix. */
    hdcd_dependence_matrix_t *dm = NULL;
    hdcd_compute_dependence_matrix(U, mask, n, d, &dm);
    fprintf(f, "  \"dependence_matrix\": [\n");
    for (size_t j = 0; j < d; j++) {
        fprintf(f, "    [");
        for (size_t k = 0; k < d; k++) {
            fprintf(f, "%.17g%s", hdcd_dependence_matrix_get(dm, j, k), (k + 1 < d) ? ", " : "");
        }
        fprintf(f, "]%s\n", (j + 1 < d) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Topology. */
    hdcd_topology_t *topo = NULL;
    hdcd_compute_topology(dm, &topo);
    const size_t *ordering = hdcd_topology_ordering(topo);
    fprintf(f, "  \"topology_ordering\": [%zu, %zu, %zu],\n", ordering[0], ordering[1], ordering[2]);
    fprintf(f, "  \"topology_mst_edges\": [\n");
    size_t n_edges = hdcd_topology_mst_edge_count(topo);
    for (size_t e = 0; e < n_edges; e++) {
        hdcd_mst_edge_t edge = hdcd_topology_mst_edge(topo, e);
        fprintf(f, "    [%zu, %zu, %.17g]%s\n", edge.j, edge.k, edge.weight, (e + 1 < n_edges) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Fixed DAG fit (manually specified, not annealed -- keeps this
     * fixture's C-side generation independent of the search's own
     * correctness, which prior milestones already test separately). */
    hdcd_dag_t *dag = NULL;
    hdcd_dag_create(d, 2, &dag);
    hdcd_dag_add_edge(dag, 0, 1);
    hdcd_dag_add_edge(dag, 1, 2);
    fprintf(f, "  \"dag_edges\": [[0, 1], [1, 2]],\n");

    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 3;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = 777;

    hdcd_dag_fit_t *dag_fit = NULL;
    hdcd_status_t fit_status = hdcd_dag_fit(U, mask, n, d, dag, &opt, &dag_fit);
    fprintf(f, "  \"dag_fit_status_ok\": %s,\n", (fit_status == HDCD_OK) ? "true" : "false");

    fprintf(f, "  \"dag_fit_holdout_scores\": [");
    for (size_t j = 0; j < d; j++) {
        const hdcd_local_fit_t *node = hdcd_dag_fit_node(dag_fit, j);
        fprintf(f, "%.17g%s", hdcd_local_fit_holdout_score(node), (j + 1 < d) ? ", " : "");
    }
    fprintf(f, "],\n");

    double point[3] = {0.3, 0.5, 0.7};
    double log_density;
    hdcd_dag_fit_joint_log_density(dag_fit, point, d, &log_density);
    fprintf(f, "  \"joint_log_density_point\": [0.3, 0.5, 0.7],\n");
    fprintf(f, "  \"joint_log_density_value\": %.17g,\n", log_density);

    fprintf(f, "  \"kl_estimate\": %.17g\n", hdcd_dag_fit_kl_estimate(dag_fit));

    fprintf(f, "}\n");
    fclose(f);

    hdcd_dag_fit_free(dag_fit);
    hdcd_dag_free(dag);
    hdcd_topology_free(topo);
    hdcd_dependence_matrix_free(dm);
    for (size_t j = 0; j < d; j++) hdcd_marginal_free(marginals[j]);
    free(X);
    free(U);
    free(mask);

    printf("wrote fixture.json\n");
    return 0;
}
