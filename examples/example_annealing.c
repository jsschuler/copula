#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Milestone 8 example: run simulated annealing (spec section 17) over
 * a 4-node synthetic dataset with a known sparse dependency structure
 * (0 -> 1, 0 -> 2, {1,2} -> 3) and confirm the search recovers it,
 * starting from the empty graph.
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

static double std_normal_cdf(double x) {
    return 0.5 * erfc(-x * 0.7071067811865476);
}

int main(void) {
    const size_t n = 600, d = 4;
    const double rho = 0.7;

    /* True structure: 0 -> 1, 0 -> 2, {1,2} -> 3 (a small diamond). */
    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    for (size_t i = 0; i < n; i++) {
        double z0 = rng_normal();
        double z1 = rho * z0 + sqrt(1.0 - rho * rho) * rng_normal();
        double z2 = rho * z0 + sqrt(1.0 - rho * rho) * rng_normal();
        double z3 = rho * 0.7 * (z1 + z2) + sqrt(1.0 - rho * rho) * rng_normal();
        u[0 * n + i] = std_normal_cdf(z0);
        u[1 * n + i] = std_normal_cdf(z1);
        u[2 * n + i] = std_normal_cdf(z2);
        u[3 * n + i] = std_normal_cdf(z3);
        for (size_t j = 0; j < d; j++) mask[j * n + i] = 1;
    }

    size_t ordering[4] = {0, 1, 2, 3}; /* consistent with the true structure */

    hdcd_annealing_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.k_max = 2;
    opt.lambda_edge = 0.05;
    opt.ordering = ordering;
    opt.local_fit_options.bernstein_degree = 3;
    opt.local_fit_options.lambda_roughness = 0.15;
    opt.local_fit_options.holdout_fraction = 0.25;
    opt.local_fit_options.seed = 7;
    opt.initial_temperature = 0.5;
    opt.cooling_rate = 0.96;
    opt.max_iterations = 150;
    opt.restarts = 3;
    opt.p_add = 1.0;
    opt.p_remove = 1.0;
    opt.p_swap = 1.0;
    opt.seed = 2026;

    hdcd_annealing_result_t *result = NULL;
    hdcd_status_t status = hdcd_run_annealing(u, mask, n, d, &opt, &result);
    if (result == NULL) {
        fprintf(stderr, "annealing failed hard: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("hdcd Milestone 8 example: simulated annealing DAG search\n");
    printf("true structure: 0->1, 0->2, 1->3, 2->3\n\n");
    printf("iterations = %zu, acceptance rate = %.2f\n", hdcd_annealing_n_iterations(result), hdcd_annealing_acceptance_rate(result));
    printf("J(empty graph) = 0.0000  (spec section 12 base case)\n");
    printf("J(best graph)  = %.4f\n\n", hdcd_annealing_best_score(result));

    const hdcd_dag_t *best = hdcd_annealing_best_dag(result);
    printf("recovered edges:\n");
    for (size_t child = 0; child < d; child++) {
        size_t np = hdcd_dag_n_parents(best, child);
        if (np == 0) continue;
        size_t parents[2];
        hdcd_dag_parents(best, child, parents);
        printf("  node %zu <- {", child);
        for (size_t k = 0; k < np; k++) printf("%zu%s", parents[k], (k + 1 < np) ? ", " : "");
        printf("}\n");
    }

    hdcd_annealing_result_free(result);
    free(u);
    free(mask);
    return 0;
}
