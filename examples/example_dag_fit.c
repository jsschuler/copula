#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Milestone 7 example: fit a 3-node chain DAG 0 -> 1 -> 2 over an exact
 * Gaussian-copula chain (spec sections 28, 14, 16) and report each
 * node's diagnostics plus the factorized joint log density.
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
    const size_t n = 600, d = 3;
    const double rho = 0.75;

    double *u = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    for (size_t i = 0; i < n; i++) {
        double z0 = rng_normal();
        double z1 = rho * z0 + sqrt(1.0 - rho * rho) * rng_normal();
        double z2 = rho * z1 + sqrt(1.0 - rho * rho) * rng_normal();
        u[0 * n + i] = std_normal_cdf(z0);
        u[1 * n + i] = std_normal_cdf(z1);
        u[2 * n + i] = std_normal_cdf(z2);
        mask[0 * n + i] = mask[1 * n + i] = mask[2 * n + i] = 1;
    }

    hdcd_dag_t *dag = NULL;
    hdcd_dag_create(d, 2, &dag);
    hdcd_dag_add_edge(dag, 0, 1);
    hdcd_dag_add_edge(dag, 1, 2);

    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 3;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = 2026;

    hdcd_dag_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_dag_fit(u, mask, n, d, dag, &opt, &fit);
    if (fit == NULL) {
        fprintf(stderr, "dag fit failed hard: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("hdcd Milestone 7 example: fixed-DAG fitting, chain 0 -> 1 -> 2\n");
    printf("fit status: %s, all nodes converged: %s\n\n",
           hdcd_status_message(status), hdcd_dag_fit_all_converged(fit) ? "yes" : "no");

    printf("%6s %10s %10s %10s %14s %10s\n", "node", "n_parents", "n_train", "n_holdout", "holdout_score", "converged");
    for (size_t j = 0; j < d; j++) {
        const hdcd_local_fit_t *nf = hdcd_dag_fit_node(fit, j);
        printf("%6zu %10zu %10zu %10zu %14.4f %10s\n",
               j, hdcd_local_fit_n_parents(nf), hdcd_local_fit_n_train(nf), hdcd_local_fit_n_holdout(nf),
               hdcd_local_fit_holdout_score(nf), hdcd_dag_fit_node_converged(fit, j) ? "yes" : "no");
    }

    double u_point[3] = {0.5, 0.5, 0.5};
    double log_density;
    hdcd_dag_fit_joint_log_density(fit, u_point, d, &log_density);
    printf("\nlog c_G(0.5, 0.5, 0.5) = %.4f  (c_G = %.4f)\n", log_density, exp(log_density));

    hdcd_dag_fit_free(fit);
    hdcd_dag_free(dag);
    free(u);
    free(mask);
    return 0;
}
