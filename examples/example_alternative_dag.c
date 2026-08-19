#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Milestone 9 example: compare a reference DAG against an alternative,
 * arbitrary candidate DAG (spec section 19) via held-out KL. The
 * candidate is built from a raw edge list via hdcd_dag_from_edges,
 * with a different topological order than the reference.
 *
 * IMPORTANT: this comparison is purely about statistical fit. It does
 * not identify causal structure -- see the disclaimer in the output.
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
    const double rho = 0.8;

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

    /* Reference DAG: the true chain 0 -> 1 -> 2. */
    hdcd_dag_t *reference_dag = NULL;
    hdcd_dag_create(d, 2, &reference_dag);
    hdcd_dag_add_edge(reference_dag, 0, 1);
    hdcd_dag_add_edge(reference_dag, 1, 2);

    /* Alternative candidate, built from a raw edge list (spec section
     * 19: "the public API must accept an arbitrary DAG G*"): the
     * empty graph, i.e. a hypothesis that these variables are
     * independent. Topological order is trivial here, but the point is
     * that hdcd_dag_from_edges accepts ANY valid edge set regardless
     * of its relationship to the reference ordering. */
    hdcd_dag_t *candidate_dag = NULL;
    hdcd_dag_from_edges(d, 2, NULL, NULL, 0, &candidate_dag);

    hdcd_local_fit_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.bernstein_degree = 3;
    opt.lambda_roughness = 0.15;
    opt.holdout_fraction = 0.25;
    opt.seed = 2026;

    hdcd_dag_fit_t *reference_fit = NULL;
    hdcd_dag_fit_t *candidate_fit = NULL;
    hdcd_dag_fit(u, mask, n, d, reference_dag, &opt, &reference_fit);
    hdcd_dag_fit(u, mask, n, d, candidate_dag, &opt, &candidate_fit);

    double delta_kl = hdcd_dag_fit_kl_difference(candidate_fit, reference_fit);

    printf("hdcd Milestone 9 example: alternative-DAG comparison\n\n");
    printf("reference: 0 -> 1 -> 2 (true chain)\n");
    printf("candidate: independence (empty graph, built via hdcd_dag_from_edges)\n\n");
    printf("Delta_KL(candidate || reference) = %.4f\n", delta_kl);
    printf("(positive means the candidate loses dependence information vs. the reference)\n\n");
    printf(
        "NOTE: this is a purely statistical, observational comparison of\n"
        "distributional fit (spec section 19). It does NOT establish that\n"
        "0 -> 1 -> 2 is a causal chain, and does NOT distinguish this DAG\n"
        "from other Markov-equivalent structures without further assumptions\n"
        "or interventions. The reference DAG is a density-estimation device,\n"
        "not a causal claim.\n"
    );

    hdcd_dag_fit_free(reference_fit);
    hdcd_dag_fit_free(candidate_fit);
    hdcd_dag_free(reference_dag);
    hdcd_dag_free(candidate_dag);
    free(u);
    free(mask);
    return 0;
}
