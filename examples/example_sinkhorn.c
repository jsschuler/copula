#include "hdcd/hdcd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Milestone 6 example: normalize a Bernstein tensor kernel (Milestone 5)
 * with Sinkhorn scaling (spec section 11) so the resulting conditional
 * density c_j(u|z) actually integrates to 1 over u for every z, and its
 * implied marginal of U is genuinely Uniform(0,1) -- properties the raw
 * (un-normalized) kernel does not have on its own.
 */

static uint64_t rng_state = 42;

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

typedef struct { size_t m; const double *theta; } bernstein_params_t;

static double raw_kernel(double u, const double *z, size_t z_dim, void *userdata) {
    (void)z_dim;
    const bernstein_params_t *p = (const bernstein_params_t *)userdata;
    double g;
    hdcd_bernstein_tensor_interaction(u, z[0], p->m, p->theta, &g);
    return exp(g);
}

int main(void) {
    size_t m = 3, dim = m + 1;
    double theta[16];
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            /* A hand-picked non-separable surface: rewards u,z agreeing. */
            theta[r * dim + s] = 1.2 * (double)((r == s) ? 1 : 0) - 0.3;
        }
    }
    bernstein_params_t params = {m, theta};

    size_t n_z = 500;
    double *z_samples = (double *)malloc(n_z * sizeof(double));
    for (size_t i = 0; i < n_z; i++) z_samples[i] = rng_uniform();

    hdcd_sinkhorn_t *sk = NULL;
    hdcd_status_t status = hdcd_sinkhorn_fit(raw_kernel, &params, z_samples, n_z, 1, NULL, &sk);
    if (status != HDCD_OK) {
        fprintf(stderr, "Sinkhorn fit failed: %s\n", hdcd_status_message(status));
        return 1;
    }

    printf("hdcd Milestone 6 example: Sinkhorn-normalized Bernstein kernel\n\n");
    printf("Sinkhorn iterations = %d, converged = %s\n", hdcd_sinkhorn_iterations(sk), hdcd_sinkhorn_converged(sk) ? "yes" : "no");
    printf("conditional-integral error  = %.2e\n", hdcd_sinkhorn_conditional_integral_error(sk));
    printf("marginal-preservation error = %.2e\n\n", hdcd_sinkhorn_marginal_preservation_error(sk));

    size_t n = 65;
    double *nodes = (double *)malloc(n * sizeof(double));
    double *weights = (double *)malloc(n * sizeof(double));
    hdcd_simpson_nodes_weights(n, nodes, weights);

    printf("integral over u of c(u|z), raw kernel vs. Sinkhorn-normalized, for a few z:\n");
    printf("%8s %16s %16s\n", "z", "raw kernel", "normalized");
    double z_points[3] = {0.1, 0.5, 0.9};
    for (int j = 0; j < 3; j++) {
        double raw_integral = 0.0, norm_integral = 0.0;
        for (size_t i = 0; i < n; i++) {
            raw_integral += weights[i] * raw_kernel(nodes[i], &z_points[j], 1, &params);
            double c;
            hdcd_sinkhorn_conditional_density(sk, nodes[i], &z_points[j], 1, &c);
            norm_integral += weights[i] * c;
        }
        printf("%8.2f %16.6f %16.6f\n", z_points[j], raw_integral, norm_integral);
    }
    free(nodes);
    free(weights);

    hdcd_sinkhorn_free(sk);
    free(z_samples);
    return 0;
}
