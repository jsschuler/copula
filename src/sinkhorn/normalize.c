#include "hdcd/sinkhorn.h"
#include "hdcd/numerics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HDCD_SINKHORN_DEFAULT_N_QUADRATURE 65
#define HDCD_SINKHORN_DEFAULT_TOL 1e-6
#define HDCD_SINKHORN_DEFAULT_MAX_ITER 500

struct hdcd_sinkhorn {
    hdcd_raw_kernel_fn kernel;
    void *kernel_userdata;

    size_t n_u;
    double *u_nodes;
    double *u_weights;

    size_t n_z;
    size_t z_dim;
    double *z_samples; /* n_z * z_dim, row-major */

    double *a; /* size n_u: a_j(u_nodes[i]) at the fitted state */
    double *b; /* size n_z: b_j(z_samples[m]) at the fitted state */

    int iterations;
    int converged;
    double conditional_integral_error;
    double marginal_preservation_error;
};

static int is_finite_positive(double x) {
    return isfinite(x) && x > 0.0;
}

hdcd_status_t hdcd_sinkhorn_fit(
    hdcd_raw_kernel_fn kernel, void *kernel_userdata,
    const double *z_samples, size_t n_z_samples, size_t z_dim,
    const hdcd_sinkhorn_options_t *options,
    hdcd_sinkhorn_t **out
) {
    if (kernel == NULL || z_samples == NULL || out == NULL || n_z_samples == 0 || z_dim == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    size_t n_u = (options != NULL && options->n_quadrature_nodes != 0)
                 ? options->n_quadrature_nodes : HDCD_SINKHORN_DEFAULT_N_QUADRATURE;
    double tol = (options != NULL && options->tol > 0.0) ? options->tol : HDCD_SINKHORN_DEFAULT_TOL;
    int max_iter = (options != NULL && options->max_iterations > 0)
                    ? options->max_iterations : HDCD_SINKHORN_DEFAULT_MAX_ITER;

    if (n_u < 3 || (n_u % 2) == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    hdcd_sinkhorn_t *sk = (hdcd_sinkhorn_t *)calloc(1, sizeof(hdcd_sinkhorn_t));
    if (sk == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    sk->kernel = kernel;
    sk->kernel_userdata = kernel_userdata;
    sk->n_u = n_u;
    sk->n_z = n_z_samples;
    sk->z_dim = z_dim;

    sk->u_nodes = (double *)malloc(n_u * sizeof(double));
    sk->u_weights = (double *)malloc(n_u * sizeof(double));
    sk->z_samples = (double *)malloc(n_z_samples * z_dim * sizeof(double));
    sk->a = (double *)malloc(n_u * sizeof(double));
    sk->b = (double *)malloc(n_z_samples * sizeof(double));
    double *kmat = (double *)malloc(n_u * n_z_samples * sizeof(double)); /* K[i][m], row-major */

    if (sk->u_nodes == NULL || sk->u_weights == NULL || sk->z_samples == NULL
        || sk->a == NULL || sk->b == NULL || kmat == NULL) {
        free(kmat);
        hdcd_sinkhorn_free(sk);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_status_t status = hdcd_simpson_nodes_weights(n_u, sk->u_nodes, sk->u_weights);
    if (status != HDCD_OK) {
        free(kmat);
        hdcd_sinkhorn_free(sk);
        return status;
    }

    memcpy(sk->z_samples, z_samples, n_z_samples * z_dim * sizeof(double));

    for (size_t i = 0; i < n_u; i++) {
        for (size_t m = 0; m < n_z_samples; m++) {
            double kval = kernel(sk->u_nodes[i], &sk->z_samples[m * z_dim], z_dim, kernel_userdata);
            if (!is_finite_positive(kval)) {
                free(kmat);
                hdcd_sinkhorn_free(sk);
                return HDCD_ERROR_NUMERICAL;
            }
            kmat[i * n_z_samples + m] = kval;
        }
    }

    for (size_t i = 0; i < n_u; i++) {
        sk->a[i] = 1.0;
    }

    int converged = 0;
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* b-update: b(z^m) <- [integral_0^1 a(t) K(t,z^m) dt]^{-1} */
        for (size_t m = 0; m < n_z_samples; m++) {
            double integral = 0.0;
            for (size_t i = 0; i < n_u; i++) {
                integral += sk->u_weights[i] * sk->a[i] * kmat[i * n_z_samples + m];
            }
            if (!is_finite_positive(integral)) {
                free(kmat);
                hdcd_sinkhorn_free(sk);
                return HDCD_ERROR_NUMERICAL;
            }
            sk->b[m] = 1.0 / integral;
        }

        /* a-update: a(u_i) <- [E_q[K(u_i,z) b(z)]]^{-1}, MC average over z_samples */
        for (size_t i = 0; i < n_u; i++) {
            double expectation = 0.0;
            for (size_t m = 0; m < n_z_samples; m++) {
                expectation += kmat[i * n_z_samples + m] * sk->b[m];
            }
            expectation /= (double)n_z_samples;
            if (!is_finite_positive(expectation)) {
                free(kmat);
                hdcd_sinkhorn_free(sk);
                return HDCD_ERROR_NUMERICAL;
            }
            sk->a[i] = 1.0 / expectation;
        }

        /* Convergence metric (spec section 11.2): both constraints
         * evaluated jointly at the current (a, b), not the half-step
         * residual either update trivially zeroes out on its own. */
        double cond_err = 0.0;
        for (size_t m = 0; m < n_z_samples; m++) {
            double integral = 0.0;
            for (size_t i = 0; i < n_u; i++) {
                integral += sk->u_weights[i] * sk->a[i] * kmat[i * n_z_samples + m];
            }
            integral *= sk->b[m];
            double err = fabs(integral - 1.0);
            if (err > cond_err) cond_err = err;
        }

        double marg_err = 0.0;
        for (size_t i = 0; i < n_u; i++) {
            double expectation = 0.0;
            for (size_t m = 0; m < n_z_samples; m++) {
                expectation += kmat[i * n_z_samples + m] * sk->b[m];
            }
            expectation = (expectation / (double)n_z_samples) * sk->a[i];
            double err = fabs(expectation - 1.0);
            if (err > marg_err) marg_err = err;
        }

        sk->conditional_integral_error = cond_err;
        sk->marginal_preservation_error = marg_err;

        if (cond_err < tol && marg_err < tol) {
            converged = 1;
            break;
        }
    }

    free(kmat);

    sk->iterations = converged ? (iter + 1) : max_iter;
    sk->converged = converged;

    *out = sk;
    return converged ? HDCD_OK : HDCD_ERROR_NOT_CONVERGED;
}

void hdcd_sinkhorn_free(hdcd_sinkhorn_t *sk) {
    if (sk == NULL) {
        return;
    }
    free(sk->u_nodes);
    free(sk->u_weights);
    free(sk->z_samples);
    free(sk->a);
    free(sk->b);
    free(sk);
}

hdcd_status_t hdcd_sinkhorn_conditional_density(
    const hdcd_sinkhorn_t *sk,
    double u, const double *z, size_t z_dim,
    double *out
) {
    if (sk == NULL || z == NULL || out == NULL || z_dim != sk->z_dim) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double k_uz = sk->kernel(u, z, z_dim, sk->kernel_userdata);
    if (!is_finite_positive(k_uz)) {
        return HDCD_ERROR_NUMERICAL;
    }

    /* a(u) via its closed form against the fitted b vector. */
    double expectation = 0.0;
    for (size_t m = 0; m < sk->n_z; m++) {
        double kval = sk->kernel(u, &sk->z_samples[m * sk->z_dim], sk->z_dim, sk->kernel_userdata);
        if (!is_finite_positive(kval)) {
            return HDCD_ERROR_NUMERICAL;
        }
        expectation += kval * sk->b[m];
    }
    expectation /= (double)sk->n_z;
    if (!is_finite_positive(expectation)) {
        return HDCD_ERROR_NUMERICAL;
    }
    double a_u = 1.0 / expectation;

    /* b(z) via its closed form against the fitted a vector. */
    double integral = 0.0;
    for (size_t i = 0; i < sk->n_u; i++) {
        double kval = sk->kernel(sk->u_nodes[i], z, z_dim, sk->kernel_userdata);
        if (!is_finite_positive(kval)) {
            return HDCD_ERROR_NUMERICAL;
        }
        integral += sk->u_weights[i] * sk->a[i] * kval;
    }
    if (!is_finite_positive(integral)) {
        return HDCD_ERROR_NUMERICAL;
    }
    double b_z = 1.0 / integral;

    *out = a_u * k_uz * b_z;
    return HDCD_OK;
}

int hdcd_sinkhorn_converged(const hdcd_sinkhorn_t *sk) {
    return (sk != NULL) ? sk->converged : 0;
}

int hdcd_sinkhorn_iterations(const hdcd_sinkhorn_t *sk) {
    return (sk != NULL) ? sk->iterations : 0;
}

double hdcd_sinkhorn_conditional_integral_error(const hdcd_sinkhorn_t *sk) {
    return (sk != NULL) ? sk->conditional_integral_error : NAN;
}

double hdcd_sinkhorn_marginal_preservation_error(const hdcd_sinkhorn_t *sk) {
    return (sk != NULL) ? sk->marginal_preservation_error : NAN;
}
