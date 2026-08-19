#ifndef HDCD_SINKHORN_H
#define HDCD_SINKHORN_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A positive raw conditional-kernel callback K_j(u,z) > 0 (spec section
 * 8): u is the child's copula coordinate, z is the (possibly
 * multi-dimensional, length z_dim) parent-vector copula coordinate.
 * Must already be numerically stable (e.g. computed by the caller as
 * exp() of a bounded log-kernel, such as the Bernstein tensor
 * interaction from Milestone 5) -- this module consumes already-
 * evaluated positive kernel values, it does not stabilize the kernel
 * itself, and it knows nothing about DAGs, edges, or how the kernel was
 * built.
 */
typedef double (*hdcd_raw_kernel_fn)(double u, const double *z, size_t z_dim, void *userdata);

typedef struct hdcd_sinkhorn_options {
    size_t n_quadrature_nodes; /* odd, >= 3; 0 selects the default (65) */
    double tol;                 /* convergence tolerance; 0 selects the default (1e-6) */
    int max_iterations;          /* 0 selects the default (500) */
} hdcd_sinkhorn_options_t;

/* Opaque fitted Sinkhorn normalization state for one conditional factor. */
typedef struct hdcd_sinkhorn hdcd_sinkhorn_t;

/*
 * Fit the copula-preserving Sinkhorn normalization (spec section 11):
 * find scaling functions a_j, b_j such that
 *   c_j(u|z) = a_j(u) K_j(u,z) b_j(z)
 * is a valid conditional density (integral_0^1 c_j(u|z) du = 1 for
 * every z) whose implied unconditional marginal of U is Uniform(0,1)
 * (integral c_j(u|z) q_j(z) dz = 1 for every u).
 *
 * The u-integral uses deterministic Simpson quadrature (spec section
 * 11.2). The expectation against q_j(z) uses a plain Monte Carlo
 * average over `z_samples` -- caller-supplied draws from q_j (spec
 * section 11.2: "the expectation over q_j(z) may use Monte Carlo
 * samples from the fitted parent distribution; permit cached parent
 * samples"). This module does not draw its own samples and has no
 * knowledge of how the parent distribution q_j was constructed; that is
 * the caller's responsibility (in later milestones, samples drawn from
 * the partially-built joint model). `z_samples` is n_z_samples x z_dim,
 * row-major, and is copied internally.
 *
 * On non-convergence within max_iterations, returns
 * HDCD_ERROR_NOT_CONVERGED (spec section 24: fail clearly rather than
 * silently continuing) but *out is still populated with the best-effort
 * fitted state and diagnostics, which the caller may free normally.
 */
hdcd_status_t hdcd_sinkhorn_fit(
    hdcd_raw_kernel_fn kernel, void *kernel_userdata,
    const double *z_samples, size_t n_z_samples, size_t z_dim,
    const hdcd_sinkhorn_options_t *options,
    hdcd_sinkhorn_t **out
);

void hdcd_sinkhorn_free(hdcd_sinkhorn_t *sk);

/*
 * Evaluate the fitted conditional density c_j(u|z) = a_j(u) K_j(u,z)
 * b_j(z) at an arbitrary (u, z) pair. a_j(u) and b_j(z) are recomputed
 * from their closed-form update rules (spec section 11.1) against the
 * fitted state (the final a-at-quadrature-nodes and b-at-z-samples
 * vectors) -- this is exact given that fitted state, not an
 * interpolation of it.
 */
hdcd_status_t hdcd_sinkhorn_conditional_density(
    const hdcd_sinkhorn_t *sk,
    double u, const double *z, size_t z_dim,
    double *out
);

int hdcd_sinkhorn_converged(const hdcd_sinkhorn_t *sk);
int hdcd_sinkhorn_iterations(const hdcd_sinkhorn_t *sk);

/* sup_z |integral_0^1 c_j(u|z) du - 1|, discretized as a max over the
 * fitted z_samples (spec section 11.2's default convergence metric). */
double hdcd_sinkhorn_conditional_integral_error(const hdcd_sinkhorn_t *sk);

/* sup_u |integral c_j(u|z) q_j(z) dz - 1|, discretized as a max over
 * the u-quadrature nodes (spec section 11.2's default convergence metric). */
double hdcd_sinkhorn_marginal_preservation_error(const hdcd_sinkhorn_t *sk);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_SINKHORN_H */
