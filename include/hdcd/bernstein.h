#ifndef HDCD_BERNSTEIN_H
#define HDCD_BERNSTEIN_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Raw Bernstein basis (spec section 9):
 *   B_{r,m}(u) = C(m,r) u^r (1-u)^{m-r},  r = 0..m.
 * Requires 0 <= u <= 1 (the copula-scale domain this basis is used on).
 * `out` must have length m+1. The binomial coefficient is built via the
 * multiplicative recurrence C(m,r+1) = C(m,r)*(m-r)/(r+1), never via
 * direct factorials (spec section 24).
 */
hdcd_status_t hdcd_bernstein_basis(double u, size_t m, double *out);

/*
 * Centered basis (spec section 9): B~_{r,m}(u) = B_{r,m}(u) - 1/(m+1),
 * since integral_0^1 B_{r,m}(u) du = 1/(m+1). `out` must have length m+1.
 */
hdcd_status_t hdcd_bernstein_basis_centered(double u, size_t m, double *out);

/*
 * d/du B_{r,m}(u) = m * (B_{r-1,m-1}(u) - B_{r,m-1}(u)), r = 0..m, with
 * the boundary convention B_{-1,m-1} = B_{m,m-1} = 0. `out` must have
 * length m+1. (Note: this is the derivative of the RAW basis; the
 * centered basis differs from it by an additive constant, so it has the
 * same derivative.)
 */
hdcd_status_t hdcd_bernstein_basis_derivative(double u, size_t m, double *out);

/*
 * Tensor interaction (spec section 9), the raw single-parent
 * conditional-kernel edge term:
 *   g(u,z) = B~(u)^T Theta B~(z) = sum_{r,s} B~_r(u) Theta[r][s] B~_s(z).
 * `theta` is (m+1) x (m+1), row-major.
 */
hdcd_status_t hdcd_bernstein_tensor_interaction(
    double u, double z, size_t m,
    const double *theta,
    double *out
);

/*
 * Gradient of g(u,z) with respect to theta (spec section 20, "raw
 * log-kernel gradients with respect to coefficients"):
 *   dg/dtheta[r][s] = B~_r(u) * B~_s(z).
 * `grad_out` is (m+1) x (m+1), row-major, fully overwritten.
 */
hdcd_status_t hdcd_bernstein_tensor_gradient(
    double u, double z, size_t m,
    double *grad_out
);

/*
 * Second-difference roughness penalty (spec section 10):
 *   R(theta) = ||Delta_u^2 theta||_F^2 + ||theta (Delta_z^2)^T||_F^2,
 * where Delta^2 is the central second-difference operator. `theta` is
 * (m+1) x (m+1), row-major. R = 0 when m < 2 (fewer than 3 rows/columns
 * means no interior second-difference triple exists).
 *
 * Equivalent to hdcd_bernstein_roughness_penalty_weighted(theta, m, 0.0, out).
 */
hdcd_status_t hdcd_bernstein_roughness_penalty(
    const double *theta, size_t m,
    double *out
);

/*
 * Gradient of the roughness penalty with respect to theta (spec section
 * 10: "The C core must expose both the raw penalty and its gradient").
 * `grad_out` is (m+1) x (m+1), row-major, fully overwritten.
 *
 * Equivalent to hdcd_bernstein_roughness_gradient_weighted(theta, m, 0.0, grad_out).
 */
hdcd_status_t hdcd_bernstein_roughness_gradient(
    const double *theta, size_t m,
    double *grad_out
);

/*
 * Corner-relaxed (anisotropic) variant of the roughness penalty (see
 * DECISIONS.md's "anisotropic (corner-relaxed) roughness penalty"
 * entry). Each interior second-difference residual, centered at grid
 * position (i,j) in the (m+1)x(m+1) Theta grid, is weighted by
 *
 *   w(i,j) = 1 - corner_relief * edge_proximity(i) * edge_proximity(j),
 *   edge_proximity(k) = 1 - min(k, dim-1-k) / ((dim-1)/2)  in [0,1],
 *
 * where dim = m+1. edge_proximity(k) is 1 exactly at either edge
 * (k=0 or k=dim-1) and falls linearly to 0 at the grid's center, so the
 * product w(i,j) is close to 1 (full, unchanged penalty) everywhere
 * except near the tensor's four CORNERS -- where u and z are BOTH
 * extreme simultaneously, exactly where tail dependence concentrates --
 * and dips toward (1 - corner_relief) right at a corner. This lets a
 * fit sharpen locally near a genuine tail-dependence corner without
 * raising `bernstein_degree` (and its coefficient count) everywhere.
 *
 * Requires 0 <= corner_relief < 1 (kept strictly below 1 so every
 * weight stays positive and the fitting objective in
 * src/optimize/local_fit.c remains well-posed even exactly at a
 * corner). corner_relief = 0 recovers the unweighted penalty exactly.
 */
hdcd_status_t hdcd_bernstein_roughness_penalty_weighted(
    const double *theta, size_t m, double corner_relief,
    double *out
);

hdcd_status_t hdcd_bernstein_roughness_gradient_weighted(
    const double *theta, size_t m, double corner_relief,
    double *grad_out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_BERNSTEIN_H */
