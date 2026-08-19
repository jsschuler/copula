#ifndef HDCD_COPULA_H
#define HDCD_COPULA_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"
#include "hdcd/marginal.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Default numerical clipping epsilon for copula coordinates (spec
 * section 4). Clipping keeps U strictly inside (0,1) so all polynomial
 * moments -- and later, the centered Bernstein basis -- remain
 * well-defined. This value must stay identical across the C core and
 * every language wrapper (spec section 4: "The clipping convention must
 * be identical across C, Python, R, and Julia.").
 */
#define HDCD_DEFAULT_COPULA_EPSILON 1e-9

/*
 * Transform observed values x[0..n-1] of ONE marginal dimension to
 * copula-scale coordinates (spec section 4):
 *
 *   u_i = clip(F_hat(x_i), epsilon),  clip(u) = min(1-epsilon, max(epsilon, u))
 *
 * `observed_mask` has the same shape as `x` (spec section 23). Where
 * observed_mask[i] == 0, x[i] is missing: u_out[i] is set to NaN as an
 * explicit sentinel and is not computed from `marginal`. The mask itself
 * is not modified -- missingness propagates unchanged, it is not
 * inferred or repaired.
 *
 * epsilon <= 0 selects HDCD_DEFAULT_COPULA_EPSILON. An explicitly
 * supplied epsilon must satisfy 0 < epsilon < 0.5.
 */
hdcd_status_t hdcd_transform_to_copula(
    const hdcd_marginal_t *marginal,
    const double *x,
    const uint8_t *observed_mask,
    size_t n,
    double epsilon,
    double *u_out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_COPULA_H */
