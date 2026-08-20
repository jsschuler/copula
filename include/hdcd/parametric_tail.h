#ifndef HDCD_PARAMETRIC_TAIL_H
#define HDCD_PARAMETRIC_TAIL_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Parametric Archimedean tail-dependence families (see DECISIONS.md's
 * "copula-level EVT tail-splice" entry), used to model the sharp corner
 * singularity a bounded Bernstein tensor cannot reach at any degree.
 * NONE means "no parametric family": the plain Bernstein tensor kernel
 * is used unmodified for that edge, exactly as before this module
 * existed.
 */
typedef enum {
    HDCD_TAIL_FAMILY_NONE = 0,
    HDCD_TAIL_FAMILY_CLAYTON = 1, /* lower-tail dependence (u,v -> 0 corner); theta > 0 */
    HDCD_TAIL_FAMILY_GUMBEL = 2   /* upper-tail dependence (u,v -> 1 corner); theta >= 1 */
} hdcd_tail_family_t;

/*
 * Clayton copula density c(u,v;theta) = (1+theta) (uv)^{-theta-1}
 * (u^{-theta}+v^{-theta}-1)^{-1/theta-2}. Requires theta > 0 and
 * u,v in (0,1).
 */
hdcd_status_t hdcd_clayton_density(double u, double v, double theta, double *out);

/*
 * Gumbel copula density, standard closed form (e.g. Joe 1997; Nelsen
 * "An Introduction to Copulas"):
 *   x = -ln(u), y = -ln(v), A = (x^theta + y^theta)^(1/theta)
 *   c(u,v;theta) = C(u,v;theta) * (xy)^{theta-1} / (uv) * A^{1-2*theta} * (A + theta - 1)
 * where C = exp(-A) is the Gumbel copula CDF. Requires theta >= 1 and
 * u,v in (0,1). theta == 1 reduces to the independence copula (c == 1),
 * checked directly in tests/test_parametric_tail.c as a closed-form
 * sanity check on the derivation.
 */
hdcd_status_t hdcd_gumbel_density(double u, double v, double theta, double *out);

/* Dispatches to hdcd_clayton_density / hdcd_gumbel_density; HDCD_TAIL_FAMILY_NONE
 * is invalid here (the caller is expected to skip parametric evaluation
 * entirely for NONE, not call this). */
hdcd_status_t hdcd_tail_family_density(hdcd_tail_family_t family, double u, double v, double theta, double *out);

/*
 * Fit `theta` by maximum likelihood (deterministic golden-section search
 * over a wide, family-appropriate bound -- spec section 24's
 * deterministic-optimizer convention, reusing hdcd_golden_section_maximize)
 * against paired copula-scale samples u[0..n-1], v[0..n-1]. Requires
 * n >= 2 and family != HDCD_TAIL_FAMILY_NONE.
 *
 * This is an ordinary full-sample copula MLE, not restricted to a
 * "tail-only" subset of the data: for a single-parameter Archimedean
 * family, tail dependence and the family's overall association strength
 * are one and the same parameter, so a full-sample fit already targets
 * the tail behavior the family exists to capture.
 */
hdcd_status_t hdcd_tail_family_fit(
    hdcd_tail_family_t family,
    const double *u, const double *v, size_t n,
    double *theta_out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_PARAMETRIC_TAIL_H */
