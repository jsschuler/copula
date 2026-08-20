#ifndef HDCD_TAIL_DEPENDENCE_H
#define HDCD_TAIL_DEPENDENCE_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Empirical nonparametric coefficient-of-tail-dependence estimator
 * (Frahm, Junker & Schmidt 2005-style): the fraction of the top-`k`
 * order-statistic exceedances of `u` that are ALSO top-`k` exceedances
 * of `v` (upper tail), and the analogous fraction at the bottom (lower
 * tail). Operates on paired COPULA-SCALE samples (already-uniform
 * margins expected, though the estimator only uses each series' own
 * ranks, so it is invariant to any strictly increasing transform of
 * either input).
 *
 * `k` is the number of extreme order statistics used per tail: the
 * classic EVT bias/variance tradeoff (too small = high variance, too
 * large = biased by non-extreme behavior). `k == 0` selects a default
 * of round(sqrt(n)), clamped to [2, n/4] -- a standard starting point
 * for order-statistic-based tail estimators, sufficient for the
 * gating/selection use this module was built for (spec section 18's "a
 * small ... grid is sufficient for version 1" spirit); a dedicated
 * threshold-selection algorithm is out of scope for v1.
 *
 * Both outputs lie in [0,1]: 0 means asymptotic independence in that
 * tail, higher means stronger tail dependence. Requires n >= 8 (need
 * enough points for even a k=2 tail estimate to be meaningful on both
 * sides) and finite, non-constant `u`/`v`.
 */
hdcd_status_t hdcd_tail_dependence_coefficient(
    const double *u, const double *v, size_t n, size_t k,
    double *out_lambda_upper, double *out_lambda_lower
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_TAIL_DEPENDENCE_H */
