#ifndef HDCD_INTERNAL_CACHE_H
#define HDCD_INTERNAL_CACHE_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/local_fit.h"

/*
 * Per-node cache of fitted local parent-set models, keyed by parent set
 * (spec section 17.3): (child, P) -> {K_hat_j(P), Theta, normalization
 * state, n_effective} -- exactly hdcd_local_fit_t's contents. Changing
 * one local edge during search only ever changes ONE node's parent
 * set, so re-scoring a proposal costs one lookup-or-fit for that node
 * alone, never a refit of the whole graph.
 */
typedef struct hdcd_local_fit_cache hdcd_local_fit_cache_t;

hdcd_status_t hdcd_internal_cache_create(size_t d, hdcd_local_fit_cache_t **out);
void hdcd_internal_cache_free(hdcd_local_fit_cache_t *cache);

/*
 * Look up node `child`'s fit for parent set `parents` (order-
 * independent, need not be sorted or pre-deduplicated by the caller
 * beyond containing no duplicates); on a miss, fits it (via
 * hdcd_local_fit_node) and stores the result before returning it. *out
 * is a BORROWED pointer, owned by the cache (valid until
 * hdcd_internal_cache_free). *was_hit (if non-NULL) reports whether
 * this call was served from the cache.
 *
 * A hard local-fit failure (not mere non-convergence) is NOT cached and
 * is propagated as this function's return status.
 */
hdcd_status_t hdcd_internal_cache_get_or_fit(
    hdcd_local_fit_cache_t *cache,
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_local_fit_options_t *options,
    const hdcd_local_fit_t **out,
    int *was_hit
);

#endif /* HDCD_INTERNAL_CACHE_H */
