#ifndef HDCD_INTERNAL_PERSISTENT_AFFINITY_H
#define HDCD_INTERNAL_PERSISTENT_AFFINITY_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/topology.h"

/*
 * Fill tau_out (d x d, row-major, pre-allocated by the caller) with the
 * single-linkage merge level tau_jk for every pair (spec section 6.1),
 * replaying d-1 MST edges in ascending-weight order. tau_jj = 0.
 *
 * For any x, y that first become connected when merging two components
 * at edge weight w, tau_xy = w exactly (the bottleneck/minimax path
 * property of a minimum spanning tree) -- every unordered pair is
 * assigned exactly once across the whole replay, so this is O(d^2)
 * total despite the nested loop.
 */
hdcd_status_t hdcd_internal_build_merge_levels(
    const hdcd_mst_edge_t *mst_edges, size_t mst_edge_count, size_t d,
    double *tau_out
);

#endif /* HDCD_INTERNAL_PERSISTENT_AFFINITY_H */
