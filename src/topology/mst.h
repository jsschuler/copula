#ifndef HDCD_INTERNAL_MST_H
#define HDCD_INTERNAL_MST_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/dcor.h"
#include "hdcd/topology.h"

/*
 * Build the minimum spanning tree of delta_jk = 1 - D_jk over the pairs
 * with a defined dCor (pairs with a NaN entry, from < 2 pairwise-complete
 * rows, are excluded as candidate edges).
 *
 * On success, *edges_out is a newly allocated array of *count_out ==
 * d - 1 edges in ascending-weight Kruskal order, ties broken by
 * (min(j,k), max(j,k)) for full determinism; the caller owns it (may be
 * NULL with count 0 when d == 1). Returns HDCD_ERROR_NUMERICAL if the
 * graph is not connected -- some variable has no usable dependence
 * measurement linking it to the rest.
 */
hdcd_status_t hdcd_internal_build_mst(
    const hdcd_dependence_matrix_t *dm,
    hdcd_mst_edge_t **edges_out,
    size_t *count_out
);

#endif /* HDCD_INTERNAL_MST_H */
