#ifndef HDCD_TOPOLOGY_H
#define HDCD_TOPOLOGY_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/dcor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hdcd_mst_edge {
    size_t j;
    size_t k;
    double weight; /* delta_jk = 1 - D_jk */
} hdcd_mst_edge_t;

/*
 * Opaque persistent-topology result: the MST of delta_jk = 1 - D_jk, the
 * single-linkage merge levels tau_jk for every pair, and the final
 * variable ordering produced by recursively walking the merge
 * dendrogram (spec section 6).
 */
typedef struct hdcd_topology hdcd_topology_t;

/*
 * Build the persistent-topology ordering from a fitted pairwise
 * dependence matrix (spec section 6): Kruskal's MST over the pairs with
 * a defined dCor, the induced single-linkage merge level tau_jk and
 * persistent affinity A_jk = 1 - tau_jk for every pair, a binary merge
 * (dendrogram) tree scored by S(C) = sum_{j in C} sum_{k in C, k != j}
 * A_jk at every internal node, and the final permutation obtained by
 * always visiting the higher-scoring child first (ties broken by the
 * smaller original column index) -- which keeps every persistent
 * component contiguous in the output by construction.
 *
 * Pairs with an undefined dCor (NaN, from < 2 pairwise-complete rows,
 * spec section 5) are excluded as MST candidate edges. If the resulting
 * graph is not connected, this returns HDCD_ERROR_NUMERICAL rather than
 * fabricating a spanning tree.
 */
hdcd_status_t hdcd_compute_topology(
    const hdcd_dependence_matrix_t *dm,
    hdcd_topology_t **out
);

void hdcd_topology_free(hdcd_topology_t *topo);

size_t hdcd_topology_dim(const hdcd_topology_t *topo);

/* MST edges: d-1 of them, in Kruskal processing (ascending weight, then
 * (min(j,k), max(j,k))) order. */
size_t hdcd_topology_mst_edge_count(const hdcd_topology_t *topo);
hdcd_mst_edge_t hdcd_topology_mst_edge(const hdcd_topology_t *topo, size_t idx);

/* tau_jk, the single-linkage merge level (spec section 6.1). NaN if
 * topo is NULL or j/k are out of range. */
double hdcd_topology_merge_level(const hdcd_topology_t *topo, size_t j, size_t k);

/* A_jk = 1 - tau_jk, the persistent affinity (spec section 6.2). */
double hdcd_topology_affinity(const hdcd_topology_t *topo, size_t j, size_t k);

/* Final permutation pi (spec section 6.3): pointer to an internal array
 * of length hdcd_topology_dim(topo), valid for the lifetime of `topo`.
 * NULL if topo is NULL. */
const size_t *hdcd_topology_ordering(const hdcd_topology_t *topo);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_TOPOLOGY_H */
