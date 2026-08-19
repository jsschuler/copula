#ifndef HDCD_INTERNAL_MERGE_TREE_H
#define HDCD_INTERNAL_MERGE_TREE_H

#include <stddef.h>
#include "hdcd/status.h"
#include "hdcd/topology.h"

typedef struct hdcd_merge_node {
    int is_leaf;
    size_t leaf_index;         /* valid if is_leaf */
    double merge_height;       /* valid if !is_leaf: delta at which children merged */
    double score;              /* S(C) for this node's component (spec section 6.3) */
    size_t size;                /* number of leaves in this subtree */
    size_t min_leaf_index;       /* smallest original column index in this subtree */
    struct hdcd_merge_node *left;
    struct hdcd_merge_node *right;
} hdcd_merge_node_t;

/*
 * Build the single-linkage merge (dendrogram) tree from d-1 MST edges in
 * ascending-weight order (spec section 6.3).
 *
 * Scores use the closed form
 *   S(C1 union C2) = S(C1) + S(C2) + 2 * |C1| * |C2| * (1 - w)
 * where w is the weight of the edge merging C1 and C2. This is exact,
 * not an approximation: every cross pair (x in C1, y in C2) has
 * tau_xy = w precisely (the same bottleneck-path fact
 * persistent_affinity.c uses), so A_xy = 1 - w for all |C1|*|C2| cross
 * pairs, and S(C) sums both (x,y) and (y,x) -- hence the factor of 2.
 * This lets the tree be scored in O(d) total, without touching the
 * d x d affinity matrix at all.
 */
hdcd_status_t hdcd_internal_build_merge_tree(
    const hdcd_mst_edge_t *mst_edges, size_t mst_edge_count, size_t d,
    hdcd_merge_node_t **root_out
);

void hdcd_internal_free_merge_tree(hdcd_merge_node_t *node);

#endif /* HDCD_INTERNAL_MERGE_TREE_H */
