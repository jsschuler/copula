#ifndef HDCD_INTERNAL_ORDERING_H
#define HDCD_INTERNAL_ORDERING_H

#include <stddef.h>
#include "hdcd/status.h"
#include "merge_tree.h"

/*
 * Recursively visit the merge tree, at every internal node placing the
 * higher-scoring child first (ties broken by the smaller
 * min_leaf_index), writing the resulting permutation into out (length
 * d). Because each recursive call emits one contiguous run of leaves
 * per subtree, persistent components land as contiguous blocks in the
 * output by construction (spec section 6.3).
 */
hdcd_status_t hdcd_internal_compute_ordering(
    const hdcd_merge_node_t *root, size_t d, size_t *out
);

#endif /* HDCD_INTERNAL_ORDERING_H */
