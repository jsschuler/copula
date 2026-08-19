#include "merge_tree.h"
#include "union_find.h"

#include <stdlib.h>

static hdcd_merge_node_t *make_leaf(size_t index) {
    hdcd_merge_node_t *node = (hdcd_merge_node_t *)malloc(sizeof(hdcd_merge_node_t));
    if (node == NULL) {
        return NULL;
    }
    node->is_leaf = 1;
    node->leaf_index = index;
    node->merge_height = 0.0;
    node->score = 0.0;
    node->size = 1;
    node->min_leaf_index = index;
    node->left = NULL;
    node->right = NULL;
    return node;
}

void hdcd_internal_free_merge_tree(hdcd_merge_node_t *node) {
    if (node == NULL) {
        return;
    }
    if (!node->is_leaf) {
        hdcd_internal_free_merge_tree(node->left);
        hdcd_internal_free_merge_tree(node->right);
    }
    free(node);
}

hdcd_status_t hdcd_internal_build_merge_tree(
    const hdcd_mst_edge_t *mst_edges, size_t mst_edge_count, size_t d,
    hdcd_merge_node_t **root_out
) {
    if (root_out == NULL || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (mst_edge_count != d - 1) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (d >= 2 && mst_edges == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    hdcd_union_find_t uf;
    if (!hdcd_uf_init(&uf, d)) {
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_merge_node_t **node_for_root = (hdcd_merge_node_t **)malloc(d * sizeof(hdcd_merge_node_t *));
    if (node_for_root == NULL) {
        hdcd_uf_free(&uf);
        return HDCD_ERROR_ALLOCATION;
    }
    for (size_t i = 0; i < d; i++) {
        node_for_root[i] = NULL;
    }

    hdcd_status_t status = HDCD_OK;
    for (size_t i = 0; i < d; i++) {
        node_for_root[i] = make_leaf(i);
        if (node_for_root[i] == NULL) {
            status = HDCD_ERROR_ALLOCATION;
            break;
        }
    }

    for (size_t e = 0; e < mst_edge_count && status == HDCD_OK; e++) {
        size_t root_j = hdcd_uf_find(&uf, mst_edges[e].j);
        size_t root_k = hdcd_uf_find(&uf, mst_edges[e].k);

        hdcd_merge_node_t *left = node_for_root[root_j];
        hdcd_merge_node_t *right = node_for_root[root_k];
        double w = mst_edges[e].weight;

        hdcd_merge_node_t *parent = (hdcd_merge_node_t *)malloc(sizeof(hdcd_merge_node_t));
        if (parent == NULL) {
            status = HDCD_ERROR_ALLOCATION;
            break;
        }
        parent->is_leaf = 0;
        parent->leaf_index = 0;
        parent->merge_height = w;
        parent->score = left->score + right->score
                         + 2.0 * (double)left->size * (double)right->size * (1.0 - w);
        parent->size = left->size + right->size;
        parent->min_leaf_index = (left->min_leaf_index < right->min_leaf_index)
                                  ? left->min_leaf_index : right->min_leaf_index;
        parent->left = left;
        parent->right = right;

        hdcd_uf_union(&uf, root_j, root_k);
        size_t new_root = hdcd_uf_find(&uf, root_j);
        node_for_root[root_j] = NULL;
        node_for_root[root_k] = NULL;
        node_for_root[new_root] = parent;
    }

    if (status != HDCD_OK) {
        /* Free whatever forest remains, once per distinct surviving
         * subtree (avoids double-freeing shared children). */
        int *freed = (int *)calloc(d, sizeof(int));
        if (freed != NULL) {
            for (size_t i = 0; i < d; i++) {
                if (node_for_root[i] != NULL) {
                    size_t r = hdcd_uf_find(&uf, i);
                    if (!freed[r]) {
                        hdcd_internal_free_merge_tree(node_for_root[r]);
                        freed[r] = 1;
                    }
                }
            }
            free(freed);
        }
        hdcd_uf_free(&uf);
        free(node_for_root);
        return status;
    }

    size_t final_root = hdcd_uf_find(&uf, 0);
    hdcd_uf_free(&uf);
    *root_out = node_for_root[final_root];
    free(node_for_root);
    return HDCD_OK;
}
