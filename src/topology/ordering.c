#include "ordering.h"

static void visit(const hdcd_merge_node_t *node, size_t *out, size_t *pos) {
    if (node->is_leaf) {
        out[*pos] = node->leaf_index;
        (*pos)++;
        return;
    }

    const hdcd_merge_node_t *first, *second;
    if (node->left->score > node->right->score) {
        first = node->left;
        second = node->right;
    } else if (node->right->score > node->left->score) {
        first = node->right;
        second = node->left;
    } else if (node->left->min_leaf_index <= node->right->min_leaf_index) {
        first = node->left;
        second = node->right;
    } else {
        first = node->right;
        second = node->left;
    }

    visit(first, out, pos);
    visit(second, out, pos);
}

hdcd_status_t hdcd_internal_compute_ordering(
    const hdcd_merge_node_t *root, size_t d, size_t *out
) {
    if (root == NULL || out == NULL || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    size_t pos = 0;
    visit(root, out, &pos);
    if (pos != d) {
        return HDCD_ERROR_NUMERICAL;
    }
    return HDCD_OK;
}
