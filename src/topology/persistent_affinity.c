#include "persistent_affinity.h"

#include <stdlib.h>

hdcd_status_t hdcd_internal_build_merge_levels(
    const hdcd_mst_edge_t *mst_edges, size_t mst_edge_count, size_t d,
    double *tau_out
) {
    if (tau_out == NULL || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    size_t expected_edges = d - 1;
    if (mst_edge_count != expected_edges) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (d >= 2 && mst_edges == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < d; i++) {
        tau_out[i * d + i] = 0.0;
    }
    if (d == 1) {
        return HDCD_OK;
    }

    /* comp_id[i]: current component representative for original index i.
     * members[r] / member_count[r]: member list for the component whose
     * representative id is currently r. */
    size_t *comp_id = (size_t *)malloc(d * sizeof(size_t));
    size_t **members = (size_t **)malloc(d * sizeof(size_t *));
    size_t *member_count = (size_t *)malloc(d * sizeof(size_t));
    if (comp_id == NULL || members == NULL || member_count == NULL) {
        free(comp_id);
        free(members);
        free(member_count);
        return HDCD_ERROR_ALLOCATION;
    }
    for (size_t i = 0; i < d; i++) {
        members[i] = NULL;
    }

    hdcd_status_t status = HDCD_OK;
    for (size_t i = 0; i < d && status == HDCD_OK; i++) {
        comp_id[i] = i;
        members[i] = (size_t *)malloc(sizeof(size_t));
        if (members[i] == NULL) {
            status = HDCD_ERROR_ALLOCATION;
            break;
        }
        members[i][0] = i;
        member_count[i] = 1;
    }

    for (size_t e = 0; e < mst_edge_count && status == HDCD_OK; e++) {
        size_t cid_j = comp_id[mst_edges[e].j];
        size_t cid_k = comp_id[mst_edges[e].k];
        double w = mst_edges[e].weight;

        for (size_t a = 0; a < member_count[cid_j]; a++) {
            size_t x = members[cid_j][a];
            for (size_t b = 0; b < member_count[cid_k]; b++) {
                size_t y = members[cid_k][b];
                tau_out[x * d + y] = w;
                tau_out[y * d + x] = w;
            }
        }

        size_t new_count = member_count[cid_j] + member_count[cid_k];
        size_t *merged = (size_t *)realloc(members[cid_j], new_count * sizeof(size_t));
        if (merged == NULL) {
            status = HDCD_ERROR_ALLOCATION;
            break;
        }
        for (size_t b = 0; b < member_count[cid_k]; b++) {
            merged[member_count[cid_j] + b] = members[cid_k][b];
        }
        free(members[cid_k]);
        members[cid_k] = NULL;
        members[cid_j] = merged;
        member_count[cid_j] = new_count;
        member_count[cid_k] = 0;

        for (size_t x = 0; x < d; x++) {
            if (comp_id[x] == cid_k) {
                comp_id[x] = cid_j;
            }
        }
    }

    for (size_t i = 0; i < d; i++) {
        free(members[i]);
    }
    free(members);
    free(member_count);
    free(comp_id);

    return status;
}
