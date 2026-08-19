#include "mst.h"
#include "union_find.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    size_t j, k;
    double weight;
} candidate_edge_t;

static int compare_candidates(const void *pa, const void *pb) {
    const candidate_edge_t *a = (const candidate_edge_t *)pa;
    const candidate_edge_t *b = (const candidate_edge_t *)pb;
    if (a->weight < b->weight) return -1;
    if (a->weight > b->weight) return 1;
    if (a->j < b->j) return -1;
    if (a->j > b->j) return 1;
    if (a->k < b->k) return -1;
    if (a->k > b->k) return 1;
    return 0;
}

hdcd_status_t hdcd_internal_build_mst(
    const hdcd_dependence_matrix_t *dm,
    hdcd_mst_edge_t **edges_out,
    size_t *count_out
) {
    if (dm == NULL || edges_out == NULL || count_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t d = hdcd_dependence_matrix_dim(dm);
    if (d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    *edges_out = NULL;
    *count_out = 0;

    if (d == 1) {
        return HDCD_OK; /* a single node is trivially already spanned */
    }

    size_t max_candidates = d * (d - 1) / 2;
    candidate_edge_t *candidates = (candidate_edge_t *)malloc(max_candidates * sizeof(candidate_edge_t));
    if (candidates == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }

    size_t n_candidates = 0;
    for (size_t j = 0; j < d; j++) {
        for (size_t k = j + 1; k < d; k++) {
            double dcor_val = hdcd_dependence_matrix_get(dm, j, k);
            if (isnan(dcor_val)) {
                continue;
            }
            candidates[n_candidates].j = j;
            candidates[n_candidates].k = k;
            candidates[n_candidates].weight = 1.0 - dcor_val;
            n_candidates++;
        }
    }

    qsort(candidates, n_candidates, sizeof(candidate_edge_t), compare_candidates);

    hdcd_union_find_t uf;
    if (!hdcd_uf_init(&uf, d)) {
        free(candidates);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_mst_edge_t *edges = (hdcd_mst_edge_t *)malloc((d - 1) * sizeof(hdcd_mst_edge_t));
    if (edges == NULL) {
        hdcd_uf_free(&uf);
        free(candidates);
        return HDCD_ERROR_ALLOCATION;
    }

    size_t n_edges = 0;
    for (size_t i = 0; i < n_candidates && n_edges < d - 1; i++) {
        if (hdcd_uf_union(&uf, candidates[i].j, candidates[i].k)) {
            edges[n_edges].j = candidates[i].j;
            edges[n_edges].k = candidates[i].k;
            edges[n_edges].weight = candidates[i].weight;
            n_edges++;
        }
    }

    hdcd_uf_free(&uf);
    free(candidates);

    if (n_edges != d - 1) {
        /* Not connected: fail clearly rather than returning a partial
         * spanning forest (spec section 24: never silently continue). */
        free(edges);
        return HDCD_ERROR_NUMERICAL;
    }

    *edges_out = edges;
    *count_out = n_edges;
    return HDCD_OK;
}
