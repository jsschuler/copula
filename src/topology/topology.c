#include "hdcd/topology.h"
#include "mst.h"
#include "persistent_affinity.h"
#include "merge_tree.h"
#include "ordering.h"

#include <math.h>
#include <stdlib.h>

struct hdcd_topology {
    size_t d;
    hdcd_mst_edge_t *mst_edges;
    size_t mst_edge_count;
    double *tau;       /* d x d, row-major */
    size_t *ordering;  /* length d */
};

hdcd_status_t hdcd_compute_topology(const hdcd_dependence_matrix_t *dm, hdcd_topology_t **out) {
    if (dm == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    size_t d = hdcd_dependence_matrix_dim(dm);
    if (d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    hdcd_mst_edge_t *mst_edges = NULL;
    size_t mst_edge_count = 0;
    hdcd_status_t status = hdcd_internal_build_mst(dm, &mst_edges, &mst_edge_count);
    if (status != HDCD_OK) {
        return status;
    }

    double *tau = (double *)malloc(d * d * sizeof(double));
    if (tau == NULL) {
        free(mst_edges);
        return HDCD_ERROR_ALLOCATION;
    }

    status = hdcd_internal_build_merge_levels(mst_edges, mst_edge_count, d, tau);
    if (status != HDCD_OK) {
        free(mst_edges);
        free(tau);
        return status;
    }

    hdcd_merge_node_t *root = NULL;
    status = hdcd_internal_build_merge_tree(mst_edges, mst_edge_count, d, &root);
    if (status != HDCD_OK) {
        free(mst_edges);
        free(tau);
        return status;
    }

    size_t *ordering = (size_t *)malloc(d * sizeof(size_t));
    if (ordering == NULL) {
        hdcd_internal_free_merge_tree(root);
        free(mst_edges);
        free(tau);
        return HDCD_ERROR_ALLOCATION;
    }

    status = hdcd_internal_compute_ordering(root, d, ordering);
    hdcd_internal_free_merge_tree(root);
    if (status != HDCD_OK) {
        free(mst_edges);
        free(tau);
        free(ordering);
        return status;
    }

    hdcd_topology_t *topo = (hdcd_topology_t *)malloc(sizeof(hdcd_topology_t));
    if (topo == NULL) {
        free(mst_edges);
        free(tau);
        free(ordering);
        return HDCD_ERROR_ALLOCATION;
    }
    topo->d = d;
    topo->mst_edges = mst_edges;
    topo->mst_edge_count = mst_edge_count;
    topo->tau = tau;
    topo->ordering = ordering;

    *out = topo;
    return HDCD_OK;
}

void hdcd_topology_free(hdcd_topology_t *topo) {
    if (topo == NULL) {
        return;
    }
    free(topo->mst_edges);
    free(topo->tau);
    free(topo->ordering);
    free(topo);
}

size_t hdcd_topology_dim(const hdcd_topology_t *topo) {
    return (topo != NULL) ? topo->d : 0;
}

size_t hdcd_topology_mst_edge_count(const hdcd_topology_t *topo) {
    return (topo != NULL) ? topo->mst_edge_count : 0;
}

hdcd_mst_edge_t hdcd_topology_mst_edge(const hdcd_topology_t *topo, size_t idx) {
    hdcd_mst_edge_t sentinel;
    sentinel.j = 0;
    sentinel.k = 0;
    sentinel.weight = NAN;
    if (topo == NULL || idx >= topo->mst_edge_count) {
        return sentinel;
    }
    return topo->mst_edges[idx];
}

double hdcd_topology_merge_level(const hdcd_topology_t *topo, size_t j, size_t k) {
    if (topo == NULL || j >= topo->d || k >= topo->d) {
        return NAN;
    }
    return topo->tau[j * topo->d + k];
}

double hdcd_topology_affinity(const hdcd_topology_t *topo, size_t j, size_t k) {
    if (topo == NULL || j >= topo->d || k >= topo->d) {
        return NAN;
    }
    return 1.0 - topo->tau[j * topo->d + k];
}

const size_t *hdcd_topology_ordering(const hdcd_topology_t *topo) {
    return (topo != NULL) ? topo->ordering : NULL;
}
