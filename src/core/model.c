#include "hdcd/dag_fit.h"

#include <math.h>
#include <stdlib.h>

struct hdcd_dag_fit {
    size_t d;
    hdcd_local_fit_t **nodes; /* size d */
};

hdcd_status_t hdcd_dag_fit(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    const hdcd_dag_t *dag,
    const hdcd_local_fit_options_t *options,
    hdcd_dag_fit_t **out
) {
    if (u == NULL || mask == NULL || dag == NULL || options == NULL || out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0 || d == 0 || hdcd_dag_dim(dag) != d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    hdcd_dag_fit_t *fit = (hdcd_dag_fit_t *)malloc(sizeof(hdcd_dag_fit_t));
    if (fit == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    fit->d = d;
    fit->nodes = (hdcd_local_fit_t **)calloc(d, sizeof(hdcd_local_fit_t *));
    if (fit->nodes == NULL) {
        free(fit);
        return HDCD_ERROR_ALLOCATION;
    }

    hdcd_status_t overall_status = HDCD_OK;
    for (size_t j = 0; j < d; j++) {
        size_t n_parents = hdcd_dag_n_parents(dag, j);
        size_t *parents = NULL;
        if (n_parents > 0) {
            parents = (size_t *)malloc(n_parents * sizeof(size_t));
            if (parents == NULL) {
                hdcd_dag_fit_free(fit);
                return HDCD_ERROR_ALLOCATION;
            }
            hdcd_dag_parents(dag, j, parents);
        }

        /* Same options for every node, except a per-node-perturbed seed
         * so each node's train/holdout split is reproducible without
         * being identical across nodes (spec section 24/36 rule 14). */
        hdcd_local_fit_options_t node_options = *options;
        node_options.seed = options->seed + (uint64_t)j * 0x9E3779B97F4A7C15ULL;

        hdcd_local_fit_t *node_fit = NULL;
        hdcd_status_t status = hdcd_local_fit_node(u, mask, n, d, j, parents, n_parents, &node_options, &node_fit);
        free(parents);

        if (node_fit == NULL) {
            /* Hard failure fitting this node aborts the whole DAG fit. */
            hdcd_dag_fit_free(fit);
            return status;
        }
        fit->nodes[j] = node_fit;
        if (status != HDCD_OK) {
            overall_status = HDCD_ERROR_NOT_CONVERGED;
        }
    }

    *out = fit;
    return overall_status;
}

void hdcd_dag_fit_free(hdcd_dag_fit_t *fit) {
    if (fit == NULL) {
        return;
    }
    if (fit->nodes != NULL) {
        for (size_t j = 0; j < fit->d; j++) {
            hdcd_local_fit_free(fit->nodes[j]);
        }
        free(fit->nodes);
    }
    free(fit);
}

size_t hdcd_dag_fit_dim(const hdcd_dag_fit_t *fit) {
    return (fit != NULL) ? fit->d : 0;
}

const hdcd_local_fit_t *hdcd_dag_fit_node(const hdcd_dag_fit_t *fit, size_t j) {
    if (fit == NULL || j >= fit->d) {
        return NULL;
    }
    return fit->nodes[j];
}

int hdcd_dag_fit_node_converged(const hdcd_dag_fit_t *fit, size_t j) {
    const hdcd_local_fit_t *node_fit = hdcd_dag_fit_node(fit, j);
    if (node_fit == NULL) {
        return 0;
    }
    return hdcd_local_fit_theta_converged(node_fit) && hdcd_local_fit_sinkhorn_converged(node_fit);
}

int hdcd_dag_fit_all_converged(const hdcd_dag_fit_t *fit) {
    if (fit == NULL) {
        return 0;
    }
    for (size_t j = 0; j < fit->d; j++) {
        if (!hdcd_dag_fit_node_converged(fit, j)) {
            return 0;
        }
    }
    return 1;
}

hdcd_status_t hdcd_dag_fit_joint_log_density(
    const hdcd_dag_fit_t *fit,
    const double *u_point, size_t d,
    double *out
) {
    if (fit == NULL || u_point == NULL || out == NULL || d != fit->d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    double total = 0.0;
    double *z_buf = NULL;
    size_t z_buf_cap = 0;

    for (size_t j = 0; j < fit->d; j++) {
        const hdcd_local_fit_t *node_fit = fit->nodes[j];
        size_t n_parents = hdcd_local_fit_n_parents(node_fit);
        const size_t *parent_order = hdcd_local_fit_parent_order(node_fit);

        if (n_parents > z_buf_cap) {
            double *grown = (double *)realloc(z_buf, n_parents * sizeof(double));
            if (grown == NULL) {
                free(z_buf);
                return HDCD_ERROR_ALLOCATION;
            }
            z_buf = grown;
            z_buf_cap = n_parents;
        }
        for (size_t k = 0; k < n_parents; k++) {
            z_buf[k] = u_point[parent_order[k]];
        }

        double log_c;
        hdcd_status_t status = hdcd_local_fit_log_density(node_fit, u_point[j], z_buf, n_parents, &log_c);
        if (status != HDCD_OK) {
            free(z_buf);
            return status;
        }
        total += log_c;
    }

    free(z_buf);
    *out = total;
    return HDCD_OK;
}

double hdcd_dag_fit_kl_estimate(const hdcd_dag_fit_t *fit) {
    if (fit == NULL) {
        return NAN;
    }
    double total = 0.0;
    for (size_t j = 0; j < fit->d; j++) {
        total += -hdcd_local_fit_holdout_score(fit->nodes[j]);
    }
    return total;
}

double hdcd_dag_fit_kl_difference(const hdcd_dag_fit_t *candidate, const hdcd_dag_fit_t *reference) {
    if (candidate == NULL || reference == NULL || candidate->d != reference->d) {
        return NAN;
    }
    return hdcd_dag_fit_kl_estimate(candidate) - hdcd_dag_fit_kl_estimate(reference);
}
