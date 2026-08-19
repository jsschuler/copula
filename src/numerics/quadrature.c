#include "hdcd/numerics.h"

hdcd_status_t hdcd_simpson_nodes_weights(size_t n_nodes, double *nodes_out, double *weights_out) {
    if (nodes_out == NULL || weights_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n_nodes < 3 || (n_nodes % 2) == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t n_intervals = n_nodes - 1;
    double h = 1.0 / (double)n_intervals;

    for (size_t i = 0; i < n_nodes; i++) {
        nodes_out[i] = h * (double)i;
        if (i == 0 || i == n_nodes - 1) {
            weights_out[i] = h / 3.0;
        } else if (i % 2 == 1) {
            weights_out[i] = 4.0 * h / 3.0;
        } else {
            weights_out[i] = 2.0 * h / 3.0;
        }
    }
    /* Clamp the endpoint exactly (avoids a tiny floating-point residual
     * from h * n_intervals != 1.0 in rare cases). */
    nodes_out[n_nodes - 1] = 1.0;

    return HDCD_OK;
}
