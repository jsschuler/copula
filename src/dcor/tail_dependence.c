#include "hdcd/tail_dependence.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    double value;
    size_t index;
} indexed_value_t;

static int compare_indexed_value(const void *a, const void *b) {
    double va = ((const indexed_value_t *)a)->value;
    double vb = ((const indexed_value_t *)b)->value;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/* ranks[i] = 0-indexed rank of x[i] among x[0..n-1] (0 = smallest). Ties
 * broken by qsort's (unspecified but deterministic-for-a-given-libc)
 * order -- acceptable here since copula-scale continuous data makes
 * exact ties vanishingly unlikely, and a tie-break either way changes
 * the resulting coefficient by at most 1/k. */
static int compute_ranks(const double *x, size_t n, size_t *ranks) {
    indexed_value_t *iv = (indexed_value_t *)malloc(n * sizeof(indexed_value_t));
    if (iv == NULL) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        iv[i].value = x[i];
        iv[i].index = i;
    }
    qsort(iv, n, sizeof(indexed_value_t), compare_indexed_value);
    for (size_t r = 0; r < n; r++) {
        ranks[iv[r].index] = r;
    }
    free(iv);
    return 1;
}

hdcd_status_t hdcd_tail_dependence_coefficient(
    const double *u, const double *v, size_t n, size_t k,
    double *out_lambda_upper, double *out_lambda_lower
) {
    if (u == NULL || v == NULL || out_lambda_upper == NULL || out_lambda_lower == NULL || n < 8) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; i++) {
        if (isnan(u[i]) || isnan(v[i]) || isinf(u[i]) || isinf(v[i])) {
            return HDCD_ERROR_INVALID_ARGUMENT;
        }
    }

    if (k == 0) {
        size_t max_k = n / 4;
        if (max_k < 2) {
            max_k = 2;
        }
        double default_k = round(sqrt((double)n));
        k = (default_k < 2.0) ? 2 : (size_t)default_k;
        if (k > max_k) {
            k = max_k;
        }
    }
    if (k < 1 || k >= n) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    size_t *rank_u = (size_t *)malloc(n * sizeof(size_t));
    size_t *rank_v = (size_t *)malloc(n * sizeof(size_t));
    if (rank_u == NULL || rank_v == NULL
        || !compute_ranks(u, n, rank_u) || !compute_ranks(v, n, rank_v)) {
        free(rank_u);
        free(rank_v);
        return HDCD_ERROR_ALLOCATION;
    }

    size_t upper_joint = 0, lower_joint = 0;
    for (size_t i = 0; i < n; i++) {
        if (rank_u[i] >= n - k && rank_v[i] >= n - k) {
            upper_joint++;
        }
        if (rank_u[i] < k && rank_v[i] < k) {
            lower_joint++;
        }
    }

    free(rank_u);
    free(rank_v);

    *out_lambda_upper = (double)upper_joint / (double)k;
    *out_lambda_lower = (double)lower_joint / (double)k;
    return HDCD_OK;
}
