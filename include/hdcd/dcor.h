#ifndef HDCD_DCOR_H
#define HDCD_DCOR_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Exact sample distance correlation between two paired univariate
 * samples (Szekely, Rizzo & Bakirov 2007), computed via double-centered
 * pairwise Euclidean (here, |.|) distance matrices (spec section 5).
 * Requires n >= 2 (dCor is undefined on 0 or 1 points).
 *
 * dCor lies in [0,1]. If either series is constant (zero distance
 * variance), the result is defined as 0 by convention rather than 0/0.
 */
hdcd_status_t hdcd_dcor_exact(const double *x, const double *y, size_t n, double *out);

/* ---- pairwise dependence matrix (spec section 5) ---------------------- */

/*
 * Opaque d x d pairwise distance-correlation matrix, together with the
 * number of pairwise-complete rows used for each entry.
 */
typedef struct hdcd_dependence_matrix hdcd_dependence_matrix_t;

/*
 * Compute the pairwise dependence matrix over copula-scale coordinates
 * `u` (n observations x d dimensions, COLUMN-MAJOR: u[j*n + i], per the
 * core layout convention in spec section 23). `observed_mask` has the
 * same shape and layout as `u`.
 *
 * For each pair (j,k), D_jk = dCor(U_j, U_k) is estimated using only the
 * pairwise-complete rows O_jk = O_j ∩ O_k (spec section 5). D_jj = 1 by
 * convention. If fewer than 2 rows are pairwise-complete for a pair,
 * D_jk is NaN (undefined) rather than a fabricated value -- the caller
 * must check the stored effective sample size.
 *
 * The resulting matrix is NOT assumed positive semidefinite and must
 * not be treated as a covariance matrix (spec section 5, section 36
 * rules 2-3).
 */
hdcd_status_t hdcd_compute_dependence_matrix(
    const double *u,
    const uint8_t *observed_mask,
    size_t n, size_t d,
    hdcd_dependence_matrix_t **out
);

void hdcd_dependence_matrix_free(hdcd_dependence_matrix_t *dm);

size_t hdcd_dependence_matrix_dim(const hdcd_dependence_matrix_t *dm);

/* D_jk. Symmetric: get(dm, j, k) == get(dm, k, j). NaN for out-of-range j/k. */
double hdcd_dependence_matrix_get(const hdcd_dependence_matrix_t *dm, size_t j, size_t k);

/* |O_jk|, the number of pairwise-complete rows used for entry (j,k). */
size_t hdcd_dependence_matrix_n_effective(const hdcd_dependence_matrix_t *dm, size_t j, size_t k);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_DCOR_H */
