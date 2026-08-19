#include "hdcd/dcor.h"
#include "hdcd/topology.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "test_utils.h"

#include <stdint.h>
#include <stdlib.h>

/* Self-contained deterministic PRNG, test-only (see the note in
 * tests/test_copula_transform.c). */
static uint64_t rng_state;

static void rng_seed(uint64_t seed) {
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t result = rng_state * 0x2545F4914F6CDD1DULL;
    double u = (double)(result >> 11) * (1.0 / 9007199254740992.0);
    if (u <= 0.0) u = 1e-12;
    if (u >= 1.0) u = 1.0 - 1e-12;
    return u;
}

static double rng_normal(void) {
    const double two_pi = 6.283185307179586;
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}

/*
 * There is no public constructor for an arbitrary dependence matrix
 * (hdcd_dependence_matrix_t is only built by computing dCor from data,
 * spec section 5), so the hand-worked scenario documented in
 * DECISIONS.md -- two groups {0,1} (looser) and {2,3} (tighter), weak
 * cross-group dependence, MST edges (2,3,0.05),(0,1,0.3),(0,2,0.9),
 * scores S({0,1})=1.4 < S({2,3})=1.9, expected ordering [2,3,0,1] --
 * is checked end-to-end here: data engineered to have that same
 * qualitative dCor shape, then asserting the same structural
 * conclusions the hand derivation reached.
 */
static void test_two_group_structure_and_ordering(void) {
    const size_t n = 300, d = 4;
    double *x = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(x != NULL && mask != NULL);

    rng_seed(2024);
    for (size_t i = 0; i < n; i++) {
        double lA = rng_normal();
        double lB = rng_normal();
        /* Group A = {0,1}: looser common factor, more idiosyncratic noise. */
        x[0 * n + i] = lA + 0.8 * rng_normal();
        x[1 * n + i] = lA + 0.8 * rng_normal();
        /* Group B = {2,3}: tighter common factor, less idiosyncratic noise. */
        x[2 * n + i] = lB + 0.15 * rng_normal();
        x[3 * n + i] = lB + 0.15 * rng_normal();
        for (size_t j = 0; j < d; j++) mask[j * n + i] = 1;
    }

    double *u = (double *)malloc(n * d * sizeof(double));
    HDCD_CHECK(u != NULL);
    for (size_t j = 0; j < d; j++) {
        hdcd_marginal_t *marginal = NULL;
        HDCD_CHECK(hdcd_marginal_fit(&x[j * n], &mask[j * n], n, -1.0, -1.0, 1e-3, 60, &marginal) == HDCD_OK);
        HDCD_CHECK(hdcd_transform_to_copula(marginal, &x[j * n], &mask[j * n], n, 0.0, &u[j * n]) == HDCD_OK);
        hdcd_marginal_free(marginal);
    }

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    /* Sanity: B tighter than A, both much tighter than cross-group. */
    double d_A = hdcd_dependence_matrix_get(dm, 0, 1);
    double d_B = hdcd_dependence_matrix_get(dm, 2, 3);
    double d_cross = hdcd_dependence_matrix_get(dm, 0, 2);
    HDCD_CHECK(d_B > d_A);
    HDCD_CHECK(d_A > d_cross);

    hdcd_topology_t *topo = NULL;
    HDCD_CHECK(hdcd_compute_topology(dm, &topo) == HDCD_OK);
    HDCD_CHECK(hdcd_topology_dim(topo) == d);
    HDCD_CHECK(hdcd_topology_mst_edge_count(topo) == d - 1);

    const size_t *ordering = hdcd_topology_ordering(topo);
    HDCD_CHECK(ordering != NULL);

    size_t pos[4];
    for (size_t p = 0; p < d; p++) pos[ordering[p]] = p;

    /* Contiguous blocks: {0,1} adjacent, {2,3} adjacent. */
    HDCD_CHECK((pos[0] > pos[1] ? pos[0] - pos[1] : pos[1] - pos[0]) == 1);
    HDCD_CHECK((pos[2] > pos[3] ? pos[2] - pos[3] : pos[3] - pos[2]) == 1);

    /* Tighter group (B = {2,3}) has higher persistent-affinity centrality
     * and must be placed before the looser group (A = {0,1}). */
    HDCD_CHECK(pos[2] < pos[0]);
    HDCD_CHECK(pos[3] < pos[0]);

    hdcd_topology_free(topo);
    hdcd_dependence_matrix_free(dm);
    free(x);
    free(u);
    free(mask);
    HDCD_PASS("two-group structure: contiguous blocks, tighter group placed first");
}

static void test_three_cluster_contiguity(void) {
    /* Spec section 31 M4 acceptance: "synthetic clustered-dependence
     * test gives expected block structure." Three independent latent
     * factors, two observed variables per factor. */
    const size_t n = 250, d = 6;
    double *x = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(x != NULL && mask != NULL);

    rng_seed(777);
    for (size_t i = 0; i < n; i++) {
        double l1 = rng_normal(), l2 = rng_normal(), l3 = rng_normal();
        x[0 * n + i] = l1 + 0.1 * rng_normal();
        x[1 * n + i] = l1 + 0.1 * rng_normal();
        x[2 * n + i] = l2 + 0.1 * rng_normal();
        x[3 * n + i] = l2 + 0.1 * rng_normal();
        x[4 * n + i] = l3 + 0.1 * rng_normal();
        x[5 * n + i] = l3 + 0.1 * rng_normal();
        for (size_t j = 0; j < d; j++) mask[j * n + i] = 1;
    }

    double *u = (double *)malloc(n * d * sizeof(double));
    HDCD_CHECK(u != NULL);
    for (size_t j = 0; j < d; j++) {
        hdcd_marginal_t *marginal = NULL;
        HDCD_CHECK(hdcd_marginal_fit(&x[j * n], &mask[j * n], n, -1.0, -1.0, 1e-3, 60, &marginal) == HDCD_OK);
        HDCD_CHECK(hdcd_transform_to_copula(marginal, &x[j * n], &mask[j * n], n, 0.0, &u[j * n]) == HDCD_OK);
        hdcd_marginal_free(marginal);
    }

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    hdcd_topology_t *topo = NULL;
    HDCD_CHECK(hdcd_compute_topology(dm, &topo) == HDCD_OK);

    const size_t *ordering = hdcd_topology_ordering(topo);
    size_t pos[6];
    for (size_t p = 0; p < d; p++) pos[ordering[p]] = p;

    HDCD_CHECK((pos[0] > pos[1] ? pos[0] - pos[1] : pos[1] - pos[0]) == 1);
    HDCD_CHECK((pos[2] > pos[3] ? pos[2] - pos[3] : pos[3] - pos[2]) == 1);
    HDCD_CHECK((pos[4] > pos[5] ? pos[4] - pos[5] : pos[5] - pos[4]) == 1);

    hdcd_topology_free(topo);
    hdcd_dependence_matrix_free(dm);
    free(x);
    free(u);
    free(mask);
    HDCD_PASS("three independent clusters each form a contiguous block");
}

static void test_reproducibility(void) {
    const size_t n = 150, d = 5;
    double *x = (double *)malloc(n * d * sizeof(double));
    uint8_t *mask = (uint8_t *)malloc(n * d);
    HDCD_CHECK(x != NULL && mask != NULL);

    rng_seed(55);
    for (size_t i = 0; i < n; i++) {
        double l = rng_normal();
        x[0 * n + i] = l + 0.2 * rng_normal();
        x[1 * n + i] = l + 0.2 * rng_normal();
        x[2 * n + i] = rng_normal();
        x[3 * n + i] = rng_normal();
        x[4 * n + i] = rng_normal();
        for (size_t j = 0; j < d; j++) mask[j * n + i] = 1;
    }

    double *u = (double *)malloc(n * d * sizeof(double));
    for (size_t j = 0; j < d; j++) {
        hdcd_marginal_t *marginal = NULL;
        HDCD_CHECK(hdcd_marginal_fit(&x[j * n], &mask[j * n], n, -1.0, -1.0, 1e-3, 60, &marginal) == HDCD_OK);
        HDCD_CHECK(hdcd_transform_to_copula(marginal, &x[j * n], &mask[j * n], n, 0.0, &u[j * n]) == HDCD_OK);
        hdcd_marginal_free(marginal);
    }

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    hdcd_topology_t *topo1 = NULL, *topo2 = NULL;
    HDCD_CHECK(hdcd_compute_topology(dm, &topo1) == HDCD_OK);
    HDCD_CHECK(hdcd_compute_topology(dm, &topo2) == HDCD_OK);

    const size_t *ord1 = hdcd_topology_ordering(topo1);
    const size_t *ord2 = hdcd_topology_ordering(topo2);
    for (size_t i = 0; i < d; i++) {
        HDCD_CHECK(ord1[i] == ord2[i]);
    }
    HDCD_CHECK(hdcd_topology_mst_edge_count(topo1) == hdcd_topology_mst_edge_count(topo2));
    for (size_t e = 0; e < hdcd_topology_mst_edge_count(topo1); e++) {
        hdcd_mst_edge_t e1 = hdcd_topology_mst_edge(topo1, e);
        hdcd_mst_edge_t e2 = hdcd_topology_mst_edge(topo2, e);
        HDCD_CHECK(e1.j == e2.j && e1.k == e2.k);
        HDCD_CHECK_NEAR(e1.weight, e2.weight, 1e-15);
    }

    hdcd_topology_free(topo1);
    hdcd_topology_free(topo2);
    hdcd_dependence_matrix_free(dm);
    free(x);
    free(u);
    free(mask);
    HDCD_PASS("topology computation is exactly reproducible");
}

static void test_disconnected_graph_fails_clearly(void) {
    /* Column 2 shares fewer than 2 pairwise-complete rows with every
     * other column, so it has no usable MST edge to the rest of the
     * graph -- must fail clearly, not fabricate a spanning tree. */
    const size_t n = 4, d = 3;
    double u[12] = {
        0.1, 0.3, 0.5, 0.7,   /* col 0 */
        0.2, 0.4, 0.6, 0.8,   /* col 1 */
        0.15, 0.0, 0.0, 0.0   /* col 2, only row 0 matters */
    };
    uint8_t mask[12] = {
        1, 1, 1, 1,   /* col 0: fully observed */
        1, 1, 1, 1,   /* col 1: fully observed */
        1, 0, 0, 0    /* col 2: only row 0 observed */
    };

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);
    HDCD_CHECK(isnan(hdcd_dependence_matrix_get(dm, 0, 2)));
    HDCD_CHECK(isnan(hdcd_dependence_matrix_get(dm, 1, 2)));

    hdcd_topology_t *topo = NULL;
    hdcd_status_t status = hdcd_compute_topology(dm, &topo);
    HDCD_CHECK(status == HDCD_ERROR_NUMERICAL);
    HDCD_CHECK(topo == NULL);

    hdcd_dependence_matrix_free(dm);
    HDCD_PASS("disconnected dependence graph fails with HDCD_ERROR_NUMERICAL");
}

static void test_single_dimension(void) {
    const size_t n = 20, d = 1;
    double u[20];
    uint8_t mask[20];
    rng_seed(3);
    for (size_t i = 0; i < n; i++) {
        u[i] = rng_uniform();
        mask[i] = 1;
    }

    hdcd_dependence_matrix_t *dm = NULL;
    HDCD_CHECK(hdcd_compute_dependence_matrix(u, mask, n, d, &dm) == HDCD_OK);

    hdcd_topology_t *topo = NULL;
    HDCD_CHECK(hdcd_compute_topology(dm, &topo) == HDCD_OK);
    HDCD_CHECK(hdcd_topology_dim(topo) == 1);
    HDCD_CHECK(hdcd_topology_mst_edge_count(topo) == 0);
    const size_t *ordering = hdcd_topology_ordering(topo);
    HDCD_CHECK(ordering[0] == 0);
    HDCD_CHECK_NEAR(hdcd_topology_merge_level(topo, 0, 0), 0.0, 1e-15);

    hdcd_topology_free(topo);
    hdcd_dependence_matrix_free(dm);
    HDCD_PASS("d=1 topology handled without special-casing errors");
}

static void test_invalid_arguments(void) {
    hdcd_topology_t *topo = NULL;
    HDCD_CHECK(hdcd_compute_topology(NULL, &topo) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(isnan(hdcd_topology_merge_level(NULL, 0, 0)));
    HDCD_CHECK(isnan(hdcd_topology_affinity(NULL, 0, 0)));
    HDCD_CHECK(hdcd_topology_ordering(NULL) == NULL);
    HDCD_CHECK(hdcd_topology_dim(NULL) == 0);
    HDCD_CHECK(hdcd_topology_mst_edge_count(NULL) == 0);
    hdcd_mst_edge_t e = hdcd_topology_mst_edge(NULL, 0);
    HDCD_CHECK(isnan(e.weight));

    hdcd_topology_free(NULL); /* must not crash */

    HDCD_PASS("topology API rejects invalid arguments / NULL handles safely");
}

int main(void) {
    test_two_group_structure_and_ordering();
    test_three_cluster_contiguity();
    test_reproducibility();
    test_disconnected_graph_fails_clearly();
    test_single_dimension();
    test_invalid_arguments();
    printf("All topology tests passed.\n");
    return 0;
}
