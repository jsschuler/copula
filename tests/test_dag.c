#include "hdcd/dag.h"
#include "test_utils.h"

#include <stdlib.h>

static void test_valid_dag_accepted(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(4, 3, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_dim(dag) == 4);
    HDCD_CHECK(hdcd_dag_k_max(dag) == 3);

    /* 0 -> 1 -> 2, 0 -> 2, 0 -> 3 */
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 2) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 2) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 3) == HDCD_OK);

    HDCD_CHECK(hdcd_dag_has_edge(dag, 0, 1));
    HDCD_CHECK(hdcd_dag_has_edge(dag, 1, 2));
    HDCD_CHECK(!hdcd_dag_has_edge(dag, 2, 1));
    HDCD_CHECK(hdcd_dag_n_parents(dag, 2) == 2);

    size_t parents[3];
    HDCD_CHECK(hdcd_dag_parents(dag, 2, parents) == HDCD_OK);
    HDCD_CHECK(parents[0] == 0 && parents[1] == 1); /* ascending order */

    size_t order[4];
    HDCD_CHECK(hdcd_dag_topological_order(dag, order) == HDCD_OK);
    /* 0 must precede 1, 2, 3; 1 must precede 2. */
    size_t pos[4];
    for (size_t i = 0; i < 4; i++) pos[order[i]] = i;
    HDCD_CHECK(pos[0] < pos[1]);
    HDCD_CHECK(pos[0] < pos[2]);
    HDCD_CHECK(pos[0] < pos[3]);
    HDCD_CHECK(pos[1] < pos[2]);

    hdcd_dag_free(dag);
    HDCD_PASS("valid DAG is accepted, queried, and topologically sortable");
}

static void test_cycle_rejected(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(3, 3, &dag) == HDCD_OK);

    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 2) == HDCD_OK);
    /* 2 -> 0 would close a cycle 0->1->2->0. */
    HDCD_CHECK(hdcd_dag_add_edge(dag, 2, 0) == HDCD_ERROR_INVALID_ARGUMENT);
    /* Direct self-loop and immediate 2-cycle must also be rejected. */
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 1) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 2, 1) == HDCD_ERROR_INVALID_ARGUMENT); /* 1 already reaches 2 */

    hdcd_dag_free(dag);
    HDCD_PASS("edges that would create a cycle are rejected");
}

static void test_k_max_enforced(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(4, 2, &dag) == HDCD_OK);

    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 3) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 3) == HDCD_OK);
    /* node 3 already has k_max=2 parents. */
    HDCD_CHECK(hdcd_dag_add_edge(dag, 2, 3) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_n_parents(dag, 3) == 2);

    hdcd_dag_free(dag);
    HDCD_PASS("k_max hard sparsity limit is enforced");
}

static void test_duplicate_and_remove(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(3, 3, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_ERROR_INVALID_ARGUMENT); /* duplicate */

    HDCD_CHECK(hdcd_dag_remove_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(!hdcd_dag_has_edge(dag, 0, 1));
    HDCD_CHECK(hdcd_dag_remove_edge(dag, 0, 1) == HDCD_OK); /* no-op, already absent */

    /* After removal, re-adding and even the edge that was previously
     * blocked by the (now-gone) reverse path should succeed. */
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 0) == HDCD_OK);

    hdcd_dag_free(dag);
    HDCD_PASS("duplicate edges rejected; removal frees up the reverse edge");
}

static void test_invalid_arguments(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(0, 3, &dag) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_create(3, 3, NULL) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_dag_create(3, 3, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 5) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 5, 0) == HDCD_ERROR_INVALID_ARGUMENT);
    HDCD_CHECK(hdcd_dag_n_parents(NULL, 0) == 0);
    HDCD_CHECK(hdcd_dag_dim(NULL) == 0);

    hdcd_dag_free(dag);
    hdcd_dag_free(NULL); /* must not crash */
    HDCD_PASS("DAG API rejects invalid arguments");
}

static void test_clone_matches_source(void) {
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_create(4, 2, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 1) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 0, 2) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 1, 3) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_add_edge(dag, 2, 3) == HDCD_OK);

    hdcd_dag_t *clone = NULL;
    HDCD_CHECK(hdcd_dag_clone(dag, &clone) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_dim(clone) == hdcd_dag_dim(dag));
    HDCD_CHECK(hdcd_dag_k_max(clone) == hdcd_dag_k_max(dag));
    for (size_t c = 0; c < 4; c++) {
        HDCD_CHECK(hdcd_dag_n_parents(clone, c) == hdcd_dag_n_parents(dag, c));
    }
    HDCD_CHECK(hdcd_dag_has_edge(clone, 0, 1) && hdcd_dag_has_edge(clone, 2, 3));

    /* Independence: mutating the clone must not affect the source. */
    HDCD_CHECK(hdcd_dag_remove_edge(clone, 0, 1) == HDCD_OK);
    HDCD_CHECK(!hdcd_dag_has_edge(clone, 0, 1));
    HDCD_CHECK(hdcd_dag_has_edge(dag, 0, 1));

    hdcd_dag_free(dag);
    hdcd_dag_free(clone);
    HDCD_PASS("hdcd_dag_clone produces an independent, structurally identical copy");
}

static void test_from_edges_accepts_different_topological_order(void) {
    /* True topological order here is [2, 0, 1] (2 has no parents; 0's
     * only parent is 2; 1's only parent is 0) -- deliberately NOT
     * ascending index order, to exercise "accepts valid DAGs with
     * different topological orders" (spec section 31 Milestone 9). */
    size_t parents[2] = {2, 0};
    size_t children[2] = {0, 1};
    hdcd_dag_t *dag = NULL;
    HDCD_CHECK(hdcd_dag_from_edges(3, 2, parents, children, 2, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_dim(dag) == 3);
    HDCD_CHECK(hdcd_dag_has_edge(dag, 2, 0));
    HDCD_CHECK(hdcd_dag_has_edge(dag, 0, 1));
    HDCD_CHECK(hdcd_dag_n_parents(dag, 2) == 0);

    size_t order[3];
    HDCD_CHECK(hdcd_dag_topological_order(dag, order) == HDCD_OK);
    size_t pos[3];
    for (size_t i = 0; i < 3; i++) pos[order[i]] = i;
    HDCD_CHECK(pos[2] < pos[0]);
    HDCD_CHECK(pos[0] < pos[1]);

    hdcd_dag_free(dag);
    HDCD_PASS("hdcd_dag_from_edges accepts a valid DAG with a non-ascending topological order");
}

static void test_from_edges_rejects_cycle(void) {
    /* 0 -> 1 -> 2 -> 0: a genuine cycle. hdcd_dag_add_edge makes this
     * unconstructible by design (each edge is checked incrementally),
     * so hdcd_dag_from_edges -- which accepts the whole edge set before
     * validating once at the end (spec section 19 step 1) -- is the
     * only way to actually exercise cycle REJECTION through the public
     * API (spec section 31 Milestone 9: "rejects cyclic graphs"). */
    size_t parents[3] = {0, 1, 2};
    size_t children[3] = {1, 2, 0};
    hdcd_dag_t *dag = NULL;
    hdcd_status_t status = hdcd_dag_from_edges(3, 3, parents, children, 3, &dag);
    HDCD_CHECK(status == HDCD_ERROR_NUMERICAL);
    HDCD_CHECK(dag == NULL);
    HDCD_PASS("hdcd_dag_from_edges rejects a genuinely cyclic edge set");
}

static void test_from_edges_rejects_malformed_input(void) {
    hdcd_dag_t *dag = NULL;

    size_t self_loop_p[1] = {1}, self_loop_c[1] = {1};
    HDCD_CHECK(hdcd_dag_from_edges(3, 3, self_loop_p, self_loop_c, 1, &dag) == HDCD_ERROR_INVALID_ARGUMENT);

    size_t dup_p[2] = {0, 0}, dup_c[2] = {1, 1};
    HDCD_CHECK(hdcd_dag_from_edges(3, 3, dup_p, dup_c, 2, &dag) == HDCD_ERROR_INVALID_ARGUMENT);

    size_t oor_p[1] = {5}, oor_c[1] = {1};
    HDCD_CHECK(hdcd_dag_from_edges(3, 3, oor_p, oor_c, 1, &dag) == HDCD_ERROR_INVALID_ARGUMENT);

    size_t kmax_p[2] = {0, 1}, kmax_c[2] = {2, 2};
    HDCD_CHECK(hdcd_dag_from_edges(3, 1, kmax_p, kmax_c, 2, &dag) == HDCD_ERROR_INVALID_ARGUMENT);

    HDCD_CHECK(hdcd_dag_from_edges(0, 3, NULL, NULL, 0, &dag) == HDCD_ERROR_INVALID_ARGUMENT);

    /* n_edges == 0 with d > 0 is valid: the empty graph. */
    HDCD_CHECK(hdcd_dag_from_edges(3, 3, NULL, NULL, 0, &dag) == HDCD_OK);
    HDCD_CHECK(hdcd_dag_n_parents(dag, 0) == 0);
    hdcd_dag_free(dag);

    HDCD_PASS("hdcd_dag_from_edges rejects self-loops, duplicates, out-of-range indices, and k_max violations");
}

int main(void) {
    test_valid_dag_accepted();
    test_cycle_rejected();
    test_k_max_enforced();
    test_duplicate_and_remove();
    test_invalid_arguments();
    test_clone_matches_source();
    test_from_edges_accepts_different_topological_order();
    test_from_edges_rejects_cycle();
    test_from_edges_rejects_malformed_input();
    printf("All dag tests passed.\n");
    return 0;
}
