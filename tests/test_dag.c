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

int main(void) {
    test_valid_dag_accepted();
    test_cycle_rejected();
    test_k_max_enforced();
    test_duplicate_and_remove();
    test_invalid_arguments();
    printf("All dag tests passed.\n");
    return 0;
}
