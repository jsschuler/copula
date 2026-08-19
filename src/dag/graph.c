#include "hdcd/dag.h"

#include <stdint.h>
#include <stdlib.h>

struct hdcd_dag {
    size_t d;
    size_t k_max;
    size_t **parents;    /* parents[child] = ascending array of parent indices */
    size_t *n_parents;   /* count per child */
    size_t *capacity;    /* allocated capacity of parents[child] */
};

hdcd_status_t hdcd_dag_create(size_t d, size_t k_max, hdcd_dag_t **out) {
    if (out == NULL || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    hdcd_dag_t *dag = (hdcd_dag_t *)malloc(sizeof(hdcd_dag_t));
    if (dag == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    dag->d = d;
    dag->k_max = k_max;
    dag->parents = (size_t **)calloc(d, sizeof(size_t *));
    dag->n_parents = (size_t *)calloc(d, sizeof(size_t));
    dag->capacity = (size_t *)calloc(d, sizeof(size_t));
    if (dag->parents == NULL || dag->n_parents == NULL || dag->capacity == NULL) {
        free(dag->parents);
        free(dag->n_parents);
        free(dag->capacity);
        free(dag);
        return HDCD_ERROR_ALLOCATION;
    }

    *out = dag;
    return HDCD_OK;
}

void hdcd_dag_free(hdcd_dag_t *dag) {
    if (dag == NULL) {
        return;
    }
    for (size_t c = 0; c < dag->d; c++) {
        free(dag->parents[c]);
    }
    free(dag->parents);
    free(dag->n_parents);
    free(dag->capacity);
    free(dag);
}

size_t hdcd_dag_dim(const hdcd_dag_t *dag) {
    return (dag != NULL) ? dag->d : 0;
}

size_t hdcd_dag_k_max(const hdcd_dag_t *dag) {
    return (dag != NULL) ? dag->k_max : 0;
}

int hdcd_dag_has_edge(const hdcd_dag_t *dag, size_t parent, size_t child) {
    if (dag == NULL || parent >= dag->d || child >= dag->d) {
        return 0;
    }
    for (size_t i = 0; i < dag->n_parents[child]; i++) {
        if (dag->parents[child][i] == parent) {
            return 1;
        }
    }
    return 0;
}

size_t hdcd_dag_n_parents(const hdcd_dag_t *dag, size_t child) {
    if (dag == NULL || child >= dag->d) {
        return 0;
    }
    return dag->n_parents[child];
}

hdcd_status_t hdcd_dag_parents(const hdcd_dag_t *dag, size_t child, size_t *out) {
    if (dag == NULL || out == NULL || child >= dag->d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < dag->n_parents[child]; i++) {
        out[i] = dag->parents[child][i];
    }
    return HDCD_OK;
}

/* Is `target` reachable from `start` by following existing edges
 * forward (start's children, their children, ...)? Used to reject an
 * edge parent->child whenever child can already reach parent, which
 * would close a cycle. */
static int is_reachable(const hdcd_dag_t *dag, size_t start, size_t target) {
    if (start == target) {
        return 1;
    }
    uint8_t *visited = (uint8_t *)calloc(dag->d, sizeof(uint8_t));
    size_t *queue = (size_t *)malloc(dag->d * sizeof(size_t));
    if (visited == NULL || queue == NULL) {
        free(visited);
        free(queue);
        return 1; /* fail safe: treat as reachable so the edge is rejected */
    }

    size_t qhead = 0, qtail = 0;
    queue[qtail++] = start;
    visited[start] = 1;
    int found = 0;

    while (qhead < qtail && !found) {
        size_t x = queue[qhead++];
        for (size_t y = 0; y < dag->d && !found; y++) {
            if (visited[y]) {
                continue;
            }
            for (size_t p = 0; p < dag->n_parents[y]; p++) {
                if (dag->parents[y][p] == x) {
                    if (y == target) {
                        found = 1;
                    }
                    visited[y] = 1;
                    queue[qtail++] = y;
                    break;
                }
            }
        }
    }

    free(visited);
    free(queue);
    return found;
}

hdcd_status_t hdcd_dag_add_edge(hdcd_dag_t *dag, size_t parent, size_t child) {
    if (dag == NULL || parent == child || parent >= dag->d || child >= dag->d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (hdcd_dag_has_edge(dag, parent, child)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (dag->n_parents[child] >= dag->k_max) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (is_reachable(dag, child, parent)) {
        return HDCD_ERROR_INVALID_ARGUMENT; /* would create a cycle */
    }

    if (dag->n_parents[child] == dag->capacity[child]) {
        size_t new_cap = (dag->capacity[child] == 0) ? 4 : dag->capacity[child] * 2;
        size_t *grown = (size_t *)realloc(dag->parents[child], new_cap * sizeof(size_t));
        if (grown == NULL) {
            return HDCD_ERROR_ALLOCATION;
        }
        dag->parents[child] = grown;
        dag->capacity[child] = new_cap;
    }

    /* Insert in ascending order. */
    size_t pos = dag->n_parents[child];
    while (pos > 0 && dag->parents[child][pos - 1] > parent) {
        dag->parents[child][pos] = dag->parents[child][pos - 1];
        pos--;
    }
    dag->parents[child][pos] = parent;
    dag->n_parents[child]++;

    return HDCD_OK;
}

hdcd_status_t hdcd_dag_remove_edge(hdcd_dag_t *dag, size_t parent, size_t child) {
    if (dag == NULL || parent >= dag->d || child >= dag->d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    size_t *list = dag->parents[child];
    size_t count = dag->n_parents[child];
    for (size_t i = 0; i < count; i++) {
        if (list[i] == parent) {
            for (size_t j = i; j + 1 < count; j++) {
                list[j] = list[j + 1];
            }
            dag->n_parents[child]--;
            return HDCD_OK;
        }
    }
    return HDCD_OK; /* already absent: no-op */
}

hdcd_status_t hdcd_dag_topological_order(const hdcd_dag_t *dag, size_t *order_out) {
    if (dag == NULL || order_out == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    size_t d = dag->d;

    size_t *indegree = (size_t *)malloc(d * sizeof(size_t));
    size_t **children = (size_t **)calloc(d, sizeof(size_t *));
    size_t *children_count = (size_t *)calloc(d, sizeof(size_t));
    size_t *children_cap = (size_t *)calloc(d, sizeof(size_t));
    size_t *queue = (size_t *)malloc(d * sizeof(size_t));
    if (indegree == NULL || children == NULL || children_count == NULL
        || children_cap == NULL || queue == NULL) {
        free(indegree);
        if (children != NULL) {
            for (size_t i = 0; i < d; i++) free(children[i]);
        }
        free(children);
        free(children_count);
        free(children_cap);
        free(queue);
        return HDCD_ERROR_ALLOCATION;
    }

    for (size_t c = 0; c < d; c++) {
        indegree[c] = dag->n_parents[c];
    }

    hdcd_status_t status = HDCD_OK;
    for (size_t c = 0; c < d && status == HDCD_OK; c++) {
        for (size_t i = 0; i < dag->n_parents[c]; i++) {
            size_t p = dag->parents[c][i];
            if (children_count[p] == children_cap[p]) {
                size_t new_cap = (children_cap[p] == 0) ? 4 : children_cap[p] * 2;
                size_t *grown = (size_t *)realloc(children[p], new_cap * sizeof(size_t));
                if (grown == NULL) {
                    status = HDCD_ERROR_ALLOCATION;
                    break;
                }
                children[p] = grown;
                children_cap[p] = new_cap;
            }
            children[p][children_count[p]++] = c;
        }
    }

    size_t out_count = 0;
    if (status == HDCD_OK) {
        size_t qhead = 0, qtail = 0;
        for (size_t x = 0; x < d; x++) {
            if (indegree[x] == 0) {
                queue[qtail++] = x;
            }
        }
        while (qhead < qtail) {
            size_t x = queue[qhead++];
            order_out[out_count++] = x;
            for (size_t i = 0; i < children_count[x]; i++) {
                size_t y = children[x][i];
                indegree[y]--;
                if (indegree[y] == 0) {
                    queue[qtail++] = y;
                }
            }
        }
    }

    free(indegree);
    for (size_t i = 0; i < d; i++) {
        free(children[i]);
    }
    free(children);
    free(children_count);
    free(children_cap);
    free(queue);

    if (status != HDCD_OK) {
        return status;
    }
    if (out_count != d) {
        return HDCD_ERROR_NUMERICAL; /* cycle detected */
    }
    return HDCD_OK;
}
