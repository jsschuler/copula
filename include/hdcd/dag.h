#ifndef HDCD_DAG_H
#define HDCD_DAG_H

#include <stddef.h>
#include "hdcd/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A directed acyclic graph over d nodes with a hard per-node parent-set
 * size limit k_max (spec section 7). Acyclicity is enforced
 * INCREMENTALLY: hdcd_dag_add_edge rejects any edge that would create a
 * cycle (checked via reachability), so a hdcd_dag_t built exclusively
 * through this API is always acyclic by construction.
 * hdcd_dag_topological_order is still provided as a general, independent
 * validator (spec section 19 will need to validate externally-supplied
 * DAGs too, once that milestone exists).
 */
typedef struct hdcd_dag hdcd_dag_t;

hdcd_status_t hdcd_dag_create(size_t d, size_t k_max, hdcd_dag_t **out);
void hdcd_dag_free(hdcd_dag_t *dag);

size_t hdcd_dag_dim(const hdcd_dag_t *dag);
size_t hdcd_dag_k_max(const hdcd_dag_t *dag);

/*
 * Add edge parent -> child (parent becomes one of child's parents).
 * Rejected (HDCD_ERROR_INVALID_ARGUMENT) if: parent == child; parent or
 * child is out of range; the edge already exists; child already has
 * k_max parents; or the edge would create a cycle (i.e. child can
 * already reach parent).
 */
hdcd_status_t hdcd_dag_add_edge(hdcd_dag_t *dag, size_t parent, size_t child);

/* Removes the edge if present; a no-op (HDCD_OK) if it was already absent. */
hdcd_status_t hdcd_dag_remove_edge(hdcd_dag_t *dag, size_t parent, size_t child);

int hdcd_dag_has_edge(const hdcd_dag_t *dag, size_t parent, size_t child);

size_t hdcd_dag_n_parents(const hdcd_dag_t *dag, size_t child);

/* Writes child's parents, in ascending index order, into out (caller-
 * allocated, length >= hdcd_dag_n_parents(dag, child)). */
hdcd_status_t hdcd_dag_parents(const hdcd_dag_t *dag, size_t child, size_t *out);

/*
 * General acyclicity validator via Kahn's algorithm, independent of how
 * the DAG was constructed. On success, order_out (length d) receives a
 * valid topological order. Returns HDCD_ERROR_NUMERICAL if the graph
 * contains a cycle (should be unreachable for a DAG built solely via
 * hdcd_dag_add_edge, but this function does not assume that).
 */
hdcd_status_t hdcd_dag_topological_order(const hdcd_dag_t *dag, size_t *order_out);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_DAG_H */
