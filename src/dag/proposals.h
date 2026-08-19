#ifndef HDCD_INTERNAL_PROPOSALS_H
#define HDCD_INTERNAL_PROPOSALS_H

#include <stddef.h>
#include "hdcd/dag.h"
#include "hdcd/rng.h"

/* Version 1 DAG proposal kinds (spec section 7). */
typedef enum {
    HDCD_PROPOSAL_ADD,
    HDCD_PROPOSAL_REMOVE,
    HDCD_PROPOSAL_SWAP
} hdcd_proposal_kind_t;

typedef struct {
    hdcd_proposal_kind_t kind;
    size_t child;
    size_t add_parent;     /* valid for ADD and SWAP */
    size_t remove_parent;  /* valid for REMOVE and SWAP */
} hdcd_proposal_t;

/*
 * Randomly draw ONE valid proposal. Candidate parents for `child` are
 * restricted to nodes strictly earlier than `child` in `ordering`
 * (spec section 7: Pa(pi_j) subseteq {pi_1,...,pi_{j-1}}) -- this alone
 * guarantees every proposal is acyclic, without needing a reachability
 * check per proposal (hdcd_dag_add_edge would reject a cycle anyway,
 * but under this restriction it never has to).
 *
 * The requested move kind is chosen by the normalized weights
 * p_add/p_remove/p_swap; if that kind has no valid instance (e.g. ADD
 * when every node is already at k_max, or REMOVE on an empty graph),
 * this falls back to trying the other kinds (fixed order: ADD, REMOVE,
 * SWAP) before giving up. Returns 1 and fills *out on success, 0 if no
 * proposal of ANY kind is currently possible.
 */
int hdcd_internal_propose_move(
    hdcd_rng_t *rng,
    const hdcd_dag_t *dag,
    const size_t *ordering, size_t d, size_t k_max,
    double p_add, double p_remove, double p_swap,
    hdcd_proposal_t *out
);

#endif /* HDCD_INTERNAL_PROPOSALS_H */
