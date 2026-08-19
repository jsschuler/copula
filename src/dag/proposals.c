#include "proposals.h"

#include <stdlib.h>

static void build_position_array(const size_t *ordering, size_t d, size_t *pos) {
    for (size_t i = 0; i < d; i++) {
        pos[ordering[i]] = i;
    }
}

static size_t pick_random_index(hdcd_rng_t *rng, size_t n) {
    size_t idx = (size_t)(hdcd_rng_uniform(rng) * (double)n);
    if (idx >= n) {
        idx = n - 1;
    }
    return idx;
}

static int has_available_candidate(
    const hdcd_dag_t *dag, const size_t *ordering, const size_t *pos, size_t child
) {
    size_t child_pos = pos[child];
    for (size_t p = 0; p < child_pos; p++) {
        if (!hdcd_dag_has_edge(dag, ordering[p], child)) {
            return 1;
        }
    }
    return 0;
}

/* Candidates for `child`: nodes earlier than `child` in `ordering` (spec
 * section 7) that are not already one of its parents. Writes into
 * out_buf (caller-allocated, length >= pos[child]) and returns the count. */
static size_t collect_available_candidates(
    const hdcd_dag_t *dag, const size_t *ordering, const size_t *pos, size_t child, size_t *out_buf
) {
    size_t count = 0;
    size_t child_pos = pos[child];
    for (size_t p = 0; p < child_pos; p++) {
        size_t candidate = ordering[p];
        if (!hdcd_dag_has_edge(dag, candidate, child)) {
            out_buf[count++] = candidate;
        }
    }
    return count;
}

static int try_add(
    hdcd_rng_t *rng, const hdcd_dag_t *dag, const size_t *ordering, const size_t *pos,
    size_t d, size_t k_max, hdcd_proposal_t *out
) {
    size_t *eligible = (size_t *)malloc(d * sizeof(size_t));
    if (eligible == NULL) {
        return 0;
    }
    size_t n_eligible = 0;
    for (size_t j = 0; j < d; j++) {
        if (hdcd_dag_n_parents(dag, j) >= k_max) {
            continue;
        }
        if (has_available_candidate(dag, ordering, pos, j)) {
            eligible[n_eligible++] = j;
        }
    }
    if (n_eligible == 0) {
        free(eligible);
        return 0;
    }
    size_t child = eligible[pick_random_index(rng, n_eligible)];
    free(eligible);

    size_t *candidates = (size_t *)malloc(pos[child] * sizeof(size_t));
    if (candidates == NULL) {
        return 0;
    }
    size_t n_candidates = collect_available_candidates(dag, ordering, pos, child, candidates);
    size_t parent = candidates[pick_random_index(rng, n_candidates)];
    free(candidates);

    out->kind = HDCD_PROPOSAL_ADD;
    out->child = child;
    out->add_parent = parent;
    return 1;
}

static int try_remove(hdcd_rng_t *rng, const hdcd_dag_t *dag, size_t d, hdcd_proposal_t *out) {
    size_t *eligible = (size_t *)malloc(d * sizeof(size_t));
    if (eligible == NULL) {
        return 0;
    }
    size_t n_eligible = 0;
    for (size_t j = 0; j < d; j++) {
        if (hdcd_dag_n_parents(dag, j) > 0) {
            eligible[n_eligible++] = j;
        }
    }
    if (n_eligible == 0) {
        free(eligible);
        return 0;
    }
    size_t child = eligible[pick_random_index(rng, n_eligible)];
    free(eligible);

    size_t n_parents = hdcd_dag_n_parents(dag, child);
    size_t *parents = (size_t *)malloc(n_parents * sizeof(size_t));
    if (parents == NULL) {
        return 0;
    }
    hdcd_dag_parents(dag, child, parents);
    size_t parent = parents[pick_random_index(rng, n_parents)];
    free(parents);

    out->kind = HDCD_PROPOSAL_REMOVE;
    out->child = child;
    out->remove_parent = parent;
    return 1;
}

static int try_swap(
    hdcd_rng_t *rng, const hdcd_dag_t *dag, const size_t *ordering, const size_t *pos,
    size_t d, hdcd_proposal_t *out
) {
    size_t *eligible = (size_t *)malloc(d * sizeof(size_t));
    if (eligible == NULL) {
        return 0;
    }
    size_t n_eligible = 0;
    for (size_t j = 0; j < d; j++) {
        if (hdcd_dag_n_parents(dag, j) == 0) {
            continue;
        }
        if (has_available_candidate(dag, ordering, pos, j)) {
            eligible[n_eligible++] = j;
        }
    }
    if (n_eligible == 0) {
        free(eligible);
        return 0;
    }
    size_t child = eligible[pick_random_index(rng, n_eligible)];
    free(eligible);

    size_t n_parents = hdcd_dag_n_parents(dag, child);
    size_t *parents = (size_t *)malloc(n_parents * sizeof(size_t));
    if (parents == NULL) {
        return 0;
    }
    hdcd_dag_parents(dag, child, parents);
    size_t remove_parent = parents[pick_random_index(rng, n_parents)];
    free(parents);

    size_t *candidates = (size_t *)malloc(pos[child] * sizeof(size_t));
    if (candidates == NULL) {
        return 0;
    }
    size_t n_candidates = collect_available_candidates(dag, ordering, pos, child, candidates);
    size_t add_parent = candidates[pick_random_index(rng, n_candidates)];
    free(candidates);

    out->kind = HDCD_PROPOSAL_SWAP;
    out->child = child;
    out->remove_parent = remove_parent;
    out->add_parent = add_parent;
    return 1;
}

int hdcd_internal_propose_move(
    hdcd_rng_t *rng,
    const hdcd_dag_t *dag,
    const size_t *ordering, size_t d, size_t k_max,
    double p_add, double p_remove, double p_swap,
    hdcd_proposal_t *out
) {
    if (rng == NULL || dag == NULL || ordering == NULL || out == NULL || d == 0) {
        return 0;
    }

    size_t *pos = (size_t *)malloc(d * sizeof(size_t));
    if (pos == NULL) {
        return 0;
    }
    build_position_array(ordering, d, pos);

    double wa = (p_add > 0.0) ? p_add : 0.0;
    double wr = (p_remove > 0.0) ? p_remove : 0.0;
    double ws = (p_swap > 0.0) ? p_swap : 0.0;
    double total = wa + wr + ws;

    hdcd_proposal_kind_t chosen;
    if (total <= 0.0) {
        chosen = HDCD_PROPOSAL_ADD; /* arbitrary: fallback below tries every kind regardless */
    } else {
        double r = hdcd_rng_uniform(rng) * total;
        if (r < wa) {
            chosen = HDCD_PROPOSAL_ADD;
        } else if (r < wa + wr) {
            chosen = HDCD_PROPOSAL_REMOVE;
        } else {
            chosen = HDCD_PROPOSAL_SWAP;
        }
    }

    hdcd_proposal_kind_t attempt_order[3];
    int n_attempts = 0;
    attempt_order[n_attempts++] = chosen;
    if (chosen != HDCD_PROPOSAL_ADD) attempt_order[n_attempts++] = HDCD_PROPOSAL_ADD;
    if (chosen != HDCD_PROPOSAL_REMOVE) attempt_order[n_attempts++] = HDCD_PROPOSAL_REMOVE;
    if (chosen != HDCD_PROPOSAL_SWAP) attempt_order[n_attempts++] = HDCD_PROPOSAL_SWAP;

    int success = 0;
    for (int i = 0; i < n_attempts && !success; i++) {
        switch (attempt_order[i]) {
            case HDCD_PROPOSAL_ADD:
                success = try_add(rng, dag, ordering, pos, d, k_max, out);
                break;
            case HDCD_PROPOSAL_REMOVE:
                success = try_remove(rng, dag, d, out);
                break;
            case HDCD_PROPOSAL_SWAP:
                success = try_swap(rng, dag, ordering, pos, d, out);
                break;
        }
    }

    free(pos);
    return success;
}
