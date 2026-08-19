#ifndef HDCD_ANNEALING_H
#define HDCD_ANNEALING_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"
#include "hdcd/dag.h"
#include "hdcd/local_fit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hdcd_annealing_options {
    size_t k_max;                   /* hard per-node parent limit (spec section 7) */
    double lambda_edge;              /* lambda_E, soft edge-count penalty (spec section 17) */
    const size_t *ordering;           /* length d: candidate parents for node ordering[j] are
                                        * restricted to {ordering[0],...,ordering[j-1]} (spec
                                        * section 7); typically hdcd_topology_ordering's output */
    hdcd_local_fit_options_t local_fit_options; /* reused for every local fit during search
                                                  * (lambda_roughness lives here, spec section 17) */

    double initial_temperature;        /* T_0 > 0 */
    double cooling_rate;                /* geometric decay per iteration: T_t = T_0 * cooling_rate^t; in (0,1) */
    size_t max_iterations;               /* per restart */
    size_t restarts;                      /* independent annealing runs (best-of); 0 or 1 = single run */

    double p_add, p_remove, p_swap;        /* proposal weights (spec section 7); need not sum to 1 */
    uint64_t seed;                           /* seeds proposal selection and Metropolis draws (spec section 24/36 rule 14) */

    const hdcd_dag_t *initial_dag;            /* optional starting graph; NULL starts from the empty (all-roots) graph */
} hdcd_annealing_options_t;

/* Result of a simulated-annealing DAG search (spec section 17). */
typedef struct hdcd_annealing_result hdcd_annealing_result_t;

/*
 * Run simulated annealing over admissible DAGs (spec sections 7, 17):
 * minimize J(G) = sum_j J_j(Pa(j)), J_j(P) = K_hat_j(P) + lambda_E|P| +
 * lambda_R * sum_{k in P} R(Theta_jk), where K_hat_j(P) is estimated as
 * the NEGATIVE of the held-out normalized local score
 * (hdcd_local_fit_holdout_score) already computed by Milestone 7's
 * local-fit pipeline. Each proposal (add/remove/swap one parent, spec
 * section 7) only ever changes one node's parent set, so only that
 * node is refit or (spec section 17.3) looked up from the internal
 * cache of already-fitted (child, parent-set) results.
 *
 * Fully deterministic given options->seed: identical inputs and options
 * reproduce an identical search trace, accepted/rejected decisions, and
 * final result (spec section 24/29.10).
 */
hdcd_status_t hdcd_run_annealing(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    const hdcd_annealing_options_t *options,
    hdcd_annealing_result_t **out
);

void hdcd_annealing_result_free(hdcd_annealing_result_t *result);

/* Best-scoring DAG found across the whole search (all restarts). Borrowed. */
const hdcd_dag_t *hdcd_annealing_best_dag(const hdcd_annealing_result_t *result);
double hdcd_annealing_best_score(const hdcd_annealing_result_t *result);

/* Final current DAG/score at the end of the search (may differ from best-so-far). Borrowed. */
const hdcd_dag_t *hdcd_annealing_current_dag(const hdcd_annealing_result_t *result);
double hdcd_annealing_current_score(const hdcd_annealing_result_t *result);

size_t hdcd_annealing_n_iterations(const hdcd_annealing_result_t *result); /* total across all restarts */

/* J(G) after iteration `iter` (whether that iteration's proposal was
 * accepted or not), 0-indexed, length hdcd_annealing_n_iterations(). */
double hdcd_annealing_score_trace(const hdcd_annealing_result_t *result, size_t iter);

/* 1 if iteration `iter`'s proposal was accepted, 0 otherwise. */
int hdcd_annealing_accepted_trace(const hdcd_annealing_result_t *result, size_t iter);

double hdcd_annealing_acceptance_rate(const hdcd_annealing_result_t *result); /* overall fraction accepted */

#ifdef __cplusplus
}
#endif

#endif /* HDCD_ANNEALING_H */
