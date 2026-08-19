#include "hdcd/annealing.h"
#include "../dag/cache.h"
#include "../dag/proposals.h"

#include <math.h>
#include <stdlib.h>

struct hdcd_annealing_result {
    hdcd_dag_t *best_dag;
    double best_score;
    hdcd_dag_t *current_dag;
    double current_score;
    size_t n_iterations;
    double *score_trace;
    uint8_t *accepted_trace;
    double acceptance_rate;
};

static hdcd_status_t compute_node_score(
    hdcd_local_fit_cache_t *cache,
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_annealing_options_t *options,
    double *out_score
) {
    const hdcd_local_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_internal_cache_get_or_fit(
        cache, u, mask, n, d, child, parents, n_parents, &options->local_fit_options, &fit, NULL
    );
    if (fit == NULL) {
        return status;
    }
    double k_hat = -hdcd_local_fit_holdout_score(fit);
    double edge_penalty = options->lambda_edge * (double)n_parents;
    double roughness = options->local_fit_options.lambda_roughness * hdcd_local_fit_roughness_penalty(fit);
    *out_score = k_hat + edge_penalty + roughness;
    return status;
}

/* Build child's parent set AFTER applying `prop` (without mutating `dag`). */
static hdcd_status_t build_new_parent_set(
    const hdcd_dag_t *dag, const hdcd_proposal_t *prop, size_t **out_parents, size_t *out_n
) {
    size_t child = prop->child;
    size_t old_n = hdcd_dag_n_parents(dag, child);
    size_t *old_parents = NULL;
    if (old_n > 0) {
        old_parents = (size_t *)malloc(old_n * sizeof(size_t));
        if (old_parents == NULL) {
            return HDCD_ERROR_ALLOCATION;
        }
        hdcd_dag_parents(dag, child, old_parents);
    }

    size_t max_new = old_n + 1;
    size_t *new_parents = (size_t *)malloc(max_new * sizeof(size_t));
    if (new_parents == NULL) {
        free(old_parents);
        return HDCD_ERROR_ALLOCATION;
    }
    size_t new_n = 0;
    for (size_t i = 0; i < old_n; i++) {
        if (prop->kind != HDCD_PROPOSAL_ADD && old_parents[i] == prop->remove_parent) {
            continue; /* dropped by REMOVE or SWAP */
        }
        new_parents[new_n++] = old_parents[i];
    }
    if (prop->kind != HDCD_PROPOSAL_REMOVE) {
        new_parents[new_n++] = prop->add_parent;
    }

    free(old_parents);
    *out_parents = new_parents;
    *out_n = new_n;
    return HDCD_OK;
}

static hdcd_status_t apply_proposal(hdcd_dag_t *dag, const hdcd_proposal_t *prop) {
    hdcd_status_t status = HDCD_OK;
    if (prop->kind == HDCD_PROPOSAL_ADD || prop->kind == HDCD_PROPOSAL_SWAP) {
        if (prop->kind == HDCD_PROPOSAL_SWAP) {
            status = hdcd_dag_remove_edge(dag, prop->remove_parent, prop->child);
            if (status != HDCD_OK) return status;
        }
        status = hdcd_dag_add_edge(dag, prop->add_parent, prop->child);
    } else {
        status = hdcd_dag_remove_edge(dag, prop->remove_parent, prop->child);
    }
    return status;
}

static hdcd_status_t validate_options(
    size_t d, const hdcd_annealing_options_t *options
) {
    if (options == NULL || options->ordering == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(options->initial_temperature > 0.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (!(options->cooling_rate > 0.0) || !(options->cooling_rate < 1.0)) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (options->max_iterations == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (options->p_add <= 0.0 && options->p_remove <= 0.0 && options->p_swap <= 0.0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }

    uint8_t *seen = (uint8_t *)calloc(d, sizeof(uint8_t));
    if (seen == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    hdcd_status_t status = HDCD_OK;
    for (size_t i = 0; i < d; i++) {
        size_t v = options->ordering[i];
        if (v >= d || seen[v]) {
            status = HDCD_ERROR_INVALID_ARGUMENT;
            break;
        }
        seen[v] = 1;
    }
    free(seen);
    return status;
}

hdcd_status_t hdcd_run_annealing(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    const hdcd_annealing_options_t *options,
    hdcd_annealing_result_t **out
) {
    if (u == NULL || mask == NULL || out == NULL || n == 0 || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    hdcd_status_t status = validate_options(d, options);
    if (status != HDCD_OK) {
        return status;
    }
    *out = NULL;

    size_t n_restarts = (options->restarts == 0) ? 1 : options->restarts;
    size_t capacity = n_restarts * options->max_iterations;

    hdcd_annealing_result_t *result = (hdcd_annealing_result_t *)calloc(1, sizeof(hdcd_annealing_result_t));
    if (result == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    result->score_trace = (double *)malloc(capacity * sizeof(double));
    result->accepted_trace = (uint8_t *)malloc(capacity * sizeof(uint8_t));
    if (result->score_trace == NULL || result->accepted_trace == NULL) {
        hdcd_annealing_result_free(result);
        return HDCD_ERROR_ALLOCATION;
    }
    result->best_score = INFINITY;

    hdcd_local_fit_cache_t *cache = NULL;
    status = hdcd_internal_cache_create(d, &cache);
    if (status != HDCD_OK) {
        hdcd_annealing_result_free(result);
        return status;
    }

    double *node_score = (double *)malloc(d * sizeof(double));
    if (node_score == NULL) {
        hdcd_internal_cache_free(cache);
        hdcd_annealing_result_free(result);
        return HDCD_ERROR_ALLOCATION;
    }

    size_t global_iter = 0;
    hdcd_status_t hard_failure = HDCD_OK;

    for (size_t restart = 0; restart < n_restarts && hard_failure == HDCD_OK; restart++) {
        hdcd_dag_t *current_dag = NULL;
        if (options->initial_dag != NULL) {
            status = hdcd_dag_clone(options->initial_dag, &current_dag);
        } else {
            status = hdcd_dag_create(d, options->k_max, &current_dag);
        }
        if (status != HDCD_OK) {
            hard_failure = status;
            break;
        }

        hdcd_rng_t rng;
        hdcd_rng_seed(&rng, options->seed + (uint64_t)restart * 0x9E3779B97F4A7C15ULL);

        double current_score = 0.0;
        for (size_t j = 0; j < d && hard_failure == HDCD_OK; j++) {
            size_t np = hdcd_dag_n_parents(current_dag, j);
            size_t *parents = NULL;
            if (np > 0) {
                parents = (size_t *)malloc(np * sizeof(size_t));
                if (parents == NULL) {
                    hard_failure = HDCD_ERROR_ALLOCATION;
                    break;
                }
                hdcd_dag_parents(current_dag, j, parents);
            }
            double score;
            status = compute_node_score(cache, u, mask, n, d, j, parents, np, options, &score);
            free(parents);
            if (status != HDCD_OK && status != HDCD_ERROR_NOT_CONVERGED) {
                hard_failure = status;
                break;
            }
            node_score[j] = score;
            current_score += score;
        }
        if (hard_failure != HDCD_OK) {
            hdcd_dag_free(current_dag);
            break;
        }

        double temperature = options->initial_temperature;

        for (size_t iter = 0; iter < options->max_iterations; iter++) {
            hdcd_proposal_t prop;
            int has_proposal = hdcd_internal_propose_move(
                &rng, current_dag, options->ordering, d, options->k_max,
                options->p_add, options->p_remove, options->p_swap, &prop
            );
            if (!has_proposal) {
                break; /* search space exhausted from this state; further iterations cannot change it */
            }

            size_t *new_parents = NULL;
            size_t new_n = 0;
            status = build_new_parent_set(current_dag, &prop, &new_parents, &new_n);
            if (status != HDCD_OK) {
                hard_failure = status;
                break;
            }

            double new_node_score;
            status = compute_node_score(cache, u, mask, n, d, prop.child, new_parents, new_n, options, &new_node_score);
            free(new_parents);
            if (status != HDCD_OK && status != HDCD_ERROR_NOT_CONVERGED) {
                hard_failure = status;
                break;
            }

            double delta = new_node_score - node_score[prop.child];
            int accept;
            if (delta <= 0.0) {
                accept = 1;
            } else {
                double accept_prob = exp(-delta / temperature);
                accept = (hdcd_rng_uniform(&rng) < accept_prob);
            }

            if (accept) {
                status = apply_proposal(current_dag, &prop);
                if (status != HDCD_OK) {
                    hard_failure = status;
                    break;
                }
                current_score += delta;
                node_score[prop.child] = new_node_score;
            }

            result->score_trace[global_iter] = current_score;
            result->accepted_trace[global_iter] = accept ? 1 : 0;
            global_iter++;

            if (current_score < result->best_score) {
                result->best_score = current_score;
                hdcd_dag_t *snapshot = NULL;
                status = hdcd_dag_clone(current_dag, &snapshot);
                if (status != HDCD_OK) {
                    hard_failure = status;
                    break;
                }
                hdcd_dag_free(result->best_dag);
                result->best_dag = snapshot;
            }

            temperature *= options->cooling_rate;
        }

        if (restart + 1 == n_restarts || hard_failure != HDCD_OK) {
            hdcd_dag_free(result->current_dag);
            result->current_dag = current_dag;
            result->current_score = current_score;
        } else {
            hdcd_dag_free(current_dag);
        }
    }

    free(node_score);
    hdcd_internal_cache_free(cache);

    if (hard_failure != HDCD_OK) {
        hdcd_annealing_result_free(result);
        return hard_failure;
    }

    result->n_iterations = global_iter;
    size_t accepted_count = 0;
    for (size_t i = 0; i < global_iter; i++) {
        if (result->accepted_trace[i]) accepted_count++;
    }
    result->acceptance_rate = (global_iter > 0) ? (double)accepted_count / (double)global_iter : 0.0;

    if (result->best_dag == NULL) {
        /* Every proposal attempt failed to even start (e.g. d==1): the
         * initial graph IS the best (and only) graph seen. */
        status = hdcd_dag_clone(result->current_dag, &result->best_dag);
        if (status != HDCD_OK) {
            hdcd_annealing_result_free(result);
            return status;
        }
        result->best_score = result->current_score;
    }

    *out = result;
    return HDCD_OK;
}

void hdcd_annealing_result_free(hdcd_annealing_result_t *result) {
    if (result == NULL) {
        return;
    }
    hdcd_dag_free(result->best_dag);
    hdcd_dag_free(result->current_dag);
    free(result->score_trace);
    free(result->accepted_trace);
    free(result);
}

const hdcd_dag_t *hdcd_annealing_best_dag(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->best_dag : NULL;
}

double hdcd_annealing_best_score(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->best_score : NAN;
}

const hdcd_dag_t *hdcd_annealing_current_dag(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->current_dag : NULL;
}

double hdcd_annealing_current_score(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->current_score : NAN;
}

size_t hdcd_annealing_n_iterations(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->n_iterations : 0;
}

double hdcd_annealing_score_trace(const hdcd_annealing_result_t *result, size_t iter) {
    if (result == NULL || iter >= result->n_iterations) {
        return NAN;
    }
    return result->score_trace[iter];
}

int hdcd_annealing_accepted_trace(const hdcd_annealing_result_t *result, size_t iter) {
    if (result == NULL || iter >= result->n_iterations) {
        return 0;
    }
    return result->accepted_trace[iter];
}

double hdcd_annealing_acceptance_rate(const hdcd_annealing_result_t *result) {
    return (result != NULL) ? result->acceptance_rate : NAN;
}
