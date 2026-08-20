#ifndef HDCD_LOCAL_FIT_H
#define HDCD_LOCAL_FIT_H

#include <stddef.h>
#include <stdint.h>
#include "hdcd/status.h"
#include "hdcd/sinkhorn.h"
#include "hdcd/parametric_tail.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hdcd_local_fit_options {
    size_t bernstein_degree;   /* m: Bernstein degree for every edge into this node (spec section 9) */
    double lambda_roughness;    /* lambda_R; must be > 0 when lambda_roughness_grid is NOT
                                  * supplied -- see DECISIONS.md for why this implementation
                                  * requires strict positivity */
    double holdout_fraction;     /* in (0,1): fraction of usable rows held out for scoring (spec section 16) */
    uint64_t seed;                 /* seeds the deterministic train/holdout split */
    size_t theta_max_iterations;    /* inner gradient-ascent budget per edge; 0 selects a default */
    double theta_tol;                /* relative per-iteration objective-improvement tolerance
                                       * for the inner Theta fit (see DECISIONS.md for why this,
                                       * not a raw gradient-norm threshold); 0 selects a default */
    hdcd_sinkhorn_options_t sinkhorn_options; /* forwarded to hdcd_sinkhorn_fit */

    /* Optional: a non-global, PER-NODE learned roughness penalty (spec
     * section 18: "lambda_R is selected by validation unless explicitly
     * provided" / "a small validation grid is sufficient for version
     * 1"), as an alternative to the single fixed `lambda_roughness`
     * above applied identically everywhere.
     *
     * NULL, or lambda_roughness_grid_size == 0 (the default), disables
     * this entirely: behavior is exactly as if this field did not
     * exist, using `lambda_roughness` verbatim. This is the ONLY way
     * these two fields interact with existing callers -- fully
     * backward compatible.
     *
     * When non-empty: hdcd_local_fit_node further splits its TRAIN rows
     * (never the outer holdout) into an inner train/validation pair
     * (see roughness_validation_fraction below), fits every candidate
     * in the grid on the inner-train rows, scores each on the inner-
     * validation rows, and only then refits the winning lambda on the
     * FULL train rows before scoring on the untouched outer holdout
     * exactly as usual. The outer holdout -- the score used for
     * hdcd_dag_fit_kl_estimate/hdcd_dag_fit_kl_difference and, when this
     * options struct is reused there, for the annealing objective -- is
     * therefore never used to pick lambda, avoiding any lookahead bias
     * from penalty selection into model comparison.
     *
     * Per spec section 18 ("Penalty selection must occur outside the
     * inner annealing loop. Do not nest an expensive continuous penalty
     * search inside every graph proposal."): hdcd_run_annealing reuses
     * one hdcd_local_fit_options_t verbatim on every proposal, so
     * enabling the grid there would repeat this search on every
     * distinct parent set the search tries. This is intentionally left
     * to the caller to avoid, not forbidden by this struct -- restrict
     * grid search to hdcd_dag_fit calls made on an already-decided DAG
     * (the reference DAG the annealing search converged on, or an
     * explicit candidate DAG), where it runs once per node. */
    const double *lambda_roughness_grid;
    size_t lambda_roughness_grid_size;
    double roughness_validation_fraction; /* in (0,1): fraction of TRAIN rows further
                                            * reserved for inner grid validation; only
                                            * used when the grid is non-empty; 0 selects
                                            * a default (0.3) */

    /* Optional: a non-global, PER-NODE learned Bernstein degree, gated by
     * an empirical tail-dependence diagnostic (see hdcd/tail_dependence.h;
     * DECISIONS.md's "tail-dependence-informed bernstein_degree
     * selection" entry). Raising `bernstein_degree` is the natural next
     * lever once `lambda_roughness_grid` above has been ruled out as the
     * fix for a sharply-peaked (Archimedean tail-dependent) edge's
     * under-fit corner -- but searching it unconditionally on every node
     * wastes the validation budget on edges (Gaussian, Frank) that gain
     * nothing from extra polynomial flexibility. This gates the search
     * per node on whether the node's data actually shows tail
     * dependence.
     *
     * NULL, or bernstein_degree_grid_size == 0 (the default), disables
     * this entirely: `bernstein_degree` is used verbatim, exactly as if
     * this field did not exist -- fully backward compatible.
     *
     * When non-empty: for each of the node's parent edges,
     * hdcd_tail_dependence_coefficient() is estimated between that
     * parent and the child over ALL of O_j(P_j) (this is a descriptive
     * diagnostic that decides SEARCH STRATEGY, not a fitted model
     * parameter, so unlike lambda/degree selection itself there is no
     * leakage concern in using the full usable set, train and holdout
     * rows both). The node's MAX coefficient across its parent edges
     * (hdcd_local_fit_max_tail_dependence()) is compared against
     * `tail_dependence_gate`:
     *   - if the max coefficient is >= the gate, bernstein_degree_grid
     *     is searched (jointly with lambda_roughness_grid, if that is
     *     ALSO non-empty -- the full cross product of both grids; if
     *     lambda_roughness_grid is empty, degree is searched alone with
     *     lambda_roughness held fixed) via the SAME inner train/
     *     validation split lambda_roughness_grid above uses;
     *   - otherwise bernstein_degree is used verbatim for this node (the
     *     search is skipped, at zero extra cost) even though the grid
     *     was supplied.
     * `tail_dependence_gate` in [0,1]; 0 (the default when the degree
     * grid is non-empty) means "always search," i.e. no gating. */
    const size_t *bernstein_degree_grid;
    size_t bernstein_degree_grid_size;
    double tail_dependence_gate;
    size_t tail_dependence_k; /* forwarded to hdcd_tail_dependence_coefficient; 0 selects its own default */

    /* Optional: anisotropic (corner-relaxed) roughness penalty (see
     * hdcd/bernstein.h's hdcd_bernstein_roughness_penalty_weighted and
     * DECISIONS.md's "anisotropic (corner-relaxed) roughness penalty"
     * entry) -- applied to EVERY edge fit by hdcd_local_fit_node
     * (including inside a lambda_roughness_grid/bernstein_degree_grid
     * search, if either is active). `corner_relief = 0` (the default)
     * recovers the original uniform roughness penalty exactly --
     * behavior is unchanged unless this field is set. Must be in
     * [0, 1). This is a FIXED scalar for v1, not itself searched by a
     * grid (unlike lambda_roughness/bernstein_degree above) -- see
     * DECISIONS.md for why, and for the logged follow-up of extending
     * it to one. */
    double corner_relief;

    /* Optional: a copula-level EVT tail-splice (see DECISIONS.md's
     * "copula-level EVT tail-splice" entry) -- the heaviest of the
     * interventions tried for the Clayton/Gumbel corner under-fit, and
     * the only one that can represent an actual corner SINGULARITY: a
     * degree-m Bernstein tensor is bounded at every degree and no amount
     * of retuning `lambda_roughness`/`corner_relief`/`bernstein_degree`
     * changes that. This blends in a genuine parametric Archimedean
     * copula (Clayton for lower-tail dependence, Gumbel for upper --
     * see hdcd/parametric_tail.h) near the corner instead.
     *
     * `evt_splice_gate = 0` (the default) disables this entirely:
     * behavior is exactly as if these fields did not exist -- fully
     * backward compatible.
     *
     * When > 0: for each parent edge, hdcd_tail_dependence_coefficient()
     * is estimated exactly as for bernstein_degree_grid's gate (over ALL
     * of O_j(P_j), train and holdout both -- a descriptive diagnostic,
     * not a fitted parameter). If the larger of that edge's
     * (lambda_upper, lambda_lower) is >= `evt_splice_gate`, a Clayton
     * (lambda_lower larger) or Gumbel (lambda_upper larger) copula is
     * fit by full-sample MLE (hdcd_tail_family_fit) on that edge's TRAIN
     * rows ONCE per node -- independent of, and computed before, any
     * lambda_roughness_grid/bernstein_degree_grid search, since the
     * parametric fit does not depend on Theta's own degree/lambda and
     * would otherwise be uselessly refit identically for every grid
     * candidate. That edge's raw kernel then becomes a per-(u,z) BLEND
     * of the Bernstein tensor term and the parametric density (see
     * DECISIONS.md for the exact blending weight and why it needs no
     * hand-enforced continuity: hdcd_sinkhorn_fit already turns any
     * positive raw kernel into a valid copula-preserving density,
     * blended or not). Gated-out edges (below the gate, or when the grid
     * is empty) use the plain Bernstein kernel unmodified, at zero extra
     * cost.
     *
     * UNLIKE `corner_relief`, this is deliberately excluded from
     * hdcd_run_annealing, the same way lambda_roughness_grid/
     * bernstein_degree_grid are (spec section 18): a per-node tail-
     * dependence-coefficient estimate plus a golden-section MLE fit is
     * real, non-negligible per-node cost (unlike corner_relief's cheap
     * weight computation), and annealing calls compute_node_score for
     * every distinct parent set a proposal tries -- so this is meant for
     * hdcd_dag_fit calls on an already-decided DAG (the reference DAG
     * the annealing search converged on using the plain Bernstein
     * kernel, or an explicit candidate DAG), where it runs once per
     * node, not the search itself. A FIXED scalar for v1 either way
     * (bandwidth and gate are not grid-searched), matching
     * `corner_relief`'s v1 scope. */
    double evt_splice_gate;      /* in [0,1]; 0 disables the splice entirely */
    double evt_splice_bandwidth; /* corner-bump width in copula-scale units; 0 selects a default (0.15) */
} hdcd_local_fit_options_t;

/* One node's fitted conditional copula factor c_j(u_j | u_Pa(j)). */
typedef struct hdcd_local_fit hdcd_local_fit_t;

/*
 * Fit node `child`'s conditional copula factor given its parent set
 * (spec sections 8, 9, 11, 16, 28):
 *   1. collect O_j(P_j): rows where child and every parent are observed;
 *   2. deterministically (seeded) split O_j(P_j) into train/holdout;
 *   3. fit one (m+1)x(m+1) Theta per parent by gradient ascent on the
 *      roughness-penalized raw-kernel objective on the TRAIN rows
 *      (see DECISIONS.md for why Theta-fitting and Sinkhorn
 *      normalization are two separate sequential steps, not one joint
 *      optimization -- this mirrors spec section 28's literal step
 *      ordering: "optimize edge coefficient matrices" then "apply
 *      Sinkhorn normalization");
 *   4. Sinkhorn-normalize the resulting raw kernel using the TRAIN
 *      rows' own parent values as the Monte Carlo z-samples (spec
 *      section 11.2 -- for this node, q_j(z) IS exactly the empirical
 *      distribution of its training parent values, so no separate
 *      sampler is needed or built here);
 *   5. score the fitted, normalized c_j on the HOLDOUT rows only (spec
 *      section 16's normalized held-out score bar-ell_j(P_j)).
 *
 * `u`/`mask` are the full copula-scale dataset, n x d, COLUMN-MAJOR
 * (spec section 23): u[col*n + row]. `parents` (length n_parents) need
 * not be sorted and are not reordered; child must not appear in it and
 * it must not contain duplicates.
 *
 * n_parents == 0 (a root node) is handled trivially per spec section
 * 12's base case p_1(u_1) = 1: c_j(u) = 1 identically, no Theta or
 * Sinkhorn fitting is performed, and the holdout score is exactly 0.
 *
 * On non-convergence of either the Theta fit or the Sinkhorn fit,
 * returns HDCD_ERROR_NOT_CONVERGED but *out is still populated with the
 * best-effort fitted state (spec section 24: fail clearly, not silently).
 */
hdcd_status_t hdcd_local_fit_node(
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_local_fit_options_t *options,
    hdcd_local_fit_t **out
);

void hdcd_local_fit_free(hdcd_local_fit_t *fit);

size_t hdcd_local_fit_n_parents(const hdcd_local_fit_t *fit);

/* Parent indices, in the exact order hdcd_local_fit_log_density expects
 * its z[] argument. Pointer valid for the lifetime of `fit`. */
const size_t *hdcd_local_fit_parent_order(const hdcd_local_fit_t *fit);

size_t hdcd_local_fit_n_observed(const hdcd_local_fit_t *fit); /* |O_j(P_j)| */
size_t hdcd_local_fit_n_train(const hdcd_local_fit_t *fit);
size_t hdcd_local_fit_n_holdout(const hdcd_local_fit_t *fit);   /* effective sample size for the held-out score */

double hdcd_local_fit_holdout_score(const hdcd_local_fit_t *fit);      /* bar-ell_j(P_j) */
double hdcd_local_fit_roughness_penalty(const hdcd_local_fit_t *fit);   /* sum_k R(Theta_jk) */

/* The lambda_roughness actually used for this node's fitted Theta:
 * options->lambda_roughness verbatim when options->lambda_roughness_grid
 * was empty/NULL, or the grid candidate selected by inner-validation
 * when it was not. NAN for a root node (nothing to select) or a NULL
 * fit. */
double hdcd_local_fit_selected_lambda_roughness(const hdcd_local_fit_t *fit);

/* The Bernstein degree actually used for this node's fitted Theta:
 * options->bernstein_degree verbatim when options->bernstein_degree_grid
 * was empty/NULL or the tail-dependence gate was not met, or the grid
 * candidate selected by inner-validation otherwise. 0 for a root node
 * (nothing to select) or a NULL fit. */
size_t hdcd_local_fit_selected_bernstein_degree(const hdcd_local_fit_t *fit);

/* This node's maximum empirical tail-dependence coefficient across its
 * parent edges (the diagnostic that gated the bernstein_degree_grid
 * search -- see hdcd_local_fit_options_t). NAN if bernstein_degree_grid
 * was empty/NULL (the diagnostic is not computed at all when it would
 * not be used), for a root node, or a NULL fit. */
double hdcd_local_fit_max_tail_dependence(const hdcd_local_fit_t *fit);

/* The parametric family spliced onto parent edge `parent_idx` (in
 * hdcd_local_fit_parent_order()'s order): HDCD_TAIL_FAMILY_NONE if
 * evt_splice_gate was 0/disabled, or that edge's tail-dependence
 * coefficient did not clear the gate. HDCD_TAIL_FAMILY_NONE for an
 * out-of-range parent_idx or a NULL fit. */
hdcd_tail_family_t hdcd_local_fit_tail_family(const hdcd_local_fit_t *fit, size_t parent_idx);

/* The fitted parametric theta for parent edge `parent_idx`. NAN if that
 * edge has no active splice (hdcd_local_fit_tail_family() == NONE), for
 * an out-of-range parent_idx, or a NULL fit. */
double hdcd_local_fit_tail_theta(const hdcd_local_fit_t *fit, size_t parent_idx);

int hdcd_local_fit_theta_converged(const hdcd_local_fit_t *fit);
int hdcd_local_fit_sinkhorn_converged(const hdcd_local_fit_t *fit); /* 1 for a root node (nothing to converge) */

/*
 * log c_j(u | z). `z` must be given in hdcd_local_fit_parent_order()'s
 * order, length hdcd_local_fit_n_parents(fit). For a root node, z_dim
 * must be 0 and the result is always 0 (c_j(u) = 1).
 */
hdcd_status_t hdcd_local_fit_log_density(
    const hdcd_local_fit_t *fit,
    double u, const double *z, size_t z_dim,
    double *out
);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_LOCAL_FIT_H */
