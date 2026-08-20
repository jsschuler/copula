# High-level R API over the .Call primitives in src/hdcd_r.c (spec
# section 26). Mirrors the Python binding's semantics as closely as
# idiomatic R permits (spec section 26's own wording): the same
# pipeline (marginals -> copula transform -> dependence matrix ->
# persistent-topology ordering -> annealed reference DAG -> DAG fit),
# but exposed as R's own idioms -- an S3 "hdcd_model" object,
# predict.hdcd_model() extending the base predict() generic (spec
# section 26's conceptual predict(model, newdata, type = "logpdf")),
# and fit_dag()/score_dag() as two separate calls rather than one
# object with a .kl_divergence_ attribute (matching spec section 26's
# own conceptual R snippet, which itself differs from section 25's
# Python snippet in exactly this way).

# ---- low-level .Call wrappers, one per hdcd_r_* C function ---------------

.new_marginal <- function(x, sigma_min = -1, sigma_max = -1, tol = 1e-6, max_iter = 200L) {
  .Call("hdcd_r_marginal_fit", as.double(x), as.double(sigma_min), as.double(sigma_max),
        as.double(tol), as.integer(max_iter))
}

.marginal_cdf <- function(marginal_ext, eval_points) {
  .Call("hdcd_r_marginal_cdf", marginal_ext, as.double(eval_points))
}

.marginal_logpdf <- function(marginal_ext, eval_points) {
  .Call("hdcd_r_marginal_logpdf", marginal_ext, as.double(eval_points))
}

.marginal_sigma <- function(marginal_ext) {
  .Call("hdcd_r_marginal_sigma", marginal_ext)
}

.transform_to_copula <- function(marginal_ext, x, epsilon = 0) {
  .Call("hdcd_r_transform_to_copula", marginal_ext, as.double(x), as.double(epsilon))
}

.compute_dependence_matrix <- function(u, n, d) {
  .Call("hdcd_r_compute_dependence_matrix", u, as.integer(n), as.integer(d))
}

.dependence_matrix_dense <- function(dm_ext, d) {
  .Call("hdcd_r_dependence_matrix_dense", dm_ext, as.integer(d))
}

.compute_topology <- function(dm_ext) {
  .Call("hdcd_r_compute_topology", dm_ext)
}

.topology_ordering <- function(topo_ext) {
  .Call("hdcd_r_topology_ordering", topo_ext)
}

.dag_create <- function(d, k_max) {
  .Call("hdcd_r_dag_create", as.integer(d), as.integer(k_max))
}

.dag_from_edges <- function(d, k_max, parents_1idx, children_1idx) {
  .Call("hdcd_r_dag_from_edges", as.integer(d), as.integer(k_max),
        as.integer(parents_1idx), as.integer(children_1idx))
}

.dag_add_edge <- function(dag_ext, parent_1idx, child_1idx) {
  invisible(.Call("hdcd_r_dag_add_edge", dag_ext, as.integer(parent_1idx), as.integer(child_1idx)))
}

.dag_clone <- function(dag_ext) {
  .Call("hdcd_r_dag_clone", dag_ext)
}

.dag_edges_matrix <- function(dag_ext, d) {
  .Call("hdcd_r_dag_edges", dag_ext, as.integer(d))
}

.dag_fit_c <- function(u, n, d, dag_ext, bernstein_degree, lambda_roughness, holdout_fraction,
                        seed, theta_max_iterations = 0L, theta_tol = 0,
                        lambda_roughness_grid = numeric(0), roughness_validation_fraction = 0,
                        bernstein_degree_grid = integer(0), tail_dependence_gate = 0,
                        tail_dependence_k = 0L, corner_relief = 0) {
  .Call("hdcd_r_dag_fit", u, as.integer(n), as.integer(d), dag_ext,
        as.integer(bernstein_degree), as.double(lambda_roughness), as.double(holdout_fraction),
        as.integer(seed), as.integer(theta_max_iterations), as.double(theta_tol),
        as.double(lambda_roughness_grid), as.double(roughness_validation_fraction),
        as.integer(bernstein_degree_grid), as.double(tail_dependence_gate),
        as.integer(tail_dependence_k), as.double(corner_relief))
}

.dag_fit_joint_log_density <- function(dag_fit_ext, u_point) {
  .Call("hdcd_r_dag_fit_joint_log_density", dag_fit_ext, as.double(u_point))
}

.dag_fit_holdout_scores <- function(dag_fit_ext, d) {
  .Call("hdcd_r_dag_fit_holdout_scores", dag_fit_ext, as.integer(d))
}

.dag_fit_all_converged <- function(dag_fit_ext) {
  .Call("hdcd_r_dag_fit_all_converged", dag_fit_ext)
}

.dag_fit_kl_estimate <- function(dag_fit_ext) {
  .Call("hdcd_r_dag_fit_kl_estimate", dag_fit_ext)
}

.dag_fit_kl_difference <- function(candidate_ext, reference_ext) {
  .Call("hdcd_r_dag_fit_kl_difference", candidate_ext, reference_ext)
}

.local_fit_n_parents <- function(dag_fit_ext, node_1idx) {
  .Call("hdcd_r_local_fit_n_parents", dag_fit_ext, as.integer(node_1idx))
}

.local_fit_parent_order <- function(dag_fit_ext, node_1idx) {
  .Call("hdcd_r_local_fit_parent_order", dag_fit_ext, as.integer(node_1idx))
}

.local_fit_selected_lambda_roughness <- function(dag_fit_ext, node_1idx) {
  .Call("hdcd_r_local_fit_selected_lambda_roughness", dag_fit_ext, as.integer(node_1idx))
}

.local_fit_selected_bernstein_degree <- function(dag_fit_ext, node_1idx) {
  .Call("hdcd_r_local_fit_selected_bernstein_degree", dag_fit_ext, as.integer(node_1idx))
}

.local_fit_max_tail_dependence <- function(dag_fit_ext, node_1idx) {
  .Call("hdcd_r_local_fit_max_tail_dependence", dag_fit_ext, as.integer(node_1idx))
}

.local_fit_conditional_log_density <- function(dag_fit_ext, node_1idx, u, z) {
  .Call("hdcd_r_local_fit_conditional_log_density", dag_fit_ext, as.integer(node_1idx),
        as.double(u), as.double(z))
}

.run_annealing_c <- function(u, n, d, ordering_1idx, k_max, lambda_edge,
                              bernstein_degree, lambda_roughness, holdout_fraction, local_seed,
                              initial_temperature, cooling_rate, max_iterations, restarts,
                              p_add, p_remove, p_swap, anneal_seed, corner_relief = 0) {
  .Call("hdcd_r_run_annealing", u, as.integer(n), as.integer(d), as.integer(ordering_1idx),
        as.integer(k_max), as.double(lambda_edge),
        as.integer(bernstein_degree), as.double(lambda_roughness), as.double(holdout_fraction),
        as.integer(local_seed),
        as.double(initial_temperature), as.double(cooling_rate),
        as.integer(max_iterations), as.integer(restarts),
        as.double(p_add), as.double(p_remove), as.double(p_swap), as.integer(anneal_seed),
        as.double(corner_relief))
}

# ---- public API -----------------------------------------------------------

#' Fit the full hdcd pipeline
#'
#' Marginals -> copula transform -> dependence matrix -> persistent-
#' topology ordering -> simulated-annealing DAG search -> fixed-DAG
#' fitting (spec section 1's pipeline, driven end to end).
#'
#' @param X a numeric matrix (n x d). `NA` marks a missing entry (spec
#'   section 26: converted to an explicit observed mask at this
#'   boundary, never passed through as a sentinel).
#' @param lambda_roughness_grid optional numeric vector of candidate
#'   roughness penalties. When non-empty, replaces the single fixed
#'   `lambda_roughness` with a PER-NODE value chosen by inner-validation
#'   from this grid (spec section 18: "lambda_R is selected by
#'   validation unless explicitly provided" / "a small validation grid
#'   is sufficient for version 1") -- see [hdcd_node_lambda_roughness()].
#'   Applies only to the final reference-DAG fit, never to the
#'   annealing search itself (spec section 18: "Penalty selection must
#'   occur outside the inner annealing loop"), which keeps using
#'   `lambda_roughness` throughout for speed. Default `numeric(0)`
#'   disables this entirely -- fully backward compatible.
#' @param roughness_validation_fraction fraction of each node's TRAIN
#'   rows further reserved for inner grid validation; only used when
#'   `lambda_roughness_grid` is non-empty. `0` selects a default (0.3).
#' @param bernstein_degree_grid optional integer vector of candidate
#'   Bernstein degrees. When non-empty, a node's `bernstein_degree` is
#'   raised from the fixed default via the same inner-validation
#'   machinery as `lambda_roughness_grid` above (jointly, if that is also
#'   supplied) -- but ONLY for nodes whose data shows real tail
#'   dependence (see `tail_dependence_gate`); see DECISIONS.md's
#'   "tail-dependence-informed bernstein_degree selection" and
#'   [hdcd_node_bernstein_degree()]/[hdcd_node_tail_dependence()].
#'   Also applies only to the final reference-DAG fit, never to the
#'   annealing search. Default `integer(0)` disables this entirely.
#' @param tail_dependence_gate in `[0,1]`: a node's `bernstein_degree_grid`
#'   is only searched when its strongest parent-edge empirical
#'   tail-dependence coefficient is at least this large; only used when
#'   `bernstein_degree_grid` is non-empty. `0` (the default) means
#'   "always search" -- no gating.
#' @param tail_dependence_k number of extreme order statistics used to
#'   estimate each tail-dependence coefficient; `0` selects a default
#'   (`round(sqrt(n))`, clamped).
#' @param corner_relief anisotropic (corner-relaxed) roughness penalty
#'   strength, in `[0, 1)` (see DECISIONS.md's "anisotropic
#'   (corner-relaxed) roughness penalty" entry): relaxes the roughness
#'   penalty near the four corners of each edge's Bernstein coefficient
#'   grid -- where `u` and `z` are both extreme simultaneously, exactly
#'   where tail dependence concentrates -- without raising
#'   `bernstein_degree` (and its coefficient count) everywhere. Applied
#'   to BOTH the annealing search and the final reference-DAG fit (unlike
#'   the two grids above, it adds no extra fit calls, so there is no
#'   reason to let the search and the final fit disagree on it). A FIXED
#'   scalar for v1, not itself grid-searched. Default `0` recovers the
#'   original uniform roughness penalty exactly.
#' @return an object of class `hdcd_model`.
#' @export
hdcd_fit <- function(X, max_parents = 2L, bernstein_degree = 3L,
                      lambda_edge = 0.05, lambda_roughness = 0.15,
                      holdout_fraction = 0.25, seed = 0L,
                      initial_temperature = 0.5, cooling_rate = 0.95,
                      annealing_iterations = 150L, annealing_restarts = 1L,
                      p_add = 1.0, p_remove = 1.0, p_swap = 1.0,
                      lambda_roughness_grid = numeric(0), roughness_validation_fraction = 0,
                      bernstein_degree_grid = integer(0), tail_dependence_gate = 0,
                      tail_dependence_k = 0L, corner_relief = 0) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  n <- nrow(X)
  d <- ncol(X)

  marginals <- vector("list", d)
  U <- matrix(NA_real_, nrow = n, ncol = d)
  for (j in seq_len(d)) {
    marginals[[j]] <- .new_marginal(X[, j])
    U[, j] <- .transform_to_copula(marginals[[j]], X[, j])
  }

  dm_ext <- .compute_dependence_matrix(U, n, d)
  topo_ext <- .compute_topology(dm_ext)
  ordering <- .topology_ordering(topo_ext)

  local_seed <- seed + 1L
  annealed <- .run_annealing_c(
    U, n, d, ordering, max_parents, lambda_edge,
    bernstein_degree, lambda_roughness, holdout_fraction, local_seed,
    initial_temperature, cooling_rate, annealing_iterations, annealing_restarts,
    p_add, p_remove, p_swap, seed, corner_relief = corner_relief
  )
  reference_dag <- annealed$dag

  dag_fit_ext <- .dag_fit_c(U, n, d, reference_dag, bernstein_degree, lambda_roughness,
                             holdout_fraction, local_seed,
                             lambda_roughness_grid = lambda_roughness_grid,
                             roughness_validation_fraction = roughness_validation_fraction,
                             bernstein_degree_grid = bernstein_degree_grid,
                             tail_dependence_gate = tail_dependence_gate,
                             tail_dependence_k = tail_dependence_k,
                             corner_relief = corner_relief)

  model <- list(
    marginals = marginals,
    dag = reference_dag,
    dag_fit = dag_fit_ext,
    dependence_matrix_ext = dm_ext,
    topology_ext = topo_ext,
    d = d,
    max_parents = max_parents,
    bernstein_degree = bernstein_degree,
    lambda_roughness = lambda_roughness,
    holdout_fraction = holdout_fraction,
    local_seed = local_seed,
    lambda_roughness_grid = lambda_roughness_grid,
    roughness_validation_fraction = roughness_validation_fraction,
    bernstein_degree_grid = bernstein_degree_grid,
    tail_dependence_gate = tail_dependence_gate,
    tail_dependence_k = tail_dependence_k,
    corner_relief = corner_relief,
    best_score = annealed$score,
    score_trace = annealed$score_trace,
    accepted_trace = annealed$accepted_trace,
    acceptance_rate = annealed$acceptance_rate,
    U = U,
    X = X
  )
  class(model) <- "hdcd_model"
  model
}

#' @export
print.hdcd_model <- function(x, ...) {
  cat(sprintf(
    "<hdcd_model> d=%d, max_parents=%d, edges=%d, best_score=%.4f\n",
    x$d, x$max_parents, nrow(hdcd_dag(x)), x$best_score
  ))
  invisible(x)
}

#' Transform new data to the copula scale using the model's fitted marginals
#' @export
hdcd_transform <- function(model, data) {
  stopifnot(inherits(model, "hdcd_model"))
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  n <- nrow(data)
  U <- matrix(NA_real_, nrow = n, ncol = model$d)
  for (j in seq_len(model$d)) {
    U[, j] <- .transform_to_copula(model$marginals[[j]], data[, j])
  }
  U
}

#' Factorized joint copula log-density log c_G(u), row-wise
#'
#' Rows with any NA are reported as NA (spec section 16: full
#' likelihood under missing coordinates is out of scope for v1).
#' @export
hdcd_copula_logpdf <- function(model, u) {
  stopifnot(inherits(model, "hdcd_model"))
  u <- as.matrix(u)
  n <- nrow(u)
  out <- numeric(n)
  for (i in seq_len(n)) {
    row <- u[i, ]
    if (anyNA(row)) {
      out[i] <- NA_real_
    } else {
      out[i] <- .dag_fit_joint_log_density(model$dag_fit, row)
    }
  }
  out
}

#' Predict from a fitted hdcd_model (spec section 26)
#'
#' @param object an `hdcd_model`.
#' @param newdata a numeric matrix (n x d), raw (not copula-scale) data.
#' @param type one of "logpdf" (log f_X(x), spec section 35),
#'   "copula_logpdf" (log c_G(u) after transforming `newdata`), or
#'   "transform" (just the copula-scale transform, no density).
#' @export
predict.hdcd_model <- function(object, newdata, type = "logpdf", ...) {
  type <- match.arg(type, c("logpdf", "copula_logpdf", "transform"))
  if (type == "transform") {
    return(hdcd_transform(object, newdata))
  }
  if (type == "copula_logpdf") {
    return(hdcd_copula_logpdf(object, hdcd_transform(object, newdata)))
  }

  newdata <- as.matrix(newdata)
  storage.mode(newdata) <- "double"
  n <- nrow(newdata)
  marginal_term <- numeric(n)
  any_missing <- rep(FALSE, n)
  for (j in seq_len(object$d)) {
    col <- newdata[, j]
    missing <- is.na(col)
    any_missing <- any_missing | missing
    if (any(!missing)) {
      marginal_term[!missing] <- marginal_term[!missing] +
        .marginal_logpdf(object$marginals[[j]], col[!missing])
    }
  }
  U <- hdcd_transform(object, newdata)
  copula_term <- hdcd_copula_logpdf(object, U)
  result <- copula_term + marginal_term
  result[any_missing] <- NA_real_
  result
}

#' Sample from the fitted joint copula model
#'
#' NOT IMPLEMENTED: the C core has no sampling routine. Spec section 21
#' names `hdcd_sample` in its architecture sketch, and section 26 shows
#' `sample(model, n)` as conceptual usage, but no version-1 milestone
#' (spec section 31) actually schedules building it in the C core --
#' see DECISIONS.md (the same gap was hit, and documented, in the
#' Python binding, Milestone 10).
#' @export
hdcd_sample <- function(model, n) {
  stop(
    "hdcd_sample is not implemented: the C core has no sampling routine ",
    "(spec section 21 names hdcd_sample in its architecture sketch, but no ",
    "version-1 milestone schedules building it -- see DECISIONS.md)."
  )
}

#' The fitted pairwise distance-correlation dependence matrix (spec section 5)
#' @export
hdcd_dependence_matrix <- function(model) {
  stopifnot(inherits(model, "hdcd_model"))
  .dependence_matrix_dense(model$dependence_matrix_ext, model$d)
}

#' The persistent-topology variable ordering (spec section 6), 1-indexed
#' @export
hdcd_ordering <- function(model) {
  stopifnot(inherits(model, "hdcd_model"))
  .topology_ordering(model$topology_ext)
}

#' The reference DAG's edges, as a two-column (parent, child) matrix, 1-indexed
#' @export
hdcd_dag <- function(model) {
  stopifnot(inherits(model, "hdcd_model"))
  edges <- .dag_edges_matrix(model$dag, model$d)
  colnames(edges) <- c("parent", "child")
  edges
}

#' Fit an alternative candidate DAG over the same training data (spec section 19)
#'
#' @param model an `hdcd_model`, providing the training data and options.
#' @param candidate_edges a two-column (parent, child) matrix, 1-indexed,
#'   or `NULL`/empty for the independence (empty) graph.
#' @param lambda_roughness_grid optional override of the roughness-
#'   selection grid to use for this candidate fit (see [hdcd_fit()]);
#'   defaults to reusing whatever `model` itself was fit with, so a
#'   candidate DAG is compared on equal footing.
#' @param roughness_validation_fraction optional override; defaults to
#'   `model`'s own setting.
#' @param bernstein_degree_grid optional override of the tail-dependence-
#'   gated degree-selection grid to use for this candidate fit (see
#'   [hdcd_fit()]); defaults to reusing whatever `model` itself was fit
#'   with, so a candidate DAG is compared on equal footing.
#' @param tail_dependence_gate optional override; defaults to `model`'s
#'   own setting.
#' @param tail_dependence_k optional override; defaults to `model`'s own
#'   setting.
#' @return an object of class `hdcd_dag_fit`; pass it to
#'   [hdcd_score_dag()] to compare against `model`'s reference DAG.
#' @export
hdcd_fit_dag <- function(model, candidate_edges,
                          lambda_roughness_grid = model$lambda_roughness_grid,
                          roughness_validation_fraction = model$roughness_validation_fraction,
                          bernstein_degree_grid = model$bernstein_degree_grid,
                          tail_dependence_gate = model$tail_dependence_gate,
                          tail_dependence_k = model$tail_dependence_k,
                          corner_relief = model$corner_relief) {
  stopifnot(inherits(model, "hdcd_model"))
  if (is.null(candidate_edges) || length(candidate_edges) == 0) {
    parents_1idx <- integer(0)
    children_1idx <- integer(0)
  } else {
    candidate_edges <- as.matrix(candidate_edges)
    parents_1idx <- as.integer(candidate_edges[, 1])
    children_1idx <- as.integer(candidate_edges[, 2])
  }
  if (is.null(lambda_roughness_grid)) lambda_roughness_grid <- numeric(0)
  if (is.null(roughness_validation_fraction)) roughness_validation_fraction <- 0
  if (is.null(bernstein_degree_grid)) bernstein_degree_grid <- integer(0)
  if (is.null(tail_dependence_gate)) tail_dependence_gate <- 0
  if (is.null(tail_dependence_k)) tail_dependence_k <- 0L
  if (is.null(corner_relief)) corner_relief <- 0
  dag_ext <- .dag_from_edges(model$d, model$max_parents, parents_1idx, children_1idx)
  fit_ext <- .dag_fit_c(model$U, nrow(model$U), model$d, dag_ext,
                         model$bernstein_degree, model$lambda_roughness, model$holdout_fraction,
                         model$local_seed,
                         lambda_roughness_grid = lambda_roughness_grid,
                         roughness_validation_fraction = roughness_validation_fraction,
                         bernstein_degree_grid = bernstein_degree_grid,
                         tail_dependence_gate = tail_dependence_gate,
                         tail_dependence_k = tail_dependence_k,
                         corner_relief = corner_relief)
  structure(list(dag = dag_ext, dag_fit = fit_ext), class = "hdcd_dag_fit")
}

#' Held-out KL comparison of a candidate DAG fit against the model's reference (spec section 19)
#'
#' IMPORTANT: this is a purely statistical, observational comparison of
#' distributional fit. It does not establish causal direction and does
#' not distinguish Markov-equivalent causal DAGs (spec section 19's
#' closing paragraph).
#'
#' @param model an `hdcd_model`.
#' @param candidate_fit the result of [hdcd_fit_dag()].
#' @return `Delta_KL = kl_estimate(candidate) - kl_estimate(reference)`;
#'   positive means the candidate fits worse than the reference.
#' @export
hdcd_score_dag <- function(model, candidate_fit) {
  stopifnot(inherits(model, "hdcd_model"))
  stopifnot(inherits(candidate_fit, "hdcd_dag_fit"))
  .dag_fit_kl_difference(candidate_fit$dag_fit, model$dag_fit)
}

#' The fitted parents of one node in the reference DAG
#'
#' @param model an `hdcd_model`.
#' @param node the 1-indexed column index of the node.
#' @return an integer vector of 1-indexed parent column indices (possibly empty).
#' @export
hdcd_node_parents <- function(model, node) {
  stopifnot(inherits(model, "hdcd_model"))
  .local_fit_parent_order(model$dag_fit, node)
}

#' The lambda_roughness actually used to fit one node's Theta
#'
#' Equal to `model$lambda_roughness` for every node when
#' `lambda_roughness_grid` was not used; otherwise the PER-NODE value
#' [hdcd_fit()]'s inner-validation grid search selected for that node
#' (spec section 18). `NA` for a root node (nothing to select).
#'
#' @param model an `hdcd_model` (or the result of [hdcd_fit_dag()]).
#' @param node the 1-indexed column index of the node.
#' @return a single number.
#' @export
hdcd_node_lambda_roughness <- function(model, node) {
  stopifnot(inherits(model, "hdcd_model") || inherits(model, "hdcd_dag_fit"))
  .local_fit_selected_lambda_roughness(model$dag_fit, node)
}

#' The bernstein_degree actually used to fit one node's Theta
#'
#' Equal to `model$bernstein_degree` for every node when
#' `bernstein_degree_grid` was not used, or gated off for that node (its
#' tail-dependence coefficient did not clear `tail_dependence_gate`);
#' otherwise the PER-NODE value [hdcd_fit()]'s inner-validation grid
#' search selected (spec section 18-style selection, applied to degree
#' instead of lambda; see DECISIONS.md). `0` for a root node.
#'
#' @param model an `hdcd_model` (or the result of [hdcd_fit_dag()]).
#' @param node the 1-indexed column index of the node.
#' @return a single integer.
#' @export
hdcd_node_bernstein_degree <- function(model, node) {
  stopifnot(inherits(model, "hdcd_model") || inherits(model, "hdcd_dag_fit"))
  .local_fit_selected_bernstein_degree(model$dag_fit, node)
}

#' One node's maximum empirical tail-dependence coefficient
#'
#' The diagnostic that gated [hdcd_fit()]'s `bernstein_degree_grid`
#' search for this node: the largest tail-dependence coefficient (upper
#' or lower) across its parent edges. `NA` if `bernstein_degree_grid` was
#' never supplied (the diagnostic is not computed at all when it would
#' not be used) or for a root node.
#'
#' @param model an `hdcd_model` (or the result of [hdcd_fit_dag()]).
#' @param node the 1-indexed column index of the node.
#' @return a single number in `[0,1]`, or `NA`.
#' @export
hdcd_node_tail_dependence <- function(model, node) {
  stopifnot(inherits(model, "hdcd_model") || inherits(model, "hdcd_dag_fit"))
  .local_fit_max_tail_dependence(model$dag_fit, node)
}

#' Conditional copula density c_j(u | z) for one node, over a grid of u
#'
#' Evaluates the fitted conditional density directly (not via
#' [hdcd_copula_logpdf()], which needs a full d-length row) -- useful
#' for plotting one edge's fitted density curve against its true
#' counterpart.
#'
#' @param model an `hdcd_model`.
#' @param node the 1-indexed column index of the node.
#' @param u a numeric vector of copula-scale evaluation points in (0,1).
#' @param z the parent(s)' copula-scale value(s): a numeric vector whose
#'   length must match `length(hdcd_node_parents(model, node))`; empty
#'   for a root node.
#' @return a numeric vector the same length as `u`: `c_node(u | z)`.
#' @export
hdcd_conditional_density <- function(model, node, u, z = numeric(0)) {
  stopifnot(inherits(model, "hdcd_model"))
  exp(.local_fit_conditional_log_density(model$dag_fit, node, u, z))
}
