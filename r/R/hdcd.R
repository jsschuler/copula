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
                        seed, theta_max_iterations = 0L, theta_tol = 0) {
  .Call("hdcd_r_dag_fit", u, as.integer(n), as.integer(d), dag_ext,
        as.integer(bernstein_degree), as.double(lambda_roughness), as.double(holdout_fraction),
        as.integer(seed), as.integer(theta_max_iterations), as.double(theta_tol))
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

.run_annealing_c <- function(u, n, d, ordering_1idx, k_max, lambda_edge,
                              bernstein_degree, lambda_roughness, holdout_fraction, local_seed,
                              initial_temperature, cooling_rate, max_iterations, restarts,
                              p_add, p_remove, p_swap, anneal_seed) {
  .Call("hdcd_r_run_annealing", u, as.integer(n), as.integer(d), as.integer(ordering_1idx),
        as.integer(k_max), as.double(lambda_edge),
        as.integer(bernstein_degree), as.double(lambda_roughness), as.double(holdout_fraction),
        as.integer(local_seed),
        as.double(initial_temperature), as.double(cooling_rate),
        as.integer(max_iterations), as.integer(restarts),
        as.double(p_add), as.double(p_remove), as.double(p_swap), as.integer(anneal_seed))
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
#' @return an object of class `hdcd_model`.
#' @export
hdcd_fit <- function(X, max_parents = 2L, bernstein_degree = 3L,
                      lambda_edge = 0.05, lambda_roughness = 0.15,
                      holdout_fraction = 0.25, seed = 0L,
                      initial_temperature = 0.5, cooling_rate = 0.95,
                      annealing_iterations = 150L, annealing_restarts = 1L,
                      p_add = 1.0, p_remove = 1.0, p_swap = 1.0) {
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
    p_add, p_remove, p_swap, seed
  )
  reference_dag <- annealed$dag

  dag_fit_ext <- .dag_fit_c(U, n, d, reference_dag, bernstein_degree, lambda_roughness,
                             holdout_fraction, local_seed)

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
    best_score = annealed$score,
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
#' @return an object of class `hdcd_dag_fit`; pass it to
#'   [hdcd_score_dag()] to compare against `model`'s reference DAG.
#' @export
hdcd_fit_dag <- function(model, candidate_edges) {
  stopifnot(inherits(model, "hdcd_model"))
  if (is.null(candidate_edges) || length(candidate_edges) == 0) {
    parents_1idx <- integer(0)
    children_1idx <- integer(0)
  } else {
    candidate_edges <- as.matrix(candidate_edges)
    parents_1idx <- as.integer(candidate_edges[, 1])
    children_1idx <- as.integer(candidate_edges[, 2])
  }
  dag_ext <- .dag_from_edges(model$d, model$max_parents, parents_1idx, children_1idx)
  fit_ext <- .dag_fit_c(model$U, nrow(model$U), model$d, dag_ext,
                         model$bernstein_degree, model$lambda_roughness, model$holdout_fraction,
                         model$local_seed)
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
