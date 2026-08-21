# Exercises hdcd_fit()'s full pipeline (the documented spec section 26
# API), not just individual .Call primitives -- catches integration bugs
# the fixture-agreement tests wouldn't (e.g. wrong argument order when
# composing calls in R/hdcd.R).

make_chain_data <- function(n = 300, rho = 0.7, seed = 1) {
  set.seed(seed)
  z0 <- rnorm(n)
  z1 <- rho * z0 + sqrt(1 - rho^2) * rnorm(n)
  z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
  cbind(z0, z1, z2)
}

test_that("hdcd_fit -> transform -> logpdf -> fit_dag round-trip works", {
  X <- make_chain_data(seed = 11)
  model <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 3L, lambda_edge = 0.05,
                     lambda_roughness = 0.15, annealing_iterations = 60L, seed = 1L)
  expect_s3_class(model, "hdcd_model")
  expect_equal(model$d, 3L)

  U <- hdcd_transform(model, X)
  expect_true(all(U > 0 & U < 1))

  clp <- hdcd_copula_logpdf(model, U)
  lp <- predict(model, X, type = "logpdf")
  expect_true(all(is.finite(clp)))
  expect_true(all(is.finite(lp)))

  dm <- hdcd_dependence_matrix(model)
  expect_equal(dim(dm), c(3L, 3L))

  ordering <- hdcd_ordering(model)
  expect_equal(sort(ordering), 1:3)

  edges <- hdcd_dag(model)
  expect_true(is.matrix(edges))
  expect_equal(colnames(edges), c("parent", "child"))
})

test_that("hdcd_fit_dag / hdcd_score_dag produce a meaningful KL comparison", {
  X <- make_chain_data(seed = 12)
  model <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 3L, annealing_iterations = 60L, seed = 2L)

  candidate <- hdcd_fit_dag(model, NULL) # independence hypothesis
  expect_s3_class(candidate, "hdcd_dag_fit")

  kl <- hdcd_score_dag(model, candidate)
  expect_true(is.numeric(kl))
  expect_false(is.nan(kl))
  # Discarding a real dependency should lose information relative to the
  # fitted reference DAG.
  expect_gt(kl, 0)
})

test_that("lambda_roughness_grid selects a per-node lambda without changing default behavior", {
  X <- make_chain_data(seed = 13)

  fixed <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 3L, lambda_roughness = 0.15,
                     annealing_iterations = 60L, seed = 3L)
  # hdcd_node_lambda_roughness() must report the fixed value verbatim
  # when lambda_roughness_grid was never supplied (the default).
  expect_equal(hdcd_node_lambda_roughness(fixed, 2), 0.15)

  auto <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 3L, lambda_roughness = 0.15,
                    annealing_iterations = 60L, seed = 3L,
                    lambda_roughness_grid = c(0.02, 0.05, 0.15, 0.3, 0.6),
                    roughness_validation_fraction = 0.3)
  expect_s3_class(auto, "hdcd_model")
  # The reference DAG search (annealing) is untouched by the grid (spec
  # section 18: selection happens outside the inner annealing loop), so
  # the two fits should find the SAME reference structure here.
  expect_equal(hdcd_dag(fixed), hdcd_dag(auto))

  for (node in 2:3) {
    if (length(hdcd_node_parents(auto, node)) > 0) {
      selected <- hdcd_node_lambda_roughness(auto, node)
      expect_true(selected %in% c(0.02, 0.05, 0.15, 0.3, 0.6))
    }
  }

  # A candidate DAG fit via hdcd_fit_dag() also accepts (and by default
  # reuses) the grid.
  candidate <- hdcd_fit_dag(auto, hdcd_dag(auto))
  expect_s3_class(candidate, "hdcd_dag_fit")
  kl <- hdcd_score_dag(auto, candidate)
  expect_true(is.numeric(kl))

  # Root node: nothing to select.
  root_nodes <- setdiff(1:3, hdcd_dag(auto)[, "child"])
  if (length(root_nodes) > 0) {
    expect_true(is.na(hdcd_node_lambda_roughness(auto, root_nodes[1])))
  }
})

make_tail_dependent_chain_data <- function(n = 800, p = 0.5, seed = 1) {
  set.seed(seed)
  w0 <- runif(n)
  # A "comonotonic-mixture" copula-scale chain (see the identical
  # construction and its derivation comment in
  # notebooks/vine_copula_recovery.Rmd / tests/test_tail_dependence.c):
  # with probability p, each step copies its predecessor exactly
  # (comonotonic, contributing to both tails); otherwise independent.
  step <- function(prev) ifelse(runif(n) < p, prev, runif(n))
  u1 <- step(w0)
  u2 <- step(u1)
  cbind(qnorm(w0), qnorm(u1), qnorm(u2)) # arbitrary (Gaussian) marginals on top
}

test_that("bernstein_degree_grid is tail-dependence-gated and selects a per-node degree", {
  X <- make_tail_dependent_chain_data(seed = 21)

  fixed <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                     annealing_iterations = 60L, seed = 4L)
  expect_equal(hdcd_node_bernstein_degree(fixed, 2), 4L)
  # The diagnostic is ALWAYS computed, even with no grid supplied (see
  # DECISIONS.md's "distinguish initial fit from diagnose from tune"
  # entry) -- this data is genuinely tail-dependent, so it should read
  # as such on a plain, untuned fit.
  expect_true(hdcd_node_tail_dependence(fixed, 2) > 0)

  gated_off <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                         annealing_iterations = 60L, seed = 4L,
                         bernstein_degree_grid = c(4L, 6L, 8L), tail_dependence_gate = 0.9)
  # A gate almost nothing clears: falls back to the fixed degree exactly.
  expect_equal(hdcd_node_bernstein_degree(gated_off, 2), 4L)
  expect_true(hdcd_node_tail_dependence(gated_off, 2) < 0.9)

  auto <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                    annealing_iterations = 60L, seed = 4L,
                    bernstein_degree_grid = c(4L, 6L, 8L), tail_dependence_gate = 0.05,
                    lambda_roughness_grid = c(0.1, 0.15, 0.3))
  expect_s3_class(auto, "hdcd_model")
  # Annealing is untouched by either grid: same reference structure.
  expect_equal(hdcd_dag(fixed), hdcd_dag(auto))

  for (node in 2:3) {
    if (length(hdcd_node_parents(auto, node)) > 0) {
      expect_true(hdcd_node_tail_dependence(auto, node) >= 0.05)
      expect_true(hdcd_node_bernstein_degree(auto, node) %in% c(4L, 6L, 8L))
      expect_true(hdcd_node_lambda_roughness(auto, node) %in% c(0.1, 0.15, 0.3))
    }
  }

  # hdcd_fit_dag() defaults to reusing the model's own degree grid too.
  candidate <- hdcd_fit_dag(auto, hdcd_dag(auto))
  expect_s3_class(candidate, "hdcd_dag_fit")
  expect_true(is.numeric(hdcd_score_dag(auto, candidate)))
})

test_that("hdcd_diagnose reports tail-dependence on a plain, untuned fit", {
  X <- make_tail_dependent_chain_data(seed = 23)

  # A PLAIN fit: no bernstein_degree_grid, no corner_relief -- exactly
  # the "initial fit" step of the fit -> diagnose -> tune workflow
  # (DECISIONS.md's "distinguish initial fit from diagnose from tune"
  # entry). hdcd_diagnose() must not require any tuning option to have
  # been supplied first.
  plain <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                     annealing_iterations = 60L, seed = 7L)

  report <- hdcd_diagnose(plain)
  expect_s3_class(report, "data.frame")
  expect_true(all(c("node", "n_parents", "tail_dependence") %in% names(report)))
  # Every non-root node appears, no root nodes, and it's genuinely
  # informative (not all NA/zero) on data built to be tail-dependent.
  expect_equal(nrow(report), sum(vapply(seq_len(plain$d), function(j) length(hdcd_node_parents(plain, j)) > 0, logical(1))))
  expect_true(all(!is.na(report$tail_dependence)))
  expect_true(max(report$tail_dependence) > 0)
  # Sorted most-tail-dependent-first.
  expect_true(all(diff(report$tail_dependence) <= 0))

  # A bare hdcd_fit_dag() result (no $d) is explicitly out of scope.
  candidate <- hdcd_fit_dag(plain, hdcd_dag(plain))
  expect_error(hdcd_diagnose(candidate))
})

test_that("corner_relief changes the fit and defaults to a no-op", {
  X <- make_tail_dependent_chain_data(seed = 22)

  flat <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                    annealing_iterations = 60L, seed = 5L)
  expect_equal(flat$corner_relief, 0)

  # NOTE: corner_relief is applied to the annealing search too (see
  # DECISIONS.md), unlike lambda_roughness_grid/bernstein_degree_grid
  # which are deliberately excluded from it -- so a fresh hdcd_fit() at a
  # different corner_relief is NOT guaranteed to find the same reference
  # structure. Isolate the effect on the FIT itself instead: refit the
  # SAME structure (flat's own reference DAG) via hdcd_fit_dag() at two
  # different corner_relief values, sidestepping any structure difference.
  true_structure <- hdcd_dag(flat)
  candidate_flat <- hdcd_fit_dag(flat, true_structure, corner_relief = 0)
  candidate_relief <- hdcd_fit_dag(flat, true_structure, corner_relief = 0.8)
  expect_s3_class(candidate_flat, "hdcd_dag_fit")
  expect_s3_class(candidate_relief, "hdcd_dag_fit")

  changed <- FALSE
  for (node in 2:3) {
    n_parents <- hdcd_node_parents(flat, node) # same structure for both candidates
    if (length(n_parents) > 0) {
      u_grid <- seq(0.1, 0.9, by = 0.2)
      z <- rep(0.5, length(n_parents)) # fixed parent value(s), reused across every u in u_grid
      d_flat <- exp(hdcd:::.local_fit_conditional_log_density(candidate_flat$dag_fit, node, u_grid, z))
      d_relief <- exp(hdcd:::.local_fit_conditional_log_density(candidate_relief$dag_fit, node, u_grid, z))
      if (any(abs(d_flat - d_relief) > 1e-8)) changed <- TRUE
    }
  }
  expect_true(changed)

  # hdcd_fit_dag() defaults to reusing the model's own corner_relief too.
  relief_model <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                            annealing_iterations = 60L, seed = 5L, corner_relief = 0.8)
  expect_equal(relief_model$corner_relief, 0.8)
  candidate <- hdcd_fit_dag(relief_model, hdcd_dag(relief_model))
  expect_s3_class(candidate, "hdcd_dag_fit")
})

test_that("corner_kde_gate selects a corner side and defaults to a no-op", {
  X <- make_tail_dependent_chain_data(seed = 24)

  flat <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                    annealing_iterations = 60L, seed = 8L)
  expect_equal(flat$corner_kde_gate, 0)
  expect_equal(hdcd_node_corner_side(flat, 2, 1), "none")

  # corner_kde_gate is excluded from annealing (real per-node KDE cost,
  # unlike corner_relief), so isolate its effect via hdcd_fit_dag() on a
  # FIXED structure, same reasoning as the bernstein_degree_grid test.
  true_structure <- hdcd_dag(flat)
  gated <- hdcd_fit_dag(flat, true_structure,
                         corner_kde_gate = 0.1, corner_kde_bandwidth = 0.08, corner_kde_weight = 10)
  expect_s3_class(gated, "hdcd_dag_fit")

  found_side <- FALSE
  for (node in 2:3) {
    n_parents <- length(hdcd_node_parents(flat, node))
    for (p in seq_len(n_parents)) {
      side <- hdcd_node_corner_side(gated, node, p)
      expect_true(side %in% c("none", "lower", "upper"))
      if (side != "none") found_side <- TRUE
    }
  }
  expect_true(found_side)

  # A large enough weight must measurably change the fit relative to the
  # uncorrected candidate -- otherwise the option would be a silent no-op.
  unweighted <- hdcd_fit_dag(flat, true_structure)
  u_grid <- seq(0.05, 0.95, by = 0.2)
  changed <- FALSE
  for (node in 2:3) {
    n_parents <- length(hdcd_node_parents(flat, node))
    if (n_parents > 0) {
      z <- rep(0.1, n_parents)
      d_flat <- exp(hdcd:::.local_fit_conditional_log_density(unweighted$dag_fit, node, u_grid, z))
      d_gated <- exp(hdcd:::.local_fit_conditional_log_density(gated$dag_fit, node, u_grid, z))
      if (any(abs(d_flat - d_gated) > 1e-8)) changed <- TRUE
    }
  }
  expect_true(changed)
})

test_that("hdcd_node_region_score computes a region-restricted held-out score", {
  X <- make_tail_dependent_chain_data(seed = 25)
  model <- hdcd_fit(X, max_parents = 2L, bernstein_degree = 4L, lambda_roughness = 0.15,
                     annealing_iterations = 60L, seed = 9L)

  node <- 2L
  n_parents <- length(hdcd_node_parents(model, node))
  skip_if(n_parents == 0, "node 2 has no parents in this fit")

  n_holdout <- 200
  set.seed(99)
  u_holdout <- runif(n_holdout)
  z_holdout <- matrix(runif(n_holdout * n_parents), ncol = n_parents)

  result <- hdcd_node_region_score(model, node, parent = 1,
                                    u_holdout = u_holdout, z_holdout = z_holdout,
                                    z_center = 0.5, z_window = 1.0) # window=1: every row qualifies
  expect_equal(result$n, n_holdout)
  expect_true(is.finite(result$mean_log_density))

  empty_result <- hdcd_node_region_score(model, node, parent = 1,
                                          u_holdout = u_holdout, z_holdout = z_holdout,
                                          z_center = 5.0, z_window = 0.01) # no row can qualify
  expect_equal(empty_result$n, 0L)
  expect_true(is.na(empty_result$mean_log_density))
})

test_that("hdcd_sample raises a clear error, not silently returning garbage", {
  X <- make_chain_data(n = 100, seed = 13)
  model <- hdcd_fit(X, max_parents = 2L, annealing_iterations = 30L, seed = 3L)
  expect_error(hdcd_sample(model, 10), "not implemented")
})

test_that("NA entries are handled via the observed mask, not passed through as sentinels", {
  X <- make_chain_data(n = 150, seed = 14)
  X[1, 1] <- NA
  X[5, 2] <- NA
  model <- hdcd_fit(X, max_parents = 2L, annealing_iterations = 30L, seed = 4L)

  lp <- predict(model, X, type = "logpdf")
  expect_true(is.na(lp[1]))
  expect_true(is.na(lp[5]))
  expect_true(all(is.finite(lp[-c(1, 5)])))
})

test_that("external pointers survive gc() and repeated use (finalizer safety)", {
  X <- make_chain_data(n = 120, seed = 15)
  model <- hdcd_fit(X, max_parents = 2L, annealing_iterations = 30L, seed = 5L)

  gc() # must not free anything still referenced by `model`
  U <- hdcd_transform(model, X)
  expect_true(all(is.finite(U)))
  lp <- predict(model, X, type = "logpdf")
  expect_true(all(is.finite(lp)))
})

test_that("a discarded model's external pointers are eventually finalized without error", {
  X <- make_chain_data(n = 100, seed = 16)
  local({
    model <- hdcd_fit(X, max_parents = 2L, annealing_iterations = 20L, seed = 6L)
    invisible(hdcd_dependence_matrix(model))
  })
  gc() # runs registered finalizers; must not error or crash the session
  expect_true(TRUE)
})
