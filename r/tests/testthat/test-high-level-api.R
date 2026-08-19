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
