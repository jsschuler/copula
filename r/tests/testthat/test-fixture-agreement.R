# Cross-checks the R (.Call) binding against C-computed reference values
# for the SAME inputs (spec section 31 Milestone 11: "agreement with C
# fixtures"). Both sides call the identical compiled hdcd static
# library, so any disagreement here can only be a .Call marshaling bug
# (SEXP conversion, indexing direction, column-major assumptions) -- not
# a statistical/algorithmic one, which the C test suite already covers.
#
# Reuses python/tests/fixture.json verbatim (spec section 29.11: "the
# same saved test fixture"), rather than a separate R-only fixture, so
# the SAME C-computed values anchor both language bindings' agreement
# tests.

fixture_path <- testthat::test_path("..", "..", "..", "python", "tests", "fixture.json")

if (!file.exists(fixture_path) || !requireNamespace("jsonlite", quietly = TRUE)) {
  test_that("fixture agreement (skipped: fixture.json or jsonlite unavailable)", {
    skip("fixture.json not found next to the source checkout, or jsonlite is not installed")
  })
} else {

fixture <- jsonlite::fromJSON(fixture_path)
n <- fixture$n
d <- fixture$d
X <- fixture$X # jsonlite simplifies the nested JSON array into an n x d matrix
storage.mode(X) <- "double"

test_that("marginal sigma, cdf, and logpdf agree with the C fixture", {
  eval_points <- fixture$marginal_eval_points
  for (j in seq_len(d)) {
    m <- hdcd:::.new_marginal(X[, j])
    expect_equal(hdcd:::.marginal_sigma(m), fixture$marginal_sigma[j], tolerance = 1e-10)

    cdf <- hdcd:::.marginal_cdf(m, eval_points)
    expect_equal(cdf, fixture$marginal_cdf[j, ], tolerance = 1e-9)

    lp <- hdcd:::.marginal_logpdf(m, eval_points)
    expect_equal(lp, fixture$marginal_logpdf[j, ], tolerance = 1e-7)
  }
})

fit_marginals_and_transform <- function() {
  marginals <- vector("list", d)
  U <- matrix(NA_real_, nrow = n, ncol = d)
  for (j in seq_len(d)) {
    marginals[[j]] <- hdcd:::.new_marginal(X[, j])
    U[, j] <- hdcd:::.transform_to_copula(marginals[[j]], X[, j])
  }
  U
}

test_that("copula transform agrees with the C fixture", {
  U <- fit_marginals_and_transform()
  expect_equal(U[1:5, ], fixture$U_first_5_rows, tolerance = 1e-9, ignore_attr = TRUE)
})

test_that("dependence matrix agrees with the C fixture", {
  U <- fit_marginals_and_transform()
  dm_ext <- hdcd:::.compute_dependence_matrix(U, n, d)
  dm <- hdcd:::.dependence_matrix_dense(dm_ext, d)
  expect_equal(dm, fixture$dependence_matrix, tolerance = 1e-9, ignore_attr = TRUE)
})

test_that("topology ordering and MST edges agree with the C fixture", {
  U <- fit_marginals_and_transform()
  dm_ext <- hdcd:::.compute_dependence_matrix(U, n, d)
  topo_ext <- hdcd:::.compute_topology(dm_ext)

  ordering_1idx <- hdcd:::.topology_ordering(topo_ext)
  expected_1idx <- fixture$topology_ordering + 1L # fixture stores C's 0-indexed ordering
  expect_equal(ordering_1idx, expected_1idx)
})

test_that("DAG fit holdout scores, joint log density, and kl_estimate agree with the C fixture", {
  U <- fit_marginals_and_transform()

  # fixture$dag_edges is 0-indexed (from C); .dag_from_edges expects
  # 1-indexed R convention, so convert once at this boundary.
  edges <- fixture$dag_edges + 1L
  dag_ext <- hdcd:::.dag_from_edges(d, 2L, edges[, 1], edges[, 2])

  # This is the highest-risk marshaling call in the whole binding: ten
  # positional arguments of mixed type (pointer, size_t, double,
  # uint64_t) passed to a single .Call -- exactly where an argument-
  # order mistake would silently misassign a value instead of crashing.
  dag_fit_ext <- hdcd:::.dag_fit_c(U, n, d, dag_ext, 3L, 0.15, 0.25, 777L)

  expect_true(hdcd:::.dag_fit_all_converged(dag_fit_ext))

  scores <- hdcd:::.dag_fit_holdout_scores(dag_fit_ext, d)
  expect_equal(scores, fixture$dag_fit_holdout_scores, tolerance = 1e-9)

  point <- fixture$joint_log_density_point
  value <- hdcd:::.dag_fit_joint_log_density(dag_fit_ext, point)
  expect_equal(value, fixture$joint_log_density_value, tolerance = 1e-9)

  expect_equal(hdcd:::.dag_fit_kl_estimate(dag_fit_ext), fixture$kl_estimate, tolerance = 1e-9)
})

} # if fixture available
