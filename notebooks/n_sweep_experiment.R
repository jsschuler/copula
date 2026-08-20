# Standalone reproducibility script for the "does more data resolve the
# bias-variance tradeoff" sweep referenced in vine_copula_recovery.Rmd's
# "Does more data resolve the bias-variance tradeoff?" section.
#
# NOT run automatically when the notebook renders: the full sweep (n in
# 2000, 4000, 8000, 16000) takes roughly 25-30 minutes on a single core,
# dominated by the largest n's tail-dependence-gated joint (degree,
# lambda) grid search (~15 minutes alone at n=16000) and by copula-scale
# data generation itself (R's copula::cCopula h-function chaining is
# O(n) but with a substantial constant factor). The notebook instead
# loads this script's saved output, n_sweep_results.csv, so ordinary
# renders stay fast; rerun this script directly (`Rscript
# n_sweep_experiment.R`) to regenerate that CSV from scratch.
#
# Uses the SAME ground-truth vine construction as the main notebook (see
# its "A known, high-dimensional, mixed-family vine copula" section) --
# duplicated here rather than sourced from the .Rmd, since this script
# is meant to be run standalone, independent of knitr.

suppressMessages({
  library(hdcd)
  library(copula)
})

d <- 10L
eps <- 1e-6
families <- list(
  claytonCopula(3.0), gumbelCopula(2.2), frankCopula(5.0), normalCopula(0.65),
  tCopula(0.5, df = 4), claytonCopula(1.8), gumbelCopula(3.0), frankCopula(3.5), normalCopula(0.45)
)
family_names <- c(
  "Clayton(3.0)", "Gumbel(2.2)", "Frank(5.0)", "Gaussian(0.65)",
  "t(0.5,df=4)", "Clayton(1.8)", "Gumbel(3.0)", "Frank(3.5)", "Gaussian(0.45)"
)
hinv <- function(u_given, w, cop) cCopula(cbind(u_given, w), copula = cop, indices = 2, inverse = TRUE)

gen_data <- function(n, seed) {
  set.seed(seed)
  w <- matrix(runif(n * d), ncol = d)
  w <- pmin(pmax(w, eps), 1 - eps)
  U <- matrix(NA_real_, n, d)
  U[, 1] <- w[, 1]
  for (i in 2:d) {
    u_prev <- pmin(pmax(U[, i - 1], eps), 1 - eps)
    U[, i] <- hinv(u_prev, w[, i], families[[i - 1]])
    U[, i] <- pmin(pmax(U[, i], eps), 1 - eps)
  }
  U
}

true_edges <- cbind(1:(d - 1), 2:d)
dag_ext <- hdcd:::.dag_from_edges(d, 2L, true_edges[, 1], true_edges[, 2])
u_grid <- seq(0.02, 0.98, length.out = 60)

# NOTE: each n draws an INDEPENDENT fresh sample (not a superset of the
# smaller n's data) -- one realization per sample size, not a replicated
# design. See the notebook section for why that matters to the
# conclusion drawn from this sweep.
n_values <- c(2000, 4000, 8000, 16000)
out_path <- "n_sweep_results.csv" # run this script from the notebooks/ directory

results <- list()
for (n in n_values) {
  cat(sprintf("=== n=%d ===\n", n))
  t0 <- Sys.time()
  U <- gen_data(n, 99)
  gen_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  data gen: %.1fs\n", gen_time))

  t0 <- Sys.time()
  fixed_fit <- hdcd:::.dag_fit_c(U, n, d, dag_ext, 4L, 0.15, 0.25, 2L)
  fixed_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  t0 <- Sys.time()
  auto_fit <- hdcd:::.dag_fit_c(
    U, n, d, dag_ext, 4L, 0.15, 0.25, 2L,
    lambda_roughness_grid = c(0.05, 0.1, 0.15, 0.3), roughness_validation_fraction = 0.3,
    bernstein_degree_grid = c(4L, 6L, 8L, 10L), tail_dependence_gate = 0.05
  )
  auto_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  fixed fit: %.1fs, auto joint-grid fit: %.1fs\n", fixed_time, auto_time))

  delta_kl <- hdcd:::.dag_fit_kl_difference(fixed_fit, auto_fit)

  for (i in c(1, 7)) { # Clayton 1->2, Gumbel 7->8: the two most severely under-fit edges
    node <- i + 1L
    q <- if (i == 7) 0.9 else 0.1
    fitted_fixed <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, node, u_grid, q))
    fitted_auto <- exp(hdcd:::.local_fit_conditional_log_density(auto_fit, node, u_grid, q))
    true_dens <- dCopula(cbind(rep(q, length(u_grid)), u_grid), families[[i]])
    row <- data.frame(
      n = n, edge = sprintf("%d->%d", i, i + 1), family = family_names[i],
      selected_degree = hdcd:::.local_fit_selected_bernstein_degree(auto_fit, node),
      selected_lambda = hdcd:::.local_fit_selected_lambda_roughness(auto_fit, node),
      cor_fixed = cor(fitted_fixed, true_dens), cor_auto = cor(fitted_auto, true_dens),
      delta_kl_total = delta_kl, gen_time = gen_time, fixed_time = fixed_time, auto_time = auto_time
    )
    results[[length(results) + 1]] <- row
    write.csv(do.call(rbind, results), out_path, row.names = FALSE)
  }
  cat(sprintf("  Delta_KL(fixed vs auto, all 9 edges) = %.4f\n", delta_kl))
}
cat("DONE\n")
