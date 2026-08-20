# Standalone reproducibility script for the properly replicated
# n x corner_relief sweep referenced in DECISIONS.md's "Trials queued
# for tonight (anisotropic roughness penalty)" entry.
#
# Unlike the earlier bernstein_degree_grid sweep (notebooks/
# n_sweep_experiment.R), corner_relief costs exactly ONE ordinary
# fit_dag call per (n, corner_relief) point -- not a multi-candidate
# grid search -- so real replication is affordable here. Estimated total
# runtime: ~500s/replicate x 6 replicates ~= 50 minutes. NOT run
# automatically when the notebook renders; results are saved to
# corner_relief_sweep_results.csv and loaded from there.
#
# Uses the SAME ground-truth vine construction as the main notebook and
# notebooks/n_sweep_experiment.R (duplicated here for the same reason:
# meant to run standalone, independent of knitr).

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

n_values <- c(2000, 4000, 8000, 16000)
corner_relief_values <- c(0, 0.3, 0.6, 0.8)
n_replicates <- 6
worst_edges <- c(1, 7) # 1->2 Clayton, 7->8 Gumbel: the two most severely under-fit edges
out_path <- "corner_relief_sweep_results.csv" # run this script from the notebooks/ directory

results <- list()
t_start <- Sys.time()
for (replicate in 1:n_replicates) {
  for (n in n_values) {
    data_seed <- 10000 * replicate + n
    t0 <- Sys.time()
    U <- gen_data(n, data_seed)
    gen_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    # Fit corner_relief=0 first: every other corner_relief value in this
    # (replicate, n) is compared against it via Delta_KL.
    local_seed <- 20000 * replicate + n
    t0 <- Sys.time()
    baseline_fit <- hdcd:::.dag_fit_c(U, n, d, dag_ext, 4L, 0.15, 0.25, local_seed, corner_relief = 0)
    baseline_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    for (cr in corner_relief_values) {
      if (cr == 0) {
        fit <- baseline_fit
        fit_time <- baseline_time
      } else {
        t0 <- Sys.time()
        fit <- hdcd:::.dag_fit_c(U, n, d, dag_ext, 4L, 0.15, 0.25, local_seed, corner_relief = cr)
        fit_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      }
      delta_kl <- hdcd:::.dag_fit_kl_difference(fit, baseline_fit)

      for (i in worst_edges) {
        node <- i + 1L
        q <- if (i == 7) 0.9 else 0.1
        fitted <- exp(hdcd:::.local_fit_conditional_log_density(fit, node, u_grid, q))
        true_dens <- dCopula(cbind(rep(q, length(u_grid)), u_grid), families[[i]])
        row <- data.frame(
          replicate = replicate, n = n, corner_relief = cr,
          edge = sprintf("%d->%d", i, i + 1), family = family_names[i],
          cor_true = cor(fitted, true_dens), delta_kl_total = delta_kl,
          node_holdout_score = hdcd:::.dag_fit_holdout_scores(fit, d)[node], # this node's own held-out log-density score
          gen_time = gen_time, fit_time = fit_time
        )
        results[[length(results) + 1]] <- row
      }
    }
    write.csv(do.call(rbind, results), out_path, row.names = FALSE)
    elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
    cat(sprintf("replicate=%d n=%d done (elapsed %.0fs)\n", replicate, n, elapsed))
  }
}
cat("DONE\n")
