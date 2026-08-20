# Standalone reproduction of the (now-removed) copula-level EVT tail-splice
# experiment, run once while evt_splice_gate/evt_splice_bandwidth still
# existed in the C core, so the notebook's writeup of this finding stays
# reproducible without the removed API. See DECISIONS.md's "copula-level
# EVT tail-splice" entry for why the feature was removed: it assumes a
# parametric copula family, which is outside this library's nonparametric
# dependence-structure design (spec section 3's EVT provision is for
# marginals only). Results saved here are loaded, not recomputed, by
# notebooks/vine_copula_recovery.Rmd.
this_file <- file.path(getwd(), "vine_copula_recovery.Rmd")
repo_root <- normalizePath(file.path(dirname(this_file), ".."), mustWork = FALSE)
suppressMessages({ library(hdcd); library(copula) })

set.seed(20260819)
d <- 10L; n <- 2000L; eps <- 1e-6
families <- list(
  claytonCopula(3.0), gumbelCopula(2.2), frankCopula(5.0), normalCopula(0.65),
  tCopula(0.5, df = 4), claytonCopula(1.8), gumbelCopula(3.0), frankCopula(3.5), normalCopula(0.45)
)
family_names <- c("Clayton(3.0)", "Gumbel(2.2)", "Frank(5.0)", "Gaussian(0.65)",
                   "t(0.5,df=4)", "Clayton(1.8)", "Gumbel(3.0)", "Frank(3.5)", "Gaussian(0.45)")
hinv <- function(u_given, w, cop) cCopula(cbind(u_given, w), copula = cop, indices = 2, inverse = TRUE)
w <- matrix(runif(n * d), ncol = d); w <- pmin(pmax(w, eps), 1 - eps)
U <- matrix(NA_real_, n, d); U[, 1] <- w[, 1]
for (i in 2:d) {
  u_prev <- pmin(pmax(U[, i - 1], eps), 1 - eps)
  U[, i] <- hinv(u_prev, w[, i], families[[i - 1]])
  U[, i] <- pmin(pmax(U[, i], eps), 1 - eps)
}
true_edges <- cbind(1:(d - 1), 2:d)
dag_ext <- hdcd:::.dag_from_edges(d, 2L, true_edges[, 1], true_edges[, 2])

bernstein_degree <- 4L; lambda_roughness <- 0.15; holdout_fraction <- 0.25; local_seed <- 2L
u_grid <- seq(0.02, 0.98, length.out = 60)
tail_edges <- c(1, 2, 6, 7)

fixed_fit <- hdcd:::.dag_fit_c(U, n, d, dag_ext, bernstein_degree, lambda_roughness, holdout_fraction, local_seed)
evt_fit_wide <- hdcd:::.dag_fit_c(U, n, d, dag_ext, bernstein_degree, lambda_roughness, holdout_fraction, local_seed,
                                   evt_splice_gate = 0.1, evt_splice_bandwidth = 0.15)
evt_fit <- hdcd:::.dag_fit_c(U, n, d, dag_ext, bernstein_degree, lambda_roughness, holdout_fraction, local_seed,
                              evt_splice_gate = 0.5, evt_splice_bandwidth = 0.08)

evt_wide_summary <- do.call(rbind, lapply(seq_len(d - 1), function(i) {
  node <- i + 1L
  q <- if (i %in% c(2, 7)) 0.9 else 0.1
  fitted_fixed <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, node, u_grid, q))
  fitted_evt <- exp(hdcd:::.local_fit_conditional_log_density(evt_fit_wide, node, u_grid, q))
  true_dens <- dCopula(cbind(rep(q, length(u_grid)), u_grid), families[[i]])
  data.frame(edge = sprintf("%d->%d", i, i + 1), family = family_names[i],
             spliced_family = hdcd:::.local_fit_tail_family(evt_fit_wide, node, 1L),
             cor_fixed = round(cor(fitted_fixed, true_dens), 3),
             cor_evt_wide = round(cor(fitted_evt, true_dens), 3))
}))

evt_summary <- do.call(rbind, lapply(seq_len(d - 1), function(i) {
  node <- i + 1L
  q <- if (i %in% c(2, 7)) 0.9 else 0.1
  fitted_fixed <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, node, u_grid, q))
  fitted_evt <- exp(hdcd:::.local_fit_conditional_log_density(evt_fit, node, u_grid, q))
  true_dens <- dCopula(cbind(rep(q, length(u_grid)), u_grid), families[[i]])
  data.frame(edge = sprintf("%d->%d", i, i + 1), family = family_names[i],
             spliced_family = hdcd:::.local_fit_tail_family(evt_fit, node, 1L),
             theta = round(hdcd:::.local_fit_tail_theta(evt_fit, node, 1L), 2),
             cor_fixed = round(cor(fitted_fixed, true_dens), 3),
             cor_evt = round(cor(fitted_evt, true_dens), 3))
}))
delta_kl_evt <- hdcd:::.dag_fit_kl_difference(fixed_fit, evt_fit)

evt_cond_rows <- list()
for (i in tail_edges) {
  node <- i + 1L
  q <- if (i %in% c(2, 7)) 0.9 else 0.1
  fitted_fixed <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, node, u_grid, q))
  fitted_evt <- exp(hdcd:::.local_fit_conditional_log_density(evt_fit, node, u_grid, q))
  true_dens <- dCopula(cbind(rep(q, length(u_grid)), u_grid), families[[i]])
  evt_cond_rows[[length(evt_cond_rows) + 1]] <- data.frame(
    edge = sprintf("%d->%d  %s  (z = %.1f)", i, i + 1, family_names[i], q), u = u_grid,
    density = c(true_dens, fitted_fixed, fitted_evt),
    source = rep(c("true", "hdcd (unspliced)", "hdcd (evt-spliced)"), each = length(u_grid)))
}
evt_cond_df <- do.call(rbind, evt_cond_rows)

u_zoom <- seq(0.01, 0.3, length.out = 60)
evt_zoom_rows <- list()
for (i in tail_edges) {
  node <- i + 1L
  is_gumbel <- i %in% c(2, 7)
  q <- if (is_gumbel) 0.98 else 0.02
  ug <- if (is_gumbel) 1 - u_zoom else u_zoom
  fitted_fixed <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, node, ug, q))
  fitted_evt <- exp(hdcd:::.local_fit_conditional_log_density(evt_fit, node, ug, q))
  true_dens <- dCopula(cbind(rep(q, length(ug)), ug), families[[i]])
  evt_zoom_rows[[length(evt_zoom_rows) + 1]] <- data.frame(
    edge = sprintf("%d->%d  %s  (z = %.2f)", i, i + 1, family_names[i], q), u = ug,
    density = c(true_dens, fitted_fixed, fitted_evt),
    source = rep(c("true", "hdcd (unspliced)", "hdcd (evt-spliced)"), each = length(ug)))
}
evt_zoom_df <- do.call(rbind, evt_zoom_rows)

u_fine <- seq(0.005, 0.3, length.out = 12)
fitted_fixed_fine <- exp(hdcd:::.local_fit_conditional_log_density(fixed_fit, 2L, u_fine, 0.02))
fitted_evt_fine <- exp(hdcd:::.local_fit_conditional_log_density(evt_fit, 2L, u_fine, 0.02))
true_dens_fine <- dCopula(cbind(rep(0.02, length(u_fine)), u_fine), families[[1]])
evt_corner_numbers <- data.frame(u = round(u_fine, 3), true = round(true_dens_fine, 2),
                                  unspliced = round(fitted_fixed_fine, 2), evt_spliced = round(fitted_evt_fine, 2))

saveRDS(list(evt_wide_summary = evt_wide_summary, evt_summary = evt_summary, delta_kl_evt = delta_kl_evt,
             evt_cond_df = evt_cond_df, evt_zoom_df = evt_zoom_df, evt_corner_numbers = evt_corner_numbers),
        file.path(repo_root, "notebooks", "evt_splice_experiment_results.rds"))
cat("Saved notebooks/evt_splice_experiment_results.rds\n")
