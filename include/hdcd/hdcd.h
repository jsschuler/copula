#ifndef HDCD_H
#define HDCD_H

/*
 * Top-level umbrella header. Through Milestone 9, the library covers
 * core numerics, the marginal Gaussian-mixture smoother, the copula
 * transform x -> u, the pairwise distance-correlation dependence
 * matrix, the MST/persistent-topology variable ordering, the centered
 * Bernstein tensor basis with its roughness penalty,
 * copula-preserving Sinkhorn normalization, fixed-DAG fitting with the
 * factorized joint copula density, simulated-annealing DAG search, and
 * arbitrary-DAG / held-out-KL comparison (spec section 31). Language
 * wrappers are deliberately absent until their respective milestones.
 *
 * IMPORTANT (spec section 19): the reference DAG and any alternative
 * DAG this library fits or compares are STATISTICAL factorizations for
 * density estimation, never causal claims. hdcd_dag_fit_kl_difference
 * answers "how much dependence information is lost by this
 * factorization," not "which edges are causal" -- observational fit
 * alone does not establish causal direction or distinguish
 * Markov-equivalent causal DAGs.
 */

#include "hdcd/status.h"
#include "hdcd/numerics.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "hdcd/dcor.h"
#include "hdcd/tail_dependence.h"
#include "hdcd/topology.h"
#include "hdcd/bernstein.h"
#include "hdcd/sinkhorn.h"
#include "hdcd/rng.h"
#include "hdcd/dag.h"
#include "hdcd/local_fit.h"
#include "hdcd/dag_fit.h"
#include "hdcd/annealing.h"

#endif /* HDCD_H */
