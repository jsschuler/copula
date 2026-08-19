#ifndef HDCD_H
#define HDCD_H

/*
 * Top-level umbrella header. Through Milestone 7, the library covers
 * core numerics, the marginal Gaussian-mixture smoother, the copula
 * transform x -> u, the pairwise distance-correlation dependence
 * matrix, the MST/persistent-topology variable ordering, the centered
 * Bernstein tensor basis with its roughness penalty,
 * copula-preserving Sinkhorn normalization, and fixed-DAG fitting with
 * the factorized joint copula density (spec section 31). Simulated
 * annealing / DAG search and language wrappers are deliberately absent
 * until their respective milestones.
 */

#include "hdcd/status.h"
#include "hdcd/numerics.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "hdcd/dcor.h"
#include "hdcd/topology.h"
#include "hdcd/bernstein.h"
#include "hdcd/sinkhorn.h"
#include "hdcd/rng.h"
#include "hdcd/dag.h"
#include "hdcd/local_fit.h"
#include "hdcd/dag_fit.h"

#endif /* HDCD_H */
