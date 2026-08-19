#ifndef HDCD_H
#define HDCD_H

/*
 * Top-level umbrella header. Through Milestone 4, the library covers
 * core numerics, the marginal Gaussian-mixture smoother, the copula
 * transform x -> u, the pairwise distance-correlation dependence
 * matrix, and the MST/persistent-topology variable ordering (spec
 * section 31). DAGs, Bernstein basis, Sinkhorn normalization, and
 * language wrappers are deliberately absent until their respective
 * milestones.
 */

#include "hdcd/status.h"
#include "hdcd/numerics.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "hdcd/dcor.h"
#include "hdcd/topology.h"

#endif /* HDCD_H */
