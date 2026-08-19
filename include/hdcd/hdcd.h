#ifndef HDCD_H
#define HDCD_H

/*
 * Top-level umbrella header. Through Milestone 3, the library covers
 * core numerics, the marginal Gaussian-mixture smoother, the copula
 * transform x -> u, and the pairwise distance-correlation dependence
 * matrix (spec section 31). DAGs, topology, Sinkhorn normalization, and
 * language wrappers are deliberately absent until their respective
 * milestones.
 */

#include "hdcd/status.h"
#include "hdcd/numerics.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "hdcd/dcor.h"

#endif /* HDCD_H */
