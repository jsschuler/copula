#ifndef HDCD_H
#define HDCD_H

/*
 * Top-level umbrella header. Through Milestone 5, the library covers
 * core numerics, the marginal Gaussian-mixture smoother, the copula
 * transform x -> u, the pairwise distance-correlation dependence
 * matrix, the MST/persistent-topology variable ordering, and the
 * centered Bernstein tensor basis with its roughness penalty (spec
 * section 31). DAGs, Sinkhorn normalization, and language wrappers are
 * deliberately absent until their respective milestones.
 */

#include "hdcd/status.h"
#include "hdcd/numerics.h"
#include "hdcd/marginal.h"
#include "hdcd/copula.h"
#include "hdcd/dcor.h"
#include "hdcd/topology.h"
#include "hdcd/bernstein.h"

#endif /* HDCD_H */
