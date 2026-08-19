# High-Dimensional Copula Density Estimation
## Implementation Specification for Claude Code

### Status
Authoritative v1 engineering and mathematical specification.

### Primary goal
Implement a high-dimensional density-estimation library with:

- a C numerical core;
- Python, R, and Julia bindings;
- nonparametric smoothed marginal CDFs;
- optional extreme-value tail modeling;
- copula transformation to \([0,1]^d\);
- pairwise dependence estimation with distance correlation;
- persistent-topology/MST-based variable ordering;
- sparse DAG structure search with simulated annealing;
- differentiable conditional copula factors based on centered Bernstein tensor bases;
- continuous Sinkhorn normalization so each factor preserves copula marginals;
- KL-based model fitting and graph comparison;
- node-wise composite likelihood under arbitrary missingness;
- support for evaluating alternative DAGs, including causal DAGs with different topological orderings.

The reference DAG is a **statistical factorization for efficient density estimation**. It is not a causal claim.

---

# 1. Mathematical Overview

Let

\[
X=(X_1,\ldots,X_d)
\]

have realizations

\[
x_i=(x_{i1},\ldots,x_{id}), \qquad i=1,\ldots,n,
\]

with arbitrary missingness.

The joint density is represented as

\[
f_X(x)
=
c\!\left(F_1(x_1),\ldots,F_d(x_d)\right)
\prod_{j=1}^{d}f_j(x_j),
\]

where \(F_j\) and \(f_j\) are marginal CDFs and densities, and \(c\) is a copula density.

The pipeline is:

\[
X
\rightarrow
\widehat F_j,\widehat f_j
\rightarrow
U_j=\widehat F_j(X_j)
\rightarrow
\text{distance correlation}
\rightarrow
\text{persistent-topology ordering}
\rightarrow
\text{sparse DAG search}
\rightarrow
\text{conditional copula factors}
\rightarrow
\widehat c(U)
\rightarrow
\widehat f_X(X).
\]

The principal design goals are:

1. avoid parametric assumptions in the body of each marginal;
2. model heavy tails separately when necessary;
3. use all available marginal observations despite missing data;
4. transform the dependence problem to a compact domain;
5. preserve nonlinear pairwise dependence information;
6. exploit persistent dependence structure to obtain a useful variable ordering;
7. factor the high-dimensional copula sparsely;
8. guarantee that the fitted density is a valid copula;
9. retain differentiability and local factorization;
10. permit direct comparison with arbitrary alternative DAG factorizations.

---

# 2. Marginal Estimation

For dimension \(j\), let

\[
\mathcal O_j
=
\{i:x_{ij}\text{ is observed}\}
\]

and

\[
n_j=|\mathcal O_j|.
\]

The default bulk CDF estimator is a Gaussian-smoothed empirical CDF:

\[
\widehat F_j(x;\sigma_j)
=
\frac{1}{n_j}
\sum_{i\in\mathcal O_j}
\Phi\!\left(
\frac{x-x_{ij}}{\sigma_j}
\right).
\]

Its density is

\[
\widehat f_j(x;\sigma_j)
=
\frac{1}{n_j\sigma_j}
\sum_{i\in\mathcal O_j}
\phi\!\left(
\frac{x-x_{ij}}{\sigma_j}
\right).
\]

The derivative of the density is

\[
\widehat f_j'(x;\sigma_j)
=
-\frac{1}{n_j\sigma_j^3}
\sum_{i\in\mathcal O_j}
(x-x_{ij})
\phi\!\left(
\frac{x-x_{ij}}{\sigma_j}
\right).
\]

The smoothing parameter \(\sigma_j\) controls marginal wiggliness.

## 2.1 Bandwidth selection

The default bandwidth estimator is leave-one-out log-likelihood cross-validation:

\[
\widehat\sigma_j
=
\arg\max_{\sigma\in[\sigma_{\min,j},\sigma_{\max,j}]}
\ell_{\mathrm{LOO},j}(\sigma),
\]

with

\[
\ell_{\mathrm{LOO},j}(\sigma)
=
\sum_{i\in\mathcal O_j}
\log
\widehat f_{j,-i}(x_{ij};\sigma),
\]

and

\[
\widehat f_{j,-i}(x;\sigma)
=
\frac{1}{(n_j-1)\sigma}
\sum_{\substack{r\in\mathcal O_j\\r\neq i}}
\phi\!\left(
\frac{x-x_{rj}}{\sigma}
\right).
\]

Implementation requirements:

- optimize over \(\eta=\log\sigma\);
- use a deterministic bounded one-dimensional optimizer;
- evaluate mixture densities with log-sum-exp stabilization;
- derive default bounds from robust scale measures such as IQR or MAD;
- expose bounds and optimization tolerances to the public API;
- if EVT tail splicing is enabled, select the bulk bandwidth using the bulk sample only.

## 2.2 Optional curvature diagnostic

Define

\[
R_j(\sigma)
=
\int
\left[
\widehat f_j'(x;\sigma)
\right]^2
dx.
\]

This is **not** part of the default bandwidth objective.

It may be:

- reported as a roughness diagnostic;
- constrained optionally as \(R_j(\sigma)\le R_{\max,j}\);
- approximated by Monte Carlo integration.

A convenient proposal distribution is the smoothed empirical marginal itself:

\[
Z=x_I+\sigma\epsilon,
\qquad
I\sim\operatorname{Unif}\{1,\ldots,n_j\},
\qquad
\epsilon\sim N(0,1).
\]

Then

\[
R_j(\sigma)
=
\mathbb E_{\widehat f_j}
\left[
\frac{
\widehat f_j'(Z;\sigma)^2
}{
\widehat f_j(Z;\sigma)
}
\right].
\]

The implementation may also expose local curvature approximation tools based on Taylor-polynomial continuation when needed for boundary extensions or diagnostics.

---

# 3. Extreme-Value Tail Extension

For heavy-tailed marginals, model the tails separately from the smoothed bulk.

For dimension \(j\), optionally choose lower and upper thresholds

\[
u_j^-<u_j^+.
\]

Then use

\[
F_j(x)=
\begin{cases}
F_{j,\mathrm{lower\,EVT}}(x), & x<u_j^-\\
F_{j,\mathrm{bulk}}(x), & u_j^-\le x\le u_j^+\\
F_{j,\mathrm{upper\,EVT}}(x), & x>u_j^+.
\end{cases}
\]

Version 1 requirements:

- generalized Pareto tails are the default EVT family;
- CDF continuity at each splice is mandatory;
- density continuity is optional but supported when parameterization permits;
- thresholds may be user-specified or selected by a separate EVT module;
- the bulk smoother must not be forced to reproduce asymptotic tail behavior;
- tail logic must be modular and optional.

---

# 4. Copula Transformation

For each observed entry,

\[
u_{ij}
=
\widehat F_j(x_{ij}).
\]

If \(x_{ij}\) is missing, \(u_{ij}\) remains missing.

All copula coordinates lie in \([0,1]\), so all polynomial moments exist.

For numerical stability define a clipping rule

\[
u
\leftarrow
\min(1-\epsilon,\max(\epsilon,u)),
\]

with a configurable small \(\epsilon>0\).

The clipping convention must be identical across C, Python, R, and Julia.

---

# 5. Pairwise Dependence Matrix

For every pair \(j,k\), define the pairwise complete set

\[
\mathcal O_{jk}
=
\mathcal O_j\cap\mathcal O_k.
\]

Estimate

\[
D_{jk}
=
\operatorname{dCor}(U_j,U_k)
\]

using only rows in \(\mathcal O_{jk}\).

The resulting matrix \(D\) is a nonlinear dependence table. It must **not** be assumed positive semidefinite.

Default dissimilarity:

\[
\delta_{jk}
=
1-D_{jk}.
\]

Requirements:

- exact distance correlation is the default for moderate \(n\);
- an accelerated approximation backend may be added for large \(n\);
- the backend must be selectable;
- the number of pairwise complete rows used for each entry must be stored.

---

# 6. Persistent-Topology Ordering

The purpose of this stage is not merely to rank variables individually.

The ordering should preserve groups of variables that remain connected over substantial portions of the dependence filtration.

Construct the filtration

\[
G_\epsilon
=
\{(j,k):\delta_{jk}\le\epsilon\}.
\]

Use the \(H_0\) connectivity structure. For version 1, exploit its equivalence with the single-linkage hierarchy / minimum spanning tree rather than introducing a general persistent-homology dependency.

## 6.1 Pairwise merge level

Define

\[
\tau_{jk}
=
\inf\{
\epsilon:
j\text{ and }k
\text{ belong to the same connected component of }G_\epsilon
\}.
\]

Equivalently,

\[
\tau_{jk}
=
\min_{\text{paths }j\rightsquigarrow k}
\max_{(r,s)\text{ on path}}
\delta_{rs}.
\]

This is the single-linkage ultrametric induced by the MST.

## 6.2 Persistent affinity

Define

\[
A_{jk}
=
1-\tau_{jk}.
\]

Equivalently,

\[
A_{jk}
=
\int_0^1
\mathbf 1\{
j\sim_\epsilon k
\}
\,d\epsilon.
\]

Thus \(A_{jk}\) is the fraction of the filtration over which \(j\) and \(k\) belong to the same connected component.

## 6.3 Ordering requirement

Persistent components must form contiguous blocks in the final variable ordering.

The ordering algorithm is recursive on the single-linkage dendrogram:

1. construct the MST and associated binary or tie-resolved merge tree;
2. for each internal node, identify child components;
3. score each child by persistent-affinity centrality;
4. place the higher-scoring child first;
5. recurse within each child;
6. break exact ties deterministically by original column index unless the user supplies another tie-breaking rule.

A default component score is

\[
S(C)
=
\sum_{j\in C}
\sum_{k\neq j}
A_{jk}.
\]

Within a component \(C\), variable-level centrality is

\[
S_j^{(C)}
=
\sum_{\substack{k\in C\\k\neq j}}
A_{jk}.
\]

The implementation must preserve block contiguity rather than simply sorting all variables by a scalar score.

The output is a permutation

\[
\pi=(\pi_1,\ldots,\pi_d).
\]

---

# 7. Admissible DAGs

Given ordering \(\pi\), parents of node \(\pi_j\) are restricted to earlier variables:

\[
\operatorname{Pa}(\pi_j)
\subseteq
\{\pi_1,\ldots,\pi_{j-1}\}.
\]

This guarantees acyclicity by construction.

Hard sparsity condition:

\[
|\operatorname{Pa}(j)|
\le k_{\max}.
\]

A separate soft edge penalty is described later.

Version 1 DAG proposals must include:

- add one parent;
- remove one parent;
- swap one parent.

Moves violating the ordering or \(k_{\max}\) are invalid.

---

# 8. Conditional Copula Kernel

For node \(j\), let

\[
z
=
u_{\operatorname{Pa}(j)}.
\]

Use a positive raw kernel

\[
K_j(u,z)>0.
\]

The version 1 kernel is log-additive over parents:

\[
\log K_j(u,z)
=
\sum_{k\in\operatorname{Pa}(j)}
g_{jk}(u,z_k).
\]

No higher-order multi-parent interaction terms are included in version 1.

---

# 9. Bernstein Basis

For degree \(m\),

\[
B_{r,m}(u)
=
\binom{m}{r}
u^r(1-u)^{m-r},
\qquad
r=0,\ldots,m.
\]

Use a centered basis

\[
\widetilde B_{r,m}(u)
=
B_{r,m}(u)
-
\frac{1}{m+1},
\]

since

\[
\int_0^1
B_{r,m}(u)\,du
=
\frac{1}{m+1}.
\]

For edge \(k\to j\),

\[
g_{jk}(u,z_k)
=
\widetilde B_m(u)^\top
\Theta_{jk}
\widetilde B_m(z_k),
\]

where

\[
\Theta_{jk}
\in
\mathbb R^{(m+1)\times(m+1)}.
\]

Therefore

\[
\boxed{
\log K_j(u,z)
=
\sum_{k\in\operatorname{Pa}(j)}
\widetilde B_m(u)^\top
\Theta_{jk}
\widetilde B_m(z_k)
}
\]

is the version 1 raw conditional kernel.

The degree \(m\):

- has a modest global default;
- may be overridden by the user;
- may optionally be tuned node-wise over a small discrete grid;
- must not be jointly searched over inside every DAG proposal by default.

Bernstein polynomials are proportional to beta densities:

\[
(m+1)B_{r,m}(u)
=
\operatorname{BetaPDF}
\left(
u;r+1,m-r+1
\right).
\]

This relationship may be exploited for stable integration and derivative code.

---

# 10. Functional Roughness Penalty

Regularize coefficient surfaces through second differences.

For an edge coefficient matrix \(\Theta_{jk}\),

\[
\mathcal R(\Theta_{jk})
=
\|\Delta_u^2\Theta_{jk}\|_F^2
+
\|\Theta_{jk}(\Delta_z^2)^\top\|_F^2.
\]

The total functional penalty is

\[
\mathcal R_G
=
\sum_j
\sum_{k\in\operatorname{Pa}(j)}
\mathcal R(\Theta_{jk}).
\]

Its tuning parameter is

\[
\lambda_R\ge0.
\]

The C core must expose both the raw penalty and its gradient.

---

# 11. Copula-Preserving Sinkhorn Normalization

Local conditional normalization alone does not ensure that every \(U_j\) remains Uniform\((0,1)\).

Let \(q_j(z)\) denote the already-constructed marginal density of the parent vector \(Z_j\).

Define

\[
c_j(u\mid z)
=
a_j(u)K_j(u,z)b_j(z).
\]

Choose \(a_j\) and \(b_j\) so that:

\[
\int_0^1
c_j(u\mid z)\,du
=
1
\qquad
\forall z,
\]

and

\[
\int
c_j(u\mid z)
q_j(z)\,dz
=
1
\qquad
\forall u.
\]

The first condition makes \(c_j\) a conditional density.

The second implies

\[
p_j(u)
=
\int
c_j(u\mid z)
q_j(z)\,dz
=
1,
\]

so the unconditional marginal of \(U_j\) is uniform.

## 11.1 Continuous Sinkhorn updates

Given \(a_j\),

\[
b_j(z)
\leftarrow
\left[
\int_0^1
a_j(t)
K_j(t,z)
\,dt
\right]^{-1}.
\]

Given \(b_j\),

\[
a_j(u)
\leftarrow
\left[
\int
K_j(u,z)
b_j(z)
q_j(z)
\,dz
\right]^{-1}.
\]

Alternate until convergence.

## 11.2 Numerical implementation

Requirements:

- the integral over \(u\) is one-dimensional;
- use deterministic quadrature by default for the \(u\)-integral;
- the expectation over \(q_j(z)\) may use Monte Carlo samples from the fitted parent distribution;
- permit cached parent samples;
- use log-domain evaluation where necessary;
- define a convergence metric on both marginal constraints;
- expose tolerance and maximum iterations;
- fail clearly if convergence is not achieved;
- prevent overflow/underflow with stable exponentiation and normalization.

A default convergence metric is

\[
\max
\left\{
\sup_z
\left|
\int c_j(u\mid z)du-1
\right|,
\;
\sup_u
\left|
\int c_j(u\mid z)q_j(z)dz-1
\right|
\right\}.
\]

A discretized approximation is acceptable in implementation.

---

# 12. Validity of the Full Copula

Start with

\[
p_1(u_1)=1.
\]

Suppose

\[
p_{1:j-1}
\]

is a valid density whose one-dimensional marginals are uniform.

Add

\[
p_{1:j}(u_{1:j})
=
p_{1:j-1}(u_{1:j-1})
c_j(u_j\mid z_j).
\]

Because

\[
\int_0^1
c_j(u_j\mid z_j)\,du_j
=
1,
\]

all existing marginals remain unchanged.

Because

\[
\int
c_j(u_j\mid z_j)
q_j(z_j)\,dz_j
=
1,
\]

the new marginal \(U_j\) is uniform.

Therefore by induction,

\[
c(u_1,\ldots,u_d)
=
\prod_{j=1}^d
c_j(
u_j\mid u_{\operatorname{Pa}(j)}
)
\]

is a valid copula density.

This property is a core invariant and must be tested numerically.

---

# 13. Conditional CDFs

For a fitted conditional density,

\[
C_j(u\mid z)
=
\int_0^u
c_j(t\mid z)
\,dt.
\]

It must satisfy:

\[
C_j(0\mid z)=0,
\]

\[
C_j(1\mid z)=1,
\]

\[
\frac{\partial}{\partial u}
C_j(u\mid z)
=
c_j(u\mid z)
\ge0.
\]

The API must expose conditional CDF evaluation.

Use one-dimensional adaptive or fixed-order quadrature initially.

An analytic or semi-analytic path may be implemented later where Bernstein structure permits.

---

# 14. Full Copula Density

For a DAG \(G\),

\[
c_G(u)
=
\prod_{j=1}^d
c_j
\left(
u_j
\mid
u_{\operatorname{Pa}_G(j)}
\right).
\]

The log density is

\[
\log c_G(u)
=
\sum_{j=1}^d
\log c_j
\left(
u_j
\mid
u_{\operatorname{Pa}_G(j)}
\right).
\]

This additive factorization must be exploited for:

- likelihood evaluation;
- KL scoring;
- gradients;
- local refitting;
- caching;
- DAG search;
- alternative DAG comparison.

---

# 15. KL Objective

Let \(c^\star\) denote the reference dependence distribution being approximated.

For candidate DAG \(G\),

\[
D_{\mathrm{KL}}
(c^\star\|c_G)
=
\int
c^\star(u)
\log
\frac{
c^\star(u)
}{
c_G(u)
}
du.
\]

Since the entropy term of \(c^\star\) does not depend on \(G\),

\[
D_{\mathrm{KL}}
(c^\star\|c_G)
=
\text{constant}
-
\mathbb E_{c^\star}
[
\log c_G(U)
].
\]

Using the DAG factorization,

\[
D_{\mathrm{KL}}
(c^\star\|c_G)
=
\text{constant}
-
\sum_j
\mathbb E_{c^\star}
\left[
\log
c_j
(
U_j\mid U_{\operatorname{Pa}_G(j)}
)
\right].
\]

Thus local parent-set scores are cacheable.

For parent set \(P\),

\[
K_j(P)
=
-
\mathbb E_{c^\star}
\left[
\log c_j(U_j\mid U_P)
\right].
\]

A Monte Carlo estimate is

\[
\widehat K_j(P)
=
-\frac{1}{M}
\sum_{m=1}^M
\log
c_j
\left(
U_j^{(m)}
\mid
U_P^{(m)}
\right).
\]

For nested parent sets, the expected KL improvement is related to conditional mutual information. This gives an information-theoretic interpretation to edge additions.

---

# 16. Missing Data

Version 1 uses **node-wise composite likelihood**, not full observed-data likelihood.

For node \(j\) and parent set \(P_j\), define

\[
\mathcal O_j(P_j)
=
\left\{
i:
u_{ij}\text{ observed and }
u_{ik}\text{ observed for all }k\in P_j
\right\}.
\]

The local log contribution is

\[
\ell_j(P_j)
=
\sum_{i\in\mathcal O_j(P_j)}
\log
c_j
\left(
u_{ij}\mid u_{iP_j}
\right).
\]

Because parent-rich models may have fewer usable observations, raw summed likelihoods are not directly comparable.

Use normalized held-out or cross-validated local score:

\[
\bar\ell_j(P_j)
=
\frac{
1
}{
|\mathcal O_j(P_j)|
}
\sum_{i\in\mathcal O_j(P_j)}
\log
c_j
\left(
u_{ij}\mid u_{iP_j}
\right).
\]

Preferred comparison hierarchy:

1. compare parent sets on a common held-out subset when feasible;
2. otherwise use node-wise normalized held-out score;
3. store effective sample size for every local score;
4. warn when comparison is based on materially different missingness subsets.

Full observed-data likelihood would require

\[
L_i
=
\int
c(
u_{i,\mathrm{obs}},
u_{\mathrm{mis}}
)
du_{\mathrm{mis}},
\]

which is explicitly out of scope for version 1.

---

# 17. Simulated Annealing Objective

The graph objective is

\[
J(G)
=
\widehat D_{\mathrm{KL}}
(c^\star\|c_G)
+
\lambda_E|E(G)|
+
\lambda_R\mathcal R_G,
\]

subject to

\[
|\operatorname{Pa}(j)|
\le
k_{\max}.
\]

Parameters:

- \(k_{\max}\): hard maximum number of parents per node;
- \(\lambda_E\): soft graph sparsity penalty;
- \(\lambda_R\): functional roughness penalty on Bernstein coefficient surfaces.

For local parent set \(P_j\),

\[
J_j(P_j)
=
\widehat K_j(P_j)
+
\lambda_E|P_j|
+
\lambda_R
\sum_{k\in P_j}
\mathcal R(\Theta_{jk}).
\]

Then

\[
J(G)
=
\sum_j
J_j(P_j)
\]

up to constants shared by all graphs.

## 17.1 Acceptance rule

For proposal \(G'\),

\[
\Delta
=
J(G')-J(G).
\]

Accept if

\[
\Delta\le0.
\]

Otherwise accept with probability

\[
\exp(-\Delta/T).
\]

## 17.2 Search requirements

The implementation must provide:

- deterministic reproducibility under a seed;
- user-configurable temperature schedule;
- initial temperature;
- cooling rate or schedule object;
- iteration budget;
- proposal probabilities;
- restart option;
- best-so-far graph tracking;
- current graph tracking;
- score trace;
- acceptance-rate trace.

## 17.3 Cache

Cache every fitted local parent-set model and score:

\[
(j,P)
\mapsto
\{
\widehat K_j(P),
\Theta,
\text{normalization state},
n_{\mathrm{effective}}
\}.
\]

Changing one local edge should require refitting only the affected child factor unless cache reuse is possible.

---

# 18. Penalty Selection

Defaults:

- \(k_{\max}\) is a direct structural user control with a conservative default;
- \(\lambda_E\) is selected by validation unless explicitly provided;
- \(\lambda_R\) is selected by validation unless explicitly provided.

Penalty selection must occur outside the inner annealing loop.

Do not nest an expensive continuous penalty search inside every graph proposal.

A small validation grid is sufficient for version 1.

---

# 19. Alternative DAG and Causal-Model Comparison

The reference DAG is optimized for density approximation and computational sparsity.

It is not a causal model.

The public API must accept an arbitrary DAG \(G^\star\), including a DAG with a different topological ordering.

For such a graph:

1. validate acyclicity independently of the reference ordering;
2. obtain a valid topological ordering for that graph;
3. fit the conditional copula factors under that graph;
4. compute held-out KL/cross-entropy score;
5. compare against the reference factorization.

The library should support quantities such as

\[
\Delta_{\mathrm{KL}}
=
\widehat D_{\mathrm{KL}}
(c^\star\|c_{G^\star})
-
\widehat D_{\mathrm{KL}}
(c^\star\|c_{\mathrm{reference}}).
\]

This answers:

> how much dependence information is lost when the joint distribution is constrained to factor according to the proposed DAG?

Observational fit alone does not establish causal direction and does not distinguish Markov-equivalent causal DAGs without additional assumptions or interventions.

---

# 20. Differentiability

The model is designed to be smooth and factorized.

Version 1 should use analytic derivatives in the C core wherever practical.

Required analytic or stable derivative support:

- Bernstein basis values;
- first derivatives of Bernstein basis functions;
- raw log-kernel gradients with respect to coefficients;
- roughness penalty gradients;
- local log-density gradients;
- gradients needed by the fixed-DAG optimizer.

Do not make optimization behavior depend on Python-, R-, or Julia-specific automatic differentiation.

Language wrappers may optionally expose higher-level AD compatibility later.

---

# 21. C Core Architecture

Use a small stable public ABI with opaque handles.

Conceptual public types:

```c
typedef struct hdcd_model hdcd_model_t;
typedef struct hdcd_options hdcd_options_t;
typedef struct hdcd_dag hdcd_dag_t;
typedef struct hdcd_result hdcd_result_t;
```

The final prefix may be changed once the package name is selected, but the ABI must remain namespace-safe.

Suggested public functions:

```c
hdcd_options_t *hdcd_options_create(void);
void hdcd_options_free(hdcd_options_t *);

hdcd_model_t *hdcd_fit(
    const double *x,
    const uint8_t *observed_mask,
    size_t n,
    size_t d,
    const hdcd_options_t *options
);

void hdcd_model_free(hdcd_model_t *);

int hdcd_logpdf(
    const hdcd_model_t *,
    const double *x,
    size_t n,
    double *out
);

int hdcd_copula_logpdf(
    const hdcd_model_t *,
    const double *u,
    size_t n,
    double *out
);

int hdcd_transform_to_copula(
    const hdcd_model_t *,
    const double *x,
    const uint8_t *observed_mask,
    size_t n,
    double *u_out
);

int hdcd_sample(
    const hdcd_model_t *,
    uint64_t seed,
    size_t n,
    double *x_out
);

int hdcd_get_dependence_matrix(
    const hdcd_model_t *,
    double *out
);

int hdcd_get_ordering(
    const hdcd_model_t *,
    size_t *out
);

int hdcd_get_reference_dag(
    const hdcd_model_t *,
    hdcd_dag_t **out
);

int hdcd_fit_dag(
    const hdcd_model_t *reference,
    const hdcd_dag_t *candidate,
    const hdcd_options_t *options,
    hdcd_result_t **out
);

int hdcd_score_dag(
    const hdcd_model_t *reference,
    const hdcd_result_t *candidate,
    double *kl_out
);
```

Exact signatures may evolve, but the public ABI should remain compact and opaque.

---

# 22. Internal Module Layout

Suggested source structure:

```text
src/
  core/
    model.c
    options.c
    errors.c
    memory.c

  marginal/
    gaussian_cdf_mix.c
    gaussian_density_mix.c
    bandwidth_cv.c
    curvature.c

  evt/
    gpd.c
    splice.c
    thresholds.c

  copula/
    transform.c
    conditional.c
    joint.c

  dcor/
    dcor_exact.c
    dcor_fast.c

  topology/
    mst.c
    union_find.c
    merge_tree.c
    persistent_affinity.c
    ordering.c

  basis/
    bernstein.c
    centered_bernstein.c
    difference_penalty.c

  sinkhorn/
    normalize.c
    quadrature.c
    monte_carlo.c

  dag/
    graph.c
    proposals.c
    validation.c
    cache.c

  optimize/
    local_fit.c
    penalties.c
    annealing.c
    schedules.c

  likelihood/
    composite.c
    kl.c
    validation.c

  rng/
    rng.c

  numerics/
    logsumexp.c
    quadrature.c
    special.c
    optimizer_1d.c
    linear_algebra.c
```

---

# 23. Memory Layout and Missingness

Use dense contiguous numeric arrays plus an explicit missingness mask.

Do not encode missingness internally solely through NaN.

Input representation:

- `double *x`
- `uint8_t *observed_mask`

with identical shape and layout.

Preferred core layout: **column-major**.

R and Julia are natively column-major, and NumPy can provide Fortran-contiguous arrays.

Wrappers must convert once at the boundary rather than forcing repeated internal transposition.

---

# 24. Numerical Requirements

The C core must:

- use log-sum-exp for mixture log densities;
- avoid exponentiating large raw kernels unnecessarily;
- use stable beta/binomial coefficient evaluation;
- avoid direct factorial evaluation for Bernstein coefficients;
- use recurrence relations where appropriate;
- handle probabilities close to 0 and 1 robustly;
- use deterministic seeded RNG;
- reject invalid options before fitting;
- propagate errors through explicit status codes;
- never silently continue after NaN/Inf creation;
- document floating-point tolerances;
- permit double precision only in v1.

---

# 25. Python Binding

The Python layer should be thin.

Preferred behavior:

```python
model = hdcd.fit(
    X,
    max_parents=...,
    bernstein_degree=...,
    lambda_edge=...,
    lambda_roughness=...
)

u = model.transform(X)
lp = model.logpdf(X)
clp = model.copula_logpdf(u)
samples = model.sample(1000)

model.dependence_matrix_
model.ordering_
model.dag_

result = model.fit_dag(candidate_dag)
result.kl_divergence_
```

Requirements:

- accept NumPy arrays;
- accept NaN input but convert to explicit mask at the wrapper boundary;
- avoid copying if input is already compatible column-major `float64`;
- expose fitted components read-only where reasonable.

---

# 26. R Binding

Use `.Call`, not `.C`.

R interface should mirror Python semantics as closely as idiomatic R permits.

Conceptual functions:

```r
fit(...)
predict(model, newdata, type = "logpdf")
copula_transform(model, data)
sample(model, n)
dependence_matrix(model)
ordering(model)
dag(model)
fit_dag(model, candidate_dag)
score_dag(model, candidate_fit)
```

Requirements:

- convert `NA_real_` into the explicit observed mask;
- preserve column-major memory where possible;
- use external pointers with finalizers for C model handles.

---

# 27. Julia Binding

Julia should use direct `ccall` bindings.

Conceptual API:

```julia
model = fit(X; ...)
u = transform_copula(model, X)
logpdf(model, X)
sample(model, n)
dependence_matrix(model)
ordering(model)
dag(model)
fit_dag(model, candidate)
score_dag(model, candidate_fit)
```

Julia arrays are already column-major.

Use a small wrapper type containing the opaque C pointer and finalizer.

---

# 28. Fixed-DAG Fitting

Before implementing annealing, implement robust fixed-DAG fitting.

For every node:

1. collect usable rows for child and selected parents;
2. build centered Bernstein features;
3. optimize edge coefficient matrices;
4. apply Sinkhorn normalization;
5. compute held-out local cross-entropy/KL score;
6. store effective sample size;
7. expose convergence diagnostics.

The fixed-DAG path is a prerequisite for DAG search.

---

# 29. Testing Invariants

Tests must be mathematical, not merely structural.

## 29.1 Marginals

For each fitted marginal:

\[
0\le \widehat F_j(x)\le1,
\]

\[
\widehat f_j(x)\ge0,
\]

\[
\widehat F_j'(x)\approx\widehat f_j(x),
\]

\[
\int \widehat f_j(x)dx\approx1.
\]

LOO bandwidth selection must be reproducible.

## 29.2 Copula transform

For simulated data from a known continuous marginal model, transformed coordinates should be approximately uniform under a correctly specified fitted marginal.

## 29.3 Conditional normalization

For every tested parent configuration:

\[
\int_0^1
c_j(u\mid z)\,du
\approx1.
\]

## 29.4 Copula marginal preservation

Numerically verify:

\[
\int
c_j(u\mid z)q_j(z)\,dz
\approx1.
\]

For the full fitted copula:

\[
\int_{[0,1]^d}
c(u)\,du
\approx1,
\]

and for every dimension \(j\),

\[
\int
c(u)\,du_{-j}
\approx1.
\]

## 29.5 Non-negativity

\[
c(u)\ge0.
\]

## 29.6 Independence benchmark

For independent uniforms, the fitted copula should approach

\[
c(u)=1
\]

and the selected graph should become sparse under positive edge penalty.

## 29.7 Known copula recovery

Test at minimum:

- Gaussian copula;
- Student-\(t\) copula;
- Clayton-type asymmetric dependence;
- Gumbel-type asymmetric dependence;
- sparse synthetic DAG-generated copula.

## 29.8 Nonlinear dependence

Include synthetic relationships where Pearson correlation is near zero but distance correlation is positive.

## 29.9 Missing data

Test:

- MCAR masks;
- structured missingness patterns;
- different pairwise sample sizes;
- different local DAG-factor sample sizes;
- stable failure when a parent set has insufficient usable rows.

## 29.10 Reproducibility

With identical seed and options:

- same bandwidths;
- same MST;
- same ordering;
- same annealing trace;
- same selected graph;
- same fitted parameters within deterministic floating-point tolerance.

## 29.11 Cross-language agreement

On the same saved test fixture, Python, R, and Julia wrappers must agree on:

- transformed \(U\);
- marginal log densities;
- copula log densities;
- dependence matrix;
- ordering;
- DAG structure;
- DAG score.

---

# 30. Benchmark Suite

Include benchmark generators for:

1. independent uniforms;
2. Gaussian copula;
3. Student-\(t\) copula;
4. nonlinear bivariate dependence with zero Pearson correlation;
5. heavy-tailed marginals with GPD tails;
6. sparse DAG copulas;
7. clustered dependence with obvious persistent blocks;
8. high-dimensional sparse dependence;
9. arbitrary missingness;
10. alternative DAG comparison.

Measure:

- wall-clock fit time;
- memory;
- distance-correlation time;
- ordering time;
- fixed-DAG fit time;
- annealing time;
- cache hit rate;
- Sinkhorn iterations;
- held-out KL;
- marginal normalization errors.

---

# 31. Implementation Milestones

Claude Code should implement in this order.

## Milestone 1: core numerics and marginal smoother

Implement:

- normal PDF/CDF wrappers;
- Gaussian mixture CDF;
- Gaussian mixture density;
- derivative of density;
- log-sum-exp;
- robust scale;
- LOO bandwidth objective;
- deterministic 1D optimizer.

Acceptance criteria:

- unit tests against known values;
- density integrates to 1;
- CDF derivative agrees numerically with density;
- bandwidth selection reproducible.

## Milestone 2: copula transform

Implement:

- fitted marginal object;
- transformation \(x\mapsto u\);
- clipping;
- missingness propagation.

Acceptance criteria:

- transformed values in allowed interval;
- missingness unchanged;
- simulation test approximately uniform.

## Milestone 3: distance correlation

Implement exact pairwise distance correlation and pairwise-complete missing-data logic.

Acceptance criteria:

- symmetry;
- diagonal equals 1;
- independence benchmark near 0;
- nonlinear dependence benchmark positive;
- effective sample sizes stored.

## Milestone 4: MST and persistent ordering

Implement:

- dissimilarity matrix;
- MST;
- union-find;
- merge tree;
- merge levels \(\tau_{jk}\);
- persistent affinities \(A_{jk}\);
- contiguous recursive ordering.

Acceptance criteria:

- deterministic under tie rule;
- persistent blocks remain contiguous;
- synthetic clustered-dependence test gives expected block structure.

## Milestone 5: Bernstein kernel

Implement:

- Bernstein basis;
- centered Bernstein basis;
- derivatives;
- tensor interaction;
- second-difference penalty;
- gradients.

Acceptance criteria:

- basis identities;
- centering integrals near 0;
- analytic gradients agree with finite differences.

## Milestone 6: Sinkhorn normalization

Implement continuous/discretized Sinkhorn normalization.

Acceptance criteria:

- conditional integral error below tolerance;
- marginal-preservation error below tolerance;
- robust convergence over benchmark kernels;
- clear non-convergence diagnostics.

## Milestone 7: fixed-DAG fitting

Implement:

- DAG validation;
- local parent-set fitting;
- composite missing-data score;
- held-out KL/cross-entropy;
- parameter storage;
- factorized log density.

Acceptance criteria:

- valid known DAG fits;
- normalization invariants;
- local score decomposition;
- missing-data tests.

## Milestone 8: simulated annealing

Implement:

- add/remove/swap proposals;
- hard parent limit;
- edge penalty;
- roughness penalty;
- temperature schedule;
- score cache;
- reproducibility.

Acceptance criteria:

- cached and uncached scores agree;
- known synthetic sparse graph improves over empty graph;
- fixed seed reproduces same search trace.

## Milestone 9: alternative DAG comparison

Implement arbitrary DAG input independent of reference ordering.

Acceptance criteria:

- accepts valid DAGs with different topological orders;
- rejects cyclic graphs;
- produces held-out KL comparison;
- does not label statistical comparison as causal identification.

## Milestone 10: Python binding

Acceptance criteria:

- installable package;
- NumPy interoperability;
- all core model methods exposed;
- no numerical disagreement with C fixture tests.

## Milestone 11: R binding

Acceptance criteria:

- `.Call` interface;
- external pointer finalization;
- matrix and NA handling;
- agreement with C fixtures.

## Milestone 12: Julia binding

Acceptance criteria:

- `ccall` interface;
- finalizer;
- native column-major path;
- agreement with C fixtures.

## Milestone 13: EVT module

Implement GPD tail splicing after core copula machinery is stable.

Acceptance criteria:

- CDF continuity;
- tail tests against known parameters;
- marginal density integration.

## Milestone 14: performance optimization

Only after numerical correctness:

- vectorization;
- OpenMP where safe;
- cache locality;
- optional fast distance correlation;
- reduced Monte Carlo variance;
- profile-driven optimization.

---

# 32. Explicit Non-Goals for Version 1

Do not implement in v1:

- higher-order parent interaction surfaces;
- unrestricted basis-degree search inside DAG annealing;
- full observed-data likelihood over arbitrary missing coordinates;
- causal discovery claims;
- GPU implementation;
- language-specific model-fitting logic outside the C core;
- general persistent homology beyond \(H_0\) unless later justified;
- automatic differentiation as a runtime dependency.

---

# 33. Required Diagnostics

Every fitted model should expose:

- marginal bandwidths;
- EVT settings if used;
- copula clipping epsilon;
- distance-correlation matrix;
- pairwise effective sample-size matrix;
- MST edges;
- persistent merge levels;
- persistent affinity matrix or compressed equivalent;
- final ordering;
- selected DAG;
- parents per node;
- \(k_{\max}\);
- \(\lambda_E\);
- \(\lambda_R\);
- Bernstein degree;
- local effective sample sizes;
- local held-out scores;
- Sinkhorn convergence diagnostics;
- annealing score trace;
- annealing acceptance trace;
- best graph score;
- random seed;
- numerical tolerances.

---

# 34. Reference vs Causal Interpretation

The library must clearly distinguish:

## Reference density model

The TDA ordering and annealed sparse DAG provide a computational representation of the observational copula density.

Edges are not causal.

## Candidate causal DAG

A user may supply a scientifically motivated DAG with a different topological order.

The library may fit that factorization and compare its KL divergence against the flexible reference density.

This comparison measures distributional adequacy of the proposed factorization.

It does not by itself establish causal direction.

---

# 35. Core Mathematical Contract

The implementation is correct only if the following statements hold numerically to configured tolerance.

### Marginals

\[
\widehat f_j(x)\ge0,
\qquad
\int \widehat f_j(x)dx=1.
\]

### Copula coordinates

\[
U_j\in(0,1)
\]

after numerical clipping.

### Conditional factors

\[
c_j(u\mid z)\ge0,
\]

\[
\int_0^1c_j(u\mid z)du=1,
\]

\[
\int c_j(u\mid z)q_j(z)dz=1.
\]

### Full copula

\[
c(u)
=
\prod_j
c_j(u_j\mid u_{\operatorname{Pa}(j)}),
\]

\[
c(u)\ge0,
\]

\[
\int_{[0,1]^d}c(u)du=1,
\]

and every univariate marginal is uniform:

\[
\int c(u)du_{-j}=1.
\]

### Full density

\[
\widehat f_X(x)
=
\widehat c(
\widehat F_1(x_1),\ldots,\widehat F_d(x_d)
)
\prod_j
\widehat f_j(x_j).
\]

### Optimization

Reference DAG search minimizes:

\[
\boxed{
J(G)
=
\widehat D_{\mathrm{KL}}
(c^\star\|c_G)
+
\lambda_E|E(G)|
+
\lambda_R\mathcal R_G
}
\]

subject to

\[
\boxed{
|\operatorname{Pa}(j)|
\le
k_{\max}.
}
\]

---

# 36. Coding Rules for Claude Code

1. Do not alter the mathematical model without documenting the change.
2. Do not substitute Pearson correlation for distance correlation.
3. Do not treat the distance-correlation matrix as a covariance matrix.
4. Do not replace the persistent-block ordering with a simple global centrality sort.
5. Do not omit Sinkhorn normalization.
6. Do not assume conditional normalization alone creates a copula.
7. Do not use complete-case deletion for the entire dataset.
8. Do not compare raw summed local likelihoods when candidate parent sets use materially different sample sizes.
9. Do not make causal claims about the reference DAG.
10. Do not add high-order interactions in v1.
11. Do not push core numerical optimization into the language wrappers.
12. Do not optimize performance before the mathematical invariants pass.
13. Every numerical optimization routine must return convergence diagnostics.
14. Every stochastic routine must accept an explicit seed.
15. Every public API function must define missing-data behavior.
16. Add tests before moving to the next milestone.

---

# 37. First Claude Code Task

Start only with Milestone 1.

Create:

- build system;
- C library skeleton;
- error/status framework;
- marginal Gaussian mixture module;
- stable log-sum-exp;
- robust scale utility;
- LOO bandwidth objective;
- deterministic bounded optimizer;
- unit tests;
- a small C example.

Do not implement DAGs, distance correlation, topology, Sinkhorn normalization, or wrappers until Milestone 1 passes its acceptance tests.

At the end of Milestone 1, report:

- files created;
- public API introduced;
- tests run;
- numerical tolerances used;
- benchmark timing;
- any deviation from this specification.

