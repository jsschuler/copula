"""
    HDCD

Julia bindings (spec section 27) for the hdcd C library: nonparametric
marginal smoothing, copula transformation, distance-correlation
dependence estimation, persistent-topology variable ordering, sparse
DAG factorization via simulated annealing, and centered Bernstein
tensor conditional copula factors.

Uses direct `ccall` bindings (spec section 27: "Julia should use direct
ccall bindings"), not a generated wrapper -- deliberately as mechanical
as the Python/R bindings (spec section 36 rule 11: no new statistical
logic in the language layer). Julia arrays are natively column-major,
exactly matching hdcd's core layout (spec section 23), so no
transposition or copy is needed anywhere in this module -- the same
property R's binding (Milestone 11) already benefited from.
"""
module HDCD

using Libdl

export hdcd_fit, transform_copula, logpdf, copula_logpdf, hdcd_sample,
       dependence_matrix, ordering, dag, fit_dag, score_dag,
       HdcdModel, HdcdDagFit, HdcdError

# ---- status codes (hdcd/status.h) -----------------------------------------

const HDCD_OK = Cint(0)
const HDCD_ERROR_NOT_CONVERGED = Cint(2)

struct HdcdError <: Exception
    message::String
end
Base.showerror(io::IO, e::HdcdError) = print(io, "HdcdError: ", e.message)

# ---- library loading --------------------------------------------------

function _candidate_library_paths()
    paths = String[]
    env = get(ENV, "HDCD_LIBRARY_PATH", "")
    !isempty(env) && push!(paths, env)

    libname = Sys.isapple() ? "libhdcd.dylib" : "libhdcd.so"
    here = @__DIR__
    # Development fallback: repo checkout with the C library built via
    # `make shared` (julia/src/ -> julia/ -> repo root -> build/).
    push!(paths, normpath(joinpath(here, "..", "..", "build", libname)))
    return paths
end

function _load_library()
    tried = String[]
    for path in _candidate_library_paths()
        push!(tried, path)
        if isfile(path)
            return path
        end
    end
    error(
        "could not find the hdcd shared library. Tried:\n  " *
        join(tried, "\n  ") *
        "\nBuild it with `make shared` in the repository root, or set " *
        "the HDCD_LIBRARY_PATH environment variable to point at " *
        "libhdcd.dylib/libhdcd.so directly."
    )
end

const HDCD_LIB = _load_library()

function status_message(status::Cint)
    ptr = ccall((:hdcd_status_message, HDCD_LIB), Ptr{UInt8}, (Cint,), status)
    return unsafe_string(ptr)
end

function check_status(status::Cint; allow_not_converged::Bool = false)
    status == HDCD_OK && return
    (allow_not_converged && status == HDCD_ERROR_NOT_CONVERGED) && return
    throw(HdcdError(status_message(status)))
end

# NaN (including Julia's NaN and any imported R/Python "missing" sentinel
# that round-trips as NaN) means "missing" (spec section 27 mirrors
# section 26's NA handling): converted to an explicit mask at this
# wrapper boundary, never passed through as a sentinel into the C core.
mask_from_nan(x::AbstractArray{Float64}) = UInt8.(.!isnan.(x))

# ---- struct layouts (must match the C headers EXACTLY) --------------------

struct HdcdBandwidthResult
    sigma::Cdouble
    eta::Cdouble
    loglik::Cdouble
    lower::Cdouble
    upper::Cdouble
    iterations::Cint
    converged::Cint
    status::Cint
end

struct HdcdSinkhornOptions
    n_quadrature_nodes::Csize_t
    tol::Cdouble
    max_iterations::Cint
end

struct HdcdLocalFitOptions
    bernstein_degree::Csize_t
    lambda_roughness::Cdouble
    holdout_fraction::Cdouble
    seed::UInt64
    theta_max_iterations::Csize_t
    theta_tol::Cdouble
    sinkhorn_options::HdcdSinkhornOptions
    # Optional non-global, per-node learned roughness penalty (spec
    # section 18; see DECISIONS.md). A NULL pointer / 0 size (the
    # default_local_fit_options default below) disables this and uses
    # `lambda_roughness` verbatim -- exactly as before these fields
    # existed.
    lambda_roughness_grid::Ptr{Cdouble}
    lambda_roughness_grid_size::Csize_t
    roughness_validation_fraction::Cdouble
    # Optional non-global, PER-NODE learned Bernstein degree, gated by an
    # empirical tail-dependence diagnostic (see DECISIONS.md's
    # "tail-dependence-informed bernstein_degree selection"). Same
    # NULL/0-size-disables-it convention as the roughness fields above.
    bernstein_degree_grid::Ptr{Csize_t}
    bernstein_degree_grid_size::Csize_t
    tail_dependence_gate::Cdouble
    tail_dependence_k::Csize_t
    # Anisotropic (corner-relaxed) roughness penalty (see DECISIONS.md's
    # "anisotropic (corner-relaxed) roughness penalty"). 0.0 (the
    # default_local_fit_options default below) recovers the original
    # uniform penalty exactly.
    corner_relief::Cdouble
    # Copula-level EVT tail-splice (see DECISIONS.md's "copula-level EVT
    # tail-splice"). 0.0 (the default_local_fit_options default below)
    # disables it entirely -- the plain Bernstein kernel is used
    # unmodified.
    evt_splice_gate::Cdouble
    evt_splice_bandwidth::Cdouble
end

function default_local_fit_options(; bernstein_degree, lambda_roughness, holdout_fraction, seed,
                                    theta_max_iterations = 0, theta_tol = 0.0)
    HdcdLocalFitOptions(
        Csize_t(bernstein_degree), Cdouble(lambda_roughness), Cdouble(holdout_fraction),
        UInt64(seed), Csize_t(theta_max_iterations), Cdouble(theta_tol),
        HdcdSinkhornOptions(Csize_t(0), Cdouble(0), Cint(0)),
        Ptr{Cdouble}(C_NULL), Csize_t(0), Cdouble(0),
        Ptr{Csize_t}(C_NULL), Csize_t(0), Cdouble(0), Csize_t(0),
        Cdouble(0),
        Cdouble(0), Cdouble(0),
    )
end

struct HdcdAnnealingOptions
    k_max::Csize_t
    lambda_edge::Cdouble
    ordering::Ptr{Csize_t}
    local_fit_options::HdcdLocalFitOptions
    initial_temperature::Cdouble
    cooling_rate::Cdouble
    max_iterations::Csize_t
    restarts::Csize_t
    p_add::Cdouble
    p_remove::Cdouble
    p_swap::Cdouble
    seed::UInt64
    initial_dag::Ptr{Cvoid}
end

struct HdcdMstEdge
    j::Csize_t
    k::Csize_t
    weight::Cdouble
end

# ---- wrapper types with finalizers (spec section 27) -----------------------

mutable struct Marginal
    ptr::Ptr{Cvoid}
    function Marginal(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do m
            if m.ptr != C_NULL
                ccall((:hdcd_marginal_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), m.ptr)
                m.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct DependenceMatrixHandle
    ptr::Ptr{Cvoid}
    function DependenceMatrixHandle(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do h
            if h.ptr != C_NULL
                ccall((:hdcd_dependence_matrix_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), h.ptr)
                h.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct TopologyHandle
    ptr::Ptr{Cvoid}
    function TopologyHandle(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do h
            if h.ptr != C_NULL
                ccall((:hdcd_topology_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), h.ptr)
                h.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct DagHandle
    ptr::Ptr{Cvoid}
    function DagHandle(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do h
            if h.ptr != C_NULL
                ccall((:hdcd_dag_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), h.ptr)
                h.ptr = C_NULL
            end
        end
        return obj
    end
end

mutable struct DagFitHandle
    ptr::Ptr{Cvoid}
    function DagFitHandle(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do h
            if h.ptr != C_NULL
                ccall((:hdcd_dag_fit_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), h.ptr)
                h.ptr = C_NULL
            end
        end
        return obj
    end
end

# ---- marginal (hdcd/marginal.h) -------------------------------------------

function fit_marginal(x::Vector{Float64}; sigma_min = -1.0, sigma_max = -1.0, tol = 1e-6, max_iter = 200)
    n = length(x)
    mask = mask_from_nan(x)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:hdcd_marginal_fit, HDCD_LIB), Cint,
        (Ptr{Cdouble}, Ptr{UInt8}, Csize_t, Cdouble, Cdouble, Cdouble, Cint, Ref{Ptr{Cvoid}}),
        x, mask, n, sigma_min, sigma_max, tol, max_iter, out_ptr,
    )
    check_status(status)
    return Marginal(out_ptr[])
end

function marginal_cdf(m::Marginal, eval_points::Vector{Float64})
    out = Vector{Float64}(undef, length(eval_points))
    status = ccall(
        (:hdcd_marginal_cdf, HDCD_LIB), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}),
        m.ptr, eval_points, length(eval_points), out,
    )
    check_status(status)
    return out
end

function marginal_logpdf(m::Marginal, eval_points::Vector{Float64})
    out = Vector{Float64}(undef, length(eval_points))
    status = ccall(
        (:hdcd_marginal_logpdf, HDCD_LIB), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}),
        m.ptr, eval_points, length(eval_points), out,
    )
    check_status(status)
    return out
end

function marginal_sigma(m::Marginal)
    bw = ccall((:hdcd_marginal_bandwidth_result, HDCD_LIB), HdcdBandwidthResult, (Ptr{Cvoid},), m.ptr)
    return bw.sigma
end

# ---- copula transform (hdcd/copula.h) --------------------------------------

function transform_to_copula(m::Marginal, x::Vector{Float64}; epsilon = 0.0)
    n = length(x)
    mask = mask_from_nan(x)
    out = Vector{Float64}(undef, n)
    status = ccall(
        (:hdcd_transform_to_copula, HDCD_LIB), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{UInt8}, Csize_t, Cdouble, Ptr{Cdouble}),
        m.ptr, x, mask, n, epsilon, out,
    )
    check_status(status)
    return out
end

# ---- dependence matrix / topology (hdcd/dcor.h, hdcd/topology.h) -----------

function compute_dependence_matrix(U::Matrix{Float64})
    n, d = size(U)
    mask = mask_from_nan(U)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:hdcd_compute_dependence_matrix, HDCD_LIB), Cint,
        (Ptr{Cdouble}, Ptr{UInt8}, Csize_t, Csize_t, Ref{Ptr{Cvoid}}),
        U, mask, n, d, out_ptr,
    )
    check_status(status)
    return DependenceMatrixHandle(out_ptr[])
end

function dependence_matrix_dense(dm::DependenceMatrixHandle, d::Integer)
    out = Matrix{Float64}(undef, d, d)
    for j in 0:(d - 1), k in 0:(d - 1)
        out[j + 1, k + 1] = ccall(
            (:hdcd_dependence_matrix_get, HDCD_LIB), Cdouble,
            (Ptr{Cvoid}, Csize_t, Csize_t), dm.ptr, j, k,
        )
    end
    return out
end

function compute_topology(dm::DependenceMatrixHandle)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall((:hdcd_compute_topology, HDCD_LIB), Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), dm.ptr, out_ptr)
    check_status(status)
    return TopologyHandle(out_ptr[])
end

function topology_ordering(topo::TopologyHandle, d::Integer)
    ptr = ccall((:hdcd_topology_ordering, HDCD_LIB), Ptr{Csize_t}, (Ptr{Cvoid},), topo.ptr)
    zero_indexed = unsafe_wrap(Array, ptr, d; own = false)
    return Int.(zero_indexed) .+ 1 # 1-indexed, and copy out of C-owned memory
end

# ---- DAG (hdcd/dag.h) -------------------------------------------------------

function dag_create(d::Integer, k_max::Integer)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:hdcd_dag_create, HDCD_LIB), Cint, (Csize_t, Csize_t, Ref{Ptr{Cvoid}}), d, k_max, out_ptr
    )
    check_status(status)
    return DagHandle(out_ptr[])
end

function dag_from_edges(d::Integer, k_max::Integer, edges_1idx::AbstractMatrix{<:Integer})
    n_edges = size(edges_1idx, 1)
    parents = Csize_t.(edges_1idx[:, 1] .- 1)
    children = Csize_t.(edges_1idx[:, 2] .- 1)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:hdcd_dag_from_edges, HDCD_LIB), Cint,
        (Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Ref{Ptr{Cvoid}}),
        d, k_max, n_edges > 0 ? parents : C_NULL, n_edges > 0 ? children : C_NULL, n_edges, out_ptr,
    )
    check_status(status)
    return DagHandle(out_ptr[])
end

function dag_add_edge!(dag::DagHandle, parent_1idx::Integer, child_1idx::Integer)
    status = ccall(
        (:hdcd_dag_add_edge, HDCD_LIB), Cint, (Ptr{Cvoid}, Csize_t, Csize_t),
        dag.ptr, parent_1idx - 1, child_1idx - 1,
    )
    check_status(status)
    return nothing
end

function dag_clone(dag::DagHandle)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall((:hdcd_dag_clone, HDCD_LIB), Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), dag.ptr, out_ptr)
    check_status(status)
    return DagHandle(out_ptr[])
end

function dag_edges(dag::DagHandle, d::Integer)
    rows = Tuple{Int,Int}[]
    for child in 0:(d - 1)
        np = ccall((:hdcd_dag_n_parents, HDCD_LIB), Csize_t, (Ptr{Cvoid}, Csize_t), dag.ptr, child)
        np == 0 && continue
        parents = Vector{Csize_t}(undef, np)
        ccall(
            (:hdcd_dag_parents, HDCD_LIB), Cint, (Ptr{Cvoid}, Csize_t, Ptr{Csize_t}),
            dag.ptr, child, parents,
        )
        for p in parents
            push!(rows, (Int(p) + 1, child + 1))
        end
    end
    out = Matrix{Int}(undef, length(rows), 2)
    for (i, (p, c)) in enumerate(rows)
        out[i, 1] = p
        out[i, 2] = c
    end
    return out
end

# ---- DAG fit (hdcd/dag_fit.h) -----------------------------------------------

function fit_dag_c(U::Matrix{Float64}, dag::DagHandle, options::HdcdLocalFitOptions)
    n, d = size(U)
    mask = mask_from_nan(U)
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        (:hdcd_dag_fit, HDCD_LIB), Cint,
        (Ptr{Cdouble}, Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid}, Ref{HdcdLocalFitOptions}, Ref{Ptr{Cvoid}}),
        U, mask, n, d, dag.ptr, Ref(options), out_ptr,
    )
    check_status(status; allow_not_converged = true)
    return DagFitHandle(out_ptr[])
end

function dag_fit_joint_log_density(fit::DagFitHandle, u_point::Vector{Float64})
    out = Ref{Cdouble}(0.0)
    status = ccall(
        (:hdcd_dag_fit_joint_log_density, HDCD_LIB), Cint,
        (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ref{Cdouble}),
        fit.ptr, u_point, length(u_point), out,
    )
    check_status(status)
    return out[]
end

function dag_fit_holdout_scores(fit::DagFitHandle, d::Integer)
    out = Vector{Float64}(undef, d)
    for j in 0:(d - 1)
        node = ccall((:hdcd_dag_fit_node, HDCD_LIB), Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t), fit.ptr, j)
        out[j + 1] = ccall((:hdcd_local_fit_holdout_score, HDCD_LIB), Cdouble, (Ptr{Cvoid},), node)
    end
    return out
end

dag_fit_all_converged(fit::DagFitHandle) =
    ccall((:hdcd_dag_fit_all_converged, HDCD_LIB), Cint, (Ptr{Cvoid},), fit.ptr) != 0

dag_fit_kl_estimate(fit::DagFitHandle) =
    ccall((:hdcd_dag_fit_kl_estimate, HDCD_LIB), Cdouble, (Ptr{Cvoid},), fit.ptr)

dag_fit_kl_difference(candidate::DagFitHandle, reference::DagFitHandle) =
    ccall((:hdcd_dag_fit_kl_difference, HDCD_LIB), Cdouble, (Ptr{Cvoid}, Ptr{Cvoid}), candidate.ptr, reference.ptr)

# ---- annealing (hdcd/annealing.h) -------------------------------------------

function run_annealing(U::Matrix{Float64}, ordering_1idx::Vector{Int};
                        k_max, lambda_edge, local_fit_options::HdcdLocalFitOptions,
                        initial_temperature, cooling_rate, max_iterations, restarts,
                        p_add, p_remove, p_swap, seed)
    n, d = size(U)
    mask = mask_from_nan(U)
    ordering0 = Csize_t.(ordering_1idx .- 1)

    options = HdcdAnnealingOptions(
        Csize_t(k_max), Cdouble(lambda_edge), pointer(ordering0), local_fit_options,
        Cdouble(initial_temperature), Cdouble(cooling_rate), Csize_t(max_iterations), Csize_t(restarts),
        Cdouble(p_add), Cdouble(p_remove), Cdouble(p_swap), UInt64(seed), C_NULL,
    )

    # `options.ordering` is a raw pointer captured manually via
    # `pointer(ordering0)` above and embedded inside the `options`
    # struct -- ccall's automatic argument-preservation only protects
    # arrays passed DIRECTLY as arguments (like U/mask below), not a
    # pointer stashed inside a struct value, so `ordering0` must be kept
    # alive explicitly for the duration of this specific call or the GC
    # could free/move it first and hdcd would read garbage.
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = GC.@preserve ordering0 ccall(
        (:hdcd_run_annealing, HDCD_LIB), Cint,
        (Ptr{Cdouble}, Ptr{UInt8}, Csize_t, Csize_t, Ref{HdcdAnnealingOptions}, Ref{Ptr{Cvoid}}),
        U, mask, n, d, Ref(options), out_ptr,
    )
    check_status(status; allow_not_converged = true)
    result_ptr = out_ptr[]

    best = ccall((:hdcd_annealing_best_dag, HDCD_LIB), Ptr{Cvoid}, (Ptr{Cvoid},), result_ptr)
    best_score = ccall((:hdcd_annealing_best_score, HDCD_LIB), Cdouble, (Ptr{Cvoid},), result_ptr)
    clone_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    clone_status = ccall((:hdcd_dag_clone, HDCD_LIB), Cint, (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), best, clone_ptr)
    ccall((:hdcd_annealing_result_free, HDCD_LIB), Cvoid, (Ptr{Cvoid},), result_ptr)
    check_status(clone_status)

    return DagHandle(clone_ptr[]), best_score
end

# ---- high-level API (spec section 27's conceptual usage) -------------------

"""
    HdcdModel

A fitted hdcd pipeline: marginals -> copula transform -> dependence
matrix -> persistent-topology ordering -> annealed reference DAG ->
fixed-DAG fit (spec section 1's pipeline, driven end to end).
"""
struct HdcdModel
    marginals::Vector{Marginal}
    dag::DagHandle
    dag_fit::DagFitHandle
    dependence_matrix_handle::DependenceMatrixHandle
    topology_handle::TopologyHandle
    d::Int
    max_parents::Int
    local_fit_options::HdcdLocalFitOptions
    U::Matrix{Float64}
    best_score::Float64
end

function Base.show(io::IO, m::HdcdModel)
    print(io, "HdcdModel(d=$(m.d), max_parents=$(m.max_parents), edges=$(size(dag(m), 1)), best_score=$(round(m.best_score, digits=4)))")
end

"""
    hdcd_fit(X; max_parents=2, bernstein_degree=3, lambda_edge=0.05,
             lambda_roughness=0.15, holdout_fraction=0.25, seed=0,
             initial_temperature=0.5, cooling_rate=0.95,
             annealing_iterations=150, annealing_restarts=1,
             p_add=1.0, p_remove=1.0, p_swap=1.0) -> HdcdModel

Fit the full pipeline. `X` is an n x d matrix; `NaN` marks a missing
entry (spec section 27, mirroring section 26's NA handling).
"""
function hdcd_fit(X::AbstractMatrix{<:Real}; max_parents = 2, bernstein_degree = 3,
                   lambda_edge = 0.05, lambda_roughness = 0.15, holdout_fraction = 0.25,
                   seed = 0, initial_temperature = 0.5, cooling_rate = 0.95,
                   annealing_iterations = 150, annealing_restarts = 1,
                   p_add = 1.0, p_remove = 1.0, p_swap = 1.0)
    X = Matrix{Float64}(X)
    n, d = size(X)

    marginals = Vector{Marginal}(undef, d)
    U = Matrix{Float64}(undef, n, d)
    for j in 1:d
        marginals[j] = fit_marginal(X[:, j])
        U[:, j] = transform_to_copula(marginals[j], X[:, j])
    end

    dm = compute_dependence_matrix(U)
    topo = compute_topology(dm)
    ord = topology_ordering(topo, d)

    local_seed = seed + 1
    local_fit_options = default_local_fit_options(
        bernstein_degree = bernstein_degree, lambda_roughness = lambda_roughness,
        holdout_fraction = holdout_fraction, seed = local_seed,
    )

    reference_dag, best_score = run_annealing(
        U, ord;
        k_max = max_parents, lambda_edge = lambda_edge, local_fit_options = local_fit_options,
        initial_temperature = initial_temperature, cooling_rate = cooling_rate,
        max_iterations = annealing_iterations, restarts = annealing_restarts,
        p_add = p_add, p_remove = p_remove, p_swap = p_swap, seed = seed,
    )

    dag_fit_handle = fit_dag_c(U, reference_dag, local_fit_options)

    return HdcdModel(marginals, reference_dag, dag_fit_handle, dm, topo, d, max_parents,
                      local_fit_options, U, best_score)
end

"""
    transform_copula(model, X) -> Matrix

Transform raw data to the copula scale using the model's fitted
marginals (spec section 4).
"""
function transform_copula(model::HdcdModel, X::AbstractMatrix{<:Real})
    X = Matrix{Float64}(X)
    n = size(X, 1)
    U = Matrix{Float64}(undef, n, model.d)
    for j in 1:model.d
        U[:, j] = transform_to_copula(model.marginals[j], X[:, j])
    end
    return U
end

"""
    copula_logpdf(model, u) -> Vector

Factorized joint copula log-density log c_G(u), row-wise (spec section
14). Rows with any NaN are reported as NaN (spec section 16: full
likelihood under missing coordinates is out of scope for v1).
"""
function copula_logpdf(model::HdcdModel, U::AbstractMatrix{<:Real})
    U = Matrix{Float64}(U)
    n = size(U, 1)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        row = U[i, :]
        if any(isnan, row)
            out[i] = NaN
        else
            out[i] = dag_fit_joint_log_density(model.dag_fit, row)
        end
    end
    return out
end

"""
    logpdf(model, X) -> Vector

log f_X(x) = log c_G(u) + sum_j log f_j(x_j) (spec section 35).
"""
function logpdf(model::HdcdModel, X::AbstractMatrix{<:Real})
    X = Matrix{Float64}(X)
    n = size(X, 1)
    marginal_term = zeros(Float64, n)
    any_missing = falses(n)
    for j in 1:model.d
        col = X[:, j]
        missing_mask = isnan.(col)
        any_missing .|= missing_mask
        observed = .!missing_mask
        if any(observed)
            marginal_term[observed] .+= marginal_logpdf(model.marginals[j], col[observed])
        end
    end
    U = transform_copula(model, X)
    copula_term = copula_logpdf(model, U)
    result = copula_term .+ marginal_term
    result[any_missing] .= NaN
    return result
end

"""
    hdcd_sample(model, n)

NOT IMPLEMENTED: the C core has no sampling routine. Spec section 21
names `hdcd_sample` in its architecture sketch, and section 27 shows
`sample(model, n)` as conceptual usage, but no version-1 milestone
(spec section 31) actually schedules building it in the C core -- the
same gap already documented (and hit) in the Python (Milestone 10) and
R (Milestone 11) bindings. See DECISIONS.md.
"""
function hdcd_sample(model::HdcdModel, n::Integer)
    throw(HdcdError(
        "hdcd_sample is not implemented: the C core has no sampling routine " *
        "(spec section 21 names hdcd_sample in its architecture sketch, but no " *
        "version-1 milestone schedules building it -- see DECISIONS.md)."
    ))
end

"""
    dependence_matrix(model) -> Matrix

The fitted pairwise distance-correlation dependence matrix (spec section 5).
"""
dependence_matrix(model::HdcdModel) = dependence_matrix_dense(model.dependence_matrix_handle, model.d)

"""
    ordering(model) -> Vector{Int}

The persistent-topology variable ordering (spec section 6), 1-indexed.
"""
ordering(model::HdcdModel) = topology_ordering(model.topology_handle, model.d)

"""
    dag(model) -> Matrix{Int}

The reference DAG's edges, as an (n_edges x 2) (parent, child) matrix, 1-indexed.
"""
dag(model::HdcdModel) = dag_edges(model.dag, model.d)

"""
    HdcdDagFit

An alternative candidate DAG's fit, from [`fit_dag`](@ref). Pass to
[`score_dag`](@ref) to compare against a model's reference DAG.
"""
struct HdcdDagFit
    dag::DagHandle
    dag_fit::DagFitHandle
end

"""
    fit_dag(model, candidate_edges) -> HdcdDagFit

Fit an alternative candidate DAG over the SAME training data (spec
section 19). `candidate_edges` is an (n_edges x 2) (parent, child)
matrix, 1-indexed, or `nothing`/an empty matrix for the independence
(empty) graph.
"""
function fit_dag(model::HdcdModel, candidate_edges::Union{Nothing,AbstractMatrix{<:Integer}})
    d = model.d
    dag_handle = if candidate_edges === nothing || size(candidate_edges, 1) == 0
        dag_from_edges(d, model.max_parents, Matrix{Int}(undef, 0, 2))
    else
        dag_from_edges(d, model.max_parents, candidate_edges)
    end
    fit_handle = fit_dag_c(model.U, dag_handle, model.local_fit_options)
    return HdcdDagFit(dag_handle, fit_handle)
end

"""
    score_dag(model, candidate_fit) -> Float64

Held-out KL comparison of a candidate DAG fit against `model`'s
reference DAG (spec section 19):
`Delta_KL = kl_estimate(candidate) - kl_estimate(reference)`; positive
means the candidate fits worse than the reference.

IMPORTANT: this is a purely statistical, observational comparison of
distributional fit. It does not establish causal direction and does not
distinguish Markov-equivalent causal DAGs (spec section 19's closing
paragraph).
"""
score_dag(model::HdcdModel, candidate_fit::HdcdDagFit) =
    dag_fit_kl_difference(candidate_fit.dag_fit, model.dag_fit)

end # module HDCD
