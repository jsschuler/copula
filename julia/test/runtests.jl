using Test
using HDCD

# ---- a tiny, self-contained JSON reader for fixture.json only -------------
#
# Deliberately not a dependency on JSON.jl/JSON3.jl: this package has no
# other dependencies beyond the stdlib Libdl, and fixture.json's shape
# is fully known (numbers, booleans, nested arrays, no escaped strings
# beyond the top-level keys) -- a general JSON library would be a lot of
# machinery for one narrow, controlled use.
module TinyJSON

function parse_value(s::AbstractString, i::Int)
    i = skip_ws(s, i)
    c = s[i]
    if c == '{'
        return parse_object(s, i)
    elseif c == '['
        return parse_array(s, i)
    elseif c == '"'
        return parse_string(s, i)
    elseif c == 't'
        return true, i + 4
    elseif c == 'f'
        return false, i + 5
    else
        return parse_number(s, i)
    end
end

skip_ws(s, i) = (while i <= lastindex(s) && isspace(s[i]); i = nextind(s, i); end; i)

function parse_object(s, i)
    obj = Dict{String,Any}()
    i = nextind(s, i) # skip '{'
    i = skip_ws(s, i)
    s[i] == '}' && return obj, nextind(s, i)
    while true
        i = skip_ws(s, i)
        key, i = parse_string(s, i)
        i = skip_ws(s, i)
        i = nextind(s, i) # skip ':'
        val, i = parse_value(s, i)
        obj[key] = val
        i = skip_ws(s, i)
        if s[i] == ','
            i = nextind(s, i)
        else
            break
        end
    end
    i = skip_ws(s, i)
    return obj, nextind(s, i) # skip '}'
end

function parse_array(s, i)
    arr = Any[]
    i = nextind(s, i) # skip '['
    i = skip_ws(s, i)
    s[i] == ']' && return arr, nextind(s, i)
    while true
        val, i = parse_value(s, i)
        push!(arr, val)
        i = skip_ws(s, i)
        if s[i] == ','
            i = nextind(s, i)
        else
            break
        end
    end
    i = skip_ws(s, i)
    return arr, nextind(s, i) # skip ']'
end

function parse_string(s, i)
    i = nextind(s, i) # skip opening '"'
    start = i
    while s[i] != '"'
        i = nextind(s, i)
    end
    return s[start:prevind(s, i)], nextind(s, i) # skip closing '"'
end

function parse_number(s, i)
    start = i
    while i <= lastindex(s) && (isdigit(s[i]) || s[i] in ('-', '+', '.', 'e', 'E'))
        i = nextind(s, i)
    end
    return parse(Float64, s[start:prevind(s, i)]), i
end

function parse_json(text::AbstractString)
    val, _ = parse_value(text, firstindex(text))
    return val
end

end # module TinyJSON

# jsonlist_to_matrix: fixture arrays-of-arrays parse as Vector{Any} of
# Vector{Any}; convert to a proper Float64/Int matrix.
function to_matrix(v)
    n = length(v)
    m = length(v[1])
    out = Matrix{Float64}(undef, n, m)
    for i in 1:n, j in 1:m
        out[i, j] = v[i][j]
    end
    return out
end
to_vector(v) = Float64.(v)

fixture_path = joinpath(@__DIR__, "..", "..", "python", "tests", "fixture.json")

if !isfile(fixture_path)
    @testset "fixture agreement (skipped)" begin
        @test_skip "fixture.json not found next to the source checkout"
    end
else

fixture = TinyJSON.parse_json(read(fixture_path, String))
n = Int(fixture["n"])
d = Int(fixture["d"])
X = to_matrix(fixture["X"])

@testset "marginal sigma, cdf, logpdf agree with the C fixture" begin
    eval_points = to_vector(fixture["marginal_eval_points"])
    expected_sigma = to_vector(fixture["marginal_sigma"])
    expected_cdf = to_matrix(fixture["marginal_cdf"])
    expected_logpdf = to_matrix(fixture["marginal_logpdf"])
    for j in 1:d
        m = HDCD.fit_marginal(X[:, j])
        @test isapprox(HDCD.marginal_sigma(m), expected_sigma[j]; atol = 1e-10)
        @test isapprox(HDCD.marginal_cdf(m, eval_points), expected_cdf[j, :]; atol = 1e-9)
        @test isapprox(HDCD.marginal_logpdf(m, eval_points), expected_logpdf[j, :]; atol = 1e-7)
    end
end

function fit_marginals_and_transform()
    marginals = [HDCD.fit_marginal(X[:, j]) for j in 1:d]
    U = Matrix{Float64}(undef, n, d)
    for j in 1:d
        U[:, j] = HDCD.transform_to_copula(marginals[j], X[:, j])
    end
    return U
end

@testset "copula transform agrees with the C fixture" begin
    U = fit_marginals_and_transform()
    expected = to_matrix(fixture["U_first_5_rows"])
    @test isapprox(U[1:5, :], expected; atol = 1e-9)
end

@testset "dependence matrix agrees with the C fixture" begin
    U = fit_marginals_and_transform()
    dm = HDCD.compute_dependence_matrix(U)
    dense = HDCD.dependence_matrix_dense(dm, d)
    expected = to_matrix(fixture["dependence_matrix"])
    @test isapprox(dense, expected; atol = 1e-9)
end

@testset "topology ordering agrees with the C fixture" begin
    U = fit_marginals_and_transform()
    dm = HDCD.compute_dependence_matrix(U)
    topo = HDCD.compute_topology(dm)
    ord = HDCD.topology_ordering(topo, d)
    expected_1idx = Int.(to_vector(fixture["topology_ordering"])) .+ 1 # fixture is 0-indexed
    @test ord == expected_1idx
end

@testset "DAG fit holdout scores, joint log density, kl_estimate agree with the C fixture" begin
    U = fit_marginals_and_transform()
    edges_0idx = to_matrix(fixture["dag_edges"])
    edges_1idx = Int.(edges_0idx) .+ 1
    dag = HDCD.dag_from_edges(d, 2, edges_1idx)

    options = HDCD.default_local_fit_options(
        bernstein_degree = 3, lambda_roughness = 0.15, holdout_fraction = 0.25, seed = 777,
    )
    dag_fit = HDCD.fit_dag_c(U, dag, options)
    @test HDCD.dag_fit_all_converged(dag_fit)

    scores = HDCD.dag_fit_holdout_scores(dag_fit, d)
    expected_scores = to_vector(fixture["dag_fit_holdout_scores"])
    @test isapprox(scores, expected_scores; atol = 1e-9)

    point = to_vector(fixture["joint_log_density_point"])
    value = HDCD.dag_fit_joint_log_density(dag_fit, point)
    @test isapprox(value, fixture["joint_log_density_value"]; atol = 1e-9)

    @test isapprox(HDCD.dag_fit_kl_estimate(dag_fit), fixture["kl_estimate"]; atol = 1e-9)
end

end # if fixture available

# ---- high-level API smoke tests --------------------------------------

function make_chain_data(; n = 300, rho = 0.7, seed = 1)
    rng_state = UInt64(seed == 0 ? 1 : seed)
    # deterministic, dependency-free normal generator (Box-Muller over a
    # tiny xorshift64*), matching the pattern used throughout the C and
    # other-language test suites -- avoids depending on Random's exact
    # stream (which is not guaranteed stable across Julia versions).
    function next_uniform()
        rng_state ⊻= rng_state >> 12
        rng_state ⊻= rng_state << 25
        rng_state ⊻= rng_state >> 27
        r = rng_state * 0x2545F4914F6CDD1D
        u = Float64(r >> 11) * (1.0 / 9007199254740992.0)
        return clamp(u, 1e-12, 1 - 1e-12)
    end
    next_normal() = sqrt(-2.0 * log(next_uniform())) * cos(2π * next_uniform())

    z0 = [next_normal() for _ in 1:n]
    z1 = [rho * z0[i] + sqrt(1 - rho^2) * next_normal() for i in 1:n]
    z2 = [rho * z1[i] + sqrt(1 - rho^2) * next_normal() for i in 1:n]
    return hcat(z0, z1, z2)
end

@testset "hdcd_fit -> transform -> logpdf -> fit_dag round-trip" begin
    X = make_chain_data(seed = 11)
    model = hdcd_fit(X; max_parents = 2, bernstein_degree = 3, lambda_edge = 0.05,
                      lambda_roughness = 0.15, annealing_iterations = 60, seed = 1)
    @test model.d == 3

    U = transform_copula(model, X)
    @test all(U .> 0) && all(U .< 1)

    clp = copula_logpdf(model, U)
    lp = logpdf(model, X)
    @test all(isfinite, clp)
    @test all(isfinite, lp)

    dm = dependence_matrix(model)
    @test size(dm) == (3, 3)

    ord = ordering(model)
    @test sort(ord) == [1, 2, 3]

    edges = dag(model)
    @test size(edges, 2) == 2
end

@testset "fit_dag / score_dag produce a meaningful KL comparison" begin
    X = make_chain_data(seed = 12)
    model = hdcd_fit(X; max_parents = 2, bernstein_degree = 3, annealing_iterations = 60, seed = 2)

    candidate = fit_dag(model, nothing) # independence hypothesis
    kl = score_dag(model, candidate)
    @test !isnan(kl)
    @test kl > 0 # discarding real dependency loses information vs. the reference
end

@testset "hdcd_sample raises a clear error" begin
    X = make_chain_data(n = 100, seed = 13)
    model = hdcd_fit(X; max_parents = 2, annealing_iterations = 30, seed = 3)
    @test_throws HDCD.HdcdError hdcd_sample(model, 10)
end

@testset "missing data (NaN) handled via the observed mask" begin
    X = make_chain_data(n = 150, seed = 14)
    X[1, 1] = NaN
    X[5, 2] = NaN
    model = hdcd_fit(X; max_parents = 2, annealing_iterations = 30, seed = 4)

    lp = logpdf(model, X)
    @test isnan(lp[1])
    @test isnan(lp[5])
    @test all(isfinite, lp[setdiff(1:150, [1, 5])])
end

@testset "external pointers survive an explicit GC.gc() (finalizer safety)" begin
    X = make_chain_data(n = 120, seed = 15)
    model = hdcd_fit(X; max_parents = 2, annealing_iterations = 30, seed = 5)

    GC.gc() # must not free anything still referenced by `model`
    U = transform_copula(model, X)
    @test all(isfinite, U)
    lp = logpdf(model, X)
    @test all(isfinite, lp)
end

@testset "a discarded model's handles are finalized without error" begin
    X = make_chain_data(n = 100, seed = 16)
    let
        model = hdcd_fit(X; max_parents = 2, annealing_iterations = 20, seed = 6)
        dependence_matrix(model)
    end
    GC.gc() # runs registered finalizers; must not error or crash the session
    @test true
end
