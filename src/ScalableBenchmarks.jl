# Smooth analytical benchmarks with three, five, and ten objectives. The GD
# solver is unconstrained, so box constraints are represented during the run by
# a differentiable cubic exterior penalty. Reported objectives omit that penalty.
const SCALABLE_BENCHMARKS = (
    "UF8",
    "UF10",
    "DTLZ2_M5",
    "DTLZ3_M5",
    "DTLZ4_M5",
    "DTLZ7_M5",
    "DTLZ3_M10",
)

const SCALABLE_BOX_PENALTY_MU = Ref(1.0e10)

@inline is_scalable_benchmark(problem::AbstractString) = problem in SCALABLE_BENCHMARKS

"""Set the common coefficient of the differentiable exterior box penalty."""
function set_scalable_box_penalty!(mu::Real)
    mu >= 0 || throw(ArgumentError("The box-penalty parameter must be nonnegative."))
    SCALABLE_BOX_PENALTY_MU[] = Float64(mu)
    return SCALABLE_BOX_PENALTY_MU[]
end

"""Return `(n,m)` for one of the bundled UF or DTLZ instances."""
function scalable_problem_dimensions(problem::AbstractString)
    problem == "UF8" && return 10, 3
    problem == "UF10" && return 10, 3
    problem == "DTLZ2_M5" && return 14, 5
    problem == "DTLZ3_M5" && return 14, 5
    problem == "DTLZ4_M5" && return 14, 5
    problem == "DTLZ7_M5" && return 24, 5
    problem == "DTLZ3_M10" && return 19, 10
    throw(ArgumentError("Unknown scalable benchmark: $(problem)."))
end

"""Return the standard lower and upper bounds of a scalable benchmark."""
function scalable_problem_bounds(problem::AbstractString, ::Type{T}=Float64) where {T<:AbstractFloat}
    n, _ = scalable_problem_dimensions(problem)
    if problem == "UF8" || problem == "UF10"
        return vcat(T[0.0, 0.0], fill(T(-2.0), n - 2)),
               vcat(T[1.0, 1.0], fill(T(2.0), n - 2))
    end
    return zeros(T, n), ones(T, n)
end

"""Generate a reproducible uniformly distributed point in the standard box."""
function scalable_inip(problem::AbstractString, seed::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    n, m = scalable_problem_dimensions(problem)
    l, u = scalable_problem_bounds(problem, T)
    rng = MersenneTwister(seed)
    x = [l[j] + (u[j] - l[j]) * rand(rng, T) for j in 1:n]
    set_problem_box_bounds!(l, u)
    return n, m, x, l, u, falses(m), true, false
end

"""Evaluate the cubic exterior penalty used to globalize the unconstrained run."""
function scalable_box_penalty(x::AbstractVector{T}, problem::AbstractString;
                              mu::Real=SCALABLE_BOX_PENALTY_MU[]) where {T<:Real}
    l, u = scalable_problem_bounds(problem)
    value = zero(T)
    mu_t = zero(T) + mu
    @inbounds for j in eachindex(x)
        if x[j] > u[j]
            value += mu_t * (x[j] - u[j])^3 / 3
        elseif x[j] < l[j]
            value += mu_t * (l[j] - x[j])^3 / 3
        end
    end
    return value
end

"""Add the gradient of the cubic exterior penalty to `g` in place."""
function add_scalable_box_penalty_gradient!(g::AbstractVector{T},
                                             x::AbstractVector{T},
                                             problem::AbstractString;
                                             mu::Real=SCALABLE_BOX_PENALTY_MU[]) where {T<:Real}
    l, u = scalable_problem_bounds(problem)
    mu_t = zero(T) + mu
    @inbounds for j in eachindex(x)
        if x[j] > u[j]
            g[j] += mu_t * (x[j] - u[j])^2
        elseif x[j] < l[j]
            g[j] -= mu_t * (l[j] - x[j])^2
        end
    end
    return g
end

"""Return the largest componentwise violation of the benchmark's standard box."""
function scalable_box_violation(problem::AbstractString, x::AbstractVector{<:Real})
    l, u = scalable_problem_bounds(problem)
    return maximum(max.(l .- x, x .- u, 0.0))
end

# UF8 and UF10 partition variables 3:n into the three objective-dependent
# groups from the original benchmark definitions.
@inline function uf_group(ind::Int)
    ind == 1 && return 1
    ind == 2 && return 2
    ind == 3 && return 0
    throw(ArgumentError("UF8 and UF10 have exactly three objectives."))
end

function uf_raw_evalf(problem::AbstractString, x::AbstractVector{T}, ind::Int) where {T<:Real}
    n = length(x)
    target = uf_group(ind)
    angle1 = T(pi / 2) * x[1]
    angle2 = T(pi / 2) * x[2]
    base = ind == 1 ? cos(angle1) * cos(angle2) :
           ind == 2 ? cos(angle1) * sin(angle2) : sin(angle1)

    total = zero(T)
    count = 0
    @inbounds for j in 3:n
        mod(j, 3) == target || continue
        y = x[j] - 2 * x[2] * sin(2 * T(pi) * x[1] + T(j * pi / n))
        total += problem == "UF8" ? y^2 : 4 * y^2 - cos(8 * T(pi) * y) + 1
        count += 1
    end
    count > 0 || error("Empty UF objective group for n=$(n), objective=$(ind).")
    return base + 2 * total / count
end

function uf_raw_evalg!(g::AbstractVector{T}, problem::AbstractString,
                       x::AbstractVector{T}, ind::Int) where {T<:Real}
    fill!(g, zero(T))
    n = length(x)
    target = uf_group(ind)
    angle1 = T(pi / 2) * x[1]
    angle2 = T(pi / 2) * x[2]
    if ind == 1
        g[1] = -T(pi / 2) * sin(angle1) * cos(angle2)
        g[2] = -T(pi / 2) * cos(angle1) * sin(angle2)
    elseif ind == 2
        g[1] = -T(pi / 2) * sin(angle1) * sin(angle2)
        g[2] = T(pi / 2) * cos(angle1) * cos(angle2)
    else
        g[1] = T(pi / 2) * cos(angle1)
    end

    group_count = Base.count(j -> mod(j, 3) == target, 3:n)
    @inbounds for j in 3:n
        mod(j, 3) == target || continue
        theta = 2 * T(pi) * x[1] + T(j * pi / n)
        y = x[j] - 2 * x[2] * sin(theta)
        hprime = problem == "UF8" ? 2 * y : 8 * y + 8 * T(pi) * sin(8 * T(pi) * y)
        factor = 2 * hprime / group_count
        g[1] += factor * (-4 * T(pi) * x[2] * cos(theta))
        g[2] += factor * (-2 * sin(theta))
        g[j] += factor
    end
    return g
end

# Shared angular product and derivatives for DTLZ2, DTLZ3, and DTLZ4.
function dtlz_shape_value_gradient(x::AbstractVector{T}, m::Int, ind::Int,
                                   alpha::Int) where {T<:Real}
    factor_values = T[]
    factor_derivatives = T[]
    factor_variables = Int[]
    cosine_count = m - ind
    @inbounds for j in 1:cosine_count
        powered = x[j]^alpha
        if !isfinite(powered)
            infinity = zero(x[j]) + Inf
            return infinity, fill(infinity, m - 1)
        end
        angle = T(pi / 2) * powered
        dangle = T(pi / 2) * alpha * x[j]^(alpha - 1)
        push!(factor_values, cos(angle))
        push!(factor_derivatives, -sin(angle) * dangle)
        push!(factor_variables, j)
    end
    if ind > 1
        j = cosine_count + 1
        powered = x[j]^alpha
        if !isfinite(powered)
            infinity = zero(x[j]) + Inf
            return infinity, fill(infinity, m - 1)
        end
        angle = T(pi / 2) * powered
        dangle = T(pi / 2) * alpha * x[j]^(alpha - 1)
        push!(factor_values, sin(angle))
        push!(factor_derivatives, cos(angle) * dangle)
        push!(factor_variables, j)
    end

    value = prod(factor_values; init=one(T))
    gradient = zeros(T, m - 1)
    @inbounds for r in eachindex(factor_values)
        product_other = one(T)
        for s in eachindex(factor_values)
            r == s || (product_other *= factor_values[s])
        end
        gradient[factor_variables[r]] = factor_derivatives[r] * product_other
    end
    return value, gradient
end

function dtlz_raw_evalf(problem::AbstractString, x::AbstractVector{T}, ind::Int,
                        m::Int) where {T<:Real}
    if problem == "DTLZ7_M5"
        ind < m && return x[ind]
        k = length(x) - m + 1
        gvalue = 1 + 9 * sum(@view x[m:end]) / k
        value = m * (1 + gvalue)
        @inbounds for j in 1:(m - 1)
            value -= x[j] * (1 + sin(3 * T(pi) * x[j]))
        end
        return value
    end

    alpha = problem == "DTLZ4_M5" ? 100 : 1
    shape, _ = dtlz_shape_value_gradient(x, m, ind, alpha)
    tail = @view x[m:end]
    if startswith(problem, "DTLZ3")
        k = length(tail)
        gvalue = 100 * (k + sum((z - T(0.5))^2 - cos(20 * T(pi) * (z - T(0.5))) for z in tail))
    else
        gvalue = sum((z - T(0.5))^2 for z in tail)
    end
    return (1 + gvalue) * shape
end

function dtlz_raw_evalg!(g::AbstractVector{T}, problem::AbstractString,
                         x::AbstractVector{T}, ind::Int, m::Int) where {T<:Real}
    fill!(g, zero(T))
    if problem == "DTLZ7_M5"
        if ind < m
            g[ind] = one(T)
            return g
        end
        @inbounds for j in 1:(m - 1)
            g[j] = -(1 + sin(3 * T(pi) * x[j]) +
                     3 * T(pi) * x[j] * cos(3 * T(pi) * x[j]))
        end
        k = length(x) - m + 1
        g[m:end] .= 9 * m / k
        return g
    end

    alpha = problem == "DTLZ4_M5" ? 100 : 1
    shape, shape_gradient = dtlz_shape_value_gradient(x, m, ind, alpha)
    tail = @view x[m:end]
    if startswith(problem, "DTLZ3")
        k = length(tail)
        gvalue = 100 * (k + sum((z - T(0.5))^2 - cos(20 * T(pi) * (z - T(0.5))) for z in tail))
        @inbounds for j in m:length(x)
            z = x[j] - T(0.5)
            g[j] = shape * 100 * (2 * z + 20 * T(pi) * sin(20 * T(pi) * z))
        end
    else
        gvalue = sum((z - T(0.5))^2 for z in tail)
        @inbounds for j in m:length(x)
            g[j] = shape * 2 * (x[j] - T(0.5))
        end
    end
    @inbounds for j in 1:(m - 1)
        g[j] = (1 + gvalue) * shape_gradient[j]
    end
    return g
end

"""Evaluate one original benchmark objective, without the exterior penalty."""
function scalable_raw_evalf(problem::AbstractString, x::AbstractVector{T}, ind::Int) where {T<:Real}
    _, m = scalable_problem_dimensions(problem)
    1 <= ind <= m || throw(BoundsError(1:m, ind))
    if problem == "UF8" || problem == "UF10"
        return uf_raw_evalf(problem, x, ind)
    end
    return dtlz_raw_evalf(problem, x, ind, m)
end

"""Evaluate the gradient of one original, unpenalized benchmark objective."""
function scalable_raw_evalg!(g::AbstractVector{T}, problem::AbstractString,
                             x::AbstractVector{T}, ind::Int) where {T<:Real}
    _, m = scalable_problem_dimensions(problem)
    1 <= ind <= m || throw(BoundsError(1:m, ind))
    if problem == "UF8" || problem == "UF10"
        return uf_raw_evalg!(g, problem, x, ind)
    end
    return dtlz_raw_evalg!(g, problem, x, ind, m)
end

"""Evaluate one objective plus the common differentiable box penalty."""
function scalable_evalf(problem::AbstractString, x::AbstractVector{T}, ind::Int) where {T<:Real}
    all(isfinite, x) || return zero(x[1]) + Inf
    penalty = scalable_box_penalty(x, problem)
    isfinite(penalty) || return penalty
    raw_value = scalable_raw_evalf(problem, x, ind)
    return raw_value + penalty
end

"""Evaluate the gradient of one penalized objective in place."""
function scalable_evalg!(g::AbstractVector{T}, problem::AbstractString,
                         x::AbstractVector{T}, ind::Int) where {T<:Real}
    if !all(isfinite, x)
        fill!(g, zero(x[1]) + Inf)
        return g
    end
    scalable_raw_evalg!(g, problem, x, ind)
    return add_scalable_box_penalty_gradient!(g, x, problem)
end

"""Evaluate a penalized objective Hessian through automatic differentiation."""
function scalable_evalh!(H::AbstractMatrix{T}, problem::AbstractString,
                         x::AbstractVector{T}, ind::Int) where {T<:AbstractFloat}
    H .= ForwardDiff.hessian(z -> scalable_evalf(problem, z, ind), x)
    return H
end

"""Return the complete original objective vector, excluding optimization penalties."""
function scalable_raw_objectives(problem::AbstractString, x::AbstractVector{T}) where {T<:Real}
    _, m = scalable_problem_dimensions(problem)
    return [scalable_raw_evalf(problem, x, ind) for ind in 1:m]
end

"""Return the complete penalized objective vector used by the solver."""
function scalable_objectives(problem::AbstractString, x::AbstractVector{T}) where {T<:Real}
    _, m = scalable_problem_dimensions(problem)
    return [scalable_evalf(problem, x, ind) for ind in 1:m]
end
