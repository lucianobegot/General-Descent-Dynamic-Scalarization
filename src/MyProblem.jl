# ======================================================================
# Problem definitions, transformations, and initialization
# ======================================================================

# The optimization method is unconstrained. Applied RE boxes are therefore
# represented by a smooth logistic map from internal variables to physical
# variables. Their violation objective is smoothed only while optimizing and
# is recomputed exactly before points are reported by `run_main.jl`.

const BOX_TRANSFORM_PROBLEMS = ("RE21", "RE34", "RE37", "RE41", "RE42", "RE61")
const BOX_MAP_EPS = 1.0e-8
const CURRENT_BOX_L = Ref(Vector{Float64}())
const CURRENT_BOX_U = Ref(Vector{Float64}())
const RE_SMOOTH_REL_MU = Ref(1.0e-6)

const RE_APPLIED_IDEAL = Dict(
    "RE41" => [15.576004, 3.58525, 10.61064375, 0.0],
    "RE42" => [-2756.2590400638524, 3962.557843228888, 1947.880856925791, 0.0],
    "RE61" => [63840.2774, 30.0, 285346.896494, 183749.967061, 7.22222222222, 0.0],
)

const RE_APPLIED_NADIR = Dict(
    "RE41" => [39.2905121788, 4.42725, 13.09138125, 9.49401929991],
    "RE42" => [-1010.5229595219643, 13827.138456300128, 2611.9668107424536, 12.437669929732023],
    "RE61" => [80896.9128355, 1350.0, 2853468.96494, 7076861.67064, 87748.6339553, 2.50994535821],
)

@inline uses_box_transform(prob::AbstractString=PROBLEM) = prob in BOX_TRANSFORM_PROBLEMS

@inline function logistic_stable(t::T) where {T<:Real}
    if t >= zero(T)
        z = exp(-t)
        return inv(one(T) + z)
    else
        z = exp(t)
        return z / (one(T) + z)
    end
end

function set_problem_box_bounds!(l::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    CURRENT_BOX_L[] = Float64.(l)
    CURRENT_BOX_U[] = Float64.(u)
    return nothing
end

function box_map(y::AbstractVector{S},
                 l::AbstractVector{<:Real},
                 u::AbstractVector{<:Real}) where {S<:Real}
    x = Vector{S}(undef, length(y))
    @inbounds for i in eachindex(y)
        li = S(l[i])
        ui = S(u[i])
        x[i] = li + (ui - li) * logistic_stable(y[i])
    end
    return x
end

function box_unmap(x::AbstractVector{T},
                   l::AbstractVector{T},
                   u::AbstractVector{T};
                   eps_frac::T = T(BOX_MAP_EPS)) where {T<:AbstractFloat}
    y = Vector{T}(undef, length(x))
    @inbounds for i in eachindex(x)
        z = (x[i] - l[i]) / (u[i] - l[i])
        z = clamp(z, eps_frac, one(T) - eps_frac)
        y[i] = log(z / (one(T) - z))
    end
    return y
end

function box_internal_bounds(n::Int, ::Type{T}=Float64;
                             eps_frac::T = T(BOX_MAP_EPS)) where {T<:AbstractFloat}
    lo = log(eps_frac / (one(T) - eps_frac))
    hi = -lo
    return fill(lo, n), fill(hi, n)
end

function set_re_smoothing_relative!(mu_rel::Real)
    mu_rel > 0 || throw(ArgumentError("The relative Huber parameter must be positive."))
    RE_SMOOTH_REL_MU[] = Float64(mu_rel)
    return RE_SMOOTH_REL_MU[]
end

function re_problem_bounds(problem::AbstractString, ::Type{T}=Float64) where {T<:AbstractFloat}
    if problem == "RE41"
        return T[0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
               T[1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]
    elseif problem == "RE42"
        return T[150.0, 20.0, 13.0, 10.0, 14.0, 0.63],
               T[274.32, 32.31, 25.0, 11.71, 18.0, 0.75]
    elseif problem == "RE61"
        return T[0.01, 0.01, 0.01], T[0.45, 0.10, 0.10]
    end
    throw(ArgumentError("No applied RE bounds are registered for $(problem)."))
end

function re_reference_ideal_nadir(problem::AbstractString)
    haskey(RE_APPLIED_IDEAL, problem) ||
        throw(ArgumentError("No applied RE reference points are registered for $(problem)."))
    return copy(RE_APPLIED_IDEAL[problem]), copy(RE_APPLIED_NADIR[problem])
end

@inline positive_part(t::T) where {T<:Real} = t > zero(t) ? t : zero(t)

@inline function huber_positive_part(t::T, mu::Real) where {T<:Real}
    mu_t = zero(t) + mu
    if t <= zero(t)
        return zero(t)
    elseif t < mu_t
        return (t * t) / (2 * mu_t)
    end
    return t - mu_t / 2
end

function re42_objectives_constraints(x::AbstractVector{S}) where {S<:Real}
    x_L, x_B, x_D, x_T, x_Vk, x_CB = x

    displacement = S(1.025) * x_L * x_B * x_T * x_CB
    velocity = S(0.5144) * x_Vk
    froude_number = velocity / sqrt(S(9.8065) * x_L)
    a = S(4977.06) * x_CB^2 - S(8105.61) * x_CB + S(4456.51)
    b = -S(10847.2) * x_CB^2 + S(12817.0) * x_CB - S(6960.32)

    power = displacement^(2.0 / 3.0) * x_Vk^3 / (a + b * froude_number)
    outfit_weight = x_L^0.8 * x_B^0.6 * x_D^0.3 * x_CB^0.1
    steel_weight = S(0.034) * x_L^1.7 * x_B^0.7 * x_D^0.4 * x_CB^0.5
    machinery_weight = S(0.17) * power^0.9
    light_ship_weight = steel_weight + outfit_weight + machinery_weight

    ship_cost = S(1.3) * (S(2000.0) * steel_weight^0.85 +
                          S(3500.0) * outfit_weight + S(2400.0) * power^0.8)
    capital_costs = S(0.2) * ship_cost
    deadweight = displacement - light_ship_weight
    running_costs = S(40000.0) * deadweight^0.3

    sea_days = (S(5000.0) / S(24.0)) * x_Vk
    daily_consumption = S(0.19 * 24.0 / 1000.0) * power + S(0.2)
    fuel_cost = S(1.05 * 100.0) * daily_consumption * sea_days
    port_cost = S(6.3) * deadweight^0.8
    fuel_carried = daily_consumption * (sea_days + S(5.0))
    miscellaneous_deadweight = S(2.0) * sqrt(deadweight)
    cargo_deadweight = deadweight - fuel_carried - miscellaneous_deadweight
    port_days = S(2.0) * (cargo_deadweight / S(8000.0) + S(0.5))
    round_trips_per_year = S(350.0) / (sea_days + port_days)
    voyage_costs = (fuel_cost + port_cost) * round_trips_per_year
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_deadweight * round_trips_per_year

    kb = S(0.53) * x_T
    bmt = (S(0.085) * x_CB - S(0.002)) * x_B^2 / (x_T * x_CB)
    kg = one(S) + S(0.52) * x_D
    constraints = S[
        x_L / x_B - S(6.0),
        -x_L / x_D + S(15.0),
        -x_L / x_T + S(19.0),
        S(0.45) * deadweight^0.31 - x_T,
        S(0.7) * x_D + S(0.7) - x_T,
        S(500000.0) - deadweight,
        deadweight - S(3000.0),
        S(0.32) - froude_number,
        kb + bmt - kg - S(0.07) * x_B,
    ]
    objectives = S[annual_costs / annual_cargo, light_ship_weight, -annual_cargo]
    return objectives, constraints
end

function re_constraint_values(problem::AbstractString,
                              x::AbstractVector{S}) where {S<:Real}
    if problem == "RE41"
        x1, x2, x3, x4, x5, x6, x7 = x
        f2 = S(4.72) - S(0.5) * x4 - S(0.19) * x2 * x3
        vmbp = S(10.58) - S(0.674) * x1 * x2 - S(0.67275) * x2
        vfd = S(16.45) - S(0.489) * x3 * x7 - S(0.843) * x5 * x6
        return S[
            one(S) - (S(1.16) - S(0.3717) * x2 * x4 - S(0.0092928) * x3),
            S(0.32) - (S(0.261) - S(0.0159) * x1 * x2 - S(0.06486) * x1 -
                       S(0.019) * x2 * x7 + S(0.0144) * x3 * x5 + S(0.0154464) * x6),
            S(0.32) - (S(0.214) + S(0.00817) * x5 - S(0.045195) * x1 -
                       S(0.0135168) * x1 + S(0.03099) * x2 * x6 - S(0.018) * x2 * x7 +
                       S(0.007176) * x3 + S(0.023232) * x3 - S(0.00364) * x5 * x6 -
                       S(0.018) * x2 * x2),
            S(0.32) - (S(0.74) - S(0.61) * x2 - S(0.031296) * x3 -
                       S(0.031872) * x7 + S(0.227) * x2 * x2),
            S(32.0) - (S(28.98) + S(3.818) * x3 - S(4.2) * x1 * x2 +
                       S(1.27296) * x6 - S(2.68065) * x7),
            S(32.0) - (S(33.86) + S(2.95) * x3 - S(5.057) * x1 * x2 -
                       S(3.795) * x2 - S(3.4431) * x7 + S(1.45728)),
            S(32.0) - (S(46.36) - S(9.9) * x2 - S(4.4505) * x1),
            S(4.0) - f2,
            S(9.9) - vmbp,
            S(15.7) - vfd,
        ]
    elseif problem == "RE42"
        _, constraints = re42_objectives_constraints(x)
        return constraints
    elseif problem == "RE61"
        x1, x2, x3 = x
        x1x2 = x1 * x2
        return S[
            one(S) - (S(0.00139) / x1x2 + S(4.94) * x3 - S(0.08)),
            one(S) - (S(0.000306) / x1x2 + S(1.082) * x3 - S(0.0986)),
            S(50000.0) - (S(12.307) / x1x2 + S(49408.24) * x3 + S(4051.02)),
            S(16000.0) - (S(2.098) / x1x2 + S(8046.33) * x3 - S(696.71)),
            S(10000.0) - (S(2.138) / x1x2 + S(7883.39) * x3 - S(705.04)),
            S(2000.0) - (S(0.417) * x1x2 + S(1721.26) * x3 - S(136.54)),
            S(550.0) - (S(0.164) / x1x2 + S(631.13) * x3 - S(54.48)),
        ]
    end
    throw(ArgumentError("Unknown applied RE problem $(problem)."))
end

function re_raw_objectives(problem::AbstractString,
                           x::AbstractVector{S};
                           smooth_violation::Bool=false,
                           mu_rel::Real=RE_SMOOTH_REL_MU[]) where {S<:Real}
    constraint_values = nothing
    if problem == "RE41"
        x1, x2, x3, x4, x5, x6, x7 = x
        f = Vector{S}(undef, 4)
        f[1] = S(1.98) + S(4.9) * x1 + S(6.67) * x2 + S(6.98) * x3 +
               S(4.01) * x4 + S(1.78) * x5 + S(0.00001) * x6 + S(2.73) * x7
        f[2] = S(4.72) - S(0.5) * x4 - S(0.19) * x2 * x3
        vmbp = S(10.58) - S(0.674) * x1 * x2 - S(0.67275) * x2
        vfd = S(16.45) - S(0.489) * x3 * x7 - S(0.843) * x5 * x6
        f[3] = (vmbp + vfd) / 2
    elseif problem == "RE42"
        objectives, constraint_values = re42_objectives_constraints(x)
        f = Vector{S}(undef, 4)
        f[1:3] .= objectives
    elseif problem == "RE61"
        x1, x2, x3 = x
        f = Vector{S}(undef, 6)
        f[1] = S(106780.37) * (x2 + x3) + S(61704.67)
        f[2] = S(3000.0) * x1
        f[3] = S(305700.0 * 2289.0) * x2 / S((0.06 * 2289.0)^0.65)
        f[4] = S(250.0 * 2289.0) * exp(-S(39.75) * x2 + S(9.9) * x3 + S(2.74))
        f[5] = S(25.0) * (S(1.39) / (x1 * x2) + S(4940.0) * x3 - S(80.0))
    else
        throw(ArgumentError("Unknown applied RE problem $(problem)."))
    end

    ideal, nadir = re_reference_ideal_nadir(problem)
    mu = mu_rel * (nadir[end] - ideal[end])
    violation = zero(S)
    constraints = constraint_values === nothing ? re_constraint_values(problem, x) : constraint_values
    for gi in constraints
        t = -gi
        violation += smooth_violation ? huber_positive_part(t, mu) : positive_part(t)
    end
    f[end] = violation
    return f
end

re_exact_objectives(problem::AbstractString, x::AbstractVector{S}) where {S<:Real} =
    re_raw_objectives(problem, x; smooth_violation=false)

re_smoothed_objectives(problem::AbstractString, x::AbstractVector{S};
                       mu_rel::Real=RE_SMOOTH_REL_MU[]) where {S<:Real} =
    re_raw_objectives(problem, x; smooth_violation=true, mu_rel=mu_rel)

function re_normalize_objectives(problem::AbstractString,
                                 f::AbstractVector{S}) where {S<:Real}
    ideal, nadir = re_reference_ideal_nadir(problem)
    return S[(f[i] - S(ideal[i])) / S(nadir[i] - ideal[i]) for i in eachindex(f)]
end

function re_physical_point(problem::AbstractString,
                           y::AbstractVector{S}) where {S<:Real}
    l, u = re_problem_bounds(problem, Float64)
    return box_map(y, l, u)
end

function re_internal_point(problem::AbstractString,
                           x::AbstractVector{T}) where {T<:AbstractFloat}
    l, u = re_problem_bounds(problem, T)
    return box_unmap(x, l, u)
end

function re_applied_evalf(problem::AbstractString,
                          y::AbstractVector{S},
                          ind::Int) where {S<:Real}
    f = re_smoothed_objectives(problem, re_physical_point(problem, y))
    ideal, nadir = re_reference_ideal_nadir(problem)
    return (f[ind] - S(ideal[ind])) / S(nadir[ind] - ideal[ind])
end

function re_phasea_evalf(problem::AbstractString,
                         y::AbstractVector{S},
                         ind::Int) where {S<:Real}
    x = box_map(y, CURRENT_BOX_L[], CURRENT_BOX_U[])

    if problem == "RE21"
        x1, x2, x3, x4 = x
        F = S(10.0)
        E = S(2.0e5)
        L = S(200.0)
        s2 = sqrt(S(2.0))

        if ind == 1
            return L * ((S(2.0) * x1) + s2 * x2 + sqrt(x3) + x4)
        elseif ind == 2
            return ((F * L) / E) * ((S(2.0) / x1) + (S(2.0) * s2 / x2) - (S(2.0) * s2 / x3) + (S(2.0) / x4))
        end
    elseif problem == "RE34"
        x1, x2, x3, x4, x5 = x
        if ind == 1
            return S(1640.2823) + S(2.3573285) * x1 + S(2.3220035) * x2 +
                   S(4.5688768) * x3 + S(7.7213633) * x4 + S(4.4559504) * x5
        elseif ind == 2
            return S(6.5856) + S(1.15) * x1 - S(1.0427) * x2 + S(0.9738) * x3 +
                   S(0.8364) * x4 - S(0.3695) * x1 * x4 + S(0.0861) * x1 * x5 +
                   S(0.3628) * x2 * x4 - S(0.1106) * x1 * x1 - S(0.3437) * x3 * x3 +
                   S(0.1764) * x4 * x4
        elseif ind == 3
            return -S(0.0551) + S(0.0181) * x1 + S(0.1024) * x2 + S(0.0421) * x3 -
                   S(0.0073) * x1 * x2 + S(0.0240) * x2 * x3 - S(0.0118) * x2 * x4 -
                   S(0.0204) * x3 * x4 - S(0.0080) * x3 * x5 - S(0.0241) * x2 * x2 +
                   S(0.0109) * x4 * x4
        end
    elseif problem == "RE37"
        xAlpha, xHA, xOA, xOPTT = x
        if ind == 1
            return S(0.692) + S(0.477) * xAlpha - S(0.687) * xHA - S(0.080) * xOA - S(0.0650) * xOPTT -
                   S(0.167) * xAlpha * xAlpha - S(0.0129) * xHA * xAlpha + S(0.0796) * xHA * xHA -
                   S(0.0634) * xOA * xAlpha - S(0.0257) * xOA * xHA + S(0.0877) * xOA * xOA -
                   S(0.0521) * xOPTT * xAlpha + S(0.00156) * xOPTT * xHA +
                   S(0.00198) * xOPTT * xOA + S(0.0184) * xOPTT * xOPTT
        elseif ind == 2
            return S(0.153) - S(0.322) * xAlpha + S(0.396) * xHA + S(0.424) * xOA + S(0.0226) * xOPTT +
                   S(0.175) * xAlpha * xAlpha + S(0.0185) * xHA * xAlpha - S(0.0701) * xHA * xHA -
                   S(0.251) * xOA * xAlpha + S(0.179) * xOA * xHA + S(0.0150) * xOA * xOA +
                   S(0.0134) * xOPTT * xAlpha + S(0.0296) * xOPTT * xHA +
                   S(0.0752) * xOPTT * xOA + S(0.0192) * xOPTT * xOPTT
        elseif ind == 3
            return S(0.370) - S(0.205) * xAlpha + S(0.0307) * xHA + S(0.108) * xOA + S(1.019) * xOPTT -
                   S(0.135) * xAlpha * xAlpha + S(0.0141) * xHA * xAlpha + S(0.0998) * xHA * xHA +
                   S(0.208) * xOA * xAlpha - S(0.0301) * xOA * xHA - S(0.226) * xOA * xOA +
                   S(0.353) * xOPTT * xAlpha - S(0.0497) * xOPTT * xOA - S(0.423) * xOPTT * xOPTT +
                   S(0.202) * xHA * xAlpha * xAlpha - S(0.281) * xOA * xAlpha * xAlpha -
                   S(0.342) * xHA * xHA * xAlpha - S(0.245) * xHA * xHA * xOA +
                   S(0.281) * xOA * xOA * xHA - S(0.184) * xOPTT * xOPTT * xAlpha -
                   S(0.281) * xHA * xAlpha * xOA
        end
    end

    error("Unknown phase-A real-world problem or objective: $(problem), objective $(ind)")
end

"""
    inip(problem, seed)

Select `problem`, generate its reproducible initial point, and return
`(n, m, x, l, u, convex, scaleF, checkder)`. Before subsequent evaluations,
the caller must set `SDProject.PROBLEM` to the same problem, as in `run_main.jl`.
"""
function inip(PROBLEM::String,SEED::Int)
    rng = MersenneTwister(SEED)
    l_phys = nothing
    u_phys = nothing

    if is_scalable_benchmark(PROBLEM)
        return scalable_inip(PROBLEM, SEED, T)
    end

    if PROBLEM == "AP1"
        n, m = 2, 3
        l = fill(T(-1e1), n)
        u = fill(T( 1e1), n)
        x = [l[i] + (u[i] - l[i]) * rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "AP2"
        n, m = 1, 2
        l = fill(T(-1e2), n)
        u = fill(T( 1e2), n)
        x = [l[i] + (u[i] - l[i]) * rand(rng, T) for i in 1:n]
        strconvex = Bool[true, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "AP3"
        n, m = 2, 2
        l = fill(T(-1e2), n)
        u = fill(T( 1e2), n)
        x = [l[i] + (u[i] - l[i]) * rand(rng, T) for i in 1:n]
        strconvex = Bool[false, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "AP4"
        n, m = 3, 3
        l = fill(T(-1e1), n)
        u = fill(T( 1e1), n)
        x = [l[i] + (u[i] - l[i]) * rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "BK1"
        n, m = 2, 2
        l = T[-5, -5]
        u = T[10, 10]
        x = [l[i] + (u[i] - l[i]) * rand(rng, T) for i in 1:n]
        strconvex = Bool[true, true]
        scaleF, checkder = true, false
    
    elseif PROBLEM == "DD1"
        # Das & Dennis (1998)
        n, m = 5, 2
        l = fill(T(-20.0), n)
        u = fill(T( 20.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        x .= 10 * rand(rng, T) .* x ./ norm(x)
        strconvex = Bool[true, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "DGO1"
        # Review of Multiobjective Test PROBLEMs
        n, m = 1, 2
        l = T[-10.0]
        u = T[13.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = Bool[false, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "DGO1M"
        # Variante modificada de DGO1 (trig + onda suave)
        n, m = 1, 2
        l = T[-10.0]
        u = T[13.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = Bool[false, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "DGO2"
        n, m = 1, 2
        l = T[-9.0]
        u = T[9.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = Bool[true, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "FA1"
        # Review of Multiobjective Test PROBLEMs (FA1)
        n, m = 3, 3
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Far1"
        # Review of Multiobjective Test PROBLEMs (Far1)
        n, m = 2, 2
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "FDS"
        # Newton’s Method for Multiobjective Optimization
        n, m = 5, 3
        l = fill(T(-2.0), n)
        u = fill(T( 2.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "FF1"
        # Fonseca & Fleming
        n, m = 2, 2
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Hil1"
        # Generalized Homotopy Approach to Multiobjective Optimization
        n, m = 2, 2
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "IKK1"
        # Review of Multiobjective Test PROBLEMs
        n, m = 2, 3
        l = fill(T(-50.0), n)
        u = fill(T( 50.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "IM1"
        # Review of Multiobjective Test PROBLEMs
        n, m = 2, 2
        l = T[1.0, 1.0]
        u = T[4.0, 2.0]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "JOS1"
        # Dynamic Weighted Aggregation for Evolutionary Multiobjective Optimization
        n, m = 2, 2
        l = fill(T(-100.0), n)
        u = fill(T( 100.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "JOS4"
        # Newton’s Method for Multiobjective Optimization
        n, m = 20, 2
        l = fill(T(1e-2), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "KW2"
        # Kim & de Weck (Adaptive weighted-sum method)
        n, m = 2, 2
        l = fill(T(-3.0), n)
        u = fill(T( 3.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "LE1"
        # Review of Multiobjective Test PROBLEMs (LE1)
        n, m = 2, 2
        l = fill(T(-5.0), n)
        u = fill(T(10.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov1"
        # Singular Continuation (PROBLEM 1)
        n, m = 2, 2
        l = fill(T(-10.0), n)
        u = fill(T( 10.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov2"
        # Singular Continuation (PROBLEM 2)
        n, m = 2, 2
        l = fill(T(-0.75), n)
        u = fill(T( 0.75), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov3"
        # Singular Continuation (PROBLEM 3)
        n, m = 2, 2
        l = fill(T(-20.0), n)
        u = fill(T( 20.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[true, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov4"
        # Singular Continuation (PROBLEM 4)
        n, m = 2, 2
        l = fill(T(-20.0), n)
        u = fill(T( 20.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov5"
        # Singular Continuation (PROBLEM 5)
        n, m = 3, 2
        l = fill(T(-2.0), n)
        u = fill(T( 2.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Lov6"
        # Singular Continuation (PROBLEM 6)
        n, m = 6, 2
        l = T[0.1; fill(-0.16, 5)...]
        u = T[0.425; fill( 0.16, 5)...]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "LTDZ"
        # Combining convergence and diversity
        n, m = 3, 3
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MGH9"
        # Gaussian test PROBLEM (More et al., 1981)
        n, m = 3, 15
        l = fill(T(-2.0), n)
        u = fill(T( 2.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MGH16"
        # Brown and Dennis test PROBLEM
        n, m = 4, 5
        l = T[-25.0, -5.0, -5.0, -1.0]
        u = T[ 25.0,  5.0,  5.0,  1.0]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MGH26"
        # Trigonometric PROBLEM
        n = 4; m = n
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MGH33"
        # Linear function (rank 1)
        n = 10; m = n
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MHHM2"
        # Review of Multiobjective Test PROBLEMs
        n, m = 2, 3
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "RE21"
        n, m = 4, 2
        l_phys = T[1.0, sqrt(T(2.0)), sqrt(T(2.0)), 1.0]
        u_phys = fill(T(3.0), n)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "RE34"
        n, m = 5, 3
        l_phys = fill(T(1.0), n)
        u_phys = fill(T(3.0), n)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "RE37"
        n, m = 4, 3
        l_phys = fill(T(0.0), n)
        u_phys = fill(T(1.0), n)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "RE41"
        n, m = 7, 4
        l_phys, u_phys = re_problem_bounds(PROBLEM, T)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = false, false

    elseif PROBLEM == "RE42"
        n, m = 6, 4
        l_phys, u_phys = re_problem_bounds(PROBLEM, T)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = false, false

    elseif PROBLEM == "RE61"
        n, m = 3, 6
        l_phys, u_phys = re_problem_bounds(PROBLEM, T)
        x_phys = [l_phys[i] + (u_phys[i] - l_phys[i]) * rand(rng, T) for i in 1:n]
        x = box_unmap(x_phys, l_phys, u_phys)
        l, u = box_internal_bounds(n, T)
        strconvex = falses(m)
        scaleF, checkder = false, false

    elseif PROBLEM == "MLF1"
        # Review of Multiobjective Test PROBLEMs
        n, m = 1, 2
        l = T[0.0]; u = T[20.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MLF2"
        # Review of Multiobjective Test PROBLEMs
        n, m = 2, 2
        l = fill(T(-100.0), n)
        u = fill(T( 100.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MMR1"
        # Box-constrained multi-objective optimization (modified)
        n, m = 2, 2
        l = T[0.1, 0.0]
        u = T[1.0, 1.0]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MMR3"
        n, m = 2, 2
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MMR4"
        n, m = 3, 2
        l = fill(T(0.0), n)
        u = fill(T(4.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MOP2"
        n, m = 2, 2
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MOP3"
        n, m = 2, 2
        l = fill(T(-π), n)
        u = fill(T( π), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "MOP5"
        n, m = 2, 3
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MOP6"
        n, m = 2, 2
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "MOP7"
        n, m = 2, 3
        l = fill(T(-400.0), n)
        u = fill(T( 400.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "PNR"
        n, m = 2, 2
        l = fill(T(-2.0), n)
        u = fill(T( 2.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "QV1"
        n, m = 10, 2
        l = fill(T(-5.0), n)
        u = fill(T( 5.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "SD"
        n, m = 4, 2
        l = T[1.0, √2, √2, 1.0]
        u = T[3.0, 3.0, 3.0, 3.0]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "SLCDT1"
        n, m = 2, 2
        l = fill(T(-1.5), n)
        u = fill(T( 1.5), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "SLCDT2"
        n, m = 10, 3
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "SP1"
        n, m = 2, 2
        l = fill(T(-100.0), n)
        u = fill(T( 100.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "SSFYY2"
        n, m = 1, 2
        l = T[-100.0]
        u = T[ 100.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "SK1"
        n, m = 1, 2
        l = T[-100.0]
        u = T[ 100.0]
        x = [l[1] + (u[1]-l[1])*rand(rng, T)]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "SK2"
        n, m = 4, 2
        l = fill(T(-10.0), n)
        u = fill(T( 10.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[true, false]
        scaleF, checkder = true, false

    elseif PROBLEM == "TKLY1"
        n, m = 4, 2
        l = T[0.1, 0.0, 0.0, 0.0]
        u = T[1.0, 1.0, 1.0, 1.0]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Toi4"
        n, m = 4, 2
        l = fill(T(-2.0), n)
        u = fill(T( 5.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Toi8"
        n = 3; m = n
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Toi9"
        n = 4; m = n
        l = fill(T(-1.0), n)
        u = fill(T( 1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "Toi10"
        n = 4; m = n - 1
        l = fill(T(-2.0), n)
        u = fill(T( 2.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "VU1"
        n, m = 2, 2
        l = fill(T(-3.0), n)
        u = fill(T( 3.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "VU2"
        n, m = 2, 2
        l = fill(T(-3.0), n)
        u = fill(T( 3.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = Bool[false, true]
        scaleF, checkder = true, false

    elseif PROBLEM == "ZDT1"
        n, m = 30, 2
        l = fill(T(1e-2), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "ZDT2"
        n, m = 30, 2
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "ZDT3"
        n, m = 30, 2
        l = fill(T(1e-2), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "ZDT4"
        n, m = 30, 2
        l = [1e-2; fill(T(-5.0), n-1)...]
        u = [1.0; fill(T(5.0), n-1)...]
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "ZDT6"
        n, m = 10, 2
        l = fill(T(0.0), n)
        u = fill(T(1.0), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = falses(m)
        scaleF, checkder = true, false

    elseif PROBLEM == "ZLT1"
        n, m = 10, 5
        l = fill(T(-1e3), n)
        u = fill(T( 1e3), n)
        x = [l[i] + (u[i]-l[i])*rand(rng, T) for i in 1:n]
        strconvex = trues(m)
        scaleF, checkder = true, false
    else
        error("Unknown PROBLEM : $PROBLEM")
    end

    if l_phys === nothing
        l_phys = copy(l)
        u_phys = copy(u)
    end
    set_problem_box_bounds!(l_phys, u_phys)

    return n, m, x, l, u, strconvex, scaleF, checkder
end

# ======================================================================
# Objective component f_i(x)
# ======================================================================

"""Evaluate objective component `ind` of the currently selected problem."""
function evalf(n::Int, x::Vector{T}, ind::Int) where {T<:AbstractFloat}
    f = zero(T)

    if is_scalable_benchmark(PROBLEM)
        f = scalable_evalf(PROBLEM, x, ind)

    elseif PROBLEM == "RE41" || PROBLEM == "RE42" || PROBLEM == "RE61"
        f = re_applied_evalf(PROBLEM, x, ind)

    elseif PROBLEM == "RE21" || PROBLEM == "RE34" || PROBLEM == "RE37"
        f = re_phasea_evalf(PROBLEM, x, ind)

    elseif PROBLEM == "AP1"
        if ind == 1
            f = 0.25T((x[1]-1)^4 + 2*(x[2]-2)^4)
        elseif ind == 2
            f = exp((x[1]+x[2])/2) + x[1]^2 + x[2]^2
        elseif ind == 3
            f = (1/6) * (exp(-x[1]) + 2*exp(-x[2]))
        end

    elseif PROBLEM == "AP2"
        f = ind == 1 ? x[1]^2 - 4 : (x[1]-1)^2

    elseif PROBLEM == "AP3"
        f = ind == 1 ? 0.25*((x[1]-1)^4 + 2*(x[2]-2)^4) :
                       (x[2]-x[1]^2)^2 + (1-x[1])^2

    elseif PROBLEM == "AP4"
        if ind == 1
            f = (1/9)*((x[1]-1)^4 + 2*(x[2]-2)^4 + 3*(x[3]-3)^4)
        elseif ind == 2
            f = exp((x[1]+x[2]+x[3])/3) + x[1]^2 + x[2]^2 + x[3]^2
        elseif ind == 3
            f = (1/12)*(3*exp(-x[1]) + 4*exp(-x[2]) + 3*exp(-x[3]))
        end

    elseif PROBLEM == "DD1"
        if ind == 1
            f = sum(x .^ 2)
        elseif ind == 2
            f = 3.0 * x[1] + 2.0 * x[2] - x[3] / 3.0 + 1e-2 * (x[4] - x[5])^3
        end

    elseif PROBLEM == "DGO1"
        if ind == 1
            f = sin(x[1])
        elseif ind == 2
            f = sin(x[1] + 0.7)
        end

    elseif PROBLEM == "DGO1M"
        if ind == 1
            f = sin(x[1]) + 0.1 * cos(5.0 * x[1]) * sin(x[1])
        elseif ind == 2
            y = x[1] + 0.7
            f = sin(y) + 0.1 * cos(5.0 * x[1]) * sin(y)
        end

    elseif PROBLEM == "DGO2"
        if ind == 1
            f = x[1]^2
        elseif ind == 2
            f = 9.0 - sqrt(81.0 - x[1]^2)
        end

    elseif PROBLEM == "FA1"
        faux = (1 - exp(-4 * x[1])) / (1 - exp(-4))
        if ind == 1
            f = faux
        elseif ind == 2
            f = (x[2] + 1) * (1 - (faux / (x[2] + 1))^0.5)
        elseif ind == 3
            f = (x[3] + 1) * (1 - (faux / (x[3] + 1))^0.1)
        end

    elseif PROBLEM == "Far1"
        if ind == 1
            f = -2 * exp(15 * (-(x[1] - 0.1)^2 - x[2]^2)) -
                exp(20 * (-(x[1] - 0.6)^2 - (x[2] - 0.6)^2)) +
                exp(20 * (-(x[1] + 0.6)^2 - (x[2] - 0.6)^2)) +
                exp(20 * (-(x[1] - 0.6)^2 - (x[2] + 0.6)^2)) +
                exp(20 * (-(x[1] + 0.6)^2 - (x[2] + 0.6)^2))
        elseif ind == 2
            f = 2 * exp(20 * (-(x[1]^2 + x[2]^2))) +
                exp(20 * (-(x[1] - 0.4)^2 - (x[2] - 0.6)^2)) -
                exp(20 * (-(x[1] + 0.5)^2 - (x[2] - 0.7)^2)) -
                exp(20 * (-(x[1] - 0.5)^2 - (x[2] + 0.7)^2)) +
                exp(20 * (-(x[1] + 0.4)^2 - (x[2] + 0.8)^2))
        end

    elseif PROBLEM == "FDS"
        if ind == 1
            f = sum([i * (x[i] - i)^4 for i in 1:n]) / n^2
        elseif ind == 2
            f = exp(sum(x) / n) + sum(x.^2)
        elseif ind == 3
            f = sum([i * (n - i + 1) * exp(-x[i]) for i in 1:n]) / (n * (n + 1))
        end

    elseif PROBLEM == "FF1"
        if ind == 1
            f = 1 - exp(-(x[1] - 1)^2 - (x[2] + 1)^2)
        elseif ind == 2
            f = 1 - exp(-(x[1] + 1)^2 - (x[2] - 1)^2)
        end

    elseif PROBLEM == "Hil1"
        a = 2π / 360 * (45 + 40 * sin(2π * x[1]) + 25 * sin(2π * x[2]))
        b = 1 + 0.5 * cos(2π * x[1])
        if ind == 1
            f = cos(a) * b
        elseif ind == 2
            f = sin(a) * b
        end

    elseif PROBLEM == "IKK1"
        if ind == 1
            f = x[1]^2
        elseif ind == 2
            f = (x[1] - 20)^2
        elseif ind == 3
            f = x[2]^2
        end

    elseif PROBLEM == "IM1"
        if ind == 1
            f = 2 * sqrt(x[1])
        elseif ind == 2
            f = x[1] * (1 - x[2]) + 5
        end

    elseif PROBLEM == "BK1"
        f = ind == 1 ? x[1]^2 + x[2]^2 : (x[1]-5)^2 + (x[2]-5)^2

    elseif PROBLEM == "JOS1"
        if ind == 1
            f = 0.0
            for i in 1:n
                f += x[i]^2
            end
            f /= n
        elseif ind == 2
            f = 0.0
            for i in 1:n
                f += (x[i] - 2.0)^2
            end
            f /= n
        end
    
    elseif PROBLEM == "JOS4"
        if ind == 1
            f = x[1]
        elseif ind == 2
            faux = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            f = faux * (1.0 - (x[1] / faux)^0.25 - (x[1] / faux)^4.0)
        end

    elseif PROBLEM == "KW2"
        if ind == 1
            f = -3.0 * (1.0 - x[1])^2 * exp(-x[1]^2 - (x[2] + 1.0)^2) +
                10.0 * (x[1] / 5.0 - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2) +
                3.0 * exp(-(x[1] + 2.0)^2 - x[2]^2) - 0.5 * (2.0 * x[1] + x[2])
        elseif ind == 2
            f = -3.0 * (1.0 + x[2])^2 * exp(-x[2]^2 - (1.0 - x[1])^2) +
                10.0 * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2) +
                3.0 * exp(-(2.0 - x[2])^2 - x[1]^2)
        end

    elseif PROBLEM == "LE1"
        if ind == 1
            f = (x[1]^2 + x[2]^2)^0.125
        elseif ind == 2
            f = ((x[1] - 0.5)^2 + (x[2] - 0.5)^2)^0.25
        end

    elseif PROBLEM == "Lov1"
        if ind == 1
            f = -(-1.05 * x[1]^2 - 0.98 * x[2]^2)
        elseif ind == 2
            f = -(-0.99 * (x[1] - 3.0)^2 - 1.03 * (x[2] - 2.5)^2)
        end

    elseif PROBLEM == "Lov2"
        if ind == 1
            f = x[2]
        elseif ind == 2
            f = -(x[2] - x[1]^3) / (x[1] + 1.0)
        end

    elseif PROBLEM == "Lov3"
        if ind == 1
            f = -(-x[1]^2 - x[2]^2)
        elseif ind == 2
            f = -(- (x[1] - 6.0)^2 + (x[2] + 0.3)^2)
        end

    elseif PROBLEM == "Lov4"
        if ind == 1
            f = -x[1]^2 - x[2]^2 - 4.0 * (exp(-(x[1] + 2.0)^2 - x[2]^2) +
                 exp(-(x[1] - 2.0)^2 - x[2]^2))
            f = -f
        elseif ind == 2
            f = -(x[1] - 6.0)^2 - (x[2] + 0.5)^2
            f = -f
        end

    elseif PROBLEM == "Lov5"
        M = [-1.0  -0.03   0.011;
             -0.03 -1.0    0.07;
              0.011 0.07  -1.01]

        p = [x[1], x[2] - 0.15, x[3]]
        a = 0.35
        A1 = sqrt(2.0 * π / a) * exp(dot(p, M * p) / a^2)

        p = [x[1], x[2] + 1.1, 0.5 * x[3]]
        a = 3.0
        A2 = sqrt(2.0 * π / a) * exp(dot(p, M * p) / a^2)

        faux = A1 + A2

        if ind == 1
            f = sqrt(2.0) / 2.0 * (x[1] + faux)
            f = -f
        elseif ind == 2
            f = sqrt(2.0) / 2.0 * (-x[1] + faux)
            f = -f
        end

    elseif PROBLEM == "Lov6"
        if ind == 1
            f = x[1]
        elseif ind == 2
            f = 1.0 - sqrt(x[1]) - x[1] * sin(10.0 * π * x[1]) +
                x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[6]^2
        end
    
    elseif PROBLEM == "LTDZ"
        if ind == 1
            f = -(3.0 - (1.0 + x[3]) * cos(x[1] * π / 2.0) * cos(x[2] * π / 2.0))
        elseif ind == 2
            f = -(3.0 - (1.0 + x[3]) * cos(x[1] * π / 2.0) * sin(x[2] * π / 2.0))
        elseif ind == 3
            f = -(3.0 - (1.0 + x[3]) * cos(x[1] * π / 2.0) * sin(x[1] * π / 2.0))
        end

    elseif PROBLEM == "MGH9"
        if ind == 1 || ind == 15
            y = 9.0e-4
        elseif ind == 2 || ind == 14
            y = 4.4e-3
        elseif ind == 3 || ind == 13
            y = 1.75e-2
        elseif ind == 4 || ind == 12
            y = 5.4e-2
        elseif ind == 5 || ind == 11
            y = 1.295e-1
        elseif ind == 6 || ind == 10
            y = 2.42e-1
        elseif ind == 7 || ind == 9
            y = 3.521e-1
        elseif ind == 8
            y = 3.989e-1
        end
        t = (8.0 - ind) / 2.0
        f = x[1] * exp(-x[2] * (t - x[3])^2 / 2.0) - y

    elseif PROBLEM == "MGH16"
        t = ind / 5.0
        f = (x[1] + t * x[2] - exp(t))^2 + (x[3] + x[4] * sin(t) - cos(t))^2

    elseif PROBLEM == "MGH26"
        t = sum(cos.(x))
        f = (n - t + ind * (1.0 - cos(x[ind])) - sin(x[ind]))^2

    elseif PROBLEM == "MGH33"
        faux = sum(i * x[i] for i in 1:n)
        f = (ind * faux - 1.0)^2

    elseif PROBLEM == "MHHM2"
        if ind == 1
            f = (x[1] - 0.8)^2 + (x[2] - 0.6)^2
        elseif ind == 2
            f = (x[1] - 0.85)^2 + (x[2] - 0.7)^2
        elseif ind == 3
            f = (x[1] - 0.9)^2 + (x[2] - 0.6)^2
        end

    elseif PROBLEM == "MLF1"
        if ind == 1
            f = (1.0 + x[1] / 20.0) * sin(x[1])
        elseif ind == 2
            f = (1.0 + x[1] / 20.0) * cos(x[1])
        end

    elseif PROBLEM == "MLF2"
        if ind == 1
            f = -(5.0 - ((x[1]^2 + x[2] - 11.0)^2 + (x[1] + x[2]^2 - 7.0)^2) / 200.0)
        elseif ind == 2
            f = -(5.0 - ((4.0 * x[1]^2 + 2.0 * x[2] - 11.0)^2 + (2.0 * x[1] + 4.0 * x[2]^2 - 7.0)^2) / 200.0)
        end

    elseif PROBLEM == "MMR1"
        if ind == 1
            f = 1.0 + x[1]^2
        elseif ind == 2
            f = (2.0 - 0.8 * exp(-((x[2] - 0.6) / 0.4)^2) - exp(-((x[2] - 0.2) / 0.04)^2)) / (1.0 + x[1]^2)
        end

    elseif PROBLEM == "MMR3"
        if ind == 1
            f = x[1]^3
        elseif ind == 2
            f = (x[2] - x[1])^3
        end

    elseif PROBLEM == "MMR4"
        if ind == 1
            f = x[1] - 2.0 * x[2] - x[3] - 36.0 / (2.0 * x[1] + x[2] + 2.0 * x[3] + 1.0)
        elseif ind == 2
            f = -3.0 * x[1] + x[2] - x[3]
        end

    elseif PROBLEM == "MOP2"
        if ind == 1
            f = 1.0 - exp(-sum((x[i] - 1.0 / sqrt(n))^2 for i in 1:n))
        elseif ind == 2
            f = 1.0 - exp(-sum((x[i] + 1.0 / sqrt(n))^2 for i in 1:n))
        end

    elseif PROBLEM == "MOP3"
        if ind == 1
            A1 = 0.5 * sin(1.0) - 2.0 * cos(1.0) + sin(2.0) - 1.5 * cos(2.0)
            A2 = 1.5 * sin(1.0) - cos(1.0) + 2.0 * sin(2.0) - 0.5 * cos(2.0)
            B1 = 0.5 * sin(x[1]) - 2.0 * cos(x[1]) + sin(x[2]) - 1.5 * cos(x[2])
            B2 = 1.5 * sin(x[1]) - cos(x[1]) + 2.0 * sin(x[2]) - 0.5 * cos(x[2])
            f = -(-1.0 - (A1 - B1)^2 - (A2 - B2)^2)
        elseif ind == 2
            f = -(-(x[1] + 3.0)^2 - (x[2] + 1.0)^2)
        end

    elseif PROBLEM == "MOP5"
        if ind == 1
            f = 0.5 * (x[1]^2 + x[2]^2) + sin(x[1]^2 + x[2]^2)
        elseif ind == 2
            f = (3.0 * x[1] - 2.0 * x[2] + 4.0)^2 / 8.0 + (x[1] - x[2] + 1.0)^2 / 27.0 + 15.0
        elseif ind == 3
            f = 1.0 / (x[1]^2 + x[2]^2 + 1.0) - 1.1 * exp(-x[1]^2 - x[2]^2)
        end

    elseif PROBLEM == "MOP6"
        if ind == 1
            f = x[1]
        elseif ind == 2
            a = 1.0 + 10.0 * x[2]
            t = x[1] / a
            f = a * (1.0 - t^2 - t * sin(8.0 * π * x[1]))
        end

    elseif PROBLEM == "MOP7"
        if ind == 1
            f = (x[1] - 2.0)^2 / 2.0 + (x[2] + 1.0)^2 / 13.0 + 3.0
        elseif ind == 2
            f = (x[1] + x[2] - 3.0)^2 / 36.0 + (-x[1] + x[2] + 2.0)^2 / 8.0 - 17.0
        elseif ind == 3
            f = (x[1] + 2.0 * x[2] - 1.0)^2 / 175.0 + (-x[1] + 2.0 * x[2])^2 / 17.0 - 13.0
        end

    elseif PROBLEM == "PNR"
        if ind == 1
            f = x[1]^4 + x[2]^4 - x[1]^2 + x[2]^2 - 10.0 * x[1] * x[2] + 20.0
        elseif ind == 2
            f = x[1]^2 + x[2]^2
        end

    elseif PROBLEM == "QV1"
        if ind == 1
            f = (sum(x[i]^2 - 10.0 * cos(2.0 * π * x[i]) + 10.0 for i in 1:n) / n)^0.25
        elseif ind == 2
            f = (sum((x[i] - 1.5)^2 - 10.0 * cos(2.0 * π * (x[i] - 1.5)) + 10.0 for i in 1:n) / n)^0.25
        end

    elseif PROBLEM == "SD"
        if ind == 1
            f = 2.0 * x[1] + sqrt(2.0) * (x[2] + x[3]) + x[4]
        elseif ind == 2
            f = 2.0 / x[1] + 2.0 * sqrt(2.0) / x[2] + 2.0 * sqrt(2.0) / x[3] + 2.0 / x[4]
        end

    elseif PROBLEM == "SLCDT1"
        if ind == 1
            f = 0.5 * (sqrt(1.0 + (x[1] + x[2])^2) + sqrt(1.0 + (x[1] - x[2])^2) + x[1] - x[2]) +
                0.85 * exp(-(x[1] + x[2])^2)
        elseif ind == 2
            f = 0.5 * (sqrt(1.0 + (x[1] + x[2])^2) + sqrt(1.0 + (x[1] - x[2])^2) - x[1] + x[2]) +
                0.85 * exp(-(x[1] + x[2])^2)
        end

    elseif PROBLEM == "SLCDT2"
        if ind == 1
            f = (x[1] - 1.0)^4 + sum((x[i] - 1.0)^2 for i in 2:n)
        elseif ind == 2
            f = (x[2] + 1.0)^4 + sum((x[i] + 1.0)^2 for i in 1:n if i != 2)
        elseif ind == 3
            f = (x[3] - 1.0)^4 + sum((x[i] - (-1.0)^(i + 1))^2 for i in 1:n if i != 3)
        end

    elseif PROBLEM == "SP1"
        if ind == 1
            f = (x[1] - 1.0)^2 + (x[1] - x[2])^2
        elseif ind == 2
            f = (x[2] - 3.0)^2 + (x[1] - x[2])^2
        end

    elseif PROBLEM == "SSFYY2"
        if ind == 1
            f = 10.0 + x[1]^2 - 10.0 * cos(x[1] * π / 2.0)
        elseif ind == 2
            f = (x[1] - 4.0)^2
        end

    elseif PROBLEM == "SK1"
        if ind == 1
            f = -(-x[1]^4 - 3.0 * x[1]^3 + 10.0 * x[1]^2 + 10.0 * x[1] + 10.0)
        elseif ind == 2
            f = -(-0.5 * x[1]^4 + 2.0 * x[1]^3 + 10.0 * x[1]^2 - 10.0 * x[1] + 5.0)
        end

    elseif PROBLEM == "SK2"
        if ind == 1
            f = -(-(x[1] - 2.0)^2 - (x[2] + 3.0)^2 - (x[3] - 5.0)^2 - (x[4] - 4.0)^2 + 5.0)
        elseif ind == 2
            f = -((sin(x[1]) + sin(x[2]) + sin(x[3]) + sin(x[4])) /
                (1.0 + (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) / 100.0))
        end

    elseif PROBLEM == "TKLY1"
        if ind == 1
            f = x[1]
        elseif ind == 2
            A1 = (2.0 - exp(-((x[2] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[2] - 0.9) / 0.4)^2))
            A2 = (2.0 - exp(-((x[3] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[3] - 0.9) / 0.4)^2))
            A3 = (2.0 - exp(-((x[4] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[4] - 0.9) / 0.4)^2))
            f = A1 * A2 * A3 / x[1]
        end

    elseif PROBLEM == "Toi4"
        if ind == 1
            f = x[1]^2 + x[2]^2 + 1.0
        elseif ind == 2
            f = 0.5 * ((x[1] - x[2])^2 + (x[3] - x[4])^2) + 1.0
        end

    elseif PROBLEM == "Toi8"
        if ind == 1
            f = (2.0 * x[1] - 1.0)^2
        elseif ind != 1
            f = ind * (2.0 * x[ind - 1] - x[ind])^2
        end

    elseif PROBLEM == "Toi9"
        if ind == 1
            f = (2.0 * x[1] - 1.0)^2 + x[2]^2
        elseif ind > 1 && ind < n
            f = ind * (2.0 * x[ind - 1] - x[ind])^2 - (ind - 1.0) * x[ind - 1]^2 + ind * x[ind]^2
        elseif ind == n
            f = n * (2.0 * x[n - 1] - x[n])^2 - (n - 1.0) * x[n - 1]^2
        end

    elseif PROBLEM == "Toi10"
        f = 100.0 * (x[ind + 1] - x[ind]^2)^2 + (x[ind + 1] - 1.0)^2

    elseif PROBLEM == "VU1"
        if ind == 1
            f = 1.0 / (x[1]^2 + x[2]^2 + 1.0)
        elseif ind == 2
            f = x[1]^2 + 3.0 * x[2]^2 + 1.0
        end

    elseif PROBLEM == "VU2"
        if ind == 1
            f = x[1] + x[2] + 1.0
        elseif ind == 2
            f = x[1]^2 + 2.0 * x[2] - 1.0
        end

    elseif PROBLEM == "ZDT1"
        if ind == 1
            f = x[1]
        elseif ind == 2
            faux = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            f = faux * (1.0 - sqrt(x[1] / faux))
        end

    elseif PROBLEM == "ZDT2"
        if ind == 1
            f = x[1]
        elseif ind == 2
            faux = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            f = faux * (1.0 - (x[1] / faux)^2)
        end

    elseif PROBLEM == "ZDT3"
        if ind == 1
            f = x[1]
        elseif ind == 2
            faux = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / faux
            f = faux * (1.0 - sqrt(t) - t * sin(10.0 * π * x[1]))
        end

    elseif PROBLEM == "ZDT4"
        if ind == 1
            f = x[1]
        elseif ind == 2
            faux = sum(x[i]^2 - 10.0 * cos(4.0 * π * x[i]) for i in 2:n) + 1.0 + 10.0 * (n - 1)
            t = x[1] / faux
            f = faux * (1.0 - sqrt(t))
        end

    elseif PROBLEM == "ZDT6"
        if ind == 1
            f = 1.0 - exp(-4.0 * x[1]) * (sin(6.0 * π * x[1]))^6
        elseif ind == 2
            t = 1.0 + 9.0 * (sum(x[2:n]) / (n - 1))^0.25
            f = t * (1.0 - ((1.0 - exp(-4.0 * x[1]) * (sin(6.0 * π * x[1]))^6) / t)^2)
        end

    elseif PROBLEM == "ZLT1"
        f = (x[ind] - 1.0)^2 + sum(x[i]^2 for i in 1:n if i != ind)

    else
        error("Unknown PROBLEM : $PROBLEM")
    end

    return f
end

# ======================================================================
# Gradient of objective component f_i(x)
# ======================================================================

"""Write the gradient of objective component `ind` into the preallocated `g`."""
function evalg!(n::Int, x::Vector{T}, g::Vector{T}, ind::Int) where {T<:AbstractFloat}

    g .= zeros(T, n)

    if is_scalable_benchmark(PROBLEM)
        scalable_evalg!(g, PROBLEM, x, ind)

    elseif PROBLEM == "RE41" || PROBLEM == "RE42" || PROBLEM == "RE61"
        g .= ForwardDiff.gradient(z -> re_applied_evalf(PROBLEM, z, ind), x)

    elseif PROBLEM == "RE21" || PROBLEM == "RE34" || PROBLEM == "RE37"
        g .= ForwardDiff.gradient(z -> re_phasea_evalf(PROBLEM, z, ind), x)

    elseif PROBLEM == "AP1"
        if ind == 1
            g .= [(x[1]-1)^3, 2*(x[2]-2)^3]
        elseif ind == 2
            t = 0.5 * exp((x[1]+x[2])/2)
            g .= [t + 2x[1], t + 2x[2]]
        elseif ind == 3
            g .= [-1/6*exp(-x[1]), -1/3*exp(-x[2])]
        end

    elseif PROBLEM == "AP2"
        g .= ind == 1 ? [2x[1]] : [2*(x[1]-1)]

    elseif PROBLEM == "AP3"
        if ind == 1
            g .= [(x[1]-1)^3, 2*(x[2]-2)^3]
        else
            g[1] = -4x[1]*(x[2]-x[1]^2) - 2*(1-x[1])
            g[2] =  2*(x[2]-x[1]^2)
        end

    elseif PROBLEM == "AP4"
        if ind == 1
            g .= [4/9*(x[1]-1)^3, 8/9*(x[2]-2)^3, 12/9*(x[3]-3)^3]
        elseif ind == 2
            t = (1/3)*exp((x[1]+x[2]+x[3])/3)
            g .= [t + 2x[1], t + 2x[2], t + 2x[3]]
        elseif ind == 3
            g .= [-1/4*exp(-x[1]), -1/3*exp(-x[2]), -1/4*exp(-x[3])]
        end

    elseif PROBLEM == "BK1"
        g .= ind == 1 ? 2 .* x : 2 .* (x .- 5)

    elseif PROBLEM == "DD1"
        if ind == 1
            g .= 2.0 .* x
        elseif ind == 2
            g[1] = 3
            g[2] = 2
            g[3] = -1 / 3
            g[4] = 3e-2 * (x[4] - x[5])^2
            g[5] = -3e-2 * (x[4] - x[5])^2
        end

    elseif PROBLEM == "DGO1"
        if ind == 1
            g[1] = cos(x[1])
        elseif ind == 2
            g[1] = cos(x[1] + 0.7)
        end

    elseif PROBLEM == "DGO1M"
        if ind == 1
            g[1] = cos(x[1]) + 0.1 * (cos(5.0 * x[1]) * cos(x[1]) - 5.0 * sin(5.0 * x[1]) * sin(x[1]))
        elseif ind == 2
            y = x[1] + 0.7
            g[1] = cos(y) + 0.1 * (cos(5.0 * x[1]) * cos(y) - 5.0 * sin(5.0 * x[1]) * sin(y))
        end

    elseif PROBLEM == "DGO2"
        if ind == 1
            g[1] = 2 * x[1]
        elseif ind == 2
            g[1] = x[1] / sqrt(81 - x[1]^2)
        end

    elseif PROBLEM == "FA1"
        if ind == 1
            g[1] = 4 * exp(-4 * x[1]) / (1 - exp(-4))
        elseif ind == 2
            a = exp(-4 * x[1])
            b = 1 - exp(-4)
            t = (1 - a) / (b * (x[2] + 1))
            g[1] = -2 * a / b * t^(-0.5)
            g[2] = 1 - 0.5 * sqrt(t)
        elseif ind == 3
            a = exp(-4 * x[1])
            b = 1 - exp(-4)
            t = (1 - a) / (b * (x[3] + 1))
            g[1] = -0.4 * t^(-0.9) * a / b
            g[3] = 1 - 0.9 * t^0.1
        end

    elseif PROBLEM == "Far1"
        if ind == 1
            g[1] = 60.0 * (x[1] - 0.1) * exp(15.0 * (-(x[1] - 0.1)^2 - x[2]^2)) +
                   40.0 * (x[1] - 0.6) * exp(20.0 * (-(x[1] - 0.6)^2 - (x[2] - 0.6)^2)) -
                   40.0 * (x[1] + 0.6) * exp(20.0 * (-(x[1] + 0.6)^2 - (x[2] - 0.6)^2)) -
                   40.0 * (x[1] - 0.6) * exp(20.0 * (-(x[1] - 0.6)^2 - (x[2] + 0.6)^2)) -
                   40.0 * (x[1] + 0.6) * exp(20.0 * (-(x[1] + 0.6)^2 - (x[2] + 0.6)^2))

            g[2] = 60.0 * x[2] * exp(15.0 * (-(x[1] - 0.1)^2 - x[2]^2)) +
                   40.0 * (x[2] - 0.6) * exp(20.0 * (-(x[1] - 0.6)^2 - (x[2] - 0.6)^2)) -
                   40.0 * (x[2] - 0.6) * exp(20.0 * (-(x[1] + 0.6)^2 - (x[2] - 0.6)^2)) -
                   40.0 * (x[2] + 0.6) * exp(20.0 * (-(x[1] - 0.6)^2 - (x[2] + 0.6)^2)) -
                   40.0 * (x[2] + 0.6) * exp(20.0 * (-(x[1] + 0.6)^2 - (x[2] + 0.6)^2))

        elseif ind == 2
            g[1] = -80.0 * x[1] * exp(20.0 * (-(x[1]^2 + x[2]^2))) -
                   40.0 * (x[1] - 0.4) * exp(20.0 * (-(x[1] - 0.4)^2 - (x[2] - 0.6)^2)) +
                   40.0 * (x[1] + 0.5) * exp(20.0 * (-(x[1] + 0.5)^2 - (x[2] - 0.7)^2)) +
                   40.0 * (x[1] - 0.5) * exp(20.0 * (-(x[1] - 0.5)^2 - (x[2] + 0.7)^2)) -
                   40.0 * (x[1] + 0.4) * exp(20.0 * (-(x[1] + 0.4)^2 - (x[2] + 0.8)^2))

            g[2] = -80.0 * x[2] * exp(20.0 * (-(x[1]^2 + x[2]^2))) -
                   40.0 * (x[2] - 0.6) * exp(20.0 * (-(x[1] - 0.4)^2 - (x[2] - 0.6)^2)) +
                   40.0 * (x[2] - 0.7) * exp(20.0 * (-(x[1] + 0.5)^2 - (x[2] - 0.7)^2)) +
                   40.0 * (x[2] + 0.7) * exp(20.0 * (-(x[1] - 0.5)^2 - (x[2] + 0.7)^2)) -
                   40.0 * (x[2] + 0.8) * exp(20.0 * (-(x[1] + 0.4)^2 - (x[2] + 0.8)^2))
        end

    elseif PROBLEM == "FDS"
        if ind == 1
            for i in 1:n
                g[i] = 4.0 * i * (x[i] - i)^3 / n^2
            end
        elseif ind == 2
            s = sum(x) / n
            for i in 1:n
                g[i] = exp(s) / n + 2.0 * x[i]
            end
        elseif ind == 3
            for i in 1:n
                g[i] = -i * (n - i + 1.0) * exp(-x[i]) / (n * (n + 1.0))
            end
        end

    elseif PROBLEM == "FF1"
        if ind == 1
            g[1] = 2.0 * (x[1] - 1.0) * exp(-((x[1] - 1.0)^2 + (x[2] + 1.0)^2))
            g[2] = 2.0 * (x[2] + 1.0) * exp(-((x[1] - 1.0)^2 + (x[2] + 1.0)^2))
        elseif ind == 2
            g[1] = 2.0 * (x[1] + 1.0) * exp(-((x[1] + 1.0)^2 + (x[2] - 1.0)^2))
            g[2] = 2.0 * (x[2] - 1.0) * exp(-((x[1] + 1.0)^2 + (x[2] - 1.0)^2))
        end

    elseif PROBLEM == "Hil1"
        a = 2π / 360 * (45.0 + 40.0 * sin(2π * x[1]) + 25.0 * sin(2π * x[2]))
        b = 1.0 + 0.5 * cos(2π * x[1])
        if ind == 1
            g[1] = -160.0 * π^2 / 360.0 * cos(2π * x[1]) * sin(a) * b -
                   π * sin(2π * x[1]) * cos(a)
            g[2] = -100.0 * π^2 / 360.0 * cos(2π * x[2]) * sin(a) * b
        elseif ind == 2
            g[1] = 160.0 * π^2 / 360.0 * cos(2π * x[1]) * cos(a) * b -
                   π * sin(2π * x[1]) * sin(a)
            g[2] = 100.0 * π^2 / 360.0 * cos(2π * x[2]) * cos(a) * b
        end

    elseif PROBLEM == "IKK1"
        if ind == 1
            g[1] = 2.0 * x[1]
            g[2] = 0.0
        elseif ind == 2
            g[1] = 2.0 * (x[1] - 20.0)
            g[2] = 0.0
        elseif ind == 3
            g[1] = 0.0
            g[2] = 2.0 * x[2]
        end

    elseif PROBLEM == "IM1"
        if ind == 1
            g[1] = 1.0 / sqrt(x[1])
            g[2] = 0.0
        elseif ind == 2
            g[1] = 1.0 - x[2]
            g[2] = -x[1]
        end

    elseif PROBLEM == "JOS1"
        if ind == 1
            for i in 1:n
                g[i] = 2.0 * x[i] / n
            end
        elseif ind == 2
            for i in 1:n
                g[i] = 2.0 * (x[i] - 2.0) / n
            end
        end

    elseif PROBLEM == "JOS4"
        if ind == 1
            g[1] = 1.0
            g[2:n] .= 0.0
        elseif ind == 2
            faux = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / faux
            g[1] = -0.25 * t^(-0.75) - 4.0 * t^3
            for i in 2:n
                g[i] = 9.0 / (n - 1) * (1.0 - 0.75 * t^0.25 + 3.0 * t^4)
            end
        end

    elseif PROBLEM == "KW2"
        if ind == 1
            g[1] = 6.0 * (1.0 - x[1]) * exp(-x[1]^2 - (x[2] + 1.0)^2) +
                   6.0 * (1.0 - x[1])^2 * exp(-x[1]^2 - (x[2] + 1.0)^2) * x[1] +
                   10.0 * (1.0 / 5.0 - 3.0 * x[1]^2) * exp(-x[1]^2 - x[2]^2) -
                   20.0 * (x[1] / 5.0 - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2) * x[1] -
                   6.0 * exp(-(x[1] + 2.0)^2 - x[2]^2) * (x[1] + 2.0) - 1.0

            g[2] = 6.0 * (1.0 - x[1])^2 * exp(-x[1]^2 - (x[2] + 1.0)^2) * (x[2] + 1.0) -
                   50.0 * x[2]^4 * exp(-x[1]^2 - x[2]^2) -
                   10.0 * (x[1] / 5.0 - x[1]^3 - x[2]^5) * exp(-x[1]^2 - x[2]^2) * 2.0 * x[2] -
                   6.0 * exp(-(x[1] + 2.0)^2 - x[2]^2) * x[2] - 0.5

        elseif ind == 2
            g[1] = -6.0 * (1.0 + x[2])^2 * exp(-x[2]^2 - (1.0 - x[1])^2) * (1.0 - x[1]) +
                   50.0 * x[1]^4 * exp(-x[1]^2 - x[2]^2) -
                   20.0 * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2) * x[1] -
                   6.0 * exp(-(2.0 - x[2])^2 - x[1]^2) * x[1]

            g[2] = -6.0 * (1.0 + x[2]) * exp(-x[2]^2 - (1.0 - x[1])^2) +
                   6.0 * (1.0 + x[2])^2 * exp(-x[2]^2 - (1.0 - x[1])^2) * x[2] +
                   10.0 * (-1.0 / 5.0 + 3.0 * x[2]^2) * exp(-x[1]^2 - x[2]^2) -
                   20.0 * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * exp(-x[1]^2 - x[2]^2) * x[2] +
                   6.0 * exp(-(2.0 - x[2])^2 - x[1]^2) * (2.0 - x[2])
        end

    elseif PROBLEM == "LE1"
        if ind == 1
            t = 0.25 * (x[1]^2 + x[2]^2)^(-0.875)
            g[1] = x[1] * t
            g[2] = x[2] * t
        elseif ind == 2
            t = 0.5 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2)^(-0.75)
            g[1] = (x[1] - 0.5) * t
            g[2] = (x[2] - 0.5) * t
        end

    elseif PROBLEM == "Lov1"
        if ind == 1
            g[1] = 2.1 * x[1]
            g[2] = 2.0 * 0.98 * x[2]
        elseif ind == 2
            g[1] = 2.0 * 0.99 * (x[1] - 3.0)
            g[2] = 2.0 * 1.03 * (x[2] - 2.5)
        end

    elseif PROBLEM == "Lov2"
        if ind == 1
            g[1] = 0.0
            g[2] = 1.0
        elseif ind == 2
            g[1] = -(-3.0 * x[1]^2 * (x[1] + 1.0) - (x[2] - x[1]^3)) / (x[1] + 1.0)^2
            g[2] = -1.0 / (x[1] + 1.0)
        end

    elseif PROBLEM == "Lov3"
        if ind == 1
            g[1] = 2.0 * x[1]
            g[2] = 2.0 * x[2]
        elseif ind == 2
            g[1] = 2.0 * (x[1] - 6.0)
            g[2] = -2.0 * (x[2] + 0.3)
        end

    elseif PROBLEM == "Lov4"
        if ind == 1
            g[1] = 2.0 * x[1] - 8.0 * ((x[1] + 2.0) * exp(-(x[1] + 2.0)^2 - x[2]^2) +
                                        (x[1] - 2.0) * exp(-(x[1] - 2.0)^2 - x[2]^2))
            g[2] = 2.0 * x[2] - 8.0 * (x[2] * exp(-(x[1] + 2.0)^2 - x[2]^2) +
                                        x[2] * exp(-(x[1] - 2.0)^2 - x[2]^2))
        elseif ind == 2
            g[1] = 2.0 * (x[1] - 6.0)
            g[2] = 2.0 * (x[2] + 0.5)
        end

    elseif PROBLEM == "Lov5"
        M = [
            -1.0   -0.03   0.011;
            -0.03  -1.0    0.07;
             0.011  0.07  -1.01
        ]

        p1 = [x[1], x[2] - 0.15, x[3]]
        a1 = 0.35
        A1 = sqrt(2π / a1) * exp(dot(p1, M * p1) / a1^2)

        p2 = [x[1], x[2] + 1.1, 0.5 * x[3]]
        a2 = 3.0
        A2 = sqrt(2π / a2) * exp(dot(p2, M * p2) / a2^2)

        if ind == 1
            g[1] = sqrt(2)/2 + sqrt(2)/2 * A1 * (2M[1,1]*x[1] + 2M[1,3]*x[3] + 2M[1,2]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (2M[1,1]*x[1] + M[1,3]*x[3] + 2M[1,2]*(x[2] + 1.1)) / a2^2
            g[1] = -g[1]

            g[2] = sqrt(2)/2 * A1 * (2M[1,2]*x[1] + 2M[2,3]*x[3] + 2M[2,2]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (2M[1,2]*x[1] + M[2,3]*x[3] + 2M[2,2]*(x[2] + 1.1)) / a2^2
            g[2] = -g[2]

            g[3] = sqrt(2)/2 * A1 * (2M[1,3]*x[1] + 2M[3,3]*x[3] + 2M[2,3]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (M[1,3]*x[1] + M[3,3]*x[3]/2 + M[2,3]*(x[2] + 1.1)) / a2^2
            g[3] = -g[3]

        elseif ind == 2
            g[1] = -sqrt(2)/2 + sqrt(2)/2 * A1 * (2M[1,1]*x[1] + 2M[1,3]*x[3] + 2M[1,2]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (2M[1,1]*x[1] + M[1,3]*x[3] + 2M[1,2]*(x[2] + 1.1)) / a2^2
            g[1] = -g[1]

            g[2] = sqrt(2)/2 * A1 * (2M[1,2]*x[1] + 2M[2,3]*x[3] + 2M[2,2]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (2M[1,2]*x[1] + M[2,3]*x[3] + 2M[2,2]*(x[2] + 1.1)) / a2^2
            g[2] = -g[2]

            g[3] = sqrt(2)/2 * A1 * (2M[1,3]*x[1] + 2M[3,3]*x[3] + 2M[2,3]*(x[2] - 0.15)) / a1^2 +
                    sqrt(2)/2 * A2 * (M[1,3]*x[1] + M[3,3]*x[3]/2 + M[2,3]*(x[2] + 1.1)) / a2^2
            g[3] = -g[3]
        end

    elseif PROBLEM == "Lov6"
        if ind == 1
            g[1] = 1.0
            g[2:6] .= 0.0
        elseif ind == 2
            g[1] = -0.5 / sqrt(x[1]) - sin(10π * x[1]) - 10π * x[1] * cos(10π * x[1])
            for i in 2:6
                g[i] = 2.0 * x[i]
            end
        end

    elseif PROBLEM == "LTDZ"
        if ind == 1
            g[1] = π / 2 * (1 + x[3]) * sin(x[1] * π / 2) * cos(x[2] * π / 2)
            g[2] = π / 2 * (1 + x[3]) * cos(x[1] * π / 2) * sin(x[2] * π / 2)
            g[3] = -cos(x[1] * π / 2) * cos(x[2] * π / 2)
            g .= -g
        elseif ind == 2
            g[1] = π / 2 * (1 + x[3]) * sin(x[1] * π / 2) * sin(x[2] * π / 2)
            g[2] = -π / 2 * (1 + x[3]) * cos(x[1] * π / 2) * cos(x[2] * π / 2)
            g[3] = -cos(x[1] * π / 2) * sin(x[2] * π / 2)
            g .= -g
        elseif ind == 3
            g[1] = π / 2 * (1 + x[3]) * sin(x[1] * π / 2)^2 - π / 2 * (1 + x[3]) * cos(x[1] * π / 2)^2
            g[2] = 0.0
            g[3] = -cos(x[1] * π / 2) * sin(x[1] * π / 2)
            g .= -g
        end

    elseif PROBLEM == "MGH9"
        t = (8.0 - ind) / 2.0
        g[1] = exp(-x[2] * (t - x[3])^2 / 2.0)
        g[2] = -x[1] * exp(-x[2] * (t - x[3])^2 / 2.0) * (t - x[3])^2 / 2.0
        g[3] = x[1] * exp(-x[2] * (t - x[3])^2 / 2.0) * x[2] * (t - x[3])

    elseif PROBLEM == "MGH16"
        t = ind / 5.0
        g[1] = 2.0 * (x[1] + t * x[2] - exp(t))
        g[2] = 2.0 * t * (x[1] + t * x[2] - exp(t))
        g[3] = 2.0 * (x[3] + x[4] * sin(t) - cos(t))
        g[4] = 2.0 * sin(t) * (x[3] + x[4] * sin(t) - cos(t))

    elseif PROBLEM == "MGH26"
        t = 0.0
        for i in 1:n
            t += cos(x[i])
        end
        gaux1 = 2.0 * (n - t + ind * (1.0 - cos(x[ind])) - sin(x[ind]))
        for i in 1:n
            g[i] = gaux1 * sin(x[i])
        end
        g[ind] += gaux1 * (ind * sin(x[ind]) - cos(x[ind]))

    elseif PROBLEM == "MGH33"
        faux = 0.0
        for i in 1:n
            faux += i * x[i]
        end
        faux = 2.0 * (ind * faux - 1.0)
        for i in 1:n
            g[i] = faux * i * ind
        end

    elseif PROBLEM == "MHHM2"
        if ind == 1
            g[1] = 2.0 * (x[1] - 0.8)
            g[2] = 2.0 * (x[2] - 0.6)
        elseif ind == 2
            g[1] = 2.0 * (x[1] - 0.85)
            g[2] = 2.0 * (x[2] - 0.7)
        elseif ind == 3
            g[1] = 2.0 * (x[1] - 0.9)
            g[2] = 2.0 * (x[2] - 0.6)
        end

    elseif PROBLEM == "MLF1"
        if ind == 1
            g[1] = sin(x[1]) / 20.0 + (1.0 + x[1] / 20.0) * cos(x[1])
        elseif ind == 2
            g[1] = cos(x[1]) / 20.0 - (1.0 + x[1] / 20.0) * sin(x[1])
        end

    elseif PROBLEM == "MLF2"
        if ind == 1
            g[1] = (2.0 * x[1] * (x[1]^2 + x[2] - 11.0) + (x[1] + x[2]^2 - 7.0)) / 100.0
            g[2] = ((x[1]^2 + x[2] - 11.0) + 2.0 * x[2] * (x[1] + x[2]^2 - 7.0)) / 100.0
        elseif ind == 2
            g[1] = (8.0 * x[1] * (4.0 * x[1]^2 + 2.0 * x[2] - 11.0) + 2.0 * (2.0 * x[1] + 4.0 * x[2]^2 - 7.0)) / 100.0
            g[2] = (2.0 * (4.0 * x[1]^2 + 2.0 * x[2] - 11.0) + 8.0 * x[2] * (2.0 * x[1] + 4.0 * x[2]^2 - 7.0)) / 100.0
        end

    elseif PROBLEM == "MMR1"
        if ind == 1
            g[1] = 2.0 * x[1]
            g[2] = 0.0
        elseif ind == 2
            g1 = 2.0 - 0.8 * exp(-((x[2] - 0.6) / 0.4)^2) - exp(-((x[2] - 0.2) / 0.04)^2)
            g[1] = -2.0 * x[1] * g1 / (1.0 + x[1]^2)^2
            g[2] = (10.0 * exp(-((x[2] - 0.6) / 0.4)^2) * (x[2] - 0.6) +
                    1250.0 * exp(-((x[2] - 0.2) / 0.04)^2) * (x[2] - 0.2)) / (1.0 + x[1]^2)
        end

    elseif PROBLEM == "MMR3"
        if ind == 1
            g[1] = 3.0 * x[1]^2
            g[2] = 0.0
        elseif ind == 2
            g[1] = -3.0 * (x[2] - x[1])^2
            g[2] = 3.0 * (x[2] - x[1])^2
        end

    elseif PROBLEM == "MMR4"
        if ind == 1
            t = (2.0 * x[1] + x[2] + 2.0 * x[3] + 1.0)^2
            g[1] = 1.0 + 72.0 / t
            g[2] = -2.0 + 36.0 / t
            g[3] = -1.0 + 72.0 / t
        elseif ind == 2
            g[1] = -3.0
            g[2] = 1.0
            g[3] = -1.0
        end

    elseif PROBLEM == "MOP2"
        if ind == 1
            faux = 0.0
            for i in 1:n
                faux += (x[i] - 1.0 / sqrt(n))^2
            end
            for i in 1:n
                g[i] = 2.0 * (x[i] - 1.0 / sqrt(n)) * exp(-faux)
            end
        elseif ind == 2
            faux = 0.0
            for i in 1:n
                faux += (x[i] + 1.0 / sqrt(n))^2
            end
            for i in 1:n
                g[i] = 2.0 * (x[i] + 1.0 / sqrt(n)) * exp(-faux)
            end
        end

    elseif PROBLEM == "MOP3"
        if ind == 1
            A1 = 0.5 * sin(1.0) - 2.0 * cos(1.0) + sin(2.0) - 1.5 * cos(2.0)
            A2 = 1.5 * sin(1.0) - cos(1.0) + 2.0 * sin(2.0) - 0.5 * cos(2.0)
            B1 = 0.5 * sin(x[1]) - 2.0 * cos(x[1]) + sin(x[2]) - 1.5 * cos(x[2])
            B2 = 1.5 * sin(x[1]) - cos(x[1]) + 2.0 * sin(x[2]) - 0.5 * cos(x[2])
            g[1] = 2.0 * (A1 - B1) * (-0.5 * cos(x[1]) - 2.0 * sin(x[1])) +
                   2.0 * (A2 - B2) * (-1.5 * cos(x[1]) - sin(x[1]))
            g[2] = 2.0 * (A1 - B1) * (-cos(x[2]) - 1.5 * sin(x[2])) +
                   2.0 * (A2 - B2) * (-2.0 * cos(x[2]) - 0.5 * sin(x[2]))
        elseif ind == 2
            g[1] = 2.0 * (x[1] + 3.0)
            g[2] = 2.0 * (x[2] + 1.0)
        end

    elseif PROBLEM == "MOP5"
        if ind == 1
            g[1] = x[1] + 2.0 * x[1] * cos(x[1]^2 + x[2]^2)
            g[2] = x[2] + 2.0 * x[2] * cos(x[1]^2 + x[2]^2)
        elseif ind == 2
            g[1] = 3.0 * (3.0 * x[1] - 2.0 * x[2] + 4.0) / 4.0 + 2.0 * (x[1] - x[2] + 1.0) / 27.0
            g[2] = -2.0 * (3.0 * x[1] - 2.0 * x[2] + 4.0) / 4.0 - 2.0 * (x[1] - x[2] + 1.0) / 27.0
        elseif ind == 3
            g[1] = -2.0 * x[1] / (x[1]^2 + x[2]^2 + 1.0)^2 + 2.2 * x[1] * exp(-(x[1]^2 + x[2]^2))
            g[2] = -2.0 * x[2] / (x[1]^2 + x[2]^2 + 1.0)^2 + 2.2 * x[2] * exp(-(x[1]^2 + x[2]^2))
        end
    elseif PROBLEM == "MOP6"
        if ind == 1
            g[1] = 1.0
            g[2] = 0.0
        elseif ind == 2
            a = 1.0 + 10.0 * x[2]
            b = sin(8.0 * π * x[1])
            t = x[1] / a
            g[1] = -2.0 * t - b - 8.0 * π * x[1] * cos(8.0 * π * x[1])
            g[2] = 10.0 * (1.0 - t^2 - t * b) + 10.0 * x[1] / a * (2.0 * t + b)
        end

    elseif PROBLEM == "MOP7"
        if ind == 1
            g[1] = x[1] - 2.0
            g[2] = 2.0 * (x[2] + 1.0) / 13.0
        elseif ind == 2
            g[1] = (x[1] + x[2] - 3.0) / 18.0 - (-x[1] + x[2] + 2.0) / 4.0
            g[2] = (x[1] + x[2] - 3.0) / 18.0 + (-x[1] + x[2] + 2.0) / 4.0
        elseif ind == 3
            g[1] = 2.0 * (x[1] + 2.0 * x[2] - 1.0) / 175.0 - 2.0 * (-x[1] + 2.0 * x[2]) / 17.0
            g[2] = 4.0 * (x[1] + 2.0 * x[2] - 1.0) / 175.0 + 4.0 * (-x[1] + 2.0 * x[2]) / 17.0
        end

    elseif PROBLEM == "PNR"
        if ind == 1
            g[1] = 4.0 * x[1]^3 - 2.0 * x[1] - 10.0 * x[2]
            g[2] = 4.0 * x[2]^3 + 2.0 * x[2] - 10.0 * x[1]
        elseif ind == 2
            g[1] = 2.0 * x[1]
            g[2] = 2.0 * x[2]
        end

    elseif PROBLEM == "QV1"
        if ind == 1
            faux = 0.0
            for i in 1:n
                faux += x[i]^2 - 10.0 * cos(2.0 * π * x[i]) + 10.0
            end
            faux = 0.25 * (faux / n)^(-0.75)
            for i in 1:n
                g[i] = faux * (2.0 * x[i] + 20.0 * π * sin(2.0 * π * x[i])) / n
            end
        elseif ind == 2
            faux = 0.0
            for i in 1:n
                faux += (x[i] - 1.5)^2 - 10.0 * cos(2.0 * π * (x[i] - 1.5)) + 10.0
            end
            faux = 0.25 * (faux / n)^(-0.75)
            for i in 1:n
                g[i] = faux * (2.0 * (x[i] - 1.5) + 20.0 * π * sin(2.0 * π * (x[i] - 1.5))) / n
            end
        end

    elseif PROBLEM == "SD"
        if ind == 1
            g[1] = 2.0
            g[2] = sqrt(2.0)
            g[3] = sqrt(2.0)
            g[4] = 1.0
        elseif ind == 2
            g[1] = -2.0 / x[1]^2
            g[2] = -2.0 * sqrt(2.0) / x[2]^2
            g[3] = -2.0 * sqrt(2.0) / x[3]^2
            g[4] = -2.0 / x[4]^2
        end

    elseif PROBLEM == "SLCDT1"
        if ind == 1
            g[1] = 0.5 * ((x[1] + x[2]) / sqrt(1.0 + (x[1] + x[2])^2) +
                          (x[1] - x[2]) / sqrt(1.0 + (x[1] - x[2])^2) + 1.0) -
                   2.0 * 0.85 * (x[1] + x[2]) * exp(-(x[1] + x[2])^2)
            g[2] = 0.5 * ((x[1] + x[2]) / sqrt(1.0 + (x[1] + x[2])^2) -
                          (x[1] - x[2]) / sqrt(1.0 + (x[1] - x[2])^2) - 1.0) -
                   2.0 * 0.85 * (x[1] + x[2]) * exp(-(x[1] + x[2])^2)
        elseif ind == 2
            g[1] = 0.5 * ((x[1] + x[2]) / sqrt(1.0 + (x[1] + x[2])^2) +
                          (x[1] - x[2]) / sqrt(1.0 + (x[1] - x[2])^2) - 1.0) -
                   2.0 * 0.85 * exp(-(x[1] + x[2])^2) * (x[1] + x[2])
            g[2] = 0.5 * ((x[1] + x[2]) / sqrt(1.0 + (x[1] + x[2])^2) -
                          (x[1] - x[2]) / sqrt(1.0 + (x[1] - x[2])^2) + 1.0) -
                   2.0 * 0.85 * (x[1] + x[2]) * exp(-(x[1] + x[2])^2)
        end

    elseif PROBLEM == "SLCDT2"
        if ind == 1
            g[1] = 4.0 * (x[1] - 1.0)^3
            for i in 2:n
                g[i] = 2.0 * (x[i] - 1.0)
            end
        elseif ind == 2
            g[2] = 4.0 * (x[2] + 1.0)^3
            for i in 1:n
                if i != 2
                    g[i] = 2.0 * (x[i] + 1.0)
                end
            end
        elseif ind == 3
            g[3] = 4.0 * (x[3] - 1.0)^3
            for i in 1:n
                if i != 3
                    g[i] = 2.0 * (x[i] - (-1.0)^(i + 1))
                end
            end
        end

    elseif PROBLEM == "SP1"
        if ind == 1
            g[1] = 2.0 * (x[1] - 1.0) + 2.0 * (x[1] - x[2])
            g[2] = -2.0 * (x[1] - x[2])
        elseif ind == 2
            g[1] = 2.0 * (x[1] - x[2])
            g[2] = 2.0 * (x[2] - 3.0) - 2.0 * (x[1] - x[2])
        end

    elseif PROBLEM == "SSFYY2"
        if ind == 1
            g[1] = 2.0 * x[1] + 5.0 * π * sin(x[1] * π / 2.0)
        elseif ind == 2
            g[1] = 2.0 * (x[1] - 4.0)
        end

    elseif PROBLEM == "SK1"
        if ind == 1
            g[1] = 4.0 * x[1]^3 + 9.0 * x[1]^2 - 20.0 * x[1] - 10.0
        elseif ind == 2
            g[1] = 2.0 * x[1]^3 - 6.0 * x[1]^2 - 20.0 * x[1] + 10.0
        end

    elseif PROBLEM == "SK2"
        if ind == 1
            g[1] = 2.0 * (x[1] - 2.0)
            g[2] = 2.0 * (x[2] + 3.0)
            g[3] = 2.0 * (x[3] - 5.0)
            g[4] = 2.0 * (x[4] - 4.0)
        elseif ind == 2
            faux = 1.0 + (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) / 100.0
            s = sin(x[1]) + sin(x[2]) + sin(x[3]) + sin(x[4])
            g[1] = (-cos(x[1]) * faux + s * x[1] / 50.0) / faux^2
            g[2] = (-cos(x[2]) * faux + s * x[2] / 50.0) / faux^2
            g[3] = (-cos(x[3]) * faux + s * x[3] / 50.0) / faux^2
            g[4] = (-cos(x[4]) * faux + s * x[4] / 50.0) / faux^2
        end

    elseif PROBLEM == "TKLY1"
        if ind == 1
            g[1] = 1.0
            g[2] = 0.0
            g[3] = 0.0
            g[4] = 0.0
        elseif ind == 2
            A1 = 2.0 - exp(-((x[2] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[2] - 0.9) / 0.4)^2)
            A2 = 2.0 - exp(-((x[3] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[3] - 0.9) / 0.4)^2)
            A3 = 2.0 - exp(-((x[4] - 0.1) / 0.004)^2) - 0.8 * exp(-((x[4] - 0.9) / 0.4)^2)
            g[1] = -A1 * A2 * A3 / x[1]^2
            g[2] = A2 * A3 / x[1] * (500.0 * exp(-((x[2] - 0.1) / 0.004)^2) * ((x[2] - 0.1) / 0.004) +
                                     4.0 * exp(-((x[2] - 0.9) / 0.4)^2) * ((x[2] - 0.9) / 0.4))
            g[3] = A1 * A3 / x[1] * (500.0 * exp(-((x[3] - 0.1) / 0.004)^2) * ((x[3] - 0.1) / 0.004) +
                                     4.0 * exp(-((x[3] - 0.9) / 0.4)^2) * ((x[3] - 0.9) / 0.4))
            g[4] = A1 * A2 / x[1] * (500.0 * exp(-((x[4] - 0.1) / 0.004)^2) * ((x[4] - 0.1) / 0.004) +
                                     4.0 * exp(-((x[4] - 0.9) / 0.4)^2) * ((x[4] - 0.9) / 0.4))
        end

    elseif PROBLEM == "Toi4"
        if ind == 1
            g[1] = 2.0 * x[1]
            g[2] = 2.0 * x[2]
            g[3] = 0.0
            g[4] = 0.0
        elseif ind == 2
            g[1] = x[1] - x[2]
            g[2] = -(x[1] - x[2])
            g[3] = x[3] - x[4]
            g[4] = -(x[3] - x[4])
        end

    elseif PROBLEM == "Toi8"
        fill!(g, 0.0)
        if ind == 1
            g[1] = 4.0 * (2.0 * x[1] - 1.0)
        elseif ind != 1
            g[ind - 1] = 4.0 * ind * (2.0 * x[ind - 1] - x[ind])
            g[ind] = -2.0 * ind * (2.0 * x[ind - 1] - x[ind])
        end

    elseif PROBLEM == "Toi9"
        fill!(g, 0.0)
        if ind == 1
            g[1] = 4.0 * (2.0 * x[1] - 1.0)
            g[2] = 2.0 * x[2]
        elseif ind > 1 && ind < n
            g[ind - 1] = 4.0 * ind * (2.0 * x[ind - 1] - x[ind]) - 2.0 * (ind - 1) * x[ind - 1]
            g[ind] = -2.0 * ind * (2.0 * x[ind - 1] - x[ind]) + 2.0 * ind * x[ind]
        elseif ind == n
            g[n - 1] = 4.0 * n * (2.0 * x[n - 1] - x[n]) - 2.0 * (n - 1) * x[n - 1]
            g[n] = -2.0 * n * (2.0 * x[n - 1] - x[n])
        end

    elseif PROBLEM == "Toi10"
        fill!(g, 0.0)
        g[ind] = -400.0 * (x[ind + 1] - x[ind]^2) * x[ind]
        g[ind + 1] = 200.0 * (x[ind + 1] - x[ind]^2) + 2.0 * (x[ind + 1] - 1.0)

    elseif PROBLEM == "VU1"
        if ind == 1
            g[1] = -2.0 * x[1] / (x[1]^2 + x[2]^2 + 1.0)^2
            g[2] = -2.0 * x[2] / (x[1]^2 + x[2]^2 + 1.0)^2
        elseif ind == 2
            g[1] = 2.0 * x[1]
            g[2] = 6.0 * x[2]
        end

    elseif PROBLEM == "VU2"
        if ind == 1
            g[1] = 1.0
            g[2] = 1.0
        elseif ind == 2
            g[1] = 2.0 * x[1]
            g[2] = 2.0
        end

    elseif PROBLEM == "ZDT1"
        if ind == 1
            g[1] = 1
        elseif ind == 2
            t = 1 + 9 * sum(x[2:end]) / (n - 1)
            u = x[1] / t
            g[1] = -0.5 * u^(-0.5)
            g[2:end] .= 9 / (n - 1) * (1 - sqrt(u) / 2)
        end

    elseif PROBLEM == "ZDT2"
        if ind == 1
            g[1] = 1
        elseif ind == 2
            t = 1 + 9 * sum(x[2:end]) / (n - 1)
            u = x[1] / t
            g[1] = -2 * u
            g[2:end] .= 9 / (n - 1) * (1 + u^2)
        end

    elseif PROBLEM == "ZDT3"
        if ind == 1
            g[1] = 1
        elseif ind == 2
            t = 1 + 9 * sum(x[2:end]) / (n - 1)
            u = x[1] / t
            g[1] = -0.5 * u^(-0.5) - sin(10π * x[1]) - 10π * x[1] * cos(10π * x[1])
            g[2:end] .= 9 / (n - 1) * (1 - 0.5 * sqrt(u))
        end

    elseif PROBLEM == "ZDT4"
        if ind == 1
            g[1] = 1
        elseif ind == 2
            t = 1 + 10 * (n - 1) + sum(x[2:end].^2 .- 10 * cos.(4π * x[2:end]))
            u = x[1] / t
            g[1] = -0.5 * u^(-0.5)
            for i in 2:n
                g[i] = (2 * x[i] + 40π * sin(4π * x[i])) * (1 - sqrt(u) / 2)
            end
        end

    elseif PROBLEM == "ZDT6"
        if ind == 1
            a = exp(-4 * x[1])
            b = sin(6π * x[1])
            g[1] = 4 * a * b^6 - 36π * a * b^5 * cos(6π * x[1])
        elseif ind == 2
            a = exp(-4 * x[1])
            b = sin(6π * x[1])
            t = 1 - a * b^6
            u = 1 + 9 * (sum(x[2:end])/(n-1))^0.25
            v = t / u
            A1 = 9 * 0.25 * (sum(x[2:end])/(n-1))^(-0.75) / (n - 1)
            g[1] = -2 * v * (4 * a * b^6 - 36π * a * b^5 * cos(6π * x[1]))
            g[2:end] .= A1 * (1 + v^2)
        end

    elseif PROBLEM == "ZLT1"
        g .= 2 * x
        g[ind] += -2 

    else
        error("Unknown PROBLEM: $PROBLEM")
    end

    return g
end

# ======================================================================
# Hessian of objective component f_i(x)
# ======================================================================

"""Write the Hessian of objective component `ind` into the preallocated `H`."""
function evalh!(n::Int, x::Vector{T}, H::Matrix{T}, ind::Int) where {T<:AbstractFloat}
    
    H .= zeros(T, n, n)

    if is_scalable_benchmark(PROBLEM)
        scalable_evalh!(H, PROBLEM, x, ind)

    elseif PROBLEM == "RE41" || PROBLEM == "RE42" || PROBLEM == "RE61"
        H .= ForwardDiff.hessian(z -> re_applied_evalf(PROBLEM, z, ind), x)

    elseif PROBLEM == "RE21" || PROBLEM == "RE34" || PROBLEM == "RE37"
        H .= ForwardDiff.hessian(z -> re_phasea_evalf(PROBLEM, z, ind), x)

    elseif PROBLEM == "AP1"
        if ind == 1
            H[1,1] = 3*(x[1]-1)^2
            H[2,2] = 6*(x[2]-2)^2
        elseif ind == 2
            t = 0.25*exp((x[1]+x[2])/2)
            H .= [t+2 t; t t+2]
        elseif ind == 3
            H[1,1] = (1/6)*exp(-x[1])
            H[2,2] = (1/3)*exp(-x[2])
        end

    elseif PROBLEM == "AP2"
        H[1,1] = 2

    elseif PROBLEM == "AP3"
        if ind == 1
            H[1,1] = 3*(x[1]-1)^2
            H[2,2] = 6*(x[2]-2)^2
        else
            H[1,1] = -4*(x[2]-x[1]^2) + 8*x[1]^2 + 2
            H[1,2] = -4*x[1]
            H[2,1] = H[1,2]
            H[2,2] = 2
        end

    elseif PROBLEM == "AP4"
        if ind == 1
            H[1,1] = 12/9*(x[1]-1)^2
            H[2,2] = 24/9*(x[2]-2)^2
            H[3,3] = 36/9*(x[3]-3)^2
        elseif ind == 2
            t = (1/9)*exp((x[1]+x[2]+x[3])/3)
            H .= t
            for i in 1:n
                H[i,i] += 2
            end
        elseif ind == 3
            H[1,1] = (1/4)*exp(-x[1])
            H[2,2] = (1/3)*exp(-x[2])
            H[3,3] = (1/4)*exp(-x[3])
        end

    elseif PROBLEM == "BK1"
        H .= 2.0 * Matrix{T}(I, n, n)
        
    elseif PROBLEM == "DD1"
        if ind == 1
            for i in 1:n
                H[i,i] = 2.0
            end
        else  
            H[4,4] = 6e-2 * (x[4] - x[5])
            H[4,5] = -6e-2 * (x[4] - x[5])
            H[5,4] = H[4,5]
            H[5,5] = 6e-2 * (x[4] - x[5])
        end

    # ----------------------------------------------------------------------
    # DGO1
    elseif PROBLEM == "DGO1"
        if ind == 1
            H[1,1] = -sin(x[1])
        elseif ind == 2
            H[1,1] = -sin(x[1] + 0.7)
        end

    elseif PROBLEM == "DGO1M"
        if ind == 1
            H[1,1] = -sin(x[1]) + 0.1 * (-10.0 * sin(5.0 * x[1]) * cos(x[1]) - 26.0 * cos(5.0 * x[1]) * sin(x[1]))
        elseif ind == 2
            y = x[1] + 0.7
            H[1,1] = -sin(y) + 0.1 * (-10.0 * sin(5.0 * x[1]) * cos(y) - 26.0 * cos(5.0 * x[1]) * sin(y))
        end

    # ----------------------------------------------------------------------
    # DGO2
    elseif PROBLEM == "DGO2"
        if ind == 1
            H[1,1] = 2.0
        elseif ind == 2
            t = sqrt(81.0 - x[1]^2)
            H[1,1] = (t + x[1]^2 / t) / (81.0 - x[1]^2)
        end

    # ----------------------------------------------------------------------
    # FA1
    elseif PROBLEM == "FA1"
        
        if ind == 1
            H[1,1] = -16.0 * exp(-4*x[1]) / (1 - exp(-4))
        elseif ind == 2
            a = exp(-4*x[1]); b = 1 - exp(-4)
            t = (1 - a) / (b * (x[2] + 1))
            H[1,1] = 8*a / b * t^(-0.5) + a / b * t^(-1.5) * (4*a / (b * (x[2] + 1)))
            H[1,2] = -a / b * t^(-0.5) / (x[2] + 1)
            H[2,1] = H[1,2]
            H[2,2] = 0.25 * t^(0.5) / (x[2] + 1)
        elseif ind == 3
            a = exp(-4*x[1]); b = 1 - exp(-4)
            t = (1 - a) / (b * (x[3] + 1))
            H[1,1] = 1.6 * t^(-0.9) * a / b + 0.36 * t^(-1.9) * a / b * (4*a / (b * (x[3] + 1)))
            H[1,3] = -0.36 * t^(-0.9) * a / b / (x[3] + 1)
            H[3,1] = H[1,3]
            H[3,3] = 0.09 * t^(0.1) / (x[3] + 1)
        end

    # ----------------------------------------------------------------------
    # Far1
    elseif PROBLEM == "Far1"
        if ind == 1
            H[1,1] = 60 * exp(15 * ( - (x[1] - 0.1)^2 - x[2]^2 )) +
                    60 * (x[1] - 0.1) * exp(15 * ( - (x[1] - 0.1)^2 - x[2]^2 )) * (-30 * (x[1] - 0.1)) +
                    40 * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] - 0.6)^2 )) +
                    40 * (x[1] - 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[1] - 0.6)) -
                    40 * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] - 0.6)^2 )) -
                    40 * (x[1] + 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[1] + 0.6)) -
                    40 * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] + 0.6)^2 )) -
                    40 * (x[1] - 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[1] - 0.6)) -
                    40 * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] + 0.6)^2 )) -
                    40 * (x[1] + 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[1] + 0.6))

            H[1,2] = 60 * (x[1] - 0.1) * exp(15 * ( - (x[1] - 0.1)^2 - x[2]^2 )) * (-30 * x[2]) +
                    40 * (x[1] - 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) -
                    40 * (x[1] + 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) -
                    40 * (x[1] - 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[2] + 0.6)) -
                    40 * (x[1] + 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[2] + 0.6))

            H[2,1] = H[1,2]

            H[2,2] = 60 * exp(15 * ( - (x[1] - 0.1)^2 - x[2]^2 )) +
                    60 * x[2] * exp(15 * ( - (x[1] - 0.1)^2 - x[2]^2 )) * (-30 * x[2]) +
                    40 * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] - 0.6)^2 )) +
                    40 * (x[2] - 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) -
                    40 * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] - 0.6)^2 )) -
                    40 * (x[2] - 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) -
                    40 * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] + 0.6)^2 )) -
                    40 * (x[2] + 0.6) * exp(20 * ( - (x[1] - 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[2] + 0.6)) -
                    40 * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] + 0.6)^2 )) -
                    40 * (x[2] + 0.6) * exp(20 * ( - (x[1] + 0.6)^2 - (x[2] + 0.6)^2 )) * (-40 * (x[2] + 0.6))

        elseif ind == 2
            H[1,1] = -80 * exp(20 * ( -x[1]^2 - x[2]^2 )) -
                    80 * x[1] * exp(20 * ( -x[1]^2 - x[2]^2 )) * (-40 * x[1]) -
                    40 * exp(20 * ( - (x[1] - 0.4)^2 - (x[2] - 0.6)^2 )) -
                    40 * (x[1] - 0.4) * exp(20 * ( - (x[1] - 0.4)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[1] - 0.4)) +
                    40 * exp(20 * ( - (x[1] + 0.5)^2 - (x[2] - 0.7)^2 )) +
                    40 * (x[1] + 0.5) * exp(20 * ( - (x[1] + 0.5)^2 - (x[2] - 0.7)^2 )) * (-40 * (x[1] + 0.5)) +
                    40 * exp(20 * ( - (x[1] - 0.5)^2 - (x[2] + 0.7)^2 )) +
                    40 * (x[1] - 0.5) * exp(20 * ( - (x[1] - 0.5)^2 - (x[2] + 0.7)^2 )) * (-40 * (x[1] - 0.5)) -
                    40 * exp(20 * ( - (x[1] + 0.4)^2 - (x[2] + 0.8)^2 )) -
                    40 * (x[1] + 0.4) * exp(20 * ( - (x[1] + 0.4)^2 - (x[2] + 0.8)^2 )) * (-40 * (x[1] + 0.4))

            H[1,2] = -80 * x[1] * exp(20 * ( -x[1]^2 - x[2]^2 )) * (-40 * x[2]) -
                    40 * (x[1] - 0.4) * exp(20 * ( - (x[1] - 0.4)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) +
                    40 * (x[1] + 0.5) * exp(20 * ( - (x[1] + 0.5)^2 - (x[2] - 0.7)^2 )) * (-40 * (x[2] - 0.7)) +
                    40 * (x[1] - 0.5) * exp(20 * ( - (x[1] - 0.5)^2 - (x[2] + 0.7)^2 )) * (-40 * (x[2] + 0.7)) -
                    40 * (x[1] + 0.4) * exp(20 * ( - (x[1] + 0.4)^2 - (x[2] + 0.8)^2 )) * (-40 * (x[2] + 0.8))

            H[2,1] = H[1,2]

            H[2,2] = -80 * exp(20 * ( -x[1]^2 - x[2]^2 )) -
                    80 * x[2] * exp(20 * ( -x[1]^2 - x[2]^2 )) * (-40 * x[2]) -
                    40 * exp(20 * ( - (x[1] - 0.4)^2 - (x[2] - 0.6)^2 )) -
                    40 * (x[2] - 0.6) * exp(20 * ( - (x[1] - 0.4)^2 - (x[2] - 0.6)^2 )) * (-40 * (x[2] - 0.6)) +
                    40 * exp(20 * ( - (x[1] + 0.5)^2 - (x[2] - 0.7)^2 )) +
                    40 * (x[2] - 0.7) * exp(20 * ( - (x[1] + 0.5)^2 - (x[2] - 0.7)^2 )) * (-40 * (x[2] - 0.7)) +
                    40 * exp(20 * ( - (x[1] - 0.5)^2 - (x[2] + 0.7)^2 )) +
                    40 * (x[2] + 0.7) * exp(20 * ( - (x[1] - 0.5)^2 - (x[2] + 0.7)^2 )) * (-40 * (x[2] + 0.7)) -
                    40 * exp(20 * ( - (x[1] + 0.4)^2 - (x[2] + 0.8)^2 )) -
                    40 * (x[2] + 0.8) * exp(20 * ( - (x[1] + 0.4)^2 - (x[2] + 0.8)^2 )) * (-40 * (x[2] + 0.8))
        end

    # ----------------------------------------------------------------------
    # FDS
    elseif PROBLEM == "FDS"
        
        if ind == 1
            for i in 1:n
                H[i,i] = 12 * i * (x[i] - i)^2 / n^2
            end
        elseif ind == 2
            t = exp(sum(x)/n) / n^2
            H .= t
            for i in 1:n
                H[i,i] += 2.0
            end
        elseif ind == 3
            for i in 1:n
                H[i,i] = i * (n - i + 1) * exp(-x[i]) / (n * (n + 1))
            end
        end

    # ----------------------------------------------------------------------
    # FF1
    elseif PROBLEM == "FF1"
        
        if ind == 1
            H[1,1] = 2 * exp(-((x[1]-1)^2 + (x[2]+1)^2)) + 2 * (x[1]-1) * exp(-((x[1]-1)^2 + (x[2]+1)^2)) * (-2 * (x[1]-1))
            H[1,2] = 2 * (x[1]-1) * exp(-((x[1]-1)^2 + (x[2]+1)^2)) * (-2 * (x[2]+1))
            H[2,1] = H[1,2]
            H[2,2] = 2 * exp(-((x[1]-1)^2 + (x[2]+1)^2)) + 2 * (x[2]+1) * exp(-((x[1]-1)^2 + (x[2]+1)^2)) * (-2 * (x[2]+1))
        elseif ind == 2
            H[1,1] = 2 * exp(-((x[1]+1)^2 + (x[2]-1)^2)) + 2 * (x[1]+1) * exp(-((x[1]+1)^2 + (x[2]-1)^2)) * (-2 * (x[1]+1))
            H[1,2] = 2 * (x[1]+1) * exp(-((x[1]+1)^2 + (x[2]-1)^2)) * (-2 * (x[2]-1))
            H[2,1] = H[1,2]
            H[2,2] = 2 * exp(-((x[1]+1)^2 + (x[2]-1)^2)) + 2 * (x[2]-1) * exp(-((x[1]+1)^2 + (x[2]-1)^2)) * (-2 * (x[2]-1))
        end

    # ----------------------------------------------------------------------
    # Hil1
    elseif PROBLEM == "Hil1"
        a = 2π / 360 * (45 + 40 * sin(2π * x[1]) + 25 * sin(2π * x[2]))
        b = 1 + 0.5 * cos(2π * x[1])

        if ind == 1
            A1 = 160 / 360 * π^2 * cos(2π * x[1])
            A2 = 100 / 360 * π^2 * cos(2π * x[2])
            A3 = -2π * sin(2π * x[1]) * sin(a) + cos(2π * x[1]) * cos(a) * A1

            H[1,1] = -160 * π^2 / 360 * b * A3 -
                    160 * π^2 / 360 * cos(2π * x[1]) * sin(a) * (-π * sin(2π * x[1])) -
                    2π^2 * cos(2π * x[1]) * cos(a) +
                    π * sin(2π * x[1]) * sin(a) * A1

            H[1,2] = -160 * π^2 / 360 * cos(2π * x[1]) * b * cos(a) * A2 +
                    π * sin(2π * x[1]) * sin(a) * A2

            H[2,1] = H[1,2]

            H[2,2] = 200 * π^3 / 360 * b * sin(2π * x[2]) * sin(a) -
                    100 * π^2 / 360 * b * cos(2π * x[2]) * cos(a) * A2

        elseif ind == 2
            A1 = 160 / 360 * π^2 * cos(2π * x[1])
            A2 = 100 / 360 * π^2 * cos(2π * x[2])
            A3 = -2π * sin(2π * x[1]) * cos(a) - cos(2π * x[1]) * sin(a) * A1

            H[1,1] = 160 * π^2 / 360 * b * A3 +
                    160 * π^2 / 360 * cos(2π * x[1]) * cos(a) * (-π * sin(2π * x[1])) -
                    2π^2 * cos(2π * x[1]) * sin(a) -
                    π * sin(2π * x[1]) * cos(a) * A1

            H[1,2] = -160 * π^2 / 360 * cos(2π * x[1]) * b * sin(a) * A2 -
                    π * sin(2π * x[1]) * cos(a) * A2

            H[2,1] = H[1,2]

            H[2,2] = -200 * π^3 / 360 * b * sin(2π * x[2]) * cos(a) -
                    100 * π^2 / 360 * b * cos(2π * x[2]) * sin(a) * A2
        end
        
    # ----------------------------------------------------------------------
    # IKK1
    elseif PROBLEM == "IKK1"
        
        if ind == 1
            H[1,1] = 2
        elseif ind == 2
            H[1,1] = 2
        elseif ind == 3
            H[2,2] = 2
        end

    # ----------------------------------------------------------------------
    # IM1
    elseif PROBLEM == "IM1"
        
        if ind == 1
            H[1,1] = -0.5 / sqrt(x[1]^3)
        elseif ind == 2
            H[1,2] = -1
            H[2,1] = -1
        end

    # ----------------------------------------------------------------------
    # JOS1
    elseif PROBLEM == "JOS1"
        
        for i in 1:n
            H[i,i] = 2.0 / n
        end

    # ----------------------------------------------------------------------
    # JOS4
    elseif PROBLEM == "JOS4"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / a
            c = -x[1] * 9.0 / (n - 1) / a^2

            H[1, 1] = (0.1875 * t^(-1.75) - 12.0 * t^2) / a

            for j in 2:n
                H[1, j] = (0.1875 * t^(-1.75) - 12.0 * t^2) * c
            end

            b = 9.0 / (n - 1) * (-0.1875 * t^(-0.75) + 12.0 * t^3) * c
            for i in 2:n
                for j in i:n
                    H[i, j] = b
                end
            end

            for i in 2:n
                for j in 1:i-1
                    H[i, j] = H[j, i]
                end
            end
        end

    # ----------------------------------------------------------------------
    # KW2
    elseif PROBLEM == "KW2"
        if ind == 1
            a = exp(-x[1]^2 - (x[2] + 1.0)^2)
            b = exp(-x[1]^2 - x[2]^2)
            c = exp(-(x[1] + 2.0)^2 - x[2]^2)

            H[1,1] = -6.0 * a - 12.0 * x[1] * (1.0 - x[1]) * a +
                      6.0 * (3.0 * x[1]^2 - 4.0 * x[1] + 1.0) * a -
                      12.0 * x[1]^2 * (1.0 - x[1])^2 * a +
                      10.0 * (-6.0 * x[1]) * b -
                      20.0 * x[1] * (1.0 / 5.0 - 3.0 * x[1]^2) * b -
                      20.0 * (2.0 / 5.0 * x[1] - 4.0 * x[1]^3 - x[2]^5) * b +
                      40.0 * x[1]^2 * (x[1] / 5.0 - x[1]^3 - x[2]^5) * b -
                      6.0 * c + 12.0 * (x[1] + 2.0)^2 * exp(-(x[1] + 2.0)^2 - x[2]^2)

            H[1,2] = 6.0 * (1.0 - x[1]) * a * (-2.0 * (x[2] + 1.0)) +
                      6.0 * x[1] * (1.0 - x[1])^2 * a * (-2.0 * (x[2] + 1.0)) +
                      10.0 * (1.0 / 5.0 - 3.0 * x[1]^2) * b * (-2.0 * x[2]) -
                      20.0 * x[1] * (-5.0 * x[2]^4) * b -
                      20.0 * x[1] * (x[1] / 5.0 - x[1]^3 - x[2]^5) * b * (-2.0 * x[2]) -
                      6.0 * (x[1] + 2.0) * c * (-2.0 * x[2])

            H[2,1] = H[1,2]

            H[2,2] = 6.0 * (1.0 - x[1])^2 * a +
                      6.0 * (1.0 - x[1])^2 * a * (x[2] + 1.0) * (-2.0 * (x[2] + 1.0)) -
                      50.0 * 4.0 * x[2]^3 * b -
                      50.0 * x[2]^4 * b * (-2.0 * x[2]) -
                      20.0 * (x[1] / 5.0 - x[1]^3 - 6.0 * x[2]^5) * b -
                      20.0 * x[2] * (x[1] / 5.0 - x[1]^3 - x[2]^5) * b * (-2.0 * x[2]) -
                      6.0 * c - 6.0 * c * x[2] * (-2.0 * x[2])

        elseif ind == 2
            a = exp(-x[2]^2 - (1.0 - x[1])^2)
            b = exp(-x[1]^2 - x[2]^2)
            c = exp(-(2.0 - x[2])^2 - x[1]^2)

            H[1,1] = 6.0 * (1.0 + x[2])^2 * a -
                      6.0 * (1.0 + x[2])^2 * a * (1.0 - x[1]) * (2.0 * (1.0 - x[1])) +
                      50.0 * 4.0 * x[1]^3 * b +
                      50.0 * x[1]^4 * b * (-2.0 * x[1]) -
                      20.0 * (-x[2] / 5.0 + x[2]^3 + 6.0 * x[1]^5) * b -
                      20.0 * x[1] * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * b * (-2.0 * x[1]) -
                      6.0 * c - 6.0 * c * x[1] * (-2.0 * x[1])

            H[1,2] = -12.0 * (1.0 + x[2]) * a * (1.0 - x[1]) -
                      6.0 * (1.0 + x[2])^2 * a * (1.0 - x[1]) * (-2.0 * x[2]) +
                      50.0 * x[1]^4 * b * (-2.0 * x[2]) -
                      20.0 * (-1.0 / 5.0 + 3.0 * x[2]^2) * b * x[1] -
                      20.0 * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * b * x[1] * (-2.0 * x[2]) -
                      6.0 * c * x[1] * (2.0 * (2.0 - x[2]))

            H[2,1] = H[1,2]

            H[2,2] = -6.0 * a -
                      6.0 * (1.0 + x[2]) * a * (-2.0 * x[2]) +
                      6.0 * (1.0 + 4.0 * x[2] + 3.0 * x[2]^2) * a +
                      6.0 * x[2] * (1.0 + x[2])^2 * a * (-2.0 * x[2]) +
                      10.0 * 6.0 * x[2] * b +
                      10.0 * (-1.0 / 5.0 + 3.0 * x[2]^2) * b * (-2.0 * x[2]) -
                      20.0 * (-2.0 / 5.0 * x[2] + 4.0 * x[2]^3 + x[1]^5) * b -
                      20.0 * x[2] * (-x[2] / 5.0 + x[2]^3 + x[1]^5) * b * (-2.0 * x[2]) -
                      6.0 * c + 6.0 * c * (2.0 - x[2]) * (2.0 * (2.0 - x[2]))
        end

    # ----------------------------------------------------------------------
    # LE1
    elseif PROBLEM == "LE1"
        if ind == 1
            t = 0.25 * (x[1]^2 + x[2]^2)^(-0.875)
            a = 0.5 * (-0.875) * (x[1]^2 + x[2]^2)^(-1.875)

            H[1,1] = t + a * x[1]^2
            H[1,2] = a * x[1] * x[2]
            H[2,1] = H[1,2]
            H[2,2] = t + a * x[2]^2

        elseif ind == 2
            t = 0.5 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2)^(-0.75)
            a = (-0.75) * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2)^(-1.75)

            H[1,1] = t + a * (x[1] - 0.5)^2
            H[1,2] = (x[1] - 0.5) * a * (x[2] - 0.5)
            H[2,1] = H[1,2]
            H[2,2] = t + a * (x[2] - 0.5)^2
        end

    # ----------------------------------------------------------------------
    # Lov1
    elseif PROBLEM == "Lov1"
        if ind == 1
            H[1,1] = 2.1
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 2.0 * 0.98
        elseif ind == 2
            H[1,1] = 2.0 * 0.99
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 2.0 * 1.03
        end

    # ----------------------------------------------------------------------
    # Lov2
    elseif PROBLEM == "Lov2"
        if ind == 1
            H[1,1] = 0.0
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 0.0

        elseif ind == 2
            H[1,1] = (6 * x[1]^2 + 6 * x[1]) * (x[1] + 1)^2 +
                    (-3 * x[1]^2 * (x[1] + 1) - (x[2] - x[1]^3)) * 2 * (x[1] + 1)
            H[1,1] /= (x[1] + 1)^4

            H[1,2] = 1 / (x[1] + 1)^2
            H[2,1] = H[1,2]
            H[2,2] = 0.0
        end

    # ----------------------------------------------------------------------
    # Lov3
    elseif PROBLEM == "Lov3"
        if ind == 1
            H[1,1] = 2.0
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 2.0
        elseif ind == 2
            H[1,1] = 2.0
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = -2.0
        end

    # ----------------------------------------------------------------------
    # Lov4
    elseif PROBLEM == "Lov4"
        if ind == 1
            H[1,1] = 2.0 -
                    8.0 * exp(-(x[1] + 2)^2 - x[2]^2) -
                    8.0 * (x[1] + 2) * exp(-(x[1] + 2)^2 - x[2]^2) * (-2 * (x[1] + 2)) -
                    8.0 * exp(-(x[1] - 2)^2 - x[2]^2) -
                    8.0 * (x[1] - 2) * exp(-(x[1] - 2)^2 - x[2]^2) * (-2 * (x[1] - 2))

            H[1,2] = -8.0 * (x[1] + 2) * exp(-(x[1] + 2)^2 - x[2]^2) * (-2 * x[2]) -
                    8.0 * (x[1] - 2) * exp(-(x[1] - 2)^2 - x[2]^2) * (-2 * x[2])

            H[2,1] = H[1,2]

            H[2,2] = 2.0 -
                    8.0 * exp(-(x[1] + 2)^2 - x[2]^2) -
                    8.0 * x[2] * exp(-(x[1] + 2)^2 - x[2]^2) * (-2 * x[2]) -
                    8.0 * exp(-(x[1] - 2)^2 - x[2]^2) -
                    8.0 * x[2] * exp(-(x[1] - 2)^2 - x[2]^2) * (-2 * x[2])

        elseif ind == 2
            H[1,1] = 2.0
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 2.0
        end

    # ----------------------------------------------------------------------
    # Lov5
    elseif PROBLEM == "Lov5"
        M = [
            -1.0   -0.03   0.011;
            -0.03  -1.0    0.07;
            0.011  0.07  -1.01
        ]

        p = [x[1], x[2] - 0.15, x[3]]
        a = 0.35
        A1 = sqrt(2π / a) * exp(dot(p, M * p) / a^2)

        p = [x[1], x[2] + 1.1, 0.5 * x[3]]
        b = 3.0
        A2 = sqrt(2π / b) * exp(dot(p, M * p) / b^2)

        H[1,1] = sqrt(2.0) * A1 * M[1,1] / a^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + 2*M[1,3]*x[3] + 2*M[1,2]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,1]*x[1] + 2*M[1,3]*x[3] + 2*M[1,2]*(x[2]-0.15)) / a^2 +
                sqrt(2.0) * A2 * M[1,1] / b^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + M[1,3]*x[3] + 2*M[1,2]*(x[2]+1.1)) / b^2 *
                A2 * (2*M[1,1]*x[1] + M[1,3]*x[3] + 2*M[1,2]*(x[2]+1.1)) / b^2

        H[1,2] = sqrt(2.0) * A1 * M[1,2] / a^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + 2*M[1,3]*x[3] + 2*M[1,2]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,2]*x[1] + 2*M[2,3]*x[3] + 2*M[2,2]*(x[2]-0.15)) / a^2 +
                sqrt(2.0) * A2 * M[1,2] / b^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + M[1,3]*x[3] + 2*M[1,2]*(x[2]+1.1)) / b^2 *
                A2 * (2*M[1,2]*x[1] + M[2,3]*x[3] + 2*M[2,2]*(x[2]+1.1)) / b^2

        H[1,3] = sqrt(2.0) * A1 * M[1,3] / a^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + 2*M[1,3]*x[3] + 2*M[1,2]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,3]*x[1] + 2*M[3,3]*x[3] + 2*M[2,3]*(x[2]-0.15)) / a^2 +
                sqrt(2.0)/2 * A2 * M[1,3] / b^2 +
                sqrt(2.0)/2 * (2*M[1,1]*x[1] + M[1,3]*x[3] + 2*M[1,2]*(x[2]+1.1)) / b^2 *
                A2 * (M[1,3]*x[1] + (M[3,3]*x[3])/2 + M[2,3]*(x[2]+1.1)) / b^2

        H[2,2] = sqrt(2.0) * A1 * M[2,3] / a^2 +
                sqrt(2.0)/2 * (2*M[1,3]*x[1] + 2*M[3,3]*x[3] + 2*M[2,3]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,2]*x[1] + 2*M[2,3]*x[3] + 2*M[2,2]*(x[2]-0.15)) / a^2 +
                sqrt(2.0) * A2 * M[2,2] / b^2 +
                sqrt(2.0)/2 * (2*M[1,2]*x[1] + M[2,3]*x[3] + 2*M[2,2]*(x[2]+1.1)) / b^2 *
                A2 * (2*M[1,2]*x[1] + M[2,3]*x[3] + 2*M[2,2]*(x[2]+1.1)) / b^2

        H[2,3] = sqrt(2.0) * A1 * M[2,3] / a^2 +
                sqrt(2.0)/2 * (2*M[1,2]*x[1] + 2*M[2,3]*x[3] + 2*M[2,2]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,3]*x[1] + 2*M[3,3]*x[3] + 2*M[2,3]*(x[2]-0.15)) / a^2 +
                sqrt(2.0)/2 * A2 * M[2,3] / b^2 +
                sqrt(2.0)/2 * (2*M[1,2]*x[1] + M[2,3]*x[3] + 2*M[2,2]*(x[2]+1.1)) / b^2 *
                A2 * (M[1,3]*x[1] + (M[3,3]*x[3])/2 + M[2,3]*(x[2]+1.1)) / b^2

        H[3,3] = sqrt(2.0) * A1 * M[3,3] / a^2 +
                sqrt(2.0)/2 * (2*M[1,3]*x[1] + 2*M[3,3]*x[3] + 2*M[2,3]*(x[2]-0.15)) / a^2 *
                A1 * (2*M[1,3]*x[1] + 2*M[3,3]*x[3] + 2*M[2,3]*(x[2]-0.15)) / a^2 +
                sqrt(2.0)/2 * A2 * M[3,3] / (2 * b^2) +
                sqrt(2.0)/2 * (M[1,3]*x[1] + M[3,3]*x[3]/2 + M[2,3]*(x[2]+1.1)) / b^2 *
                A2 * (M[1,3]*x[1] + (M[3,3]*x[3])/2 + M[2,3]*(x[2]+1.1)) / b^2

        H[2,1] = H[1,2]
        H[3,1] = H[1,3]
        H[3,2] = H[2,3]

        H .= -H

    # ----------------------------------------------------------------------
    # Lov6
    elseif PROBLEM == "Lov6"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            H .= 0.0
            H[1,1] = 0.25 / sqrt(x[1]^3) -
                    10π * cos(10π * x[1]) -
                    10π * cos(10π * x[1]) +
                    100π^2 * x[1] * sin(10π * x[1])
            for i in 2:6
                H[i,i] = 2.0
            end
        end

    # ----------------------------------------------------------------------
    # LTDZ  — Combining convergence and diversity in evolutionary multiobjective optimization
    elseif PROBLEM == "LTDZ"
        if ind == 1
            H[1,1] = (π/2) * (1 + x[3]) * cos(x[1]*π/2) * cos(x[2]*π/2) * (π/2)
            H[1,2] = (π/2) * (1 + x[3]) * sin(x[1]*π/2) * sin(x[2]*π/2) * (-π/2)
            H[1,3] = (π/2) * sin(x[1]*π/2) * cos(x[2]*π/2)

            H[2,1] = H[1,2]
            H[2,2] = (π/2) * (1 + x[3]) * cos(x[1]*π/2) * cos(x[2]*π/2) * (π/2)
            H[2,3] = (π/2) * cos(x[1]*π/2) * sin(x[2]*π/2)

            H[3,1] = H[1,3]
            H[3,2] = H[2,3]
            H[3,3] = 0.0

            H .= -H

        elseif ind == 2
            H[1,1] = (π/2) * (1 + x[3]) * cos(x[1]*π/2) * sin(x[2]*π/2) * (π/2)
            H[1,2] = (π/2) * (1 + x[3]) * sin(x[1]*π/2) * cos(x[2]*π/2) * (π/2)
            H[1,3] = (π/2) * sin(x[1]*π/2) * sin(x[2]*π/2)

            H[2,1] = H[1,2]
            H[2,2] = -(π/2) * (1 + x[3]) * cos(x[1]*π/2) * sin(x[2]*π/2) * (-π/2)
            H[2,3] = -(π/2) * cos(x[1]*π/2) * cos(x[2]*π/2)

            H[3,1] = H[1,3]
            H[3,2] = H[2,3]
            H[3,3] = 0.0

            H .= -H

        elseif ind == 3
            H[1,1] = π * (1 + x[3]) * sin(x[1]*π/2) * cos(x[1]*π/2) * (π/2) -
                    π * (1 + x[3]) * cos(x[1]*π/2) * sin(x[1]*π/2) * (-π/2)
            H[1,2] = 0.0
            H[1,3] = (π/2) * sin(x[1]*π/2)^2 - (π/2) * cos(x[1]*π/2)^2

            H[2,1] = H[1,2]
            H[2,2] = 0.0
            H[2,3] = 0.0

            H[3,1] = H[1,3]
            H[3,2] = H[2,3]
            H[3,3] = 0.0

            H .= -H
        end

        # ----------------------------------------------------------------------
    # MGH9
    elseif PROBLEM == "MGH9"
        t = (8.0 - ind) / 2.0
        a = exp(-x[2] * (t - x[3])^2 / 2.0)

        H[1,1] = 0.0
        H[1,2] = a * (-(t - x[3])^2 / 2.0)
        H[1,3] = a * (x[2] * (t - x[3]))

        H[2,1] = H[1,2]
        H[2,2] = -x[1] * a * (-(t - x[3])^2 / 2.0) * (t - x[3])^2 / 2.0
        H[2,3] = -x[1] * a * (x[2] * (t - x[3])) * (t - x[3])^2 / 2.0 +
                x[1] * a * (t - x[3])

        H[3,1] = H[1,3]
        H[3,2] = H[2,3]
        H[3,3] = x[1] * a * (x[2] * (t - x[3])) * x[2] * (t - x[3]) - x[1] * a * x[2]

    # ----------------------------------------------------------------------
    # MGH16
    elseif PROBLEM == "MGH16"
        t = ind / 5.0

        H[1,1] = 2.0
        H[1,2] = 2.0 * t
        H[2,1] = H[1,2]
        H[2,2] = 2.0 * t^2
        H[3,3] = 2.0
        H[3,4] = 2.0 * sin(t)
        H[4,3] = H[3,4]
        H[4,4] = 2.0 * sin(t)^2

    # ----------------------------------------------------------------------
    # MGH26
    elseif PROBLEM == "MGH26"
        t = sum(cos.(x))
        a = 2.0 * (length(x) - t + ind * (1.0 - cos(x[ind])) - sin(x[ind]))

        for i in 1:length(x)
            if i == ind
                for j in i:length(x)
                    if j == i
                        H[i,j] = a * (cos(x[ind]) + ind * cos(x[ind]) + sin(x[ind])) +
                                2.0 * (sin(x[ind]) + ind * sin(x[ind]) - cos(x[ind]))^2
                    else
                        H[i,j] = 2.0 * sin(x[j]) * (sin(x[ind]) + ind * sin(x[ind]) - cos(x[ind]))
                    end
                end
            else
                for j in i:length(x)
                    if j == i
                        H[i,j] = a * cos(x[j]) + 2.0 * sin(x[j])^2
                    elseif j == ind
                        H[i,j] = 2.0 * (sin(x[j]) + ind * sin(x[j]) - cos(x[j])) * sin(x[i])
                    else
                        H[i,j] = 2.0 * sin(x[j]) * sin(x[i])
                    end
                end
            end
        end

        for i in 2:length(x)
            for j in 1:(i-1)
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # MGH33
    elseif PROBLEM == "MGH33"
        for i in 1:length(x)
            t = 2.0 * i * ind^2
            for j in 1:length(x)
                H[i,j] = t * j
            end
        end

    # ----------------------------------------------------------------------
    # MHHM2
    elseif PROBLEM == "MHHM2"
        if ind in (1, 2, 3)
            H[1,1] = 2.0
            H[1,2] = 0.0
            H[2,1] = 0.0
            H[2,2] = 2.0
        end

    # ----------------------------------------------------------------------
    # MLF1
    elseif PROBLEM == "MLF1"
        if ind == 1
            H[1,1] = cos(x[1]) / 10 - (1 + x[1]/20) * sin(x[1])
        elseif ind == 2
            H[1,1] = -sin(x[1]) / 10 - (1 + x[1]/20) * cos(x[1])
        end

    # ----------------------------------------------------------------------
    # MLF2
    elseif PROBLEM == "MLF2"
        if ind == 1
            H[1,1] = (2 * (x[1]^2 + x[2] - 11) + 4 * x[1]^2 + 1) / 100
            H[1,2] = (2 * x[1] + 2 * x[2]) / 100
            H[2,1] = H[1,2]
            H[2,2] = (1 + 2 * (x[1] + x[2]^2 - 7) + 4 * x[2]^2) / 100

        elseif ind == 2
            H[1,1] = (8 * (4 * x[1]^2 + 2 * x[2] - 11) + 64 * x[1]^2 + 4) / 100
            H[1,2] = (16 * x[1] + 16 * x[2]) / 100
            H[2,1] = H[1,2]
            H[2,2] = (4 + 8 * (2 * x[1] + 4 * x[2]^2 - 7) + 64 * x[2]^2) / 100
        end

    # ----------------------------------------------------------------------
    # MMR1  – Box-constrained multi-objective optimization (gradient-like method)
    elseif PROBLEM == "MMR1"
        if ind == 1
            H .= 0.0
            H[1,1] = 2.0

        elseif ind == 2
            a = exp(-((x[2] - 0.6) / 0.4)^2)
            b = exp(-((x[2] - 0.2) / 0.04)^2)

            H[1,1] = (-2 * (1 + x[1]^2) + 8 * x[1]^2) / (1 + x[1]^2)^3
            H[1,1] *= (2.0 - 0.8 * a - b)

            H[1,2] = 10 * a * (x[2] - 0.6) + 1250 * b * (x[2] - 0.2)
            H[1,2] *= (-2 * x[1] / (1 + x[1]^2)^2)
            H[2,1] = H[1,2]

            H[2,2] = 10 * a + 10 * (x[2] - 0.6)^2 * a * (-12.5) +
                    1250 * b + 1250 * (x[2] - 0.2)^2 * b * (-1250)
            H[2,2] /= (1 + x[1]^2)
        end

    # ----------------------------------------------------------------------
    # MMR3
    elseif PROBLEM == "MMR3"
        if ind == 1
            H .= 0.0
            H[1,1] = 6 * x[1]

        elseif ind == 2
            H[1,1] = 6 * (x[2] - x[1])
            H[1,2] = -6 * (x[2] - x[1])
            H[2,1] = H[1,2]
            H[2,2] = 6 * (x[2] - x[1])
        end

    # ----------------------------------------------------------------------
    # MMR4
    elseif PROBLEM == "MMR4"
        if ind == 1
            t = (2 * x[1] + x[2] + 2 * x[3] + 1)^3

            H[1,1] = -288 / t
            H[1,2] = -144 / t
            H[1,3] = -288 / t

            H[2,1] = H[1,2]
            H[2,2] = -72 / t
            H[2,3] = -144 / t

            H[3,1] = H[1,3]
            H[3,2] = H[2,3]
            H[3,3] = -288 / t

        elseif ind == 2
            H .= 0.0
        end

    # ----------------------------------------------------------------------
    # MOP2
    elseif PROBLEM == "MOP2"
        b = sqrt(n)

        if ind == 1
            a = sum((x .- 1 / b).^2)
            t = exp(-a)
            for i in 1:n, j in 1:n
                H[i,j] = 2 * (x[i] - 1/b) * t * (-2 * (x[j] - 1/b))
            end
            for i in 1:n
                H[i,i] += 2 * t
            end

        elseif ind == 2
            a = sum((x .+ 1 / b).^2)
            t = exp(-a)
            for i in 1:n, j in 1:n
                H[i,j] = 2 * (x[i] + 1/b) * t * (-2 * (x[j] + 1/b))
            end
            for i in 1:n
                H[i,i] += 2 * t
            end
        end

    # ----------------------------------------------------------------------
    # MOP3
    elseif PROBLEM == "MOP3"
        if ind == 1
            A1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
            A2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
            B1 = 0.5 * sin(x[1]) - 2 * cos(x[1]) + sin(x[2]) - 1.5 * cos(x[2])
            B2 = 1.5 * sin(x[1]) - cos(x[1]) + 2 * sin(x[2]) - 0.5 * cos(x[2])

            H[1,1] = 2 * (A1 - B1) * (0.5 * sin(x[1]) - 2 * cos(x[1])) +
                    2 * (-0.5 * cos(x[1]) - 2 * sin(x[1]))^2 +
                    2 * (A2 - B2) * (1.5 * sin(x[1]) - cos(x[1])) +
                    2 * (-1.5 * cos(x[1]) - sin(x[1]))^2

            H[1,2] = 2 * (-cos(x[2]) - 1.5 * sin(x[2])) * (-0.5 * cos(x[1]) - 2 * sin(x[1])) +
                    2 * (-2 * cos(x[2]) - 0.5 * sin(x[2])) * (-1.5 * cos(x[1]) - sin(x[1]))

            H[2,1] = H[1,2]

            H[2,2] = 2 * (A1 - B1) * (sin(x[2]) - 1.5 * cos(x[2])) +
                    2 * (-cos(x[2]) - 1.5 * sin(x[2]))^2 +
                    2 * (A2 - B2) * (2 * sin(x[2]) - 0.5 * cos(x[2])) +
                    2 * (-2 * cos(x[2]) - 0.5 * sin(x[2]))^2

        elseif ind == 2
            H .= 0.0
            H[1,1] = 2.0
            H[2,2] = 2.0
        end

    # ----------------------------------------------------------------------
    # MOP5
    elseif PROBLEM == "MOP5"
        if ind == 1
            a = cos(x[1]^2 + x[2]^2)
            b = sin(x[1]^2 + x[2]^2)

            H[1,1] = 1 + 2 * a - 4 * x[1]^2 * b
            H[1,2] = -4 * x[1] * x[2] * b
            H[2,1] = H[1,2]
            H[2,2] = 1 + 2 * a - 4 * x[2]^2 * b

        elseif ind == 2
            H[1,1] = 9/4 + 2/27
            H[1,2] = -6/4 - 2/27
            H[2,1] = H[1,2]
            H[2,2] = 1 + 2/27

        elseif ind == 3
            t = x[1]^2 + x[2]^2 + 1
            a = exp(-x[1]^2 - x[2]^2)
            H[1,1] = (-2 * t + 8 * x[1]^2) / t^3 + 2.2 * a - 4.4 * x[1]^2 * a
            H[1,2] = 8 * x[1] * x[2] / t^3 - 4.4 * x[1] * x[2] * a
            H[2,1] = H[1,2]
            H[2,2] = (-2 * t + 8 * x[2]^2) / t^3 + 2.2 * a - 4.4 * x[2]^2 * a
        end

    # ----------------------------------------------------------------------
    # MOP6
    elseif PROBLEM == "MOP6"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = 1 + 10 * x[2]
            b = sin(8π * x[1])
            c = cos(8π * x[1])
            t = x[1] / a

            H[1,1] = -2 / a - 16π * c + 64π^2 * x[1] * b
            H[1,2] = 20 * x[1] / a^2
            H[2,1] = H[1,2]
            H[2,2] = -20 * t * (-10 * x[1] / a^2) +
                    100 * x[1] * b / a^2 -
                    100 * x[1] / a^2 * (2 * t + b) +
                    10 * x[1] / a * (-20 * x[1] / a^2)
        end

    # ----------------------------------------------------------------------
    # MOP7
    elseif PROBLEM == "MOP7"
        if ind == 1
            H .= 0.0
            H[1,1] = 1.0
            H[2,2] = 2 / 13

        elseif ind == 2
            H[1,1] = 1/18 + 1/4
            H[1,2] = 1/18 - 1/4
            H[2,1] = H[1,2]
            H[2,2] = 1/18 + 1/4

        elseif ind == 3
            H[1,1] = 2/175 + 2/17
            H[1,2] = 4/175 - 4/17
            H[2,1] = H[1,2]
            H[2,2] = 8/175 + 8/17
        end

    # ----------------------------------------------------------------------
    # PNR
    elseif PROBLEM == "PNR"
        if ind == 1
            H[1,1] = 12 * x[1]^2 - 2
            H[1,2] = -10
            H[2,1] = H[1,2]
            H[2,2] = 12 * x[2]^2 + 2
        elseif ind == 2
            H[1,1] = 2
            H[1,2] = 0
            H[2,1] = 0
            H[2,2] = 2
        end

    # ----------------------------------------------------------------------
    # QV1
    elseif PROBLEM == "QV1"
        if ind == 1
            t = sum(x.^2 .- 10 .* cos.(2π .* x) .+ 10) / n
            for i in 1:n
                a = (2 * x[i] + 20π * sin(2π * x[i])) / n
                for j in 1:n
                    H[i,j] = -0.1875 * t^(-1.75) * (2 * x[j] + 20π * sin(2π * x[j])) / n * a
                end
            end
            for i in 1:n
                H[i,i] += 0.25 * t^(-0.75) * (2 + 40π^2 * cos(2π * x[i])) / n
            end

        elseif ind == 2
            t = sum((x .- 1.5).^2 .- 10 .* cos.(2π .* (x .- 1.5)) .+ 10) / n
            for i in 1:n
                a = (2 * (x[i] - 1.5) + 20π * sin(2π * (x[i] - 1.5))) / n
                for j in 1:n
                    H[i,j] = -0.1875 * t^(-1.75) *
                            (2 * (x[j] - 1.5) + 20π * sin(2π * (x[j] - 1.5))) / n * a
                end
            end
            for i in 1:n
                H[i,i] += 0.25 * t^(-0.75) *
                        (2 + 40π^2 * cos(2π * (x[i] - 1.5))) / n
            end
        end

    # ----------------------------------------------------------------------
    # SD
    elseif PROBLEM == "SD"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            H .= 0.0
            H[1,1] = 4 / x[1]^3
            H[2,2] = 4 * sqrt(2) / x[2]^3
            H[3,3] = 4 * sqrt(2) / x[3]^3
            H[4,4] = 4 / x[4]^3
        end

    # ----------------------------------------------------------------------
    # SLCDT1
    elseif PROBLEM == "SLCDT1"
        a = 1 + (x[1] + x[2])^2
        b = 1 + (x[1] - x[2])^2
        c = exp(-(x[1] + x[2])^2)

        H[1,1] = 0.5 * (sqrt(a) - (x[1] + x[2])^2 / sqrt(a)) / a +
                0.5 * (sqrt(b) - (x[1] - x[2])^2 / sqrt(b)) / b -
                2 * 0.85 * c + 4 * 0.85 * (x[1] + x[2])^2 * c

        H[1,2] = 0.5 * (sqrt(a) - (x[1] + x[2])^2 / sqrt(a)) / a +
                0.5 * (-sqrt(b) + (x[1] - x[2])^2 / sqrt(b)) / b -
                2 * 0.85 * c + 4 * 0.85 * (x[1] + x[2])^2 * c

        H[2,1] = H[1,2]

        H[2,2] = 0.5 * (sqrt(a) - (x[1] + x[2])^2 / sqrt(a)) / a -
                0.5 * (-sqrt(b) + (x[1] - x[2])^2 / sqrt(b)) / b -
                2 * 0.85 * c + 4 * 0.85 * (x[1] + x[2])^2 * c

    # ----------------------------------------------------------------------
    # SLCDT2
    elseif PROBLEM == "SLCDT2"
        if ind == 1
            H .= 0.0
            H[1,1] = 12 * (x[1] - 1)^2
            for i in 2:n
                H[i,i] = 2.0
            end

        elseif ind == 2
            H .= 0.0
            H[2,2] = 12 * (x[2] + 1)^2
            for i in 1:n
                if i != 2
                    H[i,i] = 2.0
                end
            end

        elseif ind == 3
            H .= 0.0
            H[3,3] = 12 * (x[3] - 1)^2
            for i in 1:n
                if i != 3
                    H[i,i] = 2.0
                end
            end
        end

    # ----------------------------------------------------------------------
    # SP1
    elseif PROBLEM == "SP1"
        if ind == 1
            H[1,1] = 4
            H[1,2] = -2
            H[2,1] = -2
            H[2,2] = 2
        elseif ind == 2
            H[1,1] = 2
            H[1,2] = -2
            H[2,1] = -2
            H[2,2] = 4
        end

    # ----------------------------------------------------------------------
    # SSFYY2
    elseif PROBLEM == "SSFYY2"
        if ind == 1
            H[1,1] = 2 + 2.5 * π^2 * cos(x[1] * π / 2)
        elseif ind == 2
            H[1,1] = 2.0
        end

    # ----------------------------------------------------------------------
    # SK1
    elseif PROBLEM == "SK1"
        if ind == 1
            H[1,1] = 12 * x[1]^2 + 18 * x[1] - 20
        elseif ind == 2
            H[1,1] = 6 * x[1]^2 - 12 * x[1] - 20
        end

    # ----------------------------------------------------------------------
    # SK2
    elseif PROBLEM == "SK2"
        if ind == 1
            H .= 0.0
            for i in 1:4
                H[i,i] = 2.0
            end

        elseif ind == 2
            a = 1 + (sum(x.^2)) / 100
            b = sum(sin.(x))
            for i in 1:n
                for j in 1:n
                    H[i,j] = (-cos(x[i]) * x[j] / 50 + cos(x[j]) * x[i] / 50) * a^2 -
                            (-cos(x[i]) * a + b * x[i] / 50) * 2 * a * x[j] / 50
                    H[i,j] /= a^4
                    if i == j
                        H[i,j] += (sin(x[i]) * a + b / 50) / a^2
                    end
                end
            end
        end

    # ----------------------------------------------------------------------
    # TKLY1
    elseif PROBLEM == "TKLY1"
        if ind == 1
            H .= 0.0

        elseif ind == 2
            A1 = 2 - exp(-((x[2] - 0.1) / 4e-3)^2) -
                    0.8 * exp(-((x[2] - 0.9) / 4e-1)^2)
            A2 = 2 - exp(-((x[3] - 0.1) / 4e-3)^2) -
                    0.8 * exp(-((x[3] - 0.9) / 4e-1)^2)
            A3 = 2 - exp(-((x[4] - 0.1) / 4e-3)^2) -
                    0.8 * exp(-((x[4] - 0.9) / 4e-1)^2)

            H[1,1] = 2 * A1 * A2 * A3 / x[1]^3
            H[1,2] = -A2 * A3 / x[1]^2 *
                    (5e2 * exp(-((x[2] - 0.1)/4e-3)^2) * ((x[2] - 0.1)/4e-3) +
                    4 * exp(-((x[2] - 0.9)/4e-1)^2) * ((x[2] - 0.9)/4e-1))
            H[1,3] = -A1 * A3 / x[1]^2 *
                    (5e2 * exp(-((x[3] - 0.1)/4e-3)^2) * ((x[3] - 0.1)/4e-3) +
                    4 * exp(-((x[3] - 0.9)/4e-1)^2) * ((x[3] - 0.9)/4e-1))
            H[1,4] = -A1 * A2 / x[1]^2 *
                    (5e2 * exp(-((x[4] - 0.1)/4e-3)^2) * ((x[4] - 0.1)/4e-3) +
                    4 * exp(-((x[4] - 0.9)/4e-1)^2) * ((x[4] - 0.9)/4e-1))

            H[2,1] = H[1,2]
            H[2,2] = A2 * A3 / x[1] *
                    (5e2 * exp(-((x[2] - 0.1)/4e-3)^2) / 4e-3 -
                    (5e2)^2 * exp(-((x[2] - 0.1)/4e-3)^2) * ((x[2] - 0.1)/4e-3)^2 +
                    4 * exp(-((x[2] - 0.9)/4e-1)^2) / 4e-1 -
                    20 * exp(-((x[2] - 0.9)/4e-1)^2) * ((x[2] - 0.9)/4e-1)^2)
            H[2,3] = A3 / x[1] *
                    (5e2 * exp(-((x[2] - 0.1)/4e-3)^2) * ((x[2] - 0.1)/4e-3) +
                    4 * exp(-((x[2] - 0.9)/4e-1)^2) * ((x[2] - 0.9)/4e-1)) *
                    (5e2 * exp(-((x[3] - 0.1)/4e-3)^2) * ((x[3] - 0.1)/4e-3) +
                    4 * exp(-((x[3] - 0.9)/4e-1)^2) * ((x[3] - 0.9)/4e-1))
            H[2,4] = A2 / x[1] *
                    (5e2 * exp(-((x[2] - 0.1)/4e-3)^2) * ((x[2] - 0.1)/4e-3) +
                    4 * exp(-((x[2] - 0.9)/4e-1)^2) * ((x[2] - 0.9)/4e-1)) *
                    (5e2 * exp(-((x[4] - 0.1)/4e-3)^2) * ((x[4] - 0.1)/4e-3) +
                    4 * exp(-((x[4] - 0.9)/4e-1)^2) * ((x[4] - 0.9)/4e-1))

            H[3,1] = H[1,3]
            H[3,2] = H[2,3]
            H[3,3] = A1 * A3 / x[1] *
                    (5e2 * exp(-((x[3] - 0.1)/4e-3)^2) / 4e-3 -
                    (5e2)^2 * exp(-((x[3] - 0.1)/4e-3)^2) * ((x[3] - 0.1)/4e-3)^2 +
                    4 * exp(-((x[3] - 0.9)/4e-1)^2) / 4e-1 -
                    20 * exp(-((x[3] - 0.9)/4e-1)^2) * ((x[3] - 0.9)/4e-1)^2)
            H[3,4] = A1 / x[1] *
                    (5e2 * exp(-((x[3] - 0.1)/4e-3)^2) * ((x[3] - 0.1)/4e-3) +
                    4 * exp(-((x[3] - 0.9)/4e-1)^2) * ((x[3] - 0.9)/4e-1)) *
                    (5e2 * exp(-((x[4] - 0.1)/4e-3)^2) * ((x[4] - 0.1)/4e-3) +
                    4 * exp(-((x[4] - 0.9)/4e-1)^2) * ((x[4] - 0.9)/4e-1))

            H[4,1] = H[1,4]
            H[4,2] = H[2,4]
            H[4,3] = H[3,4]
            H[4,4] = A1 * A2 / x[1] *
                    (5e2 * exp(-((x[4] - 0.1)/4e-3)^2) / 4e-3 -
                    (5e2)^2 * exp(-((x[4] - 0.1)/4e-3)^2) * ((x[4] - 0.1)/4e-3)^2 +
                    4 * exp(-((x[4] - 0.9)/4e-1)^2) / 4e-1 -
                    20 * exp(-((x[4] - 0.9)/4e-1)^2) * ((x[4] - 0.9)/4e-1)^2)
        end

    # ----------------------------------------------------------------------
    # Toi4
    elseif PROBLEM == "Toi4"
        if ind == 1
            H .= 0.0
            for i in 1:2
                H[i,i] = 2.0
            end
        elseif ind == 2
            H .= 0.0
            H[1,1] = 1.0; H[1,2] = -1.0
            H[2,1] = -1.0; H[2,2] = 1.0
            H[3,3] = 1.0;  H[3,4] = -1.0
            H[4,3] = -1.0; H[4,4] = 1.0
        end

    # ----------------------------------------------------------------------
    # Toi8
    elseif PROBLEM == "Toi8"
        if ind == 1
            H .= 0.0
            H[1,1] = 8.0
        else
            H .= 0.0
            H[ind-1,ind-1] = 8.0 * ind
            H[ind-1,ind]   = -4.0 * ind
            H[ind,ind-1]   = H[ind-1,ind]
            H[ind,ind]     = 2.0 * ind
        end

    # ----------------------------------------------------------------------
    # Toi9
    elseif PROBLEM == "Toi9"
        if ind == 1
            H .= 0.0
            H[1,1] = 8.0
            H[2,2] = 2.0
        elseif 1 < ind < n
            H .= 0.0
            H[ind-1,ind-1] = 6.0 * ind + 2.0
            H[ind-1,ind]   = -4.0 * ind
            H[ind,ind-1]   = -4.0 * ind
            H[ind,ind]     = 4.0 * ind
        elseif ind == n
            H .= 0.0
            H[n-1,n-1] = 6.0 * n + 2.0
            H[n-1,n]   = -4.0 * n
            H[n,n-1]   = -4.0 * n
            H[n,n]     = 2.0 * n
        end

    # ----------------------------------------------------------------------
    # Toi10 (Rosenbrock)
    elseif PROBLEM == "Toi10"
        H .= 0.0
        H[ind,ind]     = 800 * x[ind]^2 - 400 * (x[ind+1] - x[ind]^2)
        H[ind,ind+1]   = -400 * x[ind]
        H[ind+1,ind]   = H[ind,ind+1]
        H[ind+1,ind+1] = 202.0

    # ----------------------------------------------------------------------
    # VU1
    elseif PROBLEM == "VU1"
        if ind == 1
            a = x[1]^2 + x[2]^2 + 1
            H[1,1] = (-2 * a + 8 * x[1]^2) / a^3
            H[1,2] = 8 * x[1] * x[2] / a^3
            H[2,1] = H[1,2]
            H[2,2] = (-2 * a + 8 * x[2]^2) / a^3
        elseif ind == 2
            H[1,1] = 2
            H[1,2] = 0
            H[2,1] = 0
            H[2,2] = 6
        end

    # ----------------------------------------------------------------------
    # VU2
    elseif PROBLEM == "VU2"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            H .= 0.0
            H[1,1] = 2.0
        end

    # ----------------------------------------------------------------------
    # ZDT1
    elseif PROBLEM == "ZDT1"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / a

            H[1,1] = 0.25 * t^(-1.5) / a
            for j in 2:n
                H[1,j] = 0.25 * t^(-1.5) * (-x[1] * 9.0 / (n - 1) / a^2)
            end

            b = -2.25 / (n - 1) * t^(-0.5) * (-x[1] * 9.0 / (n - 1) / a^2)
            for i in 2:n, j in i:n
                H[i,j] = b
            end
            for i in 2:n, j in 1:i-1
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # ZDT2
    elseif PROBLEM == "ZDT2"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / a

            H[1,1] = -2.0 / a
            for j in 2:n
                H[1,j] = -2.0 * (-x[1] * 9.0 / (n - 1) / a^2)
            end

            b = 18.0 / (n - 1) * t * (-x[1] * 9.0 / (n - 1) / a^2)
            for i in 2:n, j in i:n
                H[i,j] = b
            end
            for i in 2:n, j in 1:i-1
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # ZDT3
    elseif PROBLEM == "ZDT3"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = 1.0 + 9.0 * sum(x[2:n]) / (n - 1)
            t = x[1] / a

            H[1,1] = 0.25 * t^(-1.5) / a - 20.0 * π * cos(10π * x[1]) + 100.0 * π^2 * x[1] * sin(10π * x[1])
            for j in 2:n
                H[1,j] = 0.25 * t^(-1.5) * (-x[1] * 9.0 / (n - 1) / a^2)
            end

            b = -2.25 / (n - 1) * t^(-0.5) * (-x[1] * 9.0 / (n - 1) / a^2)
            for i in 2:n, j in i:n
                H[i,j] = b
            end
            for i in 2:n, j in 1:i-1
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # ZDT4
    elseif PROBLEM == "ZDT4"
        if ind == 1
            H .= 0.0
        elseif ind == 2
            a = sum(x[2:n].^2 .- 10.0 .* cos.(4π .* x[2:n])) + 1.0 + 10.0 * (n - 1)
            t = x[1] / a

            H[1,1] = 0.25 * t^(-1.5) / a
            for j in 2:n
                H[1,j] = 0.25 * t^(-1.5) * (-x[1] * (2.0 * x[j] + 40.0 * π * sin(4π * x[j])) / a^2)
            end

            for i in 2:n
                b = sin(4π * x[i])
                for j in i:n
                    H[i,j] = (2.0 * x[i] + 40.0 * π * b) *
                            (-0.25 * t^(-0.5) * (-x[1] * (2.0 * x[j] + 40.0 * π * sin(4π * x[j])) / a^2))
                    if j == i
                        H[i,j] += (2.0 + 160.0 * π^2 * cos(4π * x[i])) * (1.0 - sqrt(t) / 2.0)
                    end
                end
            end
            for i in 2:n, j in 1:i-1
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # ZDT6
    elseif PROBLEM == "ZDT6"
        if ind == 1
            a = exp(-4.0 * x[1])
            b = sin(6.0 * π * x[1])
            c = cos(6.0 * π * x[1])
            H .= 0.0
            H[1,1] = -4.0 * a * (4.0 * b^6 - 36.0 * π * b^5 * c) +
                    a * (144.0 * π * b^5 * c - 1080.0 * π^2 * b^4 * c^2 + 216.0 * π^2 * b^6)
        elseif ind == 2
            a = exp(-4.0 * x[1])
            b = sin(6.0 * π * x[1])
            c = cos(6.0 * π * x[1])
            gaux1 = 1.0 - a * b^6
            gaux2 = 1.0 + 9.0 * (sum(x[2:n]) / (n - 1))^0.25
            t = gaux1 / gaux2

            A1 = 2.25 / (n - 1) * (sum(x[2:n]) / (n - 1))^(-0.75)
            A2 = -2.25 * 0.75 / (n - 1)^2 * (sum(x[2:n]) / (n - 1))^(-1.75)

            H[1,1] = (8.0 * t * a - 2.0 * (4.0 * a * b^6 - 36.0 * π * a * b^5 * c) / gaux2 * a) *
                    (4.0 * b^6 - 36.0 * π * b^5 * c) -
                    2.0 * t * a * (144.0 * π * b^5 * c - 1080.0 * π^2 * b^4 * c^2 + 216.0 * π^2 * b^6)

            for j in 2:n
                H[1,j] = -2.0 * a * (4.0 * b^6 - 36.0 * π * b^5 * c) * (-gaux1 * A1 / gaux2^2)
            end

            for i in 2:n, j in i:n
                H[i,j] = A2 * (1.0 + t^2) + A1 * 2.0 * t * (-gaux1 * A1 / gaux2^2)
            end
            for i in 2:n, j in 1:i-1
                H[i,j] = H[j,i]
            end
        end

    # ----------------------------------------------------------------------
    # ZLT1
    elseif PROBLEM == "ZLT1"
        H .= 0.0
        for i in 1:n
            H[i,i] = 2.0
        end
    else
        error("Unknown PROBLEM: $PROBLEM")
    end

    return H
end

"""
    reported_nondominated_archive(problem, solver_points, n, m)

Convert solver coordinates to the variables and objective values of the
reported problem, then remove nonfinite, duplicate, and dominated points.
Internal smoothing, normalization, and box penalties are used only by the
optimization algorithms and are not included in this returned archive.
For RE41/RE42/RE61, comparisons use exact objectives normalized by the fixed
ideal/nadir vectors, as in campaign postprocessing; returned values are raw.
This conversion does not certify Pareto criticality. Apply the PC check in
solver coordinates before calling it when a certified archive is required.
"""
function reported_nondominated_archive(problem::AbstractString, solver_points,
                                       n::Int, m::Int;
                                       atol::Float64=1.0e-10,
                                       feasibility_tol::Float64=5.0e-4)
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative."))
    isfinite(feasibility_tol) && feasibility_tol >= 0 || throw(ArgumentError("Invalid feasibility tolerance."))
    candidates_x = Vector{Vector{Float64}}()
    candidates_f = Vector{Vector{Float64}}()
    comparison_values = Vector{Vector{Float64}}()
    applied_re = problem in ("RE41", "RE42", "RE61")

    for solver_point in solver_points
        solver_x = Float64.(solver_point)
        length(solver_x) == n || throw(DimensionMismatch("Unexpected decision dimension."))
        all(isfinite, solver_x) || continue

        if applied_re
            x = Float64.(re_physical_point(problem, solver_x))
            f = Float64.(re_exact_objectives(problem, x))
        elseif problem in ("RE21", "RE34", "RE37")
            x = Float64.(box_map(solver_x, CURRENT_BOX_L[], CURRENT_BOX_U[]))
            f = [Float64(evalf(n, solver_x, i)) for i in 1:m]
        elseif is_scalable_benchmark(problem)
            scalable_box_violation(problem, solver_x) <= feasibility_tol || continue
            x = copy(solver_x)
            f = Float64.(scalable_raw_objectives(problem, solver_x))
        else
            x = copy(solver_x)
            f = [Float64(evalf(n, solver_x, i)) for i in 1:m]
        end

        all(isfinite, x) && all(isfinite, f) || continue
        length(f) == m || throw(DimensionMismatch("Unexpected objective dimension."))
        comparison_f = applied_re ? Float64.(re_normalize_objectives(problem, f)) : f
        push!(candidates_x, x)
        push!(candidates_f, f)
        push!(comparison_values, comparison_f)
    end
    keep = sequential_nondominated_indices(comparison_values; atol=atol)
    archive_x = candidates_x[keep]
    # The raw objectives were evaluated above; comparisons use normalized RE values.
    archive_f = candidates_f[keep]
    return archive_x, archive_f
end
