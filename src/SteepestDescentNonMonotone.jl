"""
    SteepestDescentNonMonotone!(n, m, x; M=4, eta=1e-4, rho=0.5,
        initial_step=1.0, epsopt=1e-6, maxiter=MAXOUTITER, scaleF=false,
        iprint=true, min_step=STPMIN, max_backtracks=100, store_trace=false)

Multiobjective steepest descent with the max-type nonmonotone line search of
Mita, Fukuda and Yamashita, *Nonmonotone line searches for unconstrained
multiobjective optimization problems*, J. Global Optim. 75, 63-90 (2019).
DOI: https://doi.org/10.1007/s10898-019-00802-0

This is an independent Julia implementation of Algorithm 3, with the steepest
descent direction of Section 2.2, not a copy of the authors' MATLAB code.
`evalSD!` solves the same unconstrained direction subproblem as our existing SD.
For accepted iterates x^0,...,x^k, equations (9)-(10) read

    C_i^k = max { f_i(x^j) : max(0, k-M) <= j <= k },
    f_i(x^k + t*d^k) <= C_i^k + eta*t*grad(f_i)(x^k)'*d^k, for every i.

The search takes the first acceptable t = initial_step * rho^h, h=0,1,... .
Each component uses ITS OWN directional derivative. Only accepted iterates enter
the memory, which is local to this call; rejected trials never enter it.

# Parameters and differences from the experimental setup of the reference
- `M=4`, `eta=1e-4`, `rho=0.5`, `initial_step=1` and `epsopt=1e-6` follow
  Section 7. Here `eta` is the paper's delta, and `initial_step` is its mu.
- `M` counts PREVIOUS iterates: the full window has M+1 vectors including the
  current iterate. `M=0` gives the componentwise monotone Armijo control. It is
  not bitwise equivalent to our existing `armijo!`, which uses a common slope
  and safeguarded interpolation.
- `scaleF=false` disables extra start-dependent scaling. Objective values and
  derivatives still include any transformations defined by the problem itself
  (in particular the RE normalization, smooth violation and variable mapping).
  `scaleF=true` uses the same fixed gradient-based factors as `scalefactor`.
- No box constraints are added to the direction subproblem. The reference's
  Section 7 uses boxes in its experiments; Algorithm 3 itself is unconstrained.
- `maxiter`, `min_step`, `max_backtracks` and stagnation/finite-value checks are
  numerical safeguards. Exhausting them is NOT convergence. No time limit is set.
- The stopping test is abs(theta) < epsopt, with
  theta = max_i grad(f_i)'*d + 0.5*dot(d,d), computed using `evalSD!`'s numerical
  direction. This is not the collected-archive Pareto-criticality filter.

# Output
Returns `(xsol, stats)` and modifies `x` only after an accepted step. As in SD,
one terminal point is returned per execution; there is no collection/filtering.
The usual stats fields `outiter`, `time`, `nfev`, `ngev`, `theta`, `inform` are
provided, plus `reason`, `backtracks`, `last_step` and `trace`. `outiter` counts
accepted steps, time is elapsed wall time, and evaluations count individual
objective components (including initial scaling gradients, when requested).

`inform` is 1 for the stopping test, 2 for the iteration limit, -1 for invalid
objective/gradient values, -2 for a direction-subproblem failure, and -3 for a
line-search failure. Always inspect it before treating the output as converged.

With `store_trace=true`, `stats.trace` stores copies of each accepted step's
source/destination, objective vectors, reference, slope and direction. This is
diagnostic data, NOT an archive of nondominated/Pareto-critical points; disable
it for timing experiments to avoid its allocation and storage overhead.

# Example
```julia
SDProject.PROBLEM = "FDS"
n, m, x0, _, _, _, _, _ = inip("FDS", 2025)
xsol, stats = SteepestDescentNonMonotone!(n, m, copy(x0); M=4, scaleF=false)
```
"""
function SteepestDescentNonMonotone!(n::Int, m::Int, x::Vector{T};
        M::Int=4, eta::T=T(1e-4), rho::T=T(0.5), initial_step::T=one(T),
        epsopt::T=T(1e-6), maxiter::Int=MAXOUTITER, scaleF::Bool=false,
        iprint::Bool=true, min_step::T=T(STPMIN), max_backtracks::Int=100,
        store_trace::Bool=false) where {T<:AbstractFloat}

    n > 0 && m > 0 || throw(ArgumentError("n and m must be positive."))
    length(x) == n || throw(DimensionMismatch("length(x) must equal n."))
    0 <= M < typemax(Int) || throw(ArgumentError("M must be nonnegative and M+1 representable."))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative."))
    max_backtracks >= 0 || throw(ArgumentError("max_backtracks must be nonnegative."))
    0 < eta < 1 || throw(ArgumentError("eta must lie in (0,1)."))
    0 < rho < 1 || throw(ArgumentError("rho must lie in (0,1)."))
    isfinite(epsopt) && epsopt > 0 || throw(ArgumentError("epsopt must be finite and positive."))
    isfinite(initial_step) && initial_step > 0 || throw(ArgumentError("initial_step must be finite and positive."))
    isfinite(min_step) && 0 < min_step <= initial_step || throw(ArgumentError("Require 0 < min_step <= initial_step."))
    all(isfinite, x) || throw(ArgumentError("The initial point must be finite."))

    t0 = time_ns()
    F, g = zeros(T, m), zeros(T, n)
    d, lambda = zeros(T, n), zeros(T, m)
    JF = zeros(T, m, n)
    slope, reference = zeros(T, m), zeros(T, m)
    xtrial, Ftrial = similar(x), similar(F)
    sF = ones(T, m)
    memory = Matrix{T}(undef, m, M + 1)
    stored, slot = 1, 1
    outiter = nfev = ngev = backtracks = 0
    theta, last_step = T(NaN), zero(T)
    inform, reason = 0, :not_started
    trace = store_trace ? NamedTuple[] : nothing

    iprint && @printf("\nNonmonotone SD (max-type): n=%d, m=%d, M=%d\n", n, m, M)

    # Evaluate the initial Jacobian once. Optional scaling is fixed from here on,
    # so all stored reference values and slopes stay in the same objective units.
    while true
        valid_gradient = true
        for i in 1:m
            ngev += 1
            flag = try
                _, result = sevalg!(n, x, g, i, one(T))
                result
            catch err
                err isa DomainError || rethrow()
                -1
            end
            if flag != 1 || !all(isfinite, g)
                valid_gradient = false
                break
            end
            if outiter == 0 && scaleF
                sF[i] = max(T(MACHEPS12), one(T) / max(one(T), norm(g, Inf)))
            end
            @views JF[i, :] .= sF[i] .* g
        end
        if !valid_gradient || !all(isfinite, JF)
            inform, reason = -1, :invalid_gradient
            theta = T(NaN)
            break
        end

        if outiter == 0
            valid_objectives = true
            for i in 1:m
                F[i], flag = _sdnm_objective(n, x, i, sF[i])
                nfev += 1
                if flag != 1
                    valid_objectives = false
                    break
                end
            end
            if !valid_objectives
                inform, reason = -1, :invalid_objective
                break
            end
            memory[:, 1] .= F
        end

        # Algorithm 3, steps 2-3: the direction and stationarity measure are
        # always evaluated at the current ACCEPTED iterate, not at a trial.
        _, _, info_direction = evalSD!(n, m, d, lambda, JF)
        if info_direction != 1 || !all(isfinite, d) || !all(isfinite, lambda)
            inform, reason = -2, :direction_failure
            theta = T(NaN)
            break
        end
        mul!(slope, JF, d)
        theta = maximum(slope) + dot(d, d) / 2
        if !all(isfinite, slope) || !isfinite(theta)
            inform, reason = -2, :invalid_direction
            break
        end
        if iprint && outiter % 10 == 0
            @printf("  iter=%d  |theta|=%.3e  evalf=%d  evalg=%d\n", outiter, abs(theta), nfev, ngev)
        end
        if abs(theta) < epsopt
            inform, reason = 1, :critical
            break
        elseif outiter >= maxiter
            inform, reason = 2, :maxiter
            break
        elseif maximum(slope) >= 0
            inform, reason = -2, :not_descent
            break
        end

        # Step 4: a circular buffer retains precisely x^(max(0,k-M)),...,x^k.
        for i in 1:m
            reference[i] = maximum(@view memory[i, 1:stored])
        end
        step, evaluations, reductions, ls_reason = _sdnm_backtracking!(
            xtrial, Ftrial, n, m, x, d, slope, sF, reference;
            eta, rho, initial_step, min_step, max_backtracks)
        nfev += evaluations
        backtracks += reductions
        if ls_reason != :accepted
            inform, reason = -3, ls_reason
            break
        end

        if store_trace
            push!(trace, (iteration=outiter, x=copy(x), x_next=copy(xtrial),
                f=copy(F), f_next=copy(Ftrial), reference=copy(reference),
                direction=copy(d), slope=copy(slope), theta=theta,
                step=step, backtracks=reductions))
        end
        # Step 6: commit both x and F only after ALL components pass the test.
        x .= xtrial
        F .= Ftrial
        outiter += 1
        last_step = step
        slot = mod(slot, M + 1) + 1
        memory[:, slot] .= F
        stored = min(stored + 1, M + 1)
    end

    elapsed = (time_ns() - t0) / 1e9
    iprint && @printf("SD-NM: %s (inform=%d), iterations=%d, time=%.6f s\n",
        string(reason), inform, outiter, elapsed)
    return x, (outiter=outiter, time=elapsed, nfev=nfev, ngev=ngev,
        theta=theta, inform=inform, reason=reason, backtracks=backtracks,
        last_step=last_step, trace=trace)
end

# A domain error at a trial is a rejected trial, not permission to accept it.
# Unexpected exceptions are propagated so that programming errors remain visible.
function _sdnm_objective(n, x, i, sf)
    value, flag = try
        sevalf(n, x, i, sf)
    catch err
        err isa DomainError || rethrow()
        return oftype(sf, NaN), -1
    end
    return value, (flag == 1 && isfinite(value) ? 1 : -1)
end

function _sdnm_backtracking!(xtrial, Ftrial, n, m, x, d, slope, sF, reference;
        eta, rho, initial_step, min_step, max_backtracks)
    step = initial_step
    nfev = 0
    for h in 0:max_backtracks
        @. xtrial = x + step * d
        xtrial == x && return step, nfev, h, :no_progress
        accepted = all(isfinite, xtrial)
        if accepted
            for i in 1:m
                Ftrial[i], flag = _sdnm_objective(n, xtrial, i, sF[i])
                nfev += 1
                bound = reference[i] + eta * step * slope[i]
                if flag != 1 || !isfinite(bound) || Ftrial[i] > bound
                    accepted = false
                    break
                end
            end
        end
        accepted && return step, nfev, h, :accepted
        h == max_backtracks && return step, nfev, h, :max_backtracks
        next_step = rho * step
        if next_step < min_step || next_step == step
            return step, nfev, h, :step_too_small
        end
        step = next_step
    end
    error("Unreachable backtracking state.")
end
