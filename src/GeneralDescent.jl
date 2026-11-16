using LinearAlgebra
using Random

# Pareto check for objective gradients stored in columns.
# The same implementation is kept in the research and submission GD files.
# Projected gradient starts from uniform weights and stops when the normalized
# scalarized-gradient norm ||sum_i w_i*g_i||/max_i||g_i|| is small.
# The final decision then uses the model-value inequality stated in the paper:
#   max_i <g_i,d> + ||d||^2/2 >= -tau*max_i ||g_i||^2.
# Reference: Beck (2017), First-Order Methods in Optimization, Ch. 10.
function _gd_pareto_check(G::Matrix{T}, project_simplex;
                          sigma::T=T(1e-1), tau::T=T(1e-3),
                          relative::Bool=true, maxiter::Int=2000) where {T<:AbstractFloat}
    @assert isfinite(sigma) && sigma > 0
    @assert isfinite(tau) && tau >= 0
    @assert maxiter >= 0
    n, m = size(G)
    @assert n > 0 && m > 0
    w = fill(one(T) / T(m), m)
    invalid(iter) = (status=:invalid, reason=:nonfinite, weights=copy(w),
                     iterations=iter, alpha=T(NaN), stationarity=T(NaN),
                     residual=T(NaN), converged=false)
    all(isfinite, G) || return invalid(0)

    # One COMMON scale preserves the original simplex problem and its
    # relative tolerance. Two divisions avoid squaring tiny/huge gradients.
    scale = maximum(abs, G)
    if iszero(scale)
        return (status=:accepted, reason=:zero_gradients, weights=w,
                iterations=0, alpha=zero(T), stationarity=zero(T),
                residual=zero(T), converged=true)
    end
    B = G ./ scale
    colscale = maximum(norm(view(B, :, i)) for i in 1:m)
    B ./= colscale
    H = transpose(B) * B
    L = opnorm(H, 2)
    isfinite(L) && L > 0 || return invalid(0)
    # max_i ||B[:,i]||^2 = 1. Absolute tolerances need the same rescaling.
    threshold = relative ? tau : ((sqrt(tau) / scale) / colscale)^2
    roundoff = T(16) * eps(T) * T(max(n, m))
    v, h = zeros(T, n), zeros(T, m)

    for iter in 0:maxiter
        mul!(v, B, w)
        mul!(h, transpose(B), v)
        q = dot(v, v) / T(2)
        theta = q - minimum(h)
        all(isfinite, h) && isfinite(q) && isfinite(theta) || return invalid(iter)
        # B is scaled by max_i ||g_i||, so this is the relative norm proposed
        # as the stopping test in the manuscript.
        residual = norm(v)
        isfinite(residual) || return invalid(iter)
        converged = residual <= sigma + roundoff
        if converged || iter == maxiter
            # The model-value inequality classifies the weights obtained at
            # either the norm test or the iteration safeguard.
            status = theta + roundoff >= -threshold ? :accepted : :rejected
            reason = converged ? :scalarized_gradient_norm : :iteration_limit
            # Raw alpha is diagnostic only; it can overflow for enormous G.
            # The decision uses the corresponding finite normalized value.
            alpha = iszero(theta) ? zero(T) : ((theta * colscale) * scale * colscale) * scale
            return (status=status, reason=reason, weights=copy(w),
                    iterations=iter, alpha=alpha, stationarity=q,
                    residual=residual, converged=converged)
        end
        wnew = project_simplex(w .- h ./ L)
        all(isfinite, wnew) && all(>=(zero(T)), wnew) || return invalid(iter)
        total = sum(wnew)
        isfinite(total) && total > 0 || return invalid(iter)
        wnew ./= total
        w .= wnew
    end
end


"""
    GeneralDescent!(n, m, x; ref_policy=:max, M=10, zh_theta=0.99, gm_gamma=0.85,
                    epsopt=1e-6, beta=1.0, eta=1e-4, sigma1=0.1, sigma2=0.5,
                    iprint=true, maxiter=10_000, w0=nothing,
                    direction_rule=:grad,            # future hook
                    step2_mode=:reference,          # :reference or :monotone
                    step3_mode=:monotone,           # :monotone or :nonmonotone
                    split_alpha=nothing,            # nothing or alpha in [0,1]
                    weight_update=:stick,            # :stick, :mincomponent, :random, :unit_cycle, :drsc_uniform, :drsc_original, :hitandrun_uniform
                    unit_cycle_epsilon=0.0,
                    max_nu_decay_tau=nothing, max_nu_decay_k0=1,
                    drsc_trials=200, drsc_tol=1e-12, drsc_max_transforms=200,
                    har_burnin=20, har_steps=10, har_dir_tol=1e-12, har_feas_tol=1e-12,
                    weight_change_trigger=:always,
                    periodic_weight_window=100,
                    collect_stagnation_iters=30, collect_too_close_limit=20,
                    max_iters_without_weight_change=200, min_iters_between_weight_changes=5,
                    weight_accept_filter=:none, recent_weight_window=10, weight_novelty_tau=0.05,
                    adaptive_burnin_iters=30, adaptive_min_initial_weight_changes=20,
                    adaptive_reject_limit=20, adaptive_rescue_iters=10,
                    weight_acceptance=:descent,
                    collect=false, collect_alpha=0.05, collect_atol=1e-9,
                    collect_filter_style=nothing,    # nothing, :final, :during, or :during_novelty
                    collect_archive_mode=:final,      # legacy alias: :final, :incremental_output, :incremental_novelty
                    collect_metric=:global_span,      # :global_span or :component_span
                    collect_grad_gate=false, collect_lambda=1e-2, collect_gate_mode=:absolute,
                    collect_on_weight_change=true, force_collect_requires_novelty=false,
                    always_collect_final=true,
                    archive_collect_interval=0, archive_collect_requires_novelty=true,
                    pc_check=false, pc_sigma=1e-1, pc_tau=1e-3, pc_tau_mode=:absolute, pc_maxiter=2000,
                    preserve_extremes=false, extreme_tau=1e-4,
                    time_limit=nothing, stop_on_time_only=false)

Resolve o MOP escalarizado. Retorna (x, w, stats) onde `stats` inclui
outiter, nfev, ngev, gnorm, inform, policy, collected_x, collected_f,
nd_x, nd_f, nu_hist, rho_hist, weight_changed_hist,
weight_change_norm_hist, weight_change_iter, weight_change_norm,
x_hist, f_hist, step2_mode, step3_mode, split_alpha, forced_collect_count, pc_checked_count, pc_pass_count.

The Pareto check uses uniform initial weights and projected gradient. `pc_sigma`
is the tolerance for the normalized scalarized-gradient norm, NOT a
successive-weight tolerance. The weights obtained when this test is met or
`pc_maxiter` is reached are classified using the model-value inequality in the
paper. Invalid points are excluded and counted separately.
`pc_stationarity_hist` records half the squared normalized scalarized-gradient
norm.
See `docs/pareto_check_validation.md` for the criterion and parameter semantics.

With `collect_filter_style=:during`, all admitted candidates remain in
`collected_*` for proximity/history, including dominated and repeated points;
only `nd_*` is deduplicated. The initial point and every backtracking trial
(even rejected ones) use the CURRENT OUTER gradient gate. Forced weight-change
and final sites bypass that gate and, by default, proximity, but never finite
value checks, output deduplication, nondominance, or the enabled PC filter.
For `:component_span`, the box is updated before separation, and the denominator
is max(observed span, eps(T)), not max(observed span, 1). The campaign threshold
is 0.003. See `docs/archive_protocol.md` for the frozen collection contract.
"""
function GeneralDescent!(n::Int, m::Int, x::Vector{T};
                         ref_policy::Symbol = :max,
                         M::Int = 10,
                         zh_theta::T = T(0.99),
                         gm_gamma::T = T(0.85),
                         epsopt::T = T(1e-6),
                         beta::T = T(1.0),
                         eta::T = T(1e-4),
                         sigma1::T = T(0.1),
                         sigma2::T = T(0.5),
                         iprint::Bool = true,
                         maxiter::Int = 10_000,
                         w0::Union{Nothing,Vector{T}} = nothing,
                         direction_rule::Symbol = :grad,
                         step2_mode::Symbol = :reference,
                         step3_mode::Symbol = :monotone,
                         split_alpha::Union{Nothing,T} = nothing,
                         weight_update::Symbol = :stick,
                         unit_cycle_epsilon::T = zero(T),
                         max_nu_decay_tau::Union{Nothing,T} = nothing,
                         max_nu_decay_k0::Int = 1,
                         rng = Random.default_rng(),
                         drsc_trials::Int = 200,
                         drsc_tol::T = T(1e-12),
                         drsc_max_transforms::Int = 200,
                         har_burnin::Int = 20,
                         har_steps::Int = 10,
                         har_dir_tol::T = T(1e-12),
                         har_feas_tol::T = T(1e-12),
                         weight_change_trigger::Symbol = :always,
                         periodic_weight_window::Int = 100,
                         collect_stagnation_iters::Int = 30,
                         collect_too_close_limit::Int = 20,
                         max_iters_without_weight_change::Int = 200,
                         min_iters_between_weight_changes::Int = 5,
                         weight_accept_filter::Symbol = :none,
                         recent_weight_window::Int = 10,
                         weight_novelty_tau::T = T(0.05),
                         adaptive_burnin_iters::Int = 30,
                         adaptive_min_initial_weight_changes::Int = 20,
                         adaptive_reject_limit::Int = 20,
                         adaptive_rescue_iters::Int = 10,
                         weight_acceptance::Symbol = :descent,
                          collect::Bool = false,
                          collect_alpha::T = T(0.05),
                          collect_atol::T = T(1e-9),
                          collect_filter_style::Union{Nothing,Symbol} = nothing,
                          collect_archive_mode::Symbol = :final,
                          collect_metric::Symbol = :global_span,
                          collect_grad_gate::Bool = false,
                         collect_lambda::T = T(1e-2),
                         collect_gate_mode::Symbol = :absolute,
                         collect_on_weight_change::Bool = true,
                         force_collect_requires_novelty::Bool = false,
                         always_collect_final::Bool = true,
                         archive_collect_interval::Int = 0,
                         archive_collect_requires_novelty::Bool = true,
                         pc_check::Bool = false,
                         pc_sigma::T = T(1e-1),
                         pc_tau::T = T(1e-3),
                         pc_tau_mode::Symbol = :absolute,
                         pc_maxiter::Int = 2000,
                         preserve_extremes::Bool = false,
                         extreme_tau::T = T(1e-4),
                         time_limit = nothing,
                         stop_on_time_only::Bool = false,
                         track_iterates::Bool = false
                        ) where {T<:AbstractFloat}

    @assert 0 < eta < 1
    @assert 0 < sigma1 < sigma2 < 1
    @assert m >= 2
    @assert drsc_trials >= 1
    @assert drsc_max_transforms >= 1
    @assert har_burnin >= 0
    @assert har_steps >= 1
    @assert har_dir_tol > 0
    @assert har_feas_tol >= 0
    @assert weight_change_trigger in (:always, :collect_stagnation, :periodic)
    @assert periodic_weight_window >= 1
    @assert collect_stagnation_iters >= 1
    @assert collect_too_close_limit >= 1
    @assert max_iters_without_weight_change >= 1
    @assert min_iters_between_weight_changes >= 0
    @assert weight_accept_filter in (:none, :novelty, :adaptive_novelty_safe)
    @assert recent_weight_window >= 1
    @assert weight_novelty_tau >= 0
    @assert adaptive_burnin_iters >= 0
    @assert adaptive_min_initial_weight_changes >= 0
    @assert adaptive_reject_limit >= 1
    @assert adaptive_rescue_iters >= 0
    # Public campaign/package name; retain the historical internal branch name.
    weight_update == :uniform_admissible && (weight_update = :drsc_uniform)
    @assert weight_acceptance in (:descent, :free)
    @assert weight_update in (:stick, :mincomponent, :random, :unit_cycle, :drsc_uniform, :drsc_original, :hitandrun_uniform)
    @assert zero(T) <= unit_cycle_epsilon < one(T) / T(m)
    @assert step2_mode in (:reference, :monotone)
    @assert step3_mode in (:monotone, :nonmonotone)
    @assert split_alpha === nothing || (zero(T) <= split_alpha <= one(T))
    @assert 0 <= gm_gamma < 1
    @assert max_nu_decay_tau === nothing || (zero(T) < max_nu_decay_tau < one(T))
    @assert max_nu_decay_k0 >= 0
    @assert collect_metric in (:global_span, :component_span)
    @assert collect_filter_style === nothing || collect_filter_style in (:final, :during, :during_novelty)
    if collect_filter_style !== nothing
        collect_archive_mode = if collect_filter_style == :final
            :final
        elseif collect_filter_style == :during
            :incremental_output
        else
            :incremental_novelty
        end
    end
    @assert collect_archive_mode in (:final, :incremental_output, :incremental_novelty)
    resolved_collect_filter_style =
        collect_archive_mode == :final ? :final :
        collect_archive_mode == :incremental_output ? :during :
        :during_novelty
    @assert collect_lambda >= 0
    @assert isfinite(collect_alpha) && collect_alpha >= 0
    @assert isfinite(collect_atol) && collect_atol >= 0
    @assert collect_gate_mode in (:absolute, :relative_initial)
    @assert archive_collect_interval >= 0
    @assert isfinite(pc_sigma) && pc_sigma > 0
    @assert isfinite(pc_tau) && pc_tau >= 0
    @assert pc_tau_mode in (:absolute, :relative_grad_scale2)
    @assert pc_maxiter >= 1
    @assert extreme_tau >= 0
    @assert time_limit === nothing || time_limit > 0
    @assert !stop_on_time_only || time_limit !== nothing

    # workspaces
    g     = Vector{T}(undef, n)
    gF    = Vector{T}(undef, n)
    d     = Vector{T}(undef, n)
    fvec  = Vector{T}(undef, m)

    # weights
    w = if w0 === nothing
        fill(T(1)/T(m), m)
    else
        wt = copy(w0)
        @inbounds for i in 1:m
            wt[i] = max(zero(T), wt[i])
        end
        s = sum(wt); s == 0 ? fill(T(1)/T(m), m) : (wt ./= s)
    end

    # helpers
    function F_at(xv::Vector{T}, wv::Vector{T})::T
        s = zero(T)
        @inbounds for i in 1:m
            fi = try
                evalf(n, xv, i)
            catch err
                err isa DomainError || rethrow()
                return T(Inf)
            end
            isfinite(fi) || return T(Inf)
            s += wv[i]*fi
        end
        return s
    end
    function gradF!(xv::Vector{T}, wv::Vector{T}, out::Vector{T})
        fill!(out, zero(T))
        @inbounds for i in 1:m
            evalg!(n, xv, g, i)
            wi = wv[i]
            if wi != zero(T)
                @simd for j in 1:n
                    out[j] += wi*g[j]
                end
            end
        end
        out
    end
    function eval_objectives!(xv::Vector{T}, out::Vector{T})
        @inbounds for i in 1:m
            out[i] = evalf(n, xv, i)
        end
        out
    end
    function eval_trial_objectives!(xv::Vector{T}, out::Vector{T})
        try
            eval_objectives!(xv, out)
        catch err
            err isa DomainError || rethrow()
            fill!(out, T(Inf))
            return false
        end
        return all(isfinite, out)
    end
    function uniform_simplex_sample!(out::Vector{T})
        # Exact uniform sample on simplex Delta via Dirichlet(1,...,1).
        s = zero(T)
        @inbounds for i in 1:m
            u = max(rand(rng, T), eps(T))
            out[i] = -log(u)
            s += out[i]
        end
        if s == zero(T)
            fill!(out, one(T) / T(m))
        else
            @. out = out / s
        end
        out
    end
    function unit_cycle_weight!(out::Vector{T}, idx::Int)
        epsw = unit_cycle_epsilon
        fill!(out, epsw)
        out[idx] = one(T) - T(m - 1) * epsw
        return out
    end
    function next_unit_cycle_index(wv::Vector{T})
        imax = 1
        vmax = wv[1]
        @inbounds for i in 2:m
            if wv[i] > vmax
                vmax = wv[i]
                imax = i
            end
        end
        return imax == m ? 1 : imax + 1
    end
    function drsc_candidate_thresholds(a::Vector{T}, rhs::T)
        idxs = Int[]
        thetas = T[]
        if minimum(a) > rhs + drsc_tol
            return idxs, thetas
        end
        @inbounds for i in 1:m
            ai = a[i]
            ai > rhs + drsc_tol || continue
            amin_other = typemax(T)
            for j in 1:m
                j == i && continue
                amin_other = min(amin_other, a[j])
            end
            if !(amin_other < ai - drsc_tol)
                continue
            end
            theta = clamp((rhs - amin_other) / (ai - amin_other), zero(T), one(T))
            if theta >= one(T) - drsc_tol
                continue
            end
            min_on_simplex = muladd(theta, ai - amin_other, amin_other)
            if min_on_simplex <= rhs + drsc_tol
                theta = nextfloat(theta)
                theta >= one(T) && continue
            end
            push!(idxs, i)
            push!(thetas, theta)
        end
        return idxs, thetas
    end
    function drsc_select_nonoverlapping(idxs::Vector{Int}, thetas::Vector{T})
        k = length(idxs)
        k == 0 && return Int[], T[]
        vols = Vector{T}(undef, k)
        @inbounds for i in 1:k
            vols[i] = (one(T) - thetas[i])^(m - 1)
        end
        bestmask = 0
        bestscore = -typemax(T)
        bestcount = -1
        limit = 1 << k
        for mask in 1:(limit - 1)
            ok = true
            score = zero(T)
            count = 0
            @inbounds for i in 1:k
                ((mask >> (i - 1)) & 1) == 1 || continue
                score += vols[i]
                count += 1
                for j in (i + 1):k
                    ((mask >> (j - 1)) & 1) == 1 || continue
                    if thetas[i] + thetas[j] < one(T) - drsc_tol
                        ok = false
                        break
                    end
                end
                ok || break
            end
            ok || continue
            if score > bestscore + drsc_tol ||
               (abs(score - bestscore) <= drsc_tol && count > bestcount)
                bestmask = mask
                bestscore = score
                bestcount = count
            end
        end
        selidx = Int[]
        seltheta = T[]
        @inbounds for i in 1:k
            if ((bestmask >> (i - 1)) & 1) == 1
                push!(selidx, idxs[i])
                push!(seltheta, thetas[i])
            end
        end
        return selidx, seltheta
    end
    function drsc_affine_map!(wtrial::Vector{T}, idx::Int, theta::T)
        scale = one(T) - theta
        scale <= drsc_tol && return false
        @inbounds for j in 1:m
            if j == idx
                wtrial[j] = (wtrial[j] - theta) / scale
            else
                wtrial[j] /= scale
            end
            if wtrial[j] < zero(T) && wtrial[j] > -100drsc_tol
                wtrial[j] = zero(T)
            end
        end
        s = sum(wtrial)
        if !isfinite(s) || s <= zero(T)
            return false
        end
        wtrial ./= s
        return true
    end
    function drsc_original_sample!(wout::Vector{T}, a::Vector{T}, rhs::T)
        idxs, thetas = drsc_candidate_thresholds(a, rhs)
        idxs, thetas = drsc_select_nonoverlapping(idxs, thetas)
        wtrial = similar(wout)
        @inbounds for _ in 1:drsc_trials
            drsc_attempt_count += 1
            uniform_simplex_sample!(wtrial)
            for _ in 1:drsc_max_transforms
                if dot(a, wtrial) <= rhs + drsc_tol
                    wout .= wtrial
                    drsc_accept_count += 1
                    return true
                end
                bestp = 0
                bestmargin = -typemax(T)
                for p in eachindex(idxs)
                    margin = wtrial[idxs[p]] - thetas[p]
                    if margin >= -drsc_tol && margin > bestmargin
                        bestmargin = margin
                        bestp = p
                    end
                end
                bestp == 0 && break
                drsc_transform_count += 1
                drsc_affine_map!(wtrial, idxs[bestp], thetas[bestp]) || break
            end
        end
        return false
    end
    function project_simplex(vin::Vector{T})
        # Euclidean projection onto Delta = {w >= 0, sum(w)=1}
        u = sort(vin; rev=true)
        cssv = zero(T)
        rho = 1
        theta = zero(T)
        @inbounds for j in 1:length(u)
            cssv += u[j]
            t = (cssv - one(T)) / T(j)
            if u[j] - t > zero(T)
                rho = j
                theta = t
            end
        end
        wproj = similar(vin)
        @inbounds for i in eachindex(vin)
            wproj[i] = max(vin[i] - theta, zero(T))
        end
        wproj
    end
    function hitandrun_truncated_simplex!(wout::Vector{T}, a::Vector{T}, rhs::T)
        # Sample approximately uniformly from
        #   {w >= 0, sum(w)=1, a'w <= rhs}
        # using Hit-and-Run on the affine hull sum(w)=1.
        @assert length(wout) == m
        cur = copy(wout)
        dhr = similar(cur)
        total_steps = har_burnin + har_steps
        @inbounds for _ in 1:total_steps
            got_dir = false
            for _ in 1:20
                @inbounds for i in 1:m
                    dhr[i] = randn(rng, T)
                end
                μd = sum(dhr) / T(m)
                @inbounds for i in 1:m
                    dhr[i] -= μd
                end
                nd = norm(dhr)
                if nd > har_dir_tol
                    dhr ./= nd
                    got_dir = true
                    break
                end
            end
            got_dir || return false

            λmin = -typemax(T)
            λmax = typemax(T)
            @inbounds for i in 1:m
                di = dhr[i]
                if di > har_dir_tol
                    λmin = max(λmin, -cur[i] / di)
                elseif di < -har_dir_tol
                    λmax = min(λmax, -cur[i] / di)
                end
            end

            adot = dot(a, dhr)
            slack = rhs - dot(a, cur)
            if adot > har_dir_tol
                λmax = min(λmax, (slack + har_feas_tol) / adot)
            elseif adot < -har_dir_tol
                λmin = max(λmin, (slack + har_feas_tol) / adot)
            end

            if !(isfinite(λmin) && isfinite(λmax)) || λmin > λmax
                return false
            end

            λ = λmin + rand(rng, T) * (λmax - λmin)
            @inbounds for i in 1:m
                cur[i] += λ * dhr[i]
                if cur[i] < zero(T) && cur[i] > -har_feas_tol
                    cur[i] = zero(T)
                end
            end
            s = sum(cur)
            if !isfinite(s) || s <= zero(T)
                return false
            end
            cur ./= s
            if dot(a, cur) > rhs + 10har_feas_tol
                return false
            end
        end
        wout .= cur
        return true
    end
    function pareto_critical_check_at(xv::Vector{T})
        # Approximate solution of:
        #   min_d max_i <grad f_i(x), d> + 0.5*||d||^2
        # via the dual simplex QP:
        #   min_{w in Delta} 0.5*||G w||^2, where G=[grad f_1 ... grad f_m].
        G = Matrix{T}(undef, n, m)
        gj = Vector{T}(undef, n)
        @inbounds for i in 1:m
            evalg!(n, xv, gj, i); ngev += 1
            for j in 1:n
                G[j, i] = gj[j]
            end
        end
        return _gd_pareto_check(G, project_simplex; sigma=pc_sigma, tau=pc_tau,
                                relative=pc_tau_mode == :relative_grad_scale2, maxiter=pc_maxiter)
    end

    # Keep collection history separate from the nondominated output. With
    # :during, removing output duplicates must not change proximity memory.
    collected_x = Vector{Vector{T}}()
    collected_f = Vector{Vector{T}}()
    nd_x = Vector{Vector{T}}()
    nd_f = Vector{Vector{T}}()
    x_hist = Vector{Vector{T}}()
    f_hist = Vector{Vector{T}}()
    zmin = fill(typemax(T), m)
    zmax = fill(-typemax(T), m)
    best_extreme = fill(typemax(T), m)
    collect_invalid_count = 0
    function valid_candidate(xv, zv)
        if !all(isfinite, xv) || !all(isfinite, zv)
            collect_invalid_count += 1
            return false
        end
        return true
    end
    function update_bbox!(zv::Vector{T})
        @inbounds for i in 1:m
            zmin[i] = min(zmin[i], zv[i])
            zmax[i] = max(zmax[i], zv[i])
        end
    end
    function too_close_to_collected(zv::Vector{T})
        collect_alpha <= zero(T) && return false
        archive = collect_archive_mode == :incremental_novelty ? nd_f : collected_f
        if collect_metric == :component_span
            @inbounds for zk in archive
                dist = zero(T)
                for i in 1:m
                    denom = max(abs(zmax[i] - zmin[i]), eps(T))
                    dist = max(dist, abs(zv[i] - zk[i]) / denom)
                    dist > collect_alpha && break
                end
                dist <= collect_alpha && return true
            end
            return false
        end
        span = zero(T)
        @inbounds for i in 1:m
            span = max(span, abs(zmax[i] - zmin[i]))
        end
        δ = collect_alpha * span
        if δ == zero(T) || !isfinite(δ)
            return false
        end
        @inbounds for zk in archive
            dist = zero(T)
            for i in 1:m
                dist = max(dist, abs(zv[i] - zk[i]))
                dist > δ && break
            end
            dist <= δ && return true
        end
        return false
    end
    function improves_extreme(zv::Vector{T})
        preserve_extremes || return false
        @inbounds for i in 1:m
            span_i = max(abs(zmax[i] - zmin[i]), eps(T))
            tol_i = extreme_tau * span_i
            if zv[i] < best_extreme[i] - tol_i
                return true
            end
        end
        return false
    end
    function update_extremes!(zv::Vector{T})
        preserve_extremes || return
        @inbounds for i in 1:m
            best_extreme[i] = min(best_extreme[i], zv[i])
        end
    end
    function update_incremental_nd!(xv::Vector{T}, zv::Vector{T})
        collect_archive_mode in (:incremental_output, :incremental_novelty) || return :accepted
        remove_idx = Int[]
        @inbounds for i in eachindex(nd_f)
            relation = point_relation(nd_f[i], zv; atol=collect_atol)
            if relation == :equivalent
                return :duplicate
            elseif relation == :a_dominates
                return :dominated
            elseif relation == :b_dominates
                push!(remove_idx, i)
            end
        end
        for i in reverse(remove_idx)
            deleteat!(nd_f, i)
            deleteat!(nd_x, i)
        end
        push!(nd_x, copy(xv))
        push!(nd_f, copy(zv))
        return :accepted
    end
    function maybe_collect!(xv::Vector{T}, zv::Vector{T})
        valid_candidate(xv, zv) || return :invalid
        # Include this candidate in the observed span before separation,
        # even when separation subsequently rejects it.
        update_bbox!(zv)
        extreme = improves_extreme(zv)
        (!extreme && too_close_to_collected(zv)) && return :too_close
        nd_status = update_incremental_nd!(xv, zv)
        nd_status in (:dominated, :duplicate) && collect_archive_mode == :incremental_novelty && return nd_status
        push!(collected_x, copy(xv))
        push!(collected_f, copy(zv))
        update_extremes!(zv)
        return :accepted
    end
    function force_collect!(xv::Vector{T}, zv::Vector{T})
        # These sites bypass the outer-gradient gate and, by default,
        # separation. Output deduplication, dominance and the PC check remain.
        valid_candidate(xv, zv) || return :invalid
        update_bbox!(zv)
        if force_collect_requires_novelty && too_close_to_collected(zv)
            return :too_close
        end
        nd_status = update_incremental_nd!(xv, zv)
        nd_status in (:dominated, :duplicate) && collect_archive_mode == :incremental_novelty && return nd_status
        push!(collected_x, copy(xv))
        push!(collected_f, copy(zv))
        update_extremes!(zv)
        return :accepted
    end
    initial_gnorm = Ref(one(T))
    function collect_gate_pass(gnorm_current::T)
        !collect_grad_gate && return true
        if collect_gate_mode == :relative_initial
            return gnorm_current <= collect_lambda * max(initial_gnorm[], eps(T))
        end
        return gnorm_current <= collect_lambda
    end
    function nondominated_indices(F::Vector{Vector{T}})
        keep = fill(true, length(F))
        for i in eachindex(F)
            keep[i] || continue
            fi = F[i]
            for j in eachindex(F)
                i==j && continue
                keep[j] || continue
                relation = point_relation(F[j], fi; atol=collect_atol)
                if relation == :a_dominates ||
                   (relation == :equivalent && j < i)
                    keep[i] = false; break
                end
            end
        end
        keep
    end

    # init
    Fk = if collect
        eval_objectives!(x, fvec); dot(w, fvec)
    else
        F_at(x, w)
    end
    gradF!(x, w, gF)
    gnorm = norm(gF)
    initial_gnorm[] = max(gnorm, eps(T))
    collect && collect_gate_pass(gnorm) && maybe_collect!(x, fvec)
    if track_iterates
        if collect
            push!(f_hist, copy(fvec))
        else
            push!(f_hist, [evalf(n, x, i) for i in 1:m])
        end
        push!(x_hist, copy(x))
    end

    # referencia e log de troca de pesos
    nu_hist = T[]
    rho_hist = T[]
    weight_changed_hist = Bool[]
    weight_change_norm_hist = T[]
    weight_change_iter = Int[]
    weight_change_norm = T[]
    weight_changed_count = 0
    recent_weights = Vector{Vector{T}}()
    push!(recent_weights, copy(w))
    weight_novelty_reject_count = 0
    adaptive_filter_active_count = 0
    adaptive_rescue_count = 0
    consecutive_novelty_rejects = 0
    rescue_remaining = 0
    function remember_weight!(wv::Vector{T})
        push!(recent_weights, copy(wv))
        while length(recent_weights) > recent_weight_window
            deleteat!(recent_weights, 1)
        end
    end
    function weight_accept_filter_pass(wv::Vector{T})
        weight_accept_filter == :none && return true
        if weight_accept_filter == :adaptive_novelty_safe
            if outiter <= adaptive_burnin_iters ||
               weight_changed_count < adaptive_min_initial_weight_changes ||
               rescue_remaining > 0
                return true
            end
            adaptive_filter_active_count += 1
        end
        isempty(recent_weights) && return true
        dmin = typemax(T)
        @inbounds for wk in recent_weights
            dmin = min(dmin, norm(wv .- wk))
        end
        return dmin >= weight_novelty_tau
    end
    iters_since_collect_accept = 0
    iters_since_archive_collect = 0
    consecutive_collect_too_close = 0
    iters_since_weight_change = 0
    skipped_weight_change_count = 0
    collect_accept_count = 0
    collect_too_close_count = 0
    archive_collect_count = 0
    function should_attempt_weight_change()
        weight_change_trigger == :always && return true
        weight_change_trigger == :periodic && return iters_since_weight_change >= periodic_weight_window
        if iters_since_weight_change < min_iters_between_weight_changes
            return false
        end
        return iters_since_collect_accept >= collect_stagnation_iters ||
               consecutive_collect_too_close >= collect_too_close_limit ||
               iters_since_weight_change >= max_iters_without_weight_change
    end
    drsc_attempt_count = 0
    drsc_accept_count = 0
    drsc_transform_count = 0
    forced_collect_count = 0
    pc_checked_count = 0
    pc_pass_count = 0
    pc_alpha_hist = T[]
    pc_stationarity_hist = T[]
    pc_residual_hist = T[]
    pc_iteration_hist = Int[]
    pc_status_hist = Symbol[]
    pc_reason_hist = Symbol[]
    if ref_policy == :max
        window = T[ Fk ]
        rho = maximum(window)
        v = rho - Fk
    elseif ref_policy == :zhanghager
        q = one(T)
        rho = Fk
        v = rho - Fk
    elseif ref_policy == :guandmo
        rho = Fk
        v = rho - Fk
    else
        error("ref_policy deve ser :max, :zhanghager ou :guandmo")
    end
    push!(nu_hist, v); push!(rho_hist, rho)
    push!(weight_changed_hist, false)
    push!(weight_change_norm_hist, zero(T))

    outiter = 0; nfev = m; ngev = m; inform = 0
    start_time_ns = time_ns()
    elapsed_seconds() = (time_ns() - start_time_ns) / 1.0e9
    time_remaining() = time_limit === nothing || elapsed_seconds() < Float64(time_limit)
    keep_running() = stop_on_time_only ? time_remaining() : (gnorm > epsopt && outiter < maxiter && time_remaining())

    while keep_running()
        outiter += 1
        if rescue_remaining > 0
            rescue_remaining -= 1
        end
        iters_since_collect_accept += 1
        iters_since_archive_collect += 1
        iters_since_weight_change += 1

        # direção
        if direction_rule == :grad
            @. d = -gF
            if beta > one(T)
                ng = norm(gF)
                if ng > zero(T)
                    d .= (-beta/ng) .* gF
                end
            end
        else
            error("direction_rule $direction_rule não implementada")
        end

        t = one(T)
        gTd = dot(gF, d)
        nu_step2 = split_alpha === nothing ? (step2_mode == :monotone ? zero(T) : v) : split_alpha * v
        step2_ref = Fk + nu_step2
        while true
            xt = @. x + t*d
            if collect
                valid_trial = eval_trial_objectives!(xt, fvec); nfev += m
                Ft = valid_trial ? dot(w, fvec) : T(Inf)
                if collect_gate_pass(gnorm)
                    collect_status = maybe_collect!(xt, fvec)
                    if collect_status == :accepted
                        collect_accept_count += 1
                        iters_since_collect_accept = 0
                        iters_since_archive_collect = 0
                        consecutive_collect_too_close = 0
                    elseif collect_status == :too_close
                        collect_too_close_count += 1
                        consecutive_collect_too_close += 1
                    end
                end
            else
                Ft = F_at(xt, w); nfev += m
            end
            if Ft <= step2_ref + eta*t*gTd
                x .= xt; Fk = Ft; break
            end
            t = sigma2 * t
            if t <= eps(T)
                if isfinite(Ft)
                    x .= xt
                    Fk = Ft
                end
                break
            end
        end

        if collect && archive_collect_interval > 0 && iters_since_archive_collect >= archive_collect_interval
            if archive_collect_requires_novelty
                archive_status = maybe_collect!(x, fvec)
                if archive_status == :accepted
                    archive_collect_count += 1
                    collect_accept_count += 1
                    iters_since_collect_accept = 0
                    iters_since_archive_collect = 0
                    consecutive_collect_too_close = 0
                elseif archive_status == :too_close
                    collect_too_close_count += 1
                    consecutive_collect_too_close += 1
                end
            else
                force_collect!(x, fvec)
                archive_collect_count += 1
                collect_accept_count += 1
                iters_since_collect_accept = 0
                iters_since_archive_collect = 0
                consecutive_collect_too_close = 0
            end
        end

        # Step 3:
        # Legacy modes:
        # :monotone    -> F(x;wnew) <= F(x;wold)
        # :nonmonotone -> F(x;wnew) <= F(x;wold) + nu_k
        # Split mode:
        # split_alpha = alpha -> F(x;wnew) <= F(x;wold) + (1-alpha)nu_k
        F_current_w = Fk
        nu_allow = split_alpha === nothing ? (step3_mode == :nonmonotone ? v : zero(T)) : (one(T) - split_alpha) * v
        F_accept_rhs = F_current_w + nu_allow
        w_changed = false
        w_prev = copy(w)
        attempt_weight_change = should_attempt_weight_change()
        if !attempt_weight_change
            Fk = F_current_w
            skipped_weight_change_count += 1
        elseif weight_update == :mincomponent
            minval = typemax(T); argmin = 1
            @inbounds for i in 1:m
                fi = evalf(n, x, i); nfev += 1
                if fi < minval; minval = fi; argmin = i; end
            end
            if weight_acceptance == :free || minval <= F_accept_rhs
                w .= 0; w[argmin] = one(T)
                if weight_accept_filter_pass(w)
                    Fk = minval
                else
                    w .= w_prev
                    Fk = F_current_w
                    weight_novelty_reject_count += 1
                    consecutive_novelty_rejects += 1
                end
            end
        elseif weight_update == :random
            wtrial = rand(rng, T, m)
            wtrial ./= sum(wtrial)
            Ftrial = F_at(x, wtrial); nfev += m
            if weight_acceptance == :free || Ftrial <= F_accept_rhs
                if weight_accept_filter_pass(wtrial)
                    w .= wtrial; Fk = Ftrial
                else
                    Fk = F_current_w
                    weight_novelty_reject_count += 1
                    consecutive_novelty_rejects += 1
                end
            end
        elseif weight_update == :unit_cycle
            eval_objectives!(x, fvec); nfev += m
            wtrial = similar(w)
            unit_cycle_weight!(wtrial, next_unit_cycle_index(w))
            Ftrial = dot(fvec, wtrial)
            if weight_acceptance == :free || Ftrial <= F_accept_rhs
                if weight_accept_filter_pass(wtrial)
                    w .= wtrial
                    Fk = Ftrial
                else
                    Fk = F_current_w
                    weight_novelty_reject_count += 1
                    consecutive_novelty_rejects += 1
                end
            else
                Fk = F_current_w
            end
        elseif weight_acceptance == :free && weight_update in (:drsc_uniform, :drsc_original, :hitandrun_uniform)
            # Experimental mode: remove the scalarized acceptance constraint
            # on weights. DRSC/hit-and-run are then equivalent to drawing a
            # fresh simplex weight, because there is no truncated feasible set.
            eval_objectives!(x, fvec); nfev += m
            wtrial = similar(w)
            drsc_attempt_count += 1
            uniform_simplex_sample!(wtrial)
            if weight_accept_filter_pass(wtrial)
                w .= wtrial
                Fk = dot(fvec, wtrial)
                drsc_accept_count += 1
            else
                Fk = F_current_w
                weight_novelty_reject_count += 1
                consecutive_novelty_rejects += 1
            end
        elseif weight_update == :drsc_uniform
            # Uniform rejection on S = {w in Delta : f(x)'w <= F_accept_rhs}.
            eval_objectives!(x, fvec); nfev += m
            wtrial = similar(w)
            accepted = false
            @inbounds for _ in 1:drsc_trials
                drsc_attempt_count += 1
                uniform_simplex_sample!(wtrial)
                Ftrial = dot(fvec, wtrial)
                if Ftrial <= F_accept_rhs + drsc_tol
                    if weight_accept_filter_pass(wtrial)
                        w .= wtrial
                        Fk = Ftrial
                        drsc_accept_count += 1
                        accepted = true
                        break
                    else
                        weight_novelty_reject_count += 1
                        consecutive_novelty_rejects += 1
                    end
                end
            end
            # Fallback: keep old weight if no feasible sample is found.
            if !accepted
                Fk = F_current_w
            end
        elseif weight_update == :drsc_original
            # DRSC with induced simplices and affine maps, specialized to the
            # single linear restriction f(x)'w <= F_accept_rhs that arises in
            # Step 3 of the algorithm.
            eval_objectives!(x, fvec); nfev += m
            wtrial = similar(w)
            accepted = drsc_original_sample!(wtrial, fvec, F_accept_rhs)
            if accepted
                if weight_accept_filter_pass(wtrial)
                    w .= wtrial
                    Fk = dot(fvec, wtrial)
                else
                    Fk = F_current_w
                    weight_novelty_reject_count += 1
                    consecutive_novelty_rejects += 1
                end
            else
                Fk = F_current_w
            end
        elseif weight_update == :hitandrun_uniform
            eval_objectives!(x, fvec); nfev += m
            wtrial = copy(w)
            drsc_attempt_count += 1
            accepted = hitandrun_truncated_simplex!(wtrial, fvec, F_accept_rhs)
            if accepted
                Ftrial = dot(fvec, wtrial)
                if Ftrial <= F_accept_rhs + har_feas_tol
                    if weight_accept_filter_pass(wtrial)
                        w .= wtrial
                        Fk = Ftrial
                        drsc_accept_count += 1
                    else
                        Fk = F_current_w
                        weight_novelty_reject_count += 1
                        consecutive_novelty_rejects += 1
                    end
                else
                    Fk = F_current_w
                end
            else
                Fk = F_current_w
            end
        else
            # stick: mantém w e Fk já corresponde a w atual
        end
        Δw = norm(w .- w_prev)
        w_changed = (Δw > T(1e-12))
        if w_changed
            weight_changed_count += 1
            iters_since_weight_change = 0
            consecutive_novelty_rejects = 0
            remember_weight!(w)
            push!(weight_change_iter, outiter)
            push!(weight_change_norm, Δw)
            if collect && collect_on_weight_change
                forced_status = force_collect!(x, fvec)
                if forced_status == :accepted
                    forced_collect_count += 1
                elseif forced_status == :too_close
                    collect_too_close_count += 1
                    consecutive_collect_too_close += 1
                end
            end
        end
        if weight_accept_filter == :adaptive_novelty_safe &&
           consecutive_novelty_rejects >= adaptive_reject_limit
            rescue_remaining = adaptive_rescue_iters
            adaptive_rescue_count += 1
            consecutive_novelty_rejects = 0
        end
        push!(weight_changed_hist, w_changed)
        push!(weight_change_norm_hist, Δw)

        # referência
        v_prev = v
        if ref_policy == :max
            push!(window, Fk)
            if length(window) > M+1
                deleteat!(window, 1)
            end
            rho = maximum(window)
            v = max(rho - Fk, zero(T))
            # Optional conservative decay suggested to force nu_k -> 0 and
            # make weight changes asymptotically smaller.
            if max_nu_decay_tau !== nothing && (outiter - 1) >= max_nu_decay_k0
                v = min(v, max_nu_decay_tau * v_prev)
                rho = Fk + v
            end
        elseif ref_policy == :zhanghager
            qprev = q
            rhoprev = rho
            q = zh_theta*qprev + one(T)
            rho = (zh_theta*qprev*rhoprev + Fk) / q
            rho = max(rho, Fk)
            v = max(rho - Fk, zero(T))
        else
            # Gu-Mo moving average reference:
            # rho_{k+1} = gamma*rho_k + (1-gamma)*F_{k+1}
            # and keep v >= 0 by safeguarding rho >= F_{k+1}.
            rho = gm_gamma * rho + (one(T) - gm_gamma) * Fk
            rho = max(rho, Fk)
            v = rho - Fk
        end
        push!(nu_hist, v); push!(rho_hist, rho)

        gradF!(x, w, gF); ngev += m
        gnorm = norm(gF)
        if track_iterates
            if collect
                push!(f_hist, copy(fvec))
            else
                push!(f_hist, [evalf(n, x, i) for i in 1:m])
            end
            push!(x_hist, copy(x))
        end
    end

    time_exhausted = time_limit !== nothing && elapsed_seconds() >= Float64(time_limit)
    inform = gnorm <= epsopt ? 1 : (time_exhausted ? 3 : 2)

    if collect && always_collect_final
        eval_objectives!(x, fvec); nfev += m
        force_collect!(x, fvec)
    end

    nd_filter_time = zero(T)
    pc_filter_time = zero(T)
    if collect && !isempty(collected_f)
        if collect_archive_mode == :final
            nd_filter_time = @elapsed keep = nondominated_indices(collected_f)
            for (i,k) in enumerate(keep)
                k || continue
                push!(nd_x, collected_x[i])
                push!(nd_f, collected_f[i])
            end
        else
            nd_filter_time = zero(T)
        end
        if pc_check && !isempty(nd_x)
            pc_filter_time = @elapsed begin
                nd_x_pc = Vector{Vector{T}}()
                nd_f_pc = Vector{Vector{T}}()
                for i in eachindex(nd_x)
                    check = pareto_critical_check_at(nd_x[i])
                    push!(pc_alpha_hist, check.alpha)
                    push!(pc_stationarity_hist, check.stationarity)
                    push!(pc_residual_hist, check.residual)
                    push!(pc_iteration_hist, check.iterations)
                    push!(pc_status_hist, check.status)
                    push!(pc_reason_hist, check.reason)
                    pc_checked_count += 1
                    if check.status == :accepted
                        push!(nd_x_pc, nd_x[i])
                        push!(nd_f_pc, nd_f[i])
                    end
                end
                empty!(nd_x); append!(nd_x, nd_x_pc)
                empty!(nd_f); append!(nd_f, nd_f_pc)
            end
        end
        pc_pass_count = length(nd_x)
    end

    return x, w, (
        outiter = outiter,
        nfev = nfev,
        ngev = ngev,
        gnorm = gnorm,
        inform = inform,
        policy = ref_policy,
        step2_mode = step2_mode,
        step3_mode = step3_mode,
        split_alpha = split_alpha,
        collected_x = collect ? collected_x : Vector{Vector{T}}(),
        collected_f = collect ? collected_f : Vector{Vector{T}}(),
        nd_x = collect ? nd_x : Vector{Vector{T}}(),
        nd_f = collect ? nd_f : Vector{Vector{T}}(),
        nu_hist = nu_hist,
        rho_hist = rho_hist,
        weight_changed_hist = weight_changed_hist,
        weight_change_norm_hist = weight_change_norm_hist,
        weight_change_iter = weight_change_iter,
        weight_change_norm = weight_change_norm,
        x_hist = track_iterates ? x_hist : Vector{Vector{T}}(),
        f_hist = track_iterates ? f_hist : Vector{Vector{T}}(),
        weight_changed_count = weight_changed_count,
        weight_change_trigger = weight_change_trigger,
        periodic_weight_window = periodic_weight_window,
        unit_cycle_epsilon = unit_cycle_epsilon,
        weight_accept_filter = weight_accept_filter,
        weight_acceptance = weight_acceptance,
        recent_weight_window = recent_weight_window,
        weight_novelty_tau = weight_novelty_tau,
        weight_novelty_reject_count = weight_novelty_reject_count,
        adaptive_filter_active_count = adaptive_filter_active_count,
        adaptive_rescue_count = adaptive_rescue_count,
        adaptive_burnin_iters = adaptive_burnin_iters,
        adaptive_min_initial_weight_changes = adaptive_min_initial_weight_changes,
        adaptive_reject_limit = adaptive_reject_limit,
        adaptive_rescue_iters = adaptive_rescue_iters,
        skipped_weight_change_count = skipped_weight_change_count,
        collect_accept_count = collect_accept_count,
        collect_too_close_count = collect_too_close_count,
        archive_collect_count = archive_collect_count,
        archive_collect_interval = archive_collect_interval,
        archive_collect_requires_novelty = archive_collect_requires_novelty,
        iters_since_collect_accept = iters_since_collect_accept,
        consecutive_collect_too_close = consecutive_collect_too_close,
        drsc_attempt_count = drsc_attempt_count,
        drsc_accept_count = drsc_accept_count,
        drsc_transform_count = drsc_transform_count,
        drsc_accept_rate = drsc_attempt_count > 0 ? drsc_accept_count / drsc_attempt_count : zero(T),
        collect_grad_gate = collect_grad_gate,
        archive_rule = :equivalence_filtered_v2,
        collect_alpha = collect_alpha,
        collect_atol = collect_atol,
        collect_invalid_count = collect_invalid_count,
        collect_filter_style = resolved_collect_filter_style,
        collect_archive_mode = collect_archive_mode,
        collect_metric = collect_metric,
        collect_lambda = collect_lambda,
        collect_gate_mode = collect_gate_mode,
        collect_on_weight_change = collect_on_weight_change,
        force_collect_requires_novelty = force_collect_requires_novelty,
        always_collect_final = always_collect_final,
        forced_collect_count = forced_collect_count,
        pc_check = pc_check,
        pc_sigma = pc_sigma,
        pc_tau = pc_tau,
        pc_tau_mode = pc_tau_mode,
        pc_maxiter = pc_maxiter,
        pc_checked_count = pc_checked_count,
        pc_pass_count = pc_pass_count,
        pc_rejected_count = max(0, pc_checked_count - pc_pass_count),
        pc_rule = :scalarized_gradient_norm_v1,
        pc_negative_count = count(==(:rejected), pc_status_hist),
        pc_inconclusive_count = count(==(:inconclusive), pc_status_hist),
        pc_invalid_count = count(==(:invalid), pc_status_hist),
        pc_iteration_limit_count = count(==(:iteration_limit), pc_reason_hist),
        nd_filter_time = nd_filter_time,
        pc_filter_time = pc_filter_time,
        pc_alpha_hist = pc_alpha_hist,
        pc_stationarity_hist = pc_stationarity_hist,
        pc_residual_hist = pc_residual_hist,
        pc_iteration_hist = pc_iteration_hist,
        pc_status_hist = pc_status_hist,
        pc_reason_hist = pc_reason_hist,
        preserve_extremes = preserve_extremes,
        extreme_tau = extreme_tau,
        time_limit = time_limit,
        stop_on_time_only = stop_on_time_only,
        elapsed_time = elapsed_seconds(),
    )
end
