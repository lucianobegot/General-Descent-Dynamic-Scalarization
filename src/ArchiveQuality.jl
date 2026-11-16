"""
    pareto_critical_candidates(points, n, m; sigma=0.1, tau=0.01, maxiter=2000)

Apply the collection protocol's relative first-order check in solver coordinates.
Set `PROBLEM` and initialize it with `inip` first. This does not certify convergence
of the originating run or criticality of the original nonsmooth RE objectives.
Convert the retained points to reported objectives and refilter afterwards.
The returned diagnostics include elapsed checking time, outside solver time.
"""
function pareto_critical_candidates(points, n::Int, m::Int;
                                    sigma=0.1, tau=0.01, maxiter=2000)
    project(v) = begin
        u = sort(v; rev=true)
        c = cumsum(u) .- 1
        k = findlast(u .> c ./ (1:length(u)))
        max.(v .- c[k] / k, 0)
    end
    retained = Vector{Vector{Float64}}()
    checked = rejected = invalid = limited = ngev = 0
    elapsed = @elapsed for point in points
        length(point) == n || throw(DimensionMismatch("Unexpected decision dimension."))
        x = Float64.(point)
        checked += 1
        if !all(isfinite, x)
            invalid += 1
            continue
        end
        G = Matrix{Float64}(undef, n, m)
        g = zeros(n)
        valid = true
        for i in 1:m
            ngev += 1
            flag = try
                _, result = sevalg!(n, x, g, i, 1.0)
                result
            catch err
                err isa DomainError || rethrow()
                -1
            end
            if flag != 1 || !all(isfinite, g)
                valid = false
                break
            end
            G[:, i] .= g
        end
        if !valid
            invalid += 1
            continue
        end
        check = _gd_pareto_check(G, project; sigma=Float64(sigma),
            tau=Float64(tau), relative=true, maxiter=Int(maxiter))
        limited += check.reason == :iteration_limit
        if check.status == :accepted
            push!(retained, x)
        elseif check.status == :rejected
            rejected += 1
        else
            invalid += 1
        end
    end
    return retained, (checked=checked, accepted=length(retained), rejected=rejected,
        invalid=invalid, iteration_limits=limited, ngev=ngev, time=elapsed)
end
