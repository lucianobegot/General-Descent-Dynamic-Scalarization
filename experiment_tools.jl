using Printf
using Random

const GD_TOL = 1.0e-5
const SD_TOL = 1.0e-6
const ARCHIVE_ATOL = 1.0e-12

"Return reproducible weights uniformly distributed on the unit simplex."
function simplex_weights(m, count, seed)
    rng = MersenneTwister(seed)
    [begin
        w = randexp(rng, m)
        w ./ sum(w)
    end for _ in 1:count]
end

function add_reported_points!(archive, all_x, problem, solver_points, n, m)
    x, f = reported_nondominated_archive(problem, solver_points, n, m;
        atol=ARCHIVE_ATOL)
    for (point, value) in zip(x, f)
        comparison = problem in ("RE41", "RE42", "RE61") ?
            Float64.(re_normalize_objectives(problem, value)) : value
        push!(all_x, point)
        changed = add_to_nd_archive!(archive, comparison)
        if changed
            for id in archive.dominated
                all_x[id] = Float64[]
            end
        else
            all_x[end] = Float64[]
        end
    end
    return length(f)
end

function active_reported_archive(archive, all_x, problem)
    ids = findall(archive.active)
    x = all_x[ids]
    comparison = archive.points[ids]
    if problem in ("RE41", "RE42", "RE61")
        f = [Float64.(re_exact_objectives(problem, point)) for point in x]
    else
        f = comparison
    end
    return x, f
end

function warmup_method(method, problem)
    SDProject.PROBLEM = problem
    n, m, x0, _, _, _, _, _ = inip(problem, 990_001)
    if method == :GD
        GeneralDescent!(n, m, copy(x0); ref_policy=:max, M=10,
            w0=fill(1.0 / m, m), iprint=false, maxiter=1,
            epsopt=GD_TOL, weight_update=:uniform_admissible,
            split_alpha=0.0, rng=MersenneTwister(990_002), collect=false)
    elseif method == :SD
        SteepestDescent!(n, m, copy(x0); epsopt=SD_TOL,
            maxiter=1, scaleF=false, iprint=false)
    else
        SteepestDescentNonMonotone!(n, m, copy(x0); M=4,
            eta=1.0e-4, rho=0.5, initial_step=1.0, epsopt=SD_TOL,
            maxiter=1, scaleF=false, iprint=false)
    end
    method == :GD || pareto_critical_candidates([x0], n, m;
        sigma=0.1, tau=0.01, maxiter=1)
    _, sample_f = reported_nondominated_archive(problem, [x0], n, m;
        atol=ARCHIVE_ATOL)
    state = PersistentNDArchive(Float64, m; atol=ARCHIVE_ATOL)
    isempty(sample_f) || add_to_nd_archive!(state, first(sample_f))
    active_archive(state)
    return nothing
end

"Run one problem, method, and independent block with the paper settings."
function run_case(method::Symbol, problem::String, block::Int;
                  base_seed=20260210, nstarts=50, nweights=6, sd_runs=300,
                  maxiter=20_000, iprint=false, warmup=true, keep_stats=false)
    method in (:GD, :SD, :SDNM) || error("Method must be :GD, :SD, or :SDNM.")
    block >= 1 || error("Block must be positive.")
    SDProject.PROBLEM = problem
    set_scalable_box_penalty!(1.0e10)
    warmup && warmup_method(method, problem)

    block_seed = base_seed + 100_000 * (block - 1)
    n, m, _, _, _, _, _, _ = inip(problem, block_seed)
    archive = PersistentNDArchive(Float64, m; atol=ARCHIVE_ATOL)
    all_x = Vector{Vector{Float64}}()
    stats = Any[]
    candidates = converged = 0
    solver_time = check_time = archive_time = 0.0
    wall_start = time_ns()

    if method == :GD
        for start_index in 1:nstarts
            start_seed = block_seed + start_index - 1
            n, m, x0, _, _, _, _, _ = inip(problem, start_seed)
            weight_seed = start_seed + 31_337 + 1000m
            weights = simplex_weights(m, nweights, weight_seed)
            for (weight_index, w0) in enumerate(weights)
                rng_seed = 20260801 + 1_000_000 * block + 10_000 * start_index + weight_index
                _, _, run_stats = GeneralDescent!(n, m, copy(x0);
                    ref_policy=:max, M=10, w0=w0, iprint=iprint,
                    maxiter=maxiter, epsopt=GD_TOL,
                    weight_update=:uniform_admissible,
                    weight_acceptance=:descent,
                    step2_mode=:reference, step3_mode=:nonmonotone,
                    split_alpha=0.0, rng=MersenneTwister(rng_seed),
                    collect=true, collect_filter_style=:during,
                    collect_alpha=0.003, collect_atol=ARCHIVE_ATOL,
                    collect_metric=:component_span,
                    collect_grad_gate=true, collect_lambda=1.0e-2,
                    collect_gate_mode=:absolute,
                    collect_on_weight_change=true,
                    force_collect_requires_novelty=false,
                    always_collect_final=true,
                    pc_check=true, pc_sigma=1.0e-1, pc_tau=1.0e-2,
                    pc_tau_mode=:relative_grad_scale2, pc_maxiter=2000)
                solver_time += run_stats.elapsed_time
                local_count = 0
                archive_time += @elapsed local_count = add_reported_points!(
                    archive, all_x, problem, run_stats.nd_x, n, m)
                candidates += local_count
                converged += run_stats.inform == 1
                keep_stats && push!(stats, run_stats)
            end
        end
    else
        for run_index in 1:sd_runs
            start_seed = block_seed + run_index - 1
            n, m, x0, _, _, _, _, _ = inip(problem, start_seed)
            solution, run_stats = if method == :SD
                SteepestDescent!(n, m, copy(x0); epsopt=SD_TOL,
                    maxiter=maxiter, scaleF=false, iprint=iprint)
            else
                SteepestDescentNonMonotone!(n, m, copy(x0);
                    M=4, eta=1.0e-4, rho=0.5, initial_step=1.0,
                    epsopt=SD_TOL, maxiter=maxiter, scaleF=false, iprint=iprint)
            end
            solver_time += run_stats.time
            checked = Vector{Vector{Float64}}()
            quality = nothing
            check_time += @elapsed checked, quality = pareto_critical_candidates(
                [solution], n, m; sigma=0.1, tau=0.01, maxiter=2000)
            local_count = 0
            archive_time += @elapsed local_count = add_reported_points!(
                archive, all_x, problem, checked, n, m)
            candidates += local_count
            converged += run_stats.inform == 1
            keep_stats && push!(stats, run_stats)
        end
    end

    x, f = active_reported_archive(archive, all_x, problem)
    runs = method == :GD ? nstarts * nweights : sd_runs
    wall_time = (time_ns() - wall_start) / 1.0e9
    total_time = solver_time + check_time + archive_time
    return (method=method, problem=problem, block=block, n=n, m=m,
        x=x, f=f, stats=stats, runs=runs, candidates=candidates,
        converged=converged, solver_time=solver_time,
        check_time=check_time, archive_time=archive_time,
        total_time=total_time, wall_time=wall_time)
end

function write_case(output, result)
    mkpath(output)
    open(joinpath(output, "archive.csv"), "w") do io
        println(io, join(vcat(["x$i" for i in 1:result.n],
                             ["f$i" for i in 1:result.m]), ','))
        for (x, f) in zip(result.x, result.f)
            println(io, join(repr.(vcat(x, f)), ','))
        end
    end
    summary_path = joinpath(output, "summary.toml")
    open(summary_path, "w") do io
        println(io, "status = \"complete\"")
        println(io, "problem = \"$(result.problem)\"")
        println(io, "method = \"$(result.method)\"")
        println(io, "block = $(result.block)")
        println(io, "n = $(result.n)")
        println(io, "m = $(result.m)")
        println(io, "runs = $(result.runs)")
        println(io, "converged = $(result.converged)")
        println(io, "candidates = $(result.candidates)")
        println(io, "nondominated = $(length(result.f))")
        println(io, "solver_time = $(repr(result.solver_time))")
        println(io, "check_time = $(repr(result.check_time))")
        println(io, "archive_time = $(repr(result.archive_time))")
        println(io, "total_time = $(repr(result.total_time))")
        println(io, "wall_time = $(repr(result.wall_time))")
    end
    return summary_path
end
