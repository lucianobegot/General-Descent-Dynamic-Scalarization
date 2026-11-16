using Pkg
Pkg.activate(@__DIR__)

using Statistics
using TOML
using SDProject

root = abspath(isempty(ARGS) || startswith(ARGS[1], "--") ?
    joinpath(@__DIR__, "results", "paper_main_33") : ARGS[1])
skip_hv = "--skip-hv" in ARGS
config = TOML.parsefile(joinpath(root, "campaign.toml"))
problems = String.(config["campaign"]["problems"])
methods = String.(config["campaign"]["methods"])
blocks = 1:Int(config["campaign"]["blocks"])

function read_archive(path, n, m)
    lines = readlines(path)
    isempty(lines) && return Vector{Vector{Float64}}()
    [parse.(Float64, split(line, ','))[n+1:n+m] for line in lines[2:end] if !isempty(line)]
end

function gamma_metric(front, lower, upper)
    isempty(front) && return Inf
    maximum(begin
        span = max(upper[j] - lower[j], eps(Float64))
        values = sort([(point[j] - lower[j]) / span for point in front])
        maximum(diff(vcat(0.0, values, 1.0)))
    end for j in eachindex(lower))
end

records = NamedTuple[]
for problem in problems, block in blocks
    cases = Dict{String,Any}()
    union_front = Vector{Vector{Float64}}()
    for method in methods
        directory = joinpath(root, problem, "block_$block", lowercase(method))
        summary_path = joinpath(directory, "summary.toml")
        isfile(summary_path) || continue
        summary = TOML.parsefile(summary_path)
        get(summary, "status", "") == "complete" || continue
        front = read_archive(joinpath(directory, "archive.csv"),
            Int(summary["n"]), Int(summary["m"]))
        cases[method] = (summary=summary, front=front)
        append!(union_front, front)
    end
    isempty(cases) && continue
    isempty(union_front) && error("All reported archives are empty for $problem, block $block")
    m = length(first(union_front))
    lower = [minimum(point[j] for point in union_front) for j in 1:m]
    upper = [maximum(point[j] for point in union_front) for j in 1:m]
    reference = !skip_hv && m <= 3 && problem != "DGO2" ?
        hypervolume_reference(union_front) : nothing
    for method in methods
        haskey(cases, method) || continue
        summary, front = cases[method]
        total = Float64(summary["total_time"])
        solver = Float64(summary["solver_time"])
        hv = reference === nothing ? missing : hypervolume_metric(front, reference)
        push!(records, (problem=problem, block=block, method=method,
            nd=length(front), candidates=Int(summary["candidates"]),
            solver_time=solver,
            check_time=Float64(summary["check_time"]),
            archive_time=Float64(summary["archive_time"]), total_time=total,
            nd_per_solver_second=length(front) / max(solver, eps()),
            nd_per_total_second=length(front) / max(total, eps()),
            gamma=gamma_metric(front, lower, upper), hypervolume=hv))
    end
end

isempty(records) && error("No complete cases found in $root")

block_path = joinpath(root, "block_metrics.csv")
open(block_path, "w") do io
    println(io, "problem,block,method,ND,candidates,solver_time,check_time,archive_time,total_time,ND_per_solver_second,ND_per_total_second,Gamma,hypervolume")
    for r in records
        hv = ismissing(r.hypervolume) ? "" : repr(r.hypervolume)
        println(io, join((r.problem, r.block, r.method, r.nd, r.candidates,
            r.solver_time, r.check_time, r.archive_time, r.total_time, r.nd_per_solver_second,
            r.nd_per_total_second, r.gamma, hv), ','))
    end
end

aggregate_path = joinpath(root, "aggregate_metrics.csv")
open(aggregate_path, "w") do io
    println(io, "problem,method,observations,ND_mean,ND_sd,solver_time_mean,solver_time_sd,check_time_mean,check_time_sd,archive_time_mean,archive_time_sd,total_time_mean,total_time_sd,ND_per_solver_second_mean,ND_per_solver_second_sd,ND_per_total_second_mean,ND_per_total_second_sd,Gamma_mean,Gamma_sd,hypervolume_mean,hypervolume_sd")
    for problem in problems, method in methods
        rows = [r for r in records if r.problem == problem && r.method == method]
        isempty(rows) && continue
        stat(values) = length(values) == 1 ? (mean(values), 0.0) : (mean(values), std(values))
        nd = stat(Float64[r.nd for r in rows])
        solver = stat([r.solver_time for r in rows])
        check = stat([r.check_time for r in rows])
        archive = stat([r.archive_time for r in rows])
        total = stat([r.total_time for r in rows])
        solver_rate = stat([r.nd_per_solver_second for r in rows])
        rate = stat([r.nd_per_total_second for r in rows])
        gamma = stat([r.gamma for r in rows])
        hvs = Float64[r.hypervolume for r in rows if !ismissing(r.hypervolume)]
        hv = isempty(hvs) ? (missing, missing) : stat(hvs)
        println(io, join((problem, method, length(rows), nd[1], nd[2],
            solver[1], solver[2], check[1], check[2], archive[1], archive[2],
            total[1], total[2], solver_rate[1], solver_rate[2],
            rate[1], rate[2], gamma[1], gamma[2],
            ismissing(hv[1]) ? "" : hv[1], ismissing(hv[2]) ? "" : hv[2]), ','))
    end
end

println("Wrote: ", block_path)
println("Wrote: ", aggregate_path)
