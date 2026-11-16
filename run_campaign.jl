using Pkg
Pkg.activate(@__DIR__)

using TOML
using SHA
using SDProject
include(joinpath(@__DIR__, "experiment_tools.jl"))

function option(name, default=nothing)
    prefix = "--$name="
    value = findfirst(arg -> startswith(arg, prefix), ARGS)
    value === nothing ? default : split(ARGS[value], '='; limit=2)[2]
end

function selected(values, text, label)
    text === nothing && return values
    requested = split(text, ',')
    unknown = setdiff(requested, values)
    isempty(unknown) || error("Unknown $label: $(join(unknown, ", "))")
    return [value for value in values if value in requested]
end

config_index = findfirst(arg -> !startswith(arg, "--"), ARGS)
config_path = config_index === nothing ? joinpath(@__DIR__, "paper_campaign.toml") :
    abspath(ARGS[config_index])
config = TOML.parsefile(config_path)
campaign = config["campaign"]
budget = config["budget"]

problems = selected(String.(campaign["problems"]), option("problems"), "problem")
methods = selected(String.(campaign["methods"]), option("methods"), "method")
blocks = collect(1:Int(campaign["blocks"]))
block_text = option("blocks")
block_text === nothing || (blocks = parse.(Int, split(block_text, ',')))
output = abspath(joinpath(@__DIR__, option("output", String(campaign["output"]))))

smoke = "--smoke" in ARGS
if smoke
    problems = option("problems") === nothing ? ["AP2"] : problems
    blocks = [1]
    output = abspath(joinpath(@__DIR__, option("output", "results/smoke")))
end

gd_starts = smoke ? 1 : Int(budget["gd_starts"])
gd_weights = smoke ? 1 : Int(budget["gd_weights"])
sd_runs = smoke ? 1 : Int(budget["sd_runs"])
maxiter = smoke ? 2 : Int(budget["max_iterations"])
total_cases = length(problems) * length(methods) * length(blocks)

println("Campaign: ", campaign["name"])
println("Problems: ", join(problems, ", "))
println("Methods: ", join(methods, ", "))
println("Blocks: ", join(blocks, ", "))
println("Cases: ", total_cases)
println("Output: ", output)
println("Budget: GD=$(gd_weights)x$(gd_starts), SD/SD-NM=$(sd_runs), maxiter=$(maxiter)")

"--dry-run" in ARGS && exit()
mkpath(output)
snapshot = joinpath(output, "campaign.toml")
if isfile(snapshot) && read(snapshot, String) != read(config_path, String)
    error("The output already belongs to a different configuration. Choose another --output.")
end
isfile(snapshot) || cp(config_path, snapshot)

source_files = vcat([joinpath(@__DIR__, "experiment_tools.jl")],
    sort(filter(isfile, readdir(joinpath(@__DIR__, "src"); join=true))))
source_hash = bytes2hex(sha256(reduce(vcat, read.(source_files))))
hash_path = joinpath(output, "source.sha256")
if isfile(hash_path) && strip(read(hash_path, String)) != source_hash
    error("The source changed inside an existing output. Choose another --output.")
end
isfile(hash_path) || write(hash_path, source_hash * "\n")

for problem in problems, block in blocks, method_name in methods
    method = Symbol(method_name)
    case_dir = joinpath(output, problem, "block_$block", lowercase(method_name))
    summary_path = joinpath(case_dir, "summary.toml")
    if isfile(summary_path) && get(TOML.parsefile(summary_path), "status", "") == "complete"
        println("Skipping completed case: $problem block=$block method=$method_name")
        continue
    end
    println("Running: $problem block=$block method=$method_name")
    try
        result = run_case(method, problem, block;
            base_seed=Int(campaign["base_seed"]),
            nstarts=gd_starts, nweights=gd_weights, sd_runs=sd_runs,
            maxiter=maxiter, iprint=false)
        write_case(case_dir, result)
        println("  ND=$(length(result.f)) time=$(round(result.total_time; digits=3)) s")
    catch error
        mkpath(case_dir)
        open(summary_path, "w") do io
            println(io, "status = \"failed\"")
            println(io, "problem = \"$problem\"")
            println(io, "method = \"$method_name\"")
            println(io, "block = $block")
            println(io, "error = ", repr(sprint(showerror, error)))
        end
        rethrow()
    end
end

println("Campaign cases completed. Run analyze_campaign.jl next.")
