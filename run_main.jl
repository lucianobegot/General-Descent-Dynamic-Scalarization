using Pkg
Pkg.activate(@__DIR__)

using SDProject
include(joinpath(@__DIR__, "experiment_tools.jl"))

# A short example. Change only these values.
METHOD = :GD                 # :GD, :SD, or :SDNM
PROBLEM = "FDS"
SEED = 2025
NSTARTS = 1
NWEIGHTS = 1                # Used only by GD
MAXITER = 20_000

result = run_case(METHOD, PROBLEM, 1;
    base_seed=SEED,
    nstarts=NSTARTS,
    nweights=NWEIGHTS,
    sd_runs=NSTARTS,
    maxiter=MAXITER,
    iprint=true,
    keep_stats=true,
)

println("\nExecutions: ", result.runs)
println("Stopping tests satisfied: ", result.converged, "/", result.runs)
println("Nondominated points: ", length(result.f))

# `result.x`, `result.f`, and `result.stats` remain available in the session.
