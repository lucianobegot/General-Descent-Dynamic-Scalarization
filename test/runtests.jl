using Test
using Random
using SDProject
include(joinpath(@__DIR__, "..", "experiment_tools.jl"))

@testset "Simplex weights" begin
    weights = simplex_weights(3, 5, 123)
    @test length(weights) == 5
    @test all(w -> all(w .>= 0) && sum(w) ≈ 1.0, weights)
    @test weights == simplex_weights(3, 5, 123)
end

@testset "Persistent archive" begin
    points = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.0, 2.0]]
    state = PersistentNDArchive(Float64, 2)
    for point in points
        add_to_nd_archive!(state, point)
    end
    @test active_archive(state) == [[1.0, 2.0], [2.0, 1.0]]
    @test sequential_nondominated_indices(points) == [1, 2]

    rng = MersenneTwister(42)
    random_points = [rand(rng, 3) for _ in 1:100]
    reference = Vector{Vector{Float64}}()
    for point in random_points
        blocked = any(old -> all(old .<= point) || all(abs.(old .- point) .<= 1e-12), reference)
        blocked && continue
        filter!(old -> !all(point .<= old), reference)
        push!(reference, point)
    end
    fast = random_points[sequential_nondominated_indices(random_points)]
    @test Set(Tuple.(fast)) == Set(Tuple.(reference))
end

@testset "Three methods" begin
    for method in (:GD, :SD, :SDNM)
        result = run_case(method, "AP2", 1;
            nstarts=1, nweights=1, sd_runs=1, maxiter=1)
        @test result.runs == 1
        @test all(point -> all(isfinite, point), result.f)
        @test all(length(point) == 2 for point in result.f)
    end
end

@testset "Representative domains" begin
    for problem in ("DGO2", "DTLZ2_M5", "DTLZ3_M10")
        result = run_case(:GD, problem, 1;
            nstarts=1, nweights=1, maxiter=1)
        @test result.runs == 1
        @test all(point -> all(isfinite, point), result.f)
        @test all(length(point) == result.m for point in result.f)
    end
end

@testset "All campaign problems initialize" begin
    problems = [
        "AP2", "AP3", "BK1", "DGO1M", "DGO2", "Far1", "FF1", "JOS1",
        "MOP2", "MOP3", "QV1", "SP1", "ZDT2", "MLF1", "AP1", "AP4",
        "FDS", "IKK1", "LTDZ", "MHHM2", "MOP5", "MGH16", "MGH33",
        "ZLT1", "MOP7", "SLCDT2", "UF8", "UF10", "DTLZ2_M5",
        "DTLZ3_M5", "DTLZ4_M5", "DTLZ7_M5", "DTLZ3_M10",
    ]
    for problem in problems
        SDProject.PROBLEM = problem
        n, m, x, _, _, _, _, _ = inip(problem, 20260210)
        values = [SDProject.evalf(n, x, i) for i in 1:m]
        @test length(x) == n
        @test all(isfinite, values)
    end
end
