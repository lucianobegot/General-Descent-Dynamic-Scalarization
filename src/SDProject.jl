module SDProject

# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------
using LinearAlgebra
using Printf
using Random
using ForwardDiff
using QuadraticModels
using RipQP

# ----------------------------------------------------------------------
# Type definitions and global settings
# ----------------------------------------------------------------------

# Floating-point type (allows easy switching to higher precision)
const T = Float64

# Problem name identifier
PROBLEM = ""

# ----------------------------------------------------------------------
# Include all source files (each file defines one module or set of functions)
# ----------------------------------------------------------------------
include("AlgConst.jl")        # Algorithmic constants and default parameters
include("ScalableBenchmarks.jl") # Smooth scalable multiobjective benchmarks
include("MyProblem.jl")       # Problem definition and initialization routines
include("CheckDer.jl")        # Derivative and consistency checks
include("ScaleFactor.jl")     # Scaling utilities for the objectives
include("SafeEval.jl")        # Safe evaluation wrappers (handle NaNs/infs)
include("NDArchive.jl")       # Incremental non-dominated archive utilities
include("HypervolumeMetric.jl") # Hypervolume used by the analysis script
include("evalSD.jl")          # Evaluation of steepest descent direction
include("armijo.jl")          # Armijo line search implementation
include("SteepestDescent.jl") # Main steepest descent algorithm
include("SteepestDescentNonMonotone.jl") # Max-type nonmonotone SD
include("GeneralDescent.jl")  # General descent via scalarization (nonmonotone LS)
include("ArchiveQuality.jl")  # Common first-order filter for terminal points

# ----------------------------------------------------------------------
# Export public functions
# ----------------------------------------------------------------------
export SteepestDescent!, SteepestDescentNonMonotone!, inip, checkdF, GeneralDescent!,
       add_to_nd_archive!, dominates_point, point_relation,
       PersistentNDArchive, active_archive, sequential_nondominated_indices,
       objective_matrix, hypervolume_reference, hypervolume_metric,
       pareto_critical_candidates,
       set_re_smoothing_relative!, re_problem_bounds, re_reference_ideal_nadir,
       re_constraint_values, re_exact_objectives, re_smoothed_objectives,
       re_normalize_objectives, re_physical_point, re_internal_point,
       reported_nondominated_archive,
       is_scalable_benchmark, set_scalable_box_penalty!, scalable_problem_dimensions,
       scalable_problem_bounds, scalable_box_violation, scalable_raw_evalf,
       scalable_raw_evalg!, scalable_evalf, scalable_evalg!, scalable_raw_objectives,
       scalable_objectives

end # module SDProject
