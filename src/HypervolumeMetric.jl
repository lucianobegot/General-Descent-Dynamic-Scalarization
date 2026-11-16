import Metaheuristics.PerformanceIndicators: hypervolume

function objective_matrix(points::Vector{Vector{T}}) where {T<:AbstractFloat}
    isempty(points) && return Matrix{T}(undef, 0, 0)
    m = length(first(points))
    A = Matrix{T}(undef, length(points), m)
    @inbounds for i in eachindex(points), j in 1:m
        A[i, j] = points[i][j]
    end
    return A
end

function hypervolume_reference(points::Vector{Vector{T}}; margin::T=T(0.1)) where {T<:AbstractFloat}
    isempty(points) && return T[]
    m = length(first(points))
    zmin = [minimum(p[j] for p in points) for j in 1:m]
    zmax = [maximum(p[j] for p in points) for j in 1:m]
    ref = similar(zmax)
    @inbounds for j in 1:m
        span = zmax[j] - zmin[j]
        pad = if isfinite(span) && span > eps(T)
            margin * span
        else
            margin * max(abs(zmax[j]), one(T))
        end
        ref[j] = zmax[j] + pad
    end
    return ref
end

function hypervolume_metric(points::Vector{Vector{T}}, reference_point::AbstractVector{T}) where {T<:AbstractFloat}
    isempty(points) && return zero(T)
    front = [p for p in points if all(isfinite, p) && all(p[j] <= reference_point[j] for j in eachindex(reference_point))]
    isempty(front) && return zero(T)
    return T(hypervolume(objective_matrix(front), collect(reference_point); verbose=false))
end
