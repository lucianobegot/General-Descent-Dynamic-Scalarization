function dominates_point(a::AbstractVector{T}, b::AbstractVector{T}; atol::T=T(1e-12)) where {T<:AbstractFloat}
    all_le = true
    strict = false
    @inbounds for i in eachindex(a)
        if a[i] > b[i] + atol
            all_le = false
            break
        end
        if a[i] < b[i] - atol
            strict = true
        end
    end
    return all_le && strict
end

function point_relation(a::AbstractVector{T}, b::AbstractVector{T}; atol::T=T(1e-12)) where {T<:AbstractFloat}
    a_can_dominate = true
    b_can_dominate = true
    a_strict = false
    b_strict = false
    equivalent = true

    @inbounds for i in eachindex(a, b)
        diff = a[i] - b[i]
        if diff > atol
            equivalent = false
            a_can_dominate = false
            b_strict = true
        elseif diff < -atol
            equivalent = false
            b_can_dominate = false
            a_strict = true
        end
    end

    equivalent && return :equivalent
    a_can_dominate && a_strict && return :a_dominates
    b_can_dominate && b_strict && return :b_dominates
    return :incomparable
end

"""
Insert a finite objective vector into a minimization archive, storing a copy.
Equivalent vectors keep the first retained representative. Return false for
duplicates, dominated points, or nonfinite candidates. Existing entries must
be finite vectors of the same dimension, maintained through this function.
"""
function add_to_nd_archive!(archive::Vector{Vector{T}}, z::Vector{T}; atol::T=T(1e-12)) where {T<:AbstractFloat}
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative."))
    isempty(z) && throw(ArgumentError("An objective vector cannot be empty."))
    isempty(archive) || length(first(archive)) == length(z) || throw(DimensionMismatch("Objective dimensions differ."))
    all(isfinite, z) || return false
    remove_idx = Int[]
    @inbounds for i in eachindex(archive)
        y = archive[i]
        relation = point_relation(y, z; atol=atol)
        if relation == :equivalent || relation == :a_dominates
            return false
        elseif relation == :b_dominates
            push!(remove_idx, i)
        end
    end
    for i in reverse(remove_idx)
        deleteat!(archive, i)
    end
    push!(archive, copy(z))
    return true
end

# Spatial index used by the default global archive reducer.  Bounding boxes
# are deliberately not contracted after deletions: they may become
# conservative, but can never prune a comparison that is still required.
mutable struct _NDKDNode{T<:AbstractFloat}
    lo::Vector{T}
    hi::Vector{T}
    ids::Vector{Int}
    left::Union{Nothing,_NDKDNode{T}}
    right::Union{Nothing,_NDKDNode{T}}
    split_dim::Int
    split_value::T
end

function _ndkd_leaf(points::Vector{Vector{T}}, ids::Vector{Int}, m::Int) where {T<:AbstractFloat}
    lo, hi = fill(T(Inf), m), fill(T(-Inf), m)
    for id in ids, k in 1:m
        lo[k] = min(lo[k], points[id][k])
        hi[k] = max(hi[k], points[id][k])
    end
    return _NDKDNode{T}(lo, hi, ids, nothing, nothing, 0, zero(T))
end

function _ndkd_insert!(node::_NDKDNode{T}, points::Vector{Vector{T}}, id::Int;
                       leaf_size::Int=32) where {T<:AbstractFloat}
    point = points[id]
    @inbounds for k in eachindex(point)
        node.lo[k] = min(node.lo[k], point[k])
        node.hi[k] = max(node.hi[k], point[k])
    end
    if node.left === nothing
        push!(node.ids, id)
        length(node.ids) <= leaf_size && return node
        widths = node.hi .- node.lo
        dim = argmax(widths)
        iszero(widths[dim]) && return node
        sort!(node.ids; by=i -> points[i][dim], alg=Base.Sort.MergeSort)
        middle = div(length(node.ids), 2)
        left_ids, right_ids = node.ids[1:middle], node.ids[middle+1:end]
        node.left = _ndkd_leaf(points, left_ids, length(point))
        node.right = _ndkd_leaf(points, right_ids, length(point))
        node.split_dim = dim
        node.split_value = points[left_ids[end]][dim]
        empty!(node.ids)
        return node
    end
    child = point[node.split_dim] <= node.split_value ? node.left : node.right
    _ndkd_insert!(something(child), points, id; leaf_size)
    return node
end

@inline function _ndkd_may_block(node::_NDKDNode, z, atol)
    @inbounds for k in eachindex(z)
        node.lo[k] <= z[k] + atol || return false
    end
    return true
end

function _ndkd_has_blocker(node::_NDKDNode{T}, points, active, z,
                           atol::T) where {T<:AbstractFloat}
    _ndkd_may_block(node, z, atol) || return false
    if node.left === nothing
        for id in node.ids
            active[id] || continue
            point_relation(points[id], z; atol) in (:equivalent, :a_dominates) && return true
        end
        return false
    end
    return _ndkd_has_blocker(something(node.left), points, active, z, atol) ||
           _ndkd_has_blocker(something(node.right), points, active, z, atol)
end

@inline function _ndkd_may_be_dominated(node::_NDKDNode, z, atol)
    @inbounds for k in eachindex(z)
        node.hi[k] >= z[k] - atol || return false
    end
    return true
end

function _ndkd_dominated!(found, node::_NDKDNode{T}, points, active, z,
                          atol::T) where {T<:AbstractFloat}
    _ndkd_may_be_dominated(node, z, atol) || return found
    if node.left === nothing
        for id in node.ids
            active[id] || continue
            point_relation(points[id], z; atol) == :b_dominates && push!(found, id)
        end
        return found
    end
    _ndkd_dominated!(found, something(node.left), points, active, z, atol)
    _ndkd_dominated!(found, something(node.right), points, active, z, atol)
    return found
end

"""
    PersistentNDArchive(Float64, m; atol=1e-12)

Stateful nondominated archive for completed run-level archives.  Insertion
uses the same order, tolerance, and first-representative semantics as
`add_to_nd_archive!`, while a k-d index accelerates blocker and deletion
queries.  It is intended for campaign-level reduction, after each run has
completed its local Pareto-critical check.
"""
mutable struct PersistentNDArchive{T<:AbstractFloat}
    points::Vector{Vector{T}}
    active::BitVector
    root::_NDKDNode{T}
    dominated::Vector{Int}
    atol::T
end

function PersistentNDArchive(::Type{T}, m::Int; atol::T=T(1e-12)) where {T<:AbstractFloat}
    m > 0 || throw(ArgumentError("The objective dimension must be positive."))
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative."))
    points = Vector{Vector{T}}()
    return PersistentNDArchive{T}(points, falses(0), _ndkd_leaf(points, Int[], m), Int[], atol)
end

function PersistentNDArchive(points::Vector{Vector{T}};
                             atol::T=T(1e-12)) where {T<:AbstractFloat}
    isempty(points) && throw(ArgumentError("Use PersistentNDArchive(T, m) for an empty checkpoint."))
    state = PersistentNDArchive(T, length(first(points)); atol)
    for point in points
        add_to_nd_archive!(state, point)
    end
    return state
end

function add_to_nd_archive!(state::PersistentNDArchive{T}, z::Vector{T}) where {T<:AbstractFloat}
    length(z) == length(state.root.lo) || throw(DimensionMismatch("Objective dimensions differ."))
    all(isfinite, z) || return false
    point = copy(z)
    push!(state.points, point)
    push!(state.active, false)
    id = length(state.points)
    _ndkd_has_blocker(state.root, state.points, state.active, point, state.atol) && return false
    empty!(state.dominated)
    _ndkd_dominated!(state.dominated, state.root, state.points, state.active, point, state.atol)
    state.active[state.dominated] .= false
    state.active[id] = true
    _ndkd_insert!(state.root, state.points, id)
    return true
end

"""Return copies of active points in their canonical insertion order."""
active_archive(state::PersistentNDArchive) = copy.(state.points[findall(state.active)])

"""
    sequential_nondominated_indices(points; atol=1e-12)

Return the input indices retained by sequential insertion in the given order.
The k-d implementation preserves the tolerance and first-representative
semantics of `add_to_nd_archive!`.
"""
function sequential_nondominated_indices(points::Vector{Vector{T}};
                                         atol::T=T(1e-12)) where {T<:AbstractFloat}
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative."))
    isempty(points) && return Int[]
    m = length(first(points))
    m > 0 || throw(ArgumentError("Objective vectors cannot be empty."))
    all(length(point) == m for point in points) || throw(DimensionMismatch("Objective dimensions differ."))
    state = PersistentNDArchive(T, m; atol)
    for point in points
        add_to_nd_archive!(state, point)
    end
    return findall(state.active)
end
