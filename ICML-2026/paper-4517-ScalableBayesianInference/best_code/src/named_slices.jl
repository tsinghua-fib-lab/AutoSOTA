import Base: names
export Layout, layout, State, indices, nvars

"""
    Layout(; f=100, f_dx=100, f_int=99)
    Layout((:f=>100, :f_dx=>100, :f_int=>99))
    layout(nt::NamedTuple)

Define a 1D layout by giving lengths per name. Names become symbols; values are positive Ints.
"""
struct Layout
    names  :: Vector{Symbol}
    ranges :: Dict{Symbol, UnitRange{Int}}
    total  :: Int
end

function Layout(pairs::Pair{<:Any,<:Integer}...)
    nt = NamedTuple{Tuple(Symbol.(first.(pairs))...)}(last.(pairs))
    layout(nt)
end

function layout(nt::NamedTuple)
    nms  = collect(keys(nt)) .|> Symbol
    lens = collect(values(nt)) .|> Int
    @assert all(>(0), lens) "All layout sizes must be positive."
    ranges = Dict{Symbol,UnitRange{Int}}()
    offset = 0
    for (name, len) in zip(nms, lens)
        r = (offset+1):(offset+len)
        ranges[name] = r
        offset += len
    end
    Layout(nms, ranges, offset)
end

Base.length(L::Layout) = L.total
names(L::Layout) = copy(L.names)
nvars(L::Layout) = length(L.names)
indices(L::Layout, s::Symbol) = get(L.ranges, s) do
    throw(ArgumentError("Name $s not in layout. Known: $(L.names)"))
end

"""
    State(x::AbstractVector, L::Layout)

Vector wrapper with named, view-based slices via dot access.
Behaves as an `AbstractVector` (supports indexing, broadcasting, etc.).
"""
struct State{T,V<:AbstractVector{T}} <: AbstractVector{T}
    data   :: V
    layout :: Layout
    function State{T,V}(x::V, L::Layout) where {T,V<:AbstractVector{T}}
        @assert length(x) == length(L) "Vector length $(length(x)) must match layout length $(length(L))."
        new{T,V}(x, L)
    end
end
State(x::AbstractVector, L::Layout) = State{eltype(x), typeof(x)}(x, L)

# ---- AbstractVector interface ----
Base.eltype(::Type{State{T}}) where {T} = T
Base.size(s::State) = (length(s.data),)
Base.length(s::State) = length(s.data)
Base.axes(s::State) = (Base.OneTo(length(s)),)
Base.IndexStyle(::Type{<:State}) = Base.IndexLinear()

Base.getindex(s::State, i::Int) = s.data[i]
Base.getindex(s::State, r::UnitRange{Int}) = @view s.data[r]
Base.getindex(s::State, I) = s.data[I]  # fallback for vectors of Int, logical masks, etc.
Base.setindex!(s::State, v, i::Int) = (s.data[i] = v)
Base.setindex!(s::State, v, r::UnitRange{Int}) = (s.data[r] .= v; v)
Base.setindex!(s::State, v, I) = (s.data[I] = v)

Base.iterate(s::State) = iterate(s.data)
Base.iterate(s::State, st) = iterate(s.data, st)
Base.eachindex(s::State) = eachindex(s.data)
Base.parent(s::State) = s.data
Base.similar(s::State, ::Type{T}=eltype(s), dims::Union{Int,Tuple} = (length(s),)) where {T} =
    similar(s.data, T, dims)
Base.copy(s::State) = State(copy(s.data), s.layout)

# ---- Pretty printing ----
function Base.show(io::IO, s::State)
    print(io, "State($(length(s))-el, names=$(join(s.layout.names, ", ")))")
end

# ---- Named property access (views into the right ranges) ----
function Base.propertynames(s::State, private::Bool=false)
    (private ? (:data, :layout) : ()) |> x -> (x..., s.layout.names...)
end

function Base.getproperty(s::State, sym::Symbol)
    if sym === :data || sym === :layout
        return getfield(s, sym)
    end
    r = get(s.layout.ranges, sym) do
        getfield(s, sym)  # triggers normal field error if it exists
    end
    @view s.data[r]
end

function Base.setproperty!(s::State, sym::Symbol, val)
    if sym === :data || sym === :layout
        return setfield!(s, sym, val)
    end
    r = indices(s.layout, sym)
    @assert length(val) == length(r) "Assigning $(length(val)) elements into $(sym) of length $(length(r))."
    s.data[r] .= val
    val
end

# ---- Convenience with Symbols ----
Base.names(s::State) = names(s.layout)
indices(s::State, sym::Symbol) = indices(s.layout, sym)
Base.getindex(s::State, sym::Symbol) = @view s.data[indices(s.layout, sym)]
Base.getindex(s::State, syms::AbstractVector{Symbol}) = begin
    rs = sort!(reduce(vcat, collect(indices(s.layout, sym)) for sym in syms))
    if !isempty(rs) && rs == (minimum(rs):maximum(rs))
        @view s.data[minimum(rs):maximum(rs)]
    else
        s.data[rs]               # non-contiguous => copy
    end
end
Base.getindex(s::State, syms::Tuple{Vararg{Symbol}}) = s[collect(syms)]

########## Time-stacked states ##########

export TimeStack, ntime, state, series

struct TimeStack{T,V<:AbstractVector{T}}
    data   :: V
    layout :: Layout
    nt     :: Int
    function TimeStack{T,V}(x::V, L::Layout) where {T,V<:AbstractVector{T}}
        Ltot = length(L)
        @assert length(x) % Ltot == 0 "Vector length $(length(x)) must be a multiple of state length $(Ltot)."
        nt = length(x) ÷ Ltot
        new{T,V}(x, L, nt)
    end
end
TimeStack(x::AbstractVector, L::Layout) = TimeStack{eltype(x), typeof(x)}(x, L)

# Basic queries
Base.length(ts::TimeStack) = length(ts.data)
ntime(ts::TimeStack) = ts.nt
state_length(ts::TimeStack) = length(ts.layout)
Base.parent(ts::TimeStack) = ts.data
names(ts::TimeStack) = names(ts.layout)

# Pretty print
function Base.show(io::IO, ts::TimeStack)
    print(io, "TimeStack($(state_length(ts))-el per state, T=$(ntime(ts)), names=$(join(ts.layout.names, ", ")))")
end

# Time-slice -> State view
@inline function _time_range(ts::TimeStack, t::Int)
    @boundscheck 1 <= t <= ts.nt || throw(BoundsError(ts, t))
    L = state_length(ts)
    a = (t-1)*L + 1
    a:(a+L-1)
end

state(ts::TimeStack, t::Int) = State(@view(ts.data[_time_range(ts, t)]), ts.layout)
Base.getindex(ts::TimeStack, t::Int) = state(ts, t)

# Absolute indices for a named block at a given time
function indices(ts::TimeStack, sym::Symbol, t::Int)
    r = indices(ts.layout, sym)
    base = first(_time_range(ts, t)) - 1
    (first(r) + base) : (last(r) + base)
end

# Convenience: get a named block at time t (view)
Base.getindex(ts::TimeStack, sym::Symbol, t::Int) = @view ts.data[indices(ts, sym, t)]

# Collect a component across all times as a (len × T) matrix (copy; contiguous per column)
function series(ts::TimeStack, sym::Symbol)
    r = indices(ts.layout, sym)
    m = length(r)
    T = ts.nt
    out = similar(ts.data, m, T)
    @inbounds for t in 1:T
        out[:, t] .= @view ts.data[indices(ts, sym, t)]
    end
    out
end

# Mutate a whole component at a specific time using `setindex!`
function Base.setindex!(ts::TimeStack, vals, sym::Symbol, t::Int)
    r = indices(ts, sym, t)
    @assert length(vals) == length(r) "Assigning $(length(vals)) into $(sym) at t=$t of length $(length(r))."
    ts.data[r] .= vals
    vals
end

# --- Helpers for time selections ---
@inline function _normalize_tsel(ts::TimeStack, tsel::Union{UnitRange{Int},AbstractVector{Int},Colon})
    if tsel isa Colon
        return 1:ts.nt
    elseif tsel isa UnitRange{Int}
        @boundscheck 1 <= first(tsel) <= last(tsel) <= ts.nt || throw(BoundsError(tsel))
        return tsel
    else
        @boundscheck all(1 .<= tsel .<= ts.nt) || throw(BoundsError(tsel))
        return tsel
    end
end

# Keep this helper from before (sub can be Int/UnitRange/Vector or :)
@inline function _abs_subindices(ts::TimeStack, sym::Symbol, sub, t::Int)
    r = indices(ts.layout, sym)          # component range within a state
    base = first(_time_range(ts, t)) - 1 # offset for time block
    if sub isa Colon
        return (first(r)+base) : (last(r)+base)
    elseif isa(sub, Integer)
        @boundscheck 1 <= sub <= length(r) || throw(BoundsError(sub))
        i = first(r) + (sub - 1) + base
        return i:i
    elseif isa(sub, UnitRange{Int})
        @boundscheck 1 <= first(sub) <= last(sub) <= length(r) || throw(BoundsError(sub))
        a = first(r) + (first(sub)-1) + base
        b = first(r) + (last(sub)-1)  + base
        return a:b
    elseif isa(sub, AbstractVector{Int})
        @boundscheck all(1 .<= sub .<= length(r)) || throw(BoundsError(sub))
        return (first(r) .+ (sub .- 1)) .+ base
    else
        throw(ArgumentError("Unsupported subindex type $(typeof(sub)). Use Int, UnitRange, Vector{Int}, or :"))
    end
end

# Single time t -> view
Base.getindex(ts::TimeStack, sym::Symbol, sub::Union{Int,UnitRange{Int},AbstractVector{Int}}, t::Int) =
    @view ts.data[_abs_subindices(ts, sym, sub, t)]

# Stack subrange across a selection of times (copy), order given by tsel
function Base.getindex(ts::TimeStack,
                       sym::Symbol,
                       sub::Union{Int,UnitRange{Int},AbstractVector{Int}},
                       tsel::Union{UnitRange{Int},AbstractVector{Int},Colon})
    times = _normalize_tsel(ts, tsel)
    # how many rows inside the component?
    m = sub isa Colon ? length(indices(ts.layout, sym)) :
        sub isa Int ? 1 : length(sub)
    out = similar(ts.data, m * length(times))
    dest = 1
    @inbounds for t in times
        rng = _abs_subindices(ts, sym, sub, t)
        out[dest:dest+length(rng)-1] .= @view ts.data[rng]
        dest += length(rng)
    end
    out
end

# --- New: matrix form for whole component over a SELECTION of times (copy) ---
# ts[:f, :, tsel] -> (len(:f) × length(tsel)) matrix with columns in the order of tsel
function Base.getindex(ts::TimeStack, sym::Symbol, ::Colon, tsel::Union{UnitRange{Int},AbstractVector{Int},Colon})
    times = _normalize_tsel(ts, tsel)
    r = indices(ts.layout, sym)
    m = length(r)
    out = similar(ts.data, m, length(times))
    @inbounds for (j, t) in pairs(times)
        out[:, j] .= @view ts.data[_abs_subindices(ts, sym, :, t)]
    end
    out
end

export absindices

# 1) Single time -> pass through the precise absolute indices (range or vector)
absindices(ts::TimeStack, sym::Symbol, t::Int) =
    indices(ts, sym, t)  # you already defined this: returns UnitRange

absindices(ts::TimeStack,
           sym::Symbol,
           sub::Union{Int,UnitRange{Int},AbstractVector{Int},Colon},
           t::Int) =
    _abs_subindices(ts, sym, sub, t)  # returns UnitRange or Vector{Int}

# 2) All times for a whole component (stacked t=1..T)
function absindices(ts::TimeStack, sym::Symbol)
    r_local = indices(ts.layout, sym)
    m, T = length(r_local), ts.nt
    out = Vector{Int}(undef, m*T)
    pos = 1
    @inbounds for t in 1:T
        rng = _abs_subindices(ts, sym, :, t)  # UnitRange
        len = length(rng)
        @assert len == m
        copyto!(out, pos, collect(rng), 1, len)  # rng is a range; materialize indices
        pos += len
    end
    out
end

# 3) Subrange across a SELECTION of times (order follows tsel)
function absindices(ts::TimeStack,
                    sym::Symbol,
                    sub::Union{Int,UnitRange{Int},AbstractVector{Int},Colon},
                    tsel::Union{UnitRange{Int},AbstractVector{Int},Colon})
    times = _normalize_tsel(ts, tsel)
    # Determine length per time
    perlen = sub isa Colon ? length(indices(ts.layout, sym)) :
             sub isa Int ? 1 : length(sub)
    out = Vector{Int}(undef, perlen * length(times))
    pos = 1
    @inbounds for t in times
        rng_or_vec = _abs_subindices(ts, sym, sub, t)
        if rng_or_vec isa UnitRange{Int}
            len = length(rng_or_vec)
            copyto!(out, pos, collect(rng_or_vec), 1, len)
            pos += len
        else
            len = length(rng_or_vec)
            copyto!(out, pos, rng_or_vec, 1, len)
            pos += len
        end
    end
    out
end

# 4) Matrix-like whole-component across a time selection — returns column-wise indices
#    Useful if you want indices grouped by time. (Optional utility.)
function absindices_by_time(ts::TimeStack, sym::Symbol,
                            tsel::Union{UnitRange{Int},AbstractVector{Int},Colon})
    times = _normalize_tsel(ts, tsel)
    m = length(indices(ts.layout, sym))
    out = Matrix{Int}(undef, m, length(times))
    @inbounds for (j, t) in pairs(times)
        rng = _abs_subindices(ts, sym, :, t)  # UnitRange
        # Fill column j
        k = 1
        for i in rng
            out[k, j] = i
            k += 1
        end
    end
    out
end
