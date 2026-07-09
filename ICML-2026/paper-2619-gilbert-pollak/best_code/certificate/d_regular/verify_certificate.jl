import Pkg
Pkg.activate("../steiner")
Pkg.add("IntervalArithmetic")
Pkg.add("StaticArrays")

using IntervalArithmetic, StaticArrays
using Base.Threads
import Base.+, Base.-, Base.*, Base.<, Base.<=

const T = Interval{Float64}
const Point = Tuple{T, T}

# ==============================================================================
# Helpers
# ==============================================================================

(<)(a::T, b::T) = sup(a) < inf(b)
(<=)(a::T, b::T) = sup(a) <= inf(b)

(+)(a::Point, b::Point) = (a[1] + b[1], a[2] + b[2])
(-)(a::Point, b::Point) = (a[1] - b[1], a[2] - b[2])
(*)(a::T, b::Point) = (a * b[1], a * b[2])

norm2(a::Point)::T = a[1]^2 + a[2]^2
dist2(a::Point, b::Point)::T = norm2(a - b)
dist(a::Point, b::Point)::T = sqrt(dist2(a, b))
dot(a::Point, b::Point)::T = a[1] * b[1] + a[2] * b[2]
cross(a::Point, b::Point)::T = a[1] * b[2] - a[2] * b[1]

function angle_le_120(u::Point, v::Point)::Bool # ∠(u,v) <= 120°
	if inf(dot(u, v)) >= 0
		return true
	end
	return interval(4) * dot(u, v)^2 <= norm2(u) * norm2(v)
end

function angle_le_120(a::Point, b::Point, c::Point)::Bool # ∠ABC <= 120°
	u = a - b
	v = c - b
	return angle_le_120(u, v)
end

function tri_area(a::Point, b::Point, c::Point)::T
	u = b - a
	v = c - a
	return abs(cross(u, v) / @exact 2)
end

function smt_length(a::Point, b::Point, c::Point)::T
	if !angle_le_120(a, b, c)
		return sqrt(dist2(a, b)) + sqrt(dist2(b, c))
	end
	if !angle_le_120(b, c, a)
		return sqrt(dist2(b, c)) + sqrt(dist2(c, a))
	end
	if !angle_le_120(c, a, b)
		return sqrt(dist2(c, a)) + sqrt(dist2(a, b))
	end
	S = tri_area(a, b, c)
	return sqrt((dist2(a, b) + dist2(b, c) + dist2(c, a)) / (@exact 2) + (@exact 2) * sqrt(interval(3)::T) * S)
end

function rotate_ccw_60(u::Point)::Point
	cos = interval(1) / interval(2)
	sin = sqrt(interval(3)) / interval(2)
	return (cos * u[1] - sin * u[2], sin * u[1] + cos * u[2])
end

# ==============================================================================
# Configuration
# ==============================================================================

const rho::T = interval(8559//10000)

const Nodes = ['A', 'B', 'D', 'P', 'Q', 'R', 's', 'r', 'X', 'Y']
const NodeID = Dict(char => idx for (idx, char) in enumerate(Nodes))
const ImpNodes = [NodeID['X'], NodeID['Y']]
macro n_str(s)
    args = [NodeID[c] for c in s]
    if length(s) == 1
        return args[1]
	else
		return args
    end
end

normalized_edge(u::Int, v::Int)::Tuple{Int, Int} = u < v ? (u, v) : (v, u)
const Edges = [('A', 's'), ('B', 's'), ('r', 'P'), ('r', 'D'), ('s', 'r'), ('P', 'R'), ('P', 'Q')]
const EdgeID = Dict(normalized_edge(NodeID[u], NodeID[v]) => idx for (idx, (u, v)) in enumerate(Edges))

const Vars = ['b', 'c', 'd', 's', 'e', 'f']							# Should be arranged in the order corresponding to `Edges`.
const VarID = Dict(char => idx for (idx, char) in enumerate(Vars))

struct Split
	V_star::Vector{Int}
    S_minus::Vector{Int}
    S_plus::Vector{Int}
    T_star::Vector{Tuple{Int, Int}}
	mono_vars::Vector{Int}
end

const dir0::Point = (@exact 1, @exact 0)
const dir1::Point = (@exact 0.5, sqrt(interval(3)) / @exact 2)
const dir2::Point = (@exact 0.5, -sqrt(interval(3)) / @exact 2)

function get_coordinates(vars::Vector{T})::SVector{8, Point}
	A::Point = (@exact 0, @exact 0)
	s::Point = (@exact 1, @exact 0)
	B::Point = s + vars[1] * dir1
	r::Point = s + vars[4] * dir2
	P::Point = r + (-vars[2]) * dir1
	D::Point = r + vars[3] * dir0
	Q::Point = P + vars[6] * dir2
	R::Point = P + (-vars[5]) * dir0
	return @SVector [A, B, D, P, Q, R, s, r]
end

# ==============================================================================
# Splits Registration
# ==============================================================================

"""
Parses a literal string (e.g., "['A', 'B']") into a Julia object.
"""
function parse_literal(val_str::AbstractString)
    # 1. Parse the string as a Julia expression and evaluate it
    #    This handles nested structures like [('A', 'B'), ('C', 'D')] automatically.
    #    Meta.parse converts the string to an AST, eval executes it.
    val = eval(Meta.parse(val_str))

    # 2. Standardization
    #    S_plus comes in as a Tuple ('r', 'B') in the text file, 
    #    but we store it as a Vector ['r', 'B'] in the struct.
    if val isa Tuple
        return collect(val)
	else
		return val
    end
end

"""
Reads splits from a file.
"""
function read_splits_from_file(filename::String)::Vector{Split}
    splits = Vector{Split}()
    
    # Check if file exists
    if !isfile(filename)
        println("Error: File '$filename' not found.")
        return splits
    end

    for (line_num, line) in enumerate(eachline(filename))
        line = strip(line)
        if isempty(line)
            continue
        end

        try
            # Dictionary to hold the parsed parts for this line
            # We map "Key" => Raw Value String
            raw_parts = Dict{String, Any}()
            
            # Split by semicolon to get Key:Value pairs
            # e.g., "V_star:['A']"
            segments = split(line, ";")
            
            for seg in segments
                if !contains(seg, ":")
                    continue
                end
                
                # Split only on the first colon
                k, v = split(seg, ":", limit=2)
                k = strip(k)
                v = strip(v)
                
                # Parse the value string into actual Julia objects
                raw_parts[k] = parse_literal(v)
            end
            
            # Construct the Split object
			V_star::Vector{Int} = [NodeID[u] for u in raw_parts["V_star"]]
			S_minus::Vector{Int} = [EdgeID[normalized_edge(NodeID[u], NodeID[v])] for (u, v) in raw_parts["S_minus"]]
			S_plus::Vector{Int} = sort([NodeID[u] for u in raw_parts["S_plus"]])
			T_star::Vector{Tuple{Int, Int}} = [normalized_edge(NodeID[u], NodeID[v]) for (u, v) in raw_parts["T_star"]]
			mono_vars::Vector{Int} = [VarID[x] for x in raw_parts["mono_vars"]]
            
            s = Split(V_star, S_minus, S_plus, T_star, mono_vars)
            push!(splits, s)
        
        catch e
            println("Warning: Failed to parse line $line_num: $e")
            println(line)
        end
    end

    return splits
end

splits::Vector{Split} = read_splits_from_file("splits.txt")

# ==============================================================================
# Lemma Registration
# ==============================================================================

LEMMAS::Vector{Dict{Union{Tuple{Int, Int}, Vector{Int}}, Tuple{Function, Function}}} = []

for i in range(0, 8)
	lemma = include(joinpath("lemmas", "lemma_$i.jl"))
	if !(normalized_edge(n"D", n"Y") in keys(lemma))
		lemma[normalized_edge(n"D", n"Y")] = (Y_cond0, DY_upper_bound0)
	end
	push!(LEMMAS, lemma)
end

# ==============================================================================
# Verification
# ==============================================================================

function evaluate_split(split::Split, vars::Vector{T}, lemma_id::Int32)::T
	pos = get_coordinates(vars)
	# --- Part 1: T* ---
	t_star::T = interval(0)
	for (u, v) in split.T_star
		if u in ImpNodes || v in ImpNodes
			(cond, func) = LEMMAS[lemma_id + 1][(u, v)]
			if !cond(pos, vars)
				return interval(1)				# Fail. Return any number greater than 0.
			else
				t_star += func(pos, vars)
			end
		else
			t_star += dist(pos[u], pos[v])
		end
	end
	# --- Part 2: S- ---
	s_minus::T = interval(0)
	for idx in split.S_minus
		s_minus += idx > 1 ? vars[idx - 1] : interval(1)
	end
	# --- Part 3: S+ ---
	s_plus::T = interval(0)
	if length(split.S_plus) == 2
		s_plus = dist(pos[split.S_plus[1]], pos[split.S_plus[2]])
	elseif length(split.S_plus) == 3
		s_plus = smt_length(pos[split.S_plus[1]], pos[split.S_plus[2]], pos[split.S_plus[3]])
	elseif length(split.S_plus) > 3
		(cond, func) = LEMMAS[lemma_id + 1][split.S_plus]
		if !cond(pos, vars)
			return interval(1)					# Fail. Return any number greater than 0.
		else
			s_plus = func(pos, vars)
		end
	end
	ret = rho * t_star + s_plus - s_minus
	@assert isguaranteed(ret) && iscommon(ret)
	return ret
end

const Record = Tuple{Int32, Matrix{Float64}, Int32, Int32}

function verify_record(record::Record)
	region_id::Int32, box::Matrix{Float64}, split_id::Int32, lemma_id::Int32 = record
	if F_VAL == 1 && box[1, VarID['f']] > box[2, VarID['d']]
		return
	end

	try
		split::Split = splits[split_id]
		if F_VAL == 2 && !(VarID['f'] in split.mono_vars)
			throw("case f >= d, but 'f' is not in split[$split_id].mono_vars.")
		end
		if F_VAL == 1 && length(split.S_plus) > 3
			throw("case f <= d, but split[$split_id].S_plus has length $(length(split.S_plus)) > 3.")
		end
		for i in range(1, n)
			if isinf(box[2, i]) && !(i in split.mono_vars)
				throw("$(Vars[i])_max is inf, but '$(Vars[i])' is not in split[$split_id].mono_vars.")
			end
			if isinf(box[2, i]) && lemma_id > 0
				throw("$(Vars[i])_max is inf, but use Lemma $lemma_id.")
			end
		end
		vars = Vector{T}(undef, length(Vars))
		for mask in range(0, (1 << n) - 1)
			no_inf = true
			for i in range(1, n)
				v::Float64 = box[((mask >> i) & 1) + 1, i]
				if isinf(v)
					no_inf = false
					break
				end
				vars[i] = interval(v)
			end
			if !no_inf
				continue
			end
			if F_VAL == 2
				vars[VarID['f']] = vars[VarID['d']]
			end
			ret = evaluate_split(split, vars, lemma_id)
			if !(ret <= interval(0))
				throw("vertex check failed.")
			end
		end
	catch err
		throw("region $region_id failed.\nbox: $box\nsplit_id: $split_id\nlemma_id: $lemma_id\n$err")
	end
end

function verify_parallel(; bufsize::Int = 64)
	stop = Atomic{Bool}(false)
	errch = Channel{Any}(1)
	ch = Channel{Record}(bufsize)

	nworkers = nthreads()
	println("nworkers = $nworkers")
	workers = Vector{Task}(undef, nworkers)
	for i in range(1, nworkers)
		workers[i] = @spawn begin
			for rec in ch
				if stop[]
					continue
				end
				try
					verify_record(rec)
				catch e
					if !atomic_xchg!(stop, true)
						put!(errch, e)
					end
				end
			end
		end
	end

	cnt = 0
	open(cert_file, "r") do io
		while !stop[]
			region_id::Int32 = try
				read(io, Int32)
			catch e
				break
			end
			cnt += 1
			if cnt % 100000 == 0
				println("[info] verified $cnt regions...")
			end
			box = Matrix{Float64}(undef, 2, n)
			for i in range(1, n)
				for j in range(1, 2)
					box[j, i] = read(io, Float64)
				end
				@assert box[1, i] <= box[2, i]
			end
			split_id = read(io, Int32)
			lemma_id = read(io, Int32)
			put!(ch, (region_id, box, split_id, lemma_id))
		end
	end

	close(ch)
	for t in workers
		wait(t)
	end

	if isready(errch)
		e = take!(errch)
		println("[error] $e")
	else
		println("All $cnt regions verified.")
	end
end

F_VAL = 0
n = 0
cert_file = ""
if "--f-ge-d" in ARGS
	n = 5
	F_VAL = 2
	cert_file = "certificate_rho=0.8559_f_ge_d.bin"
elseif "--f-le-d" in ARGS
	n = 6
	F_VAL = 1
	cert_file = "certificate_rho=0.8559_f_le_d.bin"
else
	println("Usage: verify_certificate.jl {--f-ge-d|--f-le-d}")
	exit(1)
end

verify_parallel()
