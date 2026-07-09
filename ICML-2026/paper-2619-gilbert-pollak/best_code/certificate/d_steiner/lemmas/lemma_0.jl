
function X_cond0(pos::SVector{10, Point}, vars::Vector{T})::Bool
	(b, c, d, s, u, v, e, f) = vars
	return c <= interval(1)
end

function AX_upper_bound0(pos::SVector{10, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
	)
end

function Y_cond0(pos::SVector{10, Point}, vars::Vector{T})::Bool
	return F_VAL == 1
end

function VY_upper_bound0(pos::SVector{10, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"V"], pos[n"r"]),
		dist(pos[n"V"], pos[n"P"]),
		dist(pos[n"V"], pos[n"Q"]),
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond0, AX_upper_bound0),
	normalized_edge(n"V", n"Y") => (Y_cond0, VY_upper_bound0),
)
