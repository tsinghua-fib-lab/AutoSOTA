
function X_cond0(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	return c <= interval(1)
end

function AX_upper_bound0(pos::SVector{8, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
	)
end

function Y_cond0(pos::SVector{8, Point}, vars::Vector{T})::Bool
	return F_VAL == 1
end

function DY_upper_bound0(pos::SVector{8, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"D"], pos[n"P"]),
		dist(pos[n"D"], pos[n"Q"]),
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond0, AX_upper_bound0),
	normalized_edge(n"D", n"Y") => (Y_cond0, DY_upper_bound0),
)
