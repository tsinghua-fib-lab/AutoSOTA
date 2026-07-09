
function X_cond8(pos::SVector{10, Point}, vars::Vector{T})::Bool
	(b, c, d, s, u, v, e, f) = vars
	return e < s + interval(1)
end

function AX_upper_bound8(pos::SVector{10, Point}, vars::Vector{T})::T
	(b, c, d, s, u, v, e, f) = vars
	return max(
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
		dist(pos[n"A"], pos[n"R"]),
		e + c - interval(1)
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond8, AX_upper_bound8)
)
