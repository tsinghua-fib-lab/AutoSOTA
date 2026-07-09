
function X_cond2(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	return e < interval(1) + (interval(2)/sqrt(interval(3))-interval(1))*c &&
		   c + e >= interval(1)
end

function AX_upper_bound2(pos::SVector{8, Point}, vars::Vector{T})::T
	(b, c, d, s, e, f) = vars
	return max(
		dist(pos[n"A"], pos[n"s"]),
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
		dist(pos[n"A"], pos[n"R"]),
		sqrt(interval(3)) / interval(2) * (c + e - interval(1)),
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond2, AX_upper_bound2)
)
