
function X_cond1(pos::SVector{10, Point}, vars::Vector{T})::Bool
	(b, c, d, s, u, v, e, f) = vars
	return e < interval(1) - c + interval(2)/sqrt(interval(3))*s &&
		   c + e >= interval(1)
end

function AX_upper_bound1(pos::SVector{10, Point}, vars::Vector{T})::T
	(b, c, d, s, u, v, e, f) = vars
	return max(
		dist(pos[n"A"], pos[n"s"]),
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
		dist(pos[n"A"], pos[n"R"]),
		sqrt(interval(3)) / interval(2) * (c + e - interval(1)),
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond1, AX_upper_bound1)
)
