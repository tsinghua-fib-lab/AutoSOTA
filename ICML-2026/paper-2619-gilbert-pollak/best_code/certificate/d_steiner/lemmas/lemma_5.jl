
function X_cond5(pos::SVector{10, Point}, vars::Vector{T})::Bool
	(b, c, d, s, u, v, e, f) = vars
	return e < (interval(2)/sqrt(interval(3))-interval(1))*c
end

function AX_upper_bound5(pos::SVector{10, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
		dist(pos[n"A"], pos[n"R"])
	)
end

function UX_upper_bound5(pos::SVector{10, Point}, vars::Vector{T})::T
	(b, c, d, s, u, v, e, f) = vars
	H::Point = pos[n"R"] - (c - e)/interval(2)*dir2
	return max(
		dist(pos[n"U"], pos[n"r"]),
		dist(pos[n"U"], pos[n"P"]),
		dist(pos[n"U"], pos[n"R"]),
		dist(pos[n"U"], H)
	)
end

function VX_upper_bound5(pos::SVector{10, Point}, vars::Vector{T})::T
	(b, c, d, s, u, v, e, f) = vars
	H::Point = pos[n"R"] - (c - e)/interval(2)*dir2
	return max(
		dist(pos[n"V"], pos[n"r"]),
		dist(pos[n"V"], pos[n"P"]),
		dist(pos[n"V"], pos[n"R"]),
		dist(pos[n"V"], H)
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond5, AX_upper_bound5),
	normalized_edge(n"U", n"X") => (X_cond5, UX_upper_bound5),
	normalized_edge(n"V", n"X") => (X_cond5, VX_upper_bound5),
)
