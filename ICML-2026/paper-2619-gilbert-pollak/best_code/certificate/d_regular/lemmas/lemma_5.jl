
function X_cond5(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	return e < (interval(2)/sqrt(interval(3))-interval(1))*c
end

function AX_upper_bound5(pos::SVector{8, Point}, vars::Vector{T})::T
	return max(
		dist(pos[n"A"], pos[n"r"]),
		dist(pos[n"A"], pos[n"P"]),
		dist(pos[n"A"], pos[n"R"])
	)
end

function DX_upper_bound5(pos::SVector{8, Point}, vars::Vector{T})::T
	(b, c, d, s, e, f) = vars
	H::Point = pos[n"R"] - (c - e)/interval(2)*dir2
	return max(
		dist(pos[n"D"], pos[n"r"]),
		dist(pos[n"D"], pos[n"P"]),
		dist(pos[n"D"], pos[n"R"]),
		dist(pos[n"D"], H)
	)
end

Dict{Any, Any}(
	normalized_edge(n"A", n"X") => (X_cond5, AX_upper_bound5),
	normalized_edge(n"D", n"X") => (X_cond5, DX_upper_bound5),
)
