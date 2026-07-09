
function steiner_cond4(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	return c >= interval(1) && e >= interval(1)
end

function steiner_length4(pos::SVector{8, Point}, vars::Vector{T})::T
	(b, c, d, s, e, f) = vars
	S::T = interval(1) + s + c + e + interval(2)*d
	return sqrt(S^2 - S + interval(1))
end

Dict{Any, Any}(
	sort([n"A", n"D", n"Q", n"R"]) => (steiner_cond4, steiner_length4)
)
