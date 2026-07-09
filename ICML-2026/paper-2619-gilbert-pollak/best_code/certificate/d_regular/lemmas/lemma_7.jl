
function steiner_cond7(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	return d*(c + e - interval(1)) + e*(s + c) >= interval(0) &&
		   e >= (sqrt((c + s + interval(2)*d)^2 + interval(4)*d) - (c + s + interval(2)*d))/interval(2) &&
		   -(interval(2) + s - c + d)*(c + interval(2)*e + interval(2)*d) + interval(3)*c*(s + c + d) >= interval(0) &&
		   s + e >= interval(1) &&
		   c <= interval(1) + d + (s + e) / interval(2)
end

function steiner_length7(pos::SVector{8, Point}, vars::Vector{T})::T
	(b, c, d, s, e, f) = vars
	S = interval(2)*d + s + c + e
	return sqrt(S^2 + S + interval(1))
end

Dict{Any, Any}(
	sort([n"A", n"D", n"Q", n"R"]) => (steiner_cond7, steiner_length7)
)
