
function steiner_cond3(pos::SVector{8, Point}, vars::Vector{T})::Bool
	(b, c, d, s, e, f) = vars
	V::Point = pos[n"D"] + rotate_ccw_60(pos[n"Q"] - pos[n"D"])
	return c + e >= interval(1) &&
		   e <= s + c + d + interval(2) &&
		   e >= interval(2) - c - d + s &&
		   e * (interval(2) + c + interval(3)*d + s) <= interval(2) + c*c + interval(3)*d + interval(2)*s + interval(3)*c*s + interval(3)*d*s + interval(2)*s*s &&
		   e <= interval(3)*s &&
		   angle_le_120(pos[n"D"] - pos[n"R"], pos[n"Q"] - pos[n"A"]) &&
		   angle_le_120(pos[n"A"], pos[n"R"], V) &&
		   angle_le_120(pos[n"R"], pos[n"A"], V)
end

function steiner_length3(pos::SVector{8, Point}, vars::Vector{T})::T
	(b, c, d, s, e, f) = vars
	UVx::T = (interval(1) + e + interval(3)*d + interval(2)*s + interval(3)*c) / interval(2)
	UVy::T = -sqrt(interval(3))*(c + d + e - interval(1)) / interval(2)
	return sqrt(UVx^2 + UVy^2)
end

Dict{Any, Any}(
	sort([n"A", n"D", n"Q", n"R"]) => (steiner_cond3, steiner_length3)
)
