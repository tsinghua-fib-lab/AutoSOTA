
function steiner_cond6(pos::SVector{10, Point}, vars::Vector{T})::Bool
	(b, c, d, s, u, v, e, f) = vars
	return d >= b
end

function steiner_length6(pos::SVector{10, Point}, vars::Vector{T})::T
	(b, c, d, s, u, v, e, f) = vars
	t1 = interval(2)*b + c + interval(2)*d + e + s;
    t2 = c + interval(2)*d + e + s;
    return sqrt(t1^2 + interval(3)*t2^2)/interval(2)
end

Dict{Any, Any}(
	sort([n"B", n"D", n"Q", n"R"]) => (steiner_cond6, steiner_length6)
)
