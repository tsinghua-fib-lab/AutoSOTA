# include "geosteiner-5.3/geosteiner.h"
# include <bits/stdc++.h>
using namespace std;
using ld = double;
using str = string;
using strv = string_view;
const ld sqrt3 = sqrt(3.0l), INF = 1e18;
strv strip(strv s)
{
	auto i = ranges::find_if_not(s, [](char c){return isspace(c);}) - s.begin();
	auto j = ranges::find_if_not(views::reverse(s), [](char c){return isspace(c);}).base() - s.begin();
	if(i >= j) return {};
	return s.substr(i, j - i);
}
str parse_string(strv s)
{
	auto i = ranges::find_first_of(s, "\'\"") - s.begin();
	auto j = ranges::find_first_of(views::reverse(s), "\'\"").base() - s.begin();
	if(i + 1 >= j) assert(false);
	return str(s.substr(i + 1, j - i - 2));
}
tuple<str,str> parse_tuple(strv s)
{
	auto i = ranges::find_first_of(s, "([") - s.begin();
	auto j = ranges::find_first_of(views::reverse(s), ")]").base() - s.begin();
	if(i + 1 >= j) assert(false);
	s = s.substr(i + 1, j - i - 2);
	auto p = ranges::find_first_of(s, ",") - s.begin();
	return make_tuple(parse_string(s.substr(0, p)), parse_string(s.substr(p + 1)));
}
template<typename Fn = identity>
auto parse_vector(strv s, Fn &&proj = {})
{
	auto p = ranges::find_first_of(s, "([") - s.begin();
	auto q = ranges::find_first_of(views::reverse(s), ")]").base() - s.begin();
	if(p + 1 >= q) assert(false);
	s = s.substr(p + 1, q - p - 2);
	vector<decay_t<decltype(proj(s))>> ret;
	size_t i = 0, j = 0, n = s.size(), cnt = 0;
	for(;j < n; j++)
	{
		if(s[j] == ',' && !cnt)
		{
			ret.push_back(proj(s.substr(i, j - i)));
			i = j + 1;
		}
		else if(s[j] == '(' || s[j] == '[') cnt++;
		else if(s[j] == ')' || s[j] == ']') cnt--;
	}
	if(i < j) ret.push_back(proj(s.substr(i, j - i)));
	return ret;
}
struct Split
{
	vector<tuple<str,str>> S_minus;
	vector<str> S_plus;
	vector<tuple<str,str>> T_star;
	bool is_valid(auto &&valid) const
	{
		for(auto &&[u, v] : S_minus)
			if(!valid[u] || !valid[v]) return false;
		for(auto &&u : S_plus)
			if(!valid[u]) return false;
		for(auto &&[u, v] : T_star)
			if(!valid[u] || !valid[v]) return false;
		return true;
	}
	static Split from_string(strv __s)
	{
		Split ret;
		for(auto &&_s : views::split(__s, ';'))
		{
			strv s(_s);
			auto p = ranges::find_first_of(s, ":") - s.begin();
			auto key = strip(s.substr(0, p)), value = s.substr(p + 1);
			if(key == "S_minus") ret.S_minus = parse_vector(value, parse_tuple);
			else if(key == "S_plus") ret.S_plus = parse_vector(value, parse_string);
			else if(key == "T_star") ret.T_star = parse_vector(value, parse_tuple);
		}
		return ret;
	}
	str to_string() const
	{
		auto format_tuples = [](auto &&r) {
			str ret;
			for(auto &&[u, v] : r)
			{
				if(!ret.empty()) ret += ", ";
				ret += format("({}, {})", u, v);
			}
			return ret;
		};
		auto format_strings = [](auto &&r) {
			str ret;
			for(auto s : r)
			{
				if(!ret.empty()) ret += ", ";
				ret += s;
			}
			return ret;
		};
		return format("S_minus: [{}]; S_plus: ({}); T_star: [{}]",
			format_tuples(S_minus), format_strings(S_plus), format_tuples(T_star)
		);
	}
	str splus_to_string() const
	{
		auto format_strings = [](auto &&r) {
			str ret;
			for(auto s : r)
			{
				if(!ret.empty()) ret += ", ";
				ret += "'" + s + "'";
			}
			return ret;
		};
		return format("({})", format_strings(S_plus));
	}
	auto operator<=>(const Split &other) const = default;
};
struct Point
{
	ld x, y;
	Point() = default;
	Point(ld _x, ld _y):x(_x), y(_y){}
	Point operator+(Point a) const {return Point(x + a.x, y + a.y);}
	Point operator-(Point a) const {return Point(x - a.x, y - a.y);}
	Point operator-() const {return Point(-x, -y);}
	Point operator*(ld k) const {return Point(x * k, y * k);}
	ld operator*(Point a) const {return x * a.x + y * a.y;}
	Point rotate_ccw_60() const
	{
		return Point(
			(x - sqrt3 * y) / 2,
			(sqrt3 * x + y) / 2
		);
	}
	Point rotate_cw_60() const
	{
		return Point(
			(x + sqrt3 * y) / 2,
			(y - sqrt3 * x) / 2
		);
	}
	ld norm2() const {return x * x + y * y;}
};
map<str, Point> pos;
map<pair<str, str>, ld> dist2_upper_bound;
ld dist2(str u, str v)
{
	auto it = dist2_upper_bound.find({u, v});
	if(it != dist2_upper_bound.end()) return it->second;
	it = dist2_upper_bound.find({v, u});
	if(it != dist2_upper_bound.end()) return it->second;
	assert(pos.contains(u) && pos.contains(v));
	return (pos[u] - pos[v]).norm2();
}
ld L_s(const vector<str> &S_plus)
{
	int n = S_plus.size();
	vector<ld> x;
	x.reserve(2*n);
	for(auto &&u : S_plus)
	{
		x.push_back(pos[u].x);
		x.push_back(pos[u].y);
	}
	ld length;
	if(gst_esmt(n, x.data(), &length, NULL, NULL, NULL, NULL, NULL, NULL))
	{
		puts("Fail");
		return INF;
	}
	return length;
}

double angle(Point a, Point b)
{
	return acos((a * b) / (sqrt(a.norm2() * b.norm2()))) / numbers::pi * 180;
}

int main(int argc, char **argv)
{
	str file_name = "splits4.txt";
	ifstream fin(file_name);
	string line;
	vector<Split> splits;
	while(getline(fin, line)) splits.push_back(Split::from_string(line));
	cout<<splits.size()<<endl;

	if(gst_open_geosteiner())
	{
		puts("Fail");
		return 0;
	}
	ld a = 1, b, c, d, s, e, f, tmp;
	ld lower_b, upper_b;
	ld lower_c, upper_c;
	ld lower_d, upper_d;
	ld lower_s, upper_s;
	ld lower_e, upper_e;
	ifstream param(argv[1]);
	param >> tmp >> tmp;
	param >> lower_b >> upper_b;
	param >> lower_c >> upper_c;
	param >> lower_d >> upper_d;
	param >> lower_s >> upper_s;
	param >> lower_e >> upper_e;
	b = (lower_b + upper_b) / 2;
	c = (lower_c + upper_c) / 2;
	d = (lower_d + upper_d) / 2;
	s = (lower_s + upper_s) / 2;
	e = (lower_e + upper_e) / 2;
	f = d;
	cerr << b << " " << c << " " << d << " " << s << " " << e << " " << f << endl;
	Point dir1 = Point(1, 0), dir2 = dir1.rotate_ccw_60(), dir3 = dir1.rotate_cw_60();
	pos["A"] = Point(0, 0);
	pos["s"] = pos["A"] + dir1 * a;
	pos["B"] = pos["s"] + dir2 * b;
	pos["r"] = pos["s"] + dir3 * s;
	pos["D"] = pos["r"] + dir1 * d;
	pos["P"] = pos["r"] + dir2 * (-c);
	pos["Q"] = pos["P"] + dir3 * f;
	pos["R"] = pos["P"] + dir1 * (-e);

	dist2_upper_bound[{"A", "X"}] = max(max(max(dist2("A", "P"), dist2("A", "R")), dist2("A", "r")), e + c - 1);
	dist2_upper_bound[{"D", "X"}] = max(dist2("D", "P"), dist2("D", "R"));
	dist2_upper_bound[{"D", "Y"}] = max(dist2("D", "P"), dist2("D", "Q"));
	ld rho = 0;

	// valid: 哪些点存在
	map<str, bool> valid;
	for(auto u : {"A", "B", "D", "s", "r", "P", "Q", "R", "X", "Y"}) valid[u] = true;
	// valid["X"] = false;
	valid["Y"] = false;

	str best_splus;
	for(auto &&split : splits)
	{
		if(!split.is_valid(valid)) continue;
		ld L_s_minus = 0, L_t = 0;
		for(auto &&[u, v] : split.S_minus) L_s_minus += sqrt(dist2(u, v));
		for(auto &&[u, v] : split.T_star) L_t += sqrt(dist2(u, v));
		ld L_s_plus = L_s(split.S_plus);
		ld r = (L_s_minus - L_s_plus) / L_t;
		cout<<split.to_string()<<": "<<r<<endl;
		// rho = max(rho, r);
		if(r > rho) {
			rho = r;
			best_splus = split.splus_to_string();
		}
	}
	ofstream fout("tmp/s_plus");
	fout << best_splus << endl;
	cout << "rho = " << rho << endl;
	gst_close_geosteiner();
	return 0;
}
