# include <bits/stdc++.h>
using namespace std;
using ld = double;
using ull = unsigned long long;
const ld rho = 0.8559, INF = 1e18, eps = 1e-6, sqrt3 = sqrt(3.0l);
const int n = 5;
const int F_VAL = 2; // 2 for f >= d
const string suffix = format("_rho={}_{}.bin", rho, F_VAL == 1 ? "f_le_d"s : "f_ge_d"s);

template<typename... Args>
ld max(ld a, ld b, ld c, Args... args)
{
	return max({a, b, c, args...});
}

constexpr bool isinfinity(ld x)
{
	// return isinf(x);
	return bit_cast<ull>(x) == bit_cast<ull>(numeric_limits<ld>::infinity());
}

using Box = array<array<ld, 2>, n>;
using Point = array<ld, n>;

const array<string, n> vars = {"b", "c", "d", "s", "e"};

bool glob_cond(ld b, ld c, ld d, ld s, ld e)
{
	return true;
}

template<int>
struct F;

# include "formulas/F0"
# include "formulas/F1"
# include "formulas/F2"
# include "formulas/F3"
# include "formulas/F4"
# include "formulas/F5"
# include "formulas/F6"
# include "formulas/F7"
# include "formulas/F8"

const int m = M8;

template<template<int> class F, int N>
auto eval_all(ull mono_mask, ld b, ld c, ld d, ld s, ld e)
{
	ld f = d;
	auto imp = [&]<int... I>(integer_sequence<int, I...>) -> array<ld, sizeof...(I)> {
		return {F<I>{}(mono_mask, b, c, d, s, e, f)...};
	};
	return imp(make_integer_sequence<int, N>{});
}

template<template<int> class F, int N>
constexpr auto make_split_id_array()
{
	auto imp = [&]<int... I>(integer_sequence<int, I...>) -> array<int, sizeof...(I)> {
		return {F<I>::split_id...};
	};
	return imp(make_integer_sequence<int, N>{});
}
constexpr auto split_ids = make_split_id_array<F, m>();

template<template<int> class F, int N>
constexpr auto make_lemma_id_array()
{
	auto imp = [&]<int... I>(integer_sequence<int, I...>) -> array<int, sizeof...(I)> {
		return {F<I>::lemma_id...};
	};
	return imp(make_integer_sequence<int, N>{});
}
constexpr auto lemma_ids = make_lemma_id_array<F, m>();

template<typename T>
struct TaskPool
{
	queue<pair<size_t, T>> que;
	size_t working, tot;
	bool stop;
	mutex mut;
	condition_variable cv;

	TaskPool() : working(0), tot(0), stop(false) {}
	optional<pair<size_t, T>> try_pop()
	{
		unique_lock lock(mut);
		cv.wait(lock, [&]{return stop || !que.empty();});
		if(stop) return {};
		auto head = que.front(); que.pop();
		working++;
		return head;
	}
	template<typename Arg>
	size_t push(Arg &&arg)
	{
		lock_guard lock(mut);
		size_t ID = ++ tot;
		que.emplace(ID, forward<Arg>(arg));
		cv.notify_one();
		return ID;
	}
	template<typename Arg>
	size_t _unlocked_push(Arg &&arg)
	{
		int ID = ++ tot;
		que.emplace(ID, forward<Arg>(arg));
		return ID;
	}
	template<typename... Args>
	auto push_children(Args&&... args)
	{
		lock_guard lock(mut);
		auto ret = make_tuple(_unlocked_push(forward<Args>(args)) ...);
		working--;
		if(que.empty() && working == 0) stop = true;
		cv.notify_all();
		return ret;
	}
};

int main()
{
	ull iter = 0;
	TaskPool<Box> pool;
	for(ull mask = 0; mask < 1ull<<n; mask++)
	{
		Box box;
		for(int i = 0; i < n; i++)
			if((mask >> i) & 1) box[i][0] = 1, box[i][1] = numeric_limits<ld>::infinity();
			else box[i][0] = 0, box[i][1] = 1;
		pool.push(move(box));
	}

	vector<tuple<size_t, Box, int>> certified;
	vector<pair<size_t, size_t>> child;

	auto worker = [&]()
	{
		vector<Point> corners;
		corners.reserve(1ull<<n);
		array<ld, m> maxn;
		unique_lock lock(pool.mut, defer_lock);
		while(true)
		{
			auto h = pool.try_pop();
			if(!h.has_value()) break;
			auto [ID, box] = *h;
			lock.lock();
			iter++;
			if(iter % 100000 == 0) cerr<<iter<<" "<<certified.size()<<endl;
			if(iter > 1e9) break;
			lock.unlock();
			bool not_in_domain = true;
			corners.clear();
			for(ull mask = 0; mask < 1ull<<n; mask++)
			{
				Point p;
				bool no_inf = true;
				for(int i = 0; i < n; i++)
				{
					p[i] = (mask >> i) & 1 ? box[i][0] : box[i][1];
					if(isinfinity(p[i])) no_inf = false;
				}
				if(no_inf) corners.push_back(move(p));
				if(apply(glob_cond, p)) not_in_domain = false;
			}
			if(not_in_domain)
			{
				lock.lock();
				certified.emplace_back(ID, move(box), -1);
				lock.unlock();
				pool.push_children();
				continue;
			}
			ull mono_mask = 0;
			for(int i = 0; i < n; i++)
				if(isinfinity(box[i][1])) mono_mask |= 1ull<<i;
			maxn.fill(-INF);
			auto eval = [mono_mask]<typename... Args>(Args&&... args) {
				return eval_all<F, m>(mono_mask, forward<Args>(args)...);
			};
			for(auto &&p : corners)
			{
				auto val = apply(eval, p);
				for(int i = 0; i < m; i++) maxn[i] = max(maxn[i], val[i]);
			}
			auto id = min_element(maxn.begin(), maxn.end()) - maxn.begin();
			if(maxn[id] < -eps)
			{
				lock.lock();
				certified.emplace_back(ID, move(box), id);
				lock.unlock();
				pool.push_children();
				continue;
			}
			auto dim = max_element(box.begin(), box.end(), [](auto &&x, auto &&y) {
				ld sepx = isinfinity(x[1]) ? 1 / x[0] : x[1] - x[0];
				ld sepy = isinfinity(y[1]) ? 1 / y[0] : y[1] - y[0];
				return sepx < sepy;
			}) - box.begin();
			Box boxl = box, boxr = box;
			if(isinfinity(box[dim][1])) boxl[dim][1] = boxr[dim][0] = 2 * box[dim][0];
			else boxl[dim][1] = boxr[dim][0] = (box[dim][0] + box[dim][1]) / 2;
			auto [left_ID, right_ID] = pool.push_children(move(boxl), move(boxr));
			lock_guard lock(pool.mut);
			if(child.size() < ID + 1) child.resize(ID + 1);
			child[ID] = {left_ID, right_ID};
		}
	};

	auto n_threads = thread::hardware_concurrency();
	cout<<"n_threads = "<<n_threads<<endl;
	vector<thread> workers;
	workers.reserve(n_threads);
	for(unsigned _ = 0; _ < n_threads; _++) workers.emplace_back(worker);
	for(auto &&t : workers) t.join();
	child.resize(pool.tot + 1);

	cout<<fixed<<setprecision(20);
	Point maxn, minn;
	maxn.fill(0); minn.fill(INF);
	cout<<certified.size()<<" "<<pool.que.size()<<endl;
	ld unproven_area = 0, box_area = 1;
	while(!pool.que.empty())
	{
		auto [_, box] = pool.que.front(); pool.que.pop();
		ld vol = 1;
		for(int i = 0; i < n; i++)
		{
			vol *= box[i][1] - box[i][0];
			minn[i] = min(minn[i], box[i][0]);
			maxn[i] = max(maxn[i], box[i][1]);
		}
		unproven_area += vol;
	}
	for(int i = 0; i < n; i++) box_area *= max(maxn[i] - minn[i], 0.);
	cout<<unproven_area<<" "<<box_area<<endl;
	for(int i = 0; i < n; i++) cout<<vars[i]<<" in ["<<minn[i]<<", "<<maxn[i]<<"]"<<endl;
	ofstream fcert("certificate" + suffix, ios::binary);
	for(auto &&[ID, box, id] : certified)
	{
		fcert.write((char *)&ID, 4);
		for(int i = 0; i < n; i++)
		{
			fcert.write((char *)&box[i][0], sizeof(box[i][0]));
			fcert.write((char *)&box[i][1], sizeof(box[i][1]));
		}
		int split_id = -1, lemma_id = -1;
		if(id >= 0)
		{
			split_id = split_ids[id];
			lemma_id = lemma_ids[id];
		}
		fcert.write((char *)&split_id, 4);
		fcert.write((char *)&lemma_id, 4);
	}
	ofstream fchild("child" + suffix, ios::binary);
	for(size_t i = pool.tot; i >= 1; i--)
	{
		fchild.write((char *)&child[i].first, 4);
		fchild.write((char *)&child[i].second, 4);
	}
	return 0;
}