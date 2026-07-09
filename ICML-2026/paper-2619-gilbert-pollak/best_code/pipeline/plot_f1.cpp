#include <bits/stdc++.h>
using namespace std;
using ld = double;
using ull = unsigned long long;
const ld INF = 1e18, eps = 1e-6, sqrt3 = sqrt(3.0l);
const int n = 5;
const int F_VAL = 1; // 1 for f = 0
ld rho;
template<typename... Args>
ld max(ld a, ld b, ld c, Args... args)
{
	return max({a, b, c, args...});
}

template<typename To, typename From>
To bit_cast(const From& from)
{
	To to;
	memcpy(&to, &from, sizeof(To));
	return to;
}
bool isinfinity(ld x)
{
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

#include "formulas/F0"
#include "formulas/F1"
#include "formulas/F2"
#include "formulas/F3"
#include "formulas/F4"
#include "formulas/F5"
#include "formulas/F6"
#include "formulas/F7"
#include "formulas/F8"
const int m = M8;

template<template<int> class F, int N>
auto eval_all(ull mono_mask, ld b, ld c, ld d, ld s, ld e)
{
	ld f = F_VAL == 1 ? 0 : d;
	auto imp = [&]<int... I>(integer_sequence<int, I...>) -> array<ld, sizeof...(I)> {
		return {F<I>{}(mono_mask, b, c, d, s, e, f)...};
	};
	return imp(make_integer_sequence<int, N>{});
}

template<typename T>
struct TaskPool
{
	queue<T> que;
	size_t working;
	bool stop;
	mutex mut;
	condition_variable cv;

	TaskPool() : working(0), stop(false) {}
	optional<T> try_pop()
	{
		unique_lock lock(mut);
		cv.wait(lock, [&]{return stop || !que.empty();});
		if(stop) return {};
		auto head = que.front(); que.pop();
		working++;
		return head;
	}
	template<typename Arg>
	void push(Arg &&arg)
	{
		lock_guard lock(mut);
		que.push(forward<Arg>(arg));
		cv.notify_one();
	}
	template<typename... Args>
	void push_children(Args&&... args)
	{
		lock_guard lock(mut);
		(que.push(forward<Args>(args)), ...);
		working--;
		if(que.empty() && working == 0) stop = true;
		cv.notify_all();
	}
};

int main(int argc, char **argv)
{
	// argv[1] == 8282 -> rho = 0.8282
	// argv[2] == iteration budget (default 2e7)
	rho = atoi(argv[1]) / 10000.0;
	ull iter_budget = (argc >= 3) ? atoll(argv[2]) : 20000000;
	cerr << "ploting " << fixed << setprecision(4) << rho << " F_VAL=1 (f=0)" << endl; 
	cerr << "iter_budget = " << iter_budget << endl;
	ostringstream oss; oss << "unbounded_rho=" << fixed << setprecision(4) << rho << "_f=0.csv"; string file_name = oss.str();
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

	vector<pair<Box, int>> certified;

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
			auto box = *h;
			lock.lock();
			iter++;
			if(iter % 100000 == 0) cerr<<iter<<" "<<certified.size()<<endl;
			if(iter > iter_budget) break;
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
				certified.emplace_back(move(box), -1);
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
				certified.emplace_back(move(box), id);
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
			pool.push_children(move(boxl), move(boxr));
		}
	};

	auto n_threads = thread::hardware_concurrency();
	cerr<<"n_threads = "<<n_threads<<endl;
	vector<thread> workers;
	workers.reserve(n_threads);
	for(unsigned _ = 0; _ < n_threads; _++) workers.emplace_back(worker);
	for(auto &&t : workers) t.join();

	cout<<fixed<<setprecision(8);
	Point maxn, minn;
	maxn.fill(0); minn.fill(INF);
	cout<<certified.size()<<" "<<pool.que.size()<<endl;
	ld unproven_area = 0, box_area = 1;

	if(pool.que.empty()) {
		cerr << "Proof Success" << endl;
		return 0;
	}

	cerr << "Proof Failed" << endl;
	while(!pool.que.empty())
	{
		auto box = pool.que.front(); pool.que.pop();
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
	cerr<<"Area:"<<fixed<<setprecision(8)<<unproven_area<<" "<<box_area<<endl;
	for(int i = 0; i < n; i++) cout<<minn[i]<<" "<<maxn[i]<<endl;
	return 0;
}
