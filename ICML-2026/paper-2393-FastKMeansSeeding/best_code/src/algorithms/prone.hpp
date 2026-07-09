#pragma once

#include "src/core/dataset.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>

namespace fastcluster {

/* ============================================================
   Segment Tree for exact D^2 sampling (PRONE)
   ============================================================ */
class D2SegmentTree {
public:
    explicit D2SegmentTree(size_t n)
        : n_(n), tree_(4 * n, 0.0f) {}

    void build(const std::vector<float>& p) {
        build_rec(1, 0, n_ - 1, p);
    }

    void update(size_t idx, float value) {
        update_rec(1, 0, n_ - 1, idx, value);
    }

    float total() const {
        return tree_[1];
    }

    size_t sample(float u) const {
        return sample_rec(1, 0, n_ - 1, u);
    }

private:
    size_t n_;
    std::vector<float> tree_;

    void build_rec(size_t v, size_t l, size_t r,
                   const std::vector<float>& p) {
        if (l == r) {
            tree_[v] = p[l];
        } else {
            size_t m = (l + r) / 2;
            build_rec(v * 2, l, m, p);
            build_rec(v * 2 + 1, m + 1, r, p);
            tree_[v] = tree_[v * 2] + tree_[v * 2 + 1];
        }
    }

    void update_rec(size_t v, size_t l, size_t r,
                    size_t idx, float value) {
        if (l == r) {
            tree_[v] = value;
        } else {
            size_t m = (l + r) / 2;
            if (idx <= m)
                update_rec(v * 2, l, m, idx, value);
            else
                update_rec(v * 2 + 1, m + 1, r, idx, value);
            tree_[v] = tree_[v * 2] + tree_[v * 2 + 1];
        }
    }

    size_t sample_rec(size_t v, size_t l, size_t r, float u) const {
        if (l == r) return l;
        float left_sum = tree_[v * 2];
        size_t m = (l + r) / 2;
        if (u <= left_sum)
            return sample_rec(v * 2, l, m, u);
        else
            return sample_rec(v * 2 + 1, m + 1, r, u - left_sum);
    }
};

/* ============================================================
   PRONE: exact O(nd + n log n) implementation
   ============================================================ */
class PRONE {
public:
    explicit PRONE(uint64_t seed = 42)
        : rng_(seed) {}

    Dataset run(const Dataset& X, size_t k) {
        const size_t n = X.size();
        const size_t d = X.dim();
        assert(k <= n);

        /* ----------------------------
           Step 1: random 1D projection
           ---------------------------- */
        std::vector<float> v(d);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t j = 0; j < d; ++j)
            v[j] = gauss(rng_);

        std::vector<P1D> pts(n);
        for (size_t i = 0; i < n; ++i) {
            float dot = 0.0f;
            const float* xi = X.row_ptr(i);
            for (size_t j = 0; j < d; ++j)
                dot += xi[j] * v[j];
            pts[i] = { dot, i };
        }

        std::sort(pts.begin(), pts.end(),
                  [](const P1D& a, const P1D& b) {
                      return a.x < b.x;
                  });

        /* ----------------------------
           Step 2: initialize distances
           ---------------------------- */
        std::vector<float> p(n,
            std::numeric_limits<float>::infinity());

        std::vector<size_t> centers;
        centers.reserve(k);

        std::uniform_int_distribution<size_t> unif(0, n - 1);
        size_t first = unif(rng_);
        centers.push_back(first);

        update_interval(pts, p, first, nullptr);

        D2SegmentTree tree(n);
        tree.build(p);

        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);

        /* ----------------------------
           Step 3: k-means++ loop
           ---------------------------- */
        while (centers.size() < k) {
            float Z = tree.total();
            float u = unif01(rng_) * Z;
            size_t c = tree.sample(u);

            centers.push_back(c);
            update_interval(pts, p, c, &tree);
        }

        /* ----------------------------
           Step 4: lift centers back
           ---------------------------- */
        Dataset C(k, d);
        for (size_t i = 0; i < k; ++i) {
            size_t idx = pts[centers[i]].idx;
            std::copy(
                X.row_ptr(idx),
                X.row_ptr(idx) + d,
                C.row_ptr(i)
            );
        }

        return C;
    }

private:
    struct P1D {
        float x;
        size_t idx;
    };

    std::mt19937_64 rng_;

    static void update_interval(
        const std::vector<P1D>& pts,
        std::vector<float>& p,
        size_t c,
        D2SegmentTree* tree
    ) {
        const float cx = pts[c].x;

        /* ---- right scan ---- */
        for (size_t i = c; i < pts.size(); ++i) {
            float d = pts[i].x - cx;
            float d2 = d * d;
            if (d2 < p[i]) {
                p[i] = d2;
                if (tree) tree->update(i, d2);
            } else {
                break;
            }
        }

        /* ---- left scan ---- */
        for (size_t i = c; i-- > 0;) {
            float d = pts[i].x - cx;
            float d2 = d * d;
            if (d2 < p[i]) {
                p[i] = d2;
                if (tree) tree->update(i, d2);
            } else {
                break;
            }
        }
    }
};

} // namespace fastcluster
