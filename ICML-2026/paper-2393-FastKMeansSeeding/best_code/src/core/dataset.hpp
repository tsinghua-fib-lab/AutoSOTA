#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastcluster {

/* ================================
   Point view (zero-cost abstraction)
   ================================ */
struct PointView {
    const float* ptr;
    size_t d;

    inline float operator[](size_t i) const {
        return ptr[i];
    }
};

/* ================================
   Dataset
   ================================ */
class Dataset {
private:
    size_t n_;     // number of points
    size_t d_;     // dimension
    std::vector<float> data_;  // contiguous: n_ * d_

public:
    Dataset() : n_(0), d_(0) {}

    Dataset(size_t n, size_t d)
        : n_(n), d_(d), data_(n * d) {}

    /* ----------------------------
       Basic accessors
       ---------------------------- */
    inline size_t size() const { return n_; }
    inline size_t dim()  const { return d_; }

    inline const float* row_ptr(size_t i) const {
        return data_.data() + i * d_;
    }

    inline float* row_ptr(size_t i) {
        return data_.data() + i * d_;
    }

    inline PointView point(size_t i) const {
        return PointView{ row_ptr(i), d_ };
    }

    /* ----------------------------
       Load from text file
       Assumes: one point per line,
       space-separated floats
       ---------------------------- */
    void fromTxt(const std::string& filePath) {
        std::ifstream file(filePath);
        assert(file && "Failed to open file");

        std::vector<float> temp;
        std::string line;
        size_t inferred_d = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            float val;
            size_t cnt = 0;
            while (ss >> val) {
                temp.push_back(val);
                cnt++;
            }
            if (inferred_d == 0) inferred_d = cnt;
            else assert(cnt == inferred_d && "Inconsistent dimensions");
        }

        d_ = inferred_d;
        n_ = temp.size() / d_;
        data_.swap(temp);
    }

    /* ================================
       Core computational kernels
       ================================ */

    /* --------------------------------
       Squared L2 distance
       -------------------------------- */
    static inline float l2_sq(
        const float* __restrict__ x,
        const float* __restrict__ y,
        size_t d
    ) {
        float acc = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            float diff = x[i] - y[i];
            acc += diff * diff;
        }
        return acc;
    }

    inline float distance(size_t i, size_t j) const {
        return std::sqrt(l2_sq(row_ptr(i), row_ptr(j), d_));
    }

    /* --------------------------------
       Min distance of point i to centers
       -------------------------------- */
    float min_distance(
        size_t i,
        const Dataset& centers
    ) const {
        assert(d_ == centers.d_);

        const float* x = row_ptr(i);
        float best = std::numeric_limits<float>::max();

        for (size_t c = 0; c < centers.n_; ++c) {
            float dist = l2_sq(x, centers.row_ptr(c), d_);
            if (dist < best) best = dist;
        }
        return std::sqrt(best);
    }

    /* --------------------------------
       Clustering cost:
       sum_i min_c ||x_i - c||^2
       -------------------------------- */
    float clustering_cost(const Dataset& centers) const {
        assert(d_ == centers.d_);

        float total = 0.0f;

        #pragma omp parallel for reduction(+:total) schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            const float* x = row_ptr(i);
            float best = std::numeric_limits<float>::max();

            for (size_t c = 0; c < centers.n_; ++c) {
                float dist = l2_sq(x, centers.row_ptr(c), d_);
                if (dist < best) best = dist;
            }
            total += best;
        }

        return total;
    }
};

} // namespace fastcluster
