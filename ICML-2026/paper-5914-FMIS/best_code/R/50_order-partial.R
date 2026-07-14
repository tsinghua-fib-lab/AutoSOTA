# Partial ordering: get top-k indices efficiently

if (requireNamespace("Rcpp", quietly = TRUE)) {
  old_cxxflags <- Sys.getenv("PKG_CXXFLAGS", unset = NA)
  old_cflags <- Sys.getenv("PKG_CFLAGS", unset = NA)
  Sys.setenv("PKG_CXXFLAGS" = "-O3 -march=native -mtune=native -funroll-loops")
  Sys.setenv("PKG_CFLAGS" = "-O3 -march=native -mtune=native -funroll-loops")
  Rcpp::sourceCpp(
    code = '
#include <Rcpp.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace Rcpp;

struct __attribute__((packed)) Node {
  double v;
  int idx;
};

struct BetterDec {
  inline bool operator()(const Node& a, const Node& b) const noexcept {
    if (a.v > b.v) return true;
    if (a.v < b.v) return false;
    return a.idx < b.idx;
  }
};

struct BetterInc {
  inline bool operator()(const Node& a, const Node& b) const noexcept {
    if (a.v < b.v) return true;
    if (a.v > b.v) return false;
    return a.idx < b.idx;
  }
};

template <class Better>
static inline IntegerVector topk_heap(const double* xp, R_xlen_t n, int k, const Better& better) {
  if (n == 0 || k <= 0) return IntegerVector();
  if (k > (int)n) k = (int)n;
  if (k == 1) {
    R_xlen_t best = 0;
    for (R_xlen_t i = 1; i < n; ++i) {
      Node a{xp[i], (int)i}; Node b{xp[best], (int)best};
      if (better(a, b)) best = i;
    }
    IntegerVector out(1); out[0] = (int)best + 1; return out;
  }
  std::priority_queue<Node, std::vector<Node>, Better> pq(better);
  for (int i = 0; i < k; ++i) pq.push(Node{xp[i], i});
  for (R_xlen_t i = (R_xlen_t)k; i < n; ++i) {
    Node cand{xp[i], (int)i}; const Node& worst = pq.top();
    if (better(cand, worst)) { pq.pop(); pq.push(cand); }
  }
  std::vector<Node> res; res.reserve((size_t)k);
  while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
  std::sort(res.begin(), res.end(), [&](const Node& a, const Node& b) { return better(a, b); });
  IntegerVector out(k);
  for (int i = 0; i < k; ++i) out[i] = res[(size_t)i].idx + 1;
  return out;
}

// [[Rcpp::export]]
IntegerVector order_partial_cpp(SEXP xSEXP, int k, bool decreasing) {
  if (TYPEOF(xSEXP) != REALSXP) stop("Input must be a double (REALSXP)");
  const R_xlen_t n = XLENGTH(xSEXP);
  if (n > (R_xlen_t)std::numeric_limits<int>::max())
    stop("x is too long: indices must fit in 32-bit integer");
  const double* xp = REAL(xSEXP);
  if (decreasing) return topk_heap(xp, n, k, BetterDec{});
  else return topk_heap(xp, n, k, BetterInc{});
}

// [[Rcpp::export]]
List dinkelbach_mis_xr(NumericVector X, NumericVector R, int sign, int k, double tol, int max_iter) {
  int n = X.size();
  const double* __restrict__ pX = REAL(X);
  const double* __restrict__ pR = REAL(R);

  std::vector<double> num_vec(n);
  std::vector<double> den_vec(n);
  double* __restrict__ pnum = num_vec.data();
  double* __restrict__ pden = den_vec.data();
  double tot = 0.0;
  double s = (double)sign;

  #pragma GCC ivdep
  for (int i = 0; i < n; ++i) {
    double xi = pX[i];
    pnum[i] = xi * pR[i] * s;
    double d = xi * xi;
    pden[i] = d;
    tot += d;
  }

  // Smart init: compute ratios and find top-k by ratio for warm start
  std::vector<double> ratios(n);
  #pragma GCC ivdep
  for (int i = 0; i < n; ++i) {
    ratios[i] = pnum[i] / (tot - pden[i]);
  }
  std::nth_element(ratios.begin(), ratios.begin() + (k - 1), ratios.end(),
                   [](double a, double b) { return a > b; });
  double threshold = ratios[k - 1];

  double num_init = 0.0, den_init = 0.0;
  int init_count = 0;
  for (int i = 0; i < n && init_count < k; ++i) {
    double r = pnum[i] / (tot - pden[i]);
    if (r >= threshold) {
      num_init += pnum[i];
      den_init += pden[i];
      ++init_count;
    }
  }
  // Fallback: if we got fewer than k due to floating-point ties
  while (init_count < k) {
    for (int i = 0; i < n && init_count < k; ++i) {
      double r = pnum[i] / (tot - pden[i]);
      if (r < threshold) {
        num_init += pnum[i];
        den_init += pden[i];
        ++init_count;
      }
    }
    threshold *= 0.999999;
  }

  double lambda = num_init / (tot - den_init);

  // Pre-allocated working arrays for Dinkelbach iterations
  std::vector<Node> nodes(n);
  IntegerVector S_prev, S_i(k);

  for (int iter = 0; iter < max_iter; ++iter) {
    // Build node array: fused score computation with prefetch hints
    Node* __restrict__ pnodes = nodes.data();
    #pragma GCC ivdep
    for (int i = 0; i < n; ++i) {
      pnodes[i].v = pnum[i] + lambda * pden[i];
      pnodes[i].idx = i;
    }

    // nth_element + sort for top-k (decreasing)
    auto dec_cmp = [](const Node& a, const Node& b) {
      if (a.v > b.v) return true;
      if (a.v < b.v) return false;
      return a.idx < b.idx;
    };
    std::nth_element(nodes.begin(), nodes.begin() + (k - 1), nodes.end(), dec_cmp);
    std::sort(nodes.begin(), nodes.begin() + k, dec_cmp);

    // Subset sum with software prefetch
    double num_sum = 0.0, den_sum = 0.0;
    const int PREFETCH_DIST = 16;
    for (int i = 0; i < PREFETCH_DIST && i < k; ++i) {
      __builtin_prefetch(&pnum[nodes[i].idx], 0, 3);
      __builtin_prefetch(&pden[nodes[i].idx], 0, 3);
    }
    for (int i = 0; i < k; ++i) {
      if (i + PREFETCH_DIST < k) {
        __builtin_prefetch(&pnum[nodes[i + PREFETCH_DIST].idx], 0, 3);
        __builtin_prefetch(&pden[nodes[i + PREFETCH_DIST].idx], 0, 3);
      }
      int idx = nodes[i].idx;
      num_sum += pnum[idx];
      den_sum += pden[idx];
      S_i[i] = idx + 1;
    }

    double tot_i = tot - den_sum;
    if (tot_i <= 0.0) stop("Non-positive denominator reached -- check rank.");
    double lambda_i = num_sum / tot_i;

    bool converged = false;
    if (S_prev.size() > 0) {
      bool same_set = (S_i.size() == S_prev.size());
      if (same_set) {
        for (int i = 0; i < S_i.size() && same_set; ++i) {
          if (S_i[i] != S_prev[i]) same_set = false;
        }
      }
      if (same_set || std::abs(lambda_i - lambda) < tol) converged = true;
    }

    if (converged) {
      return List::create(
        Named("best_S") = S_i, Named("best_value") = lambda_i, Named("iter") = iter + 1);
    }

    lambda = lambda_i;
    S_prev = clone(S_i);
  }

  Rcpp::warning("Did not converge; increase max_iter or check data");
  return List::create(
    Named("best_S") = S_i, Named("best_value") = lambda, Named("iter") = max_iter);
}
'  ,
    rebuild = TRUE
  )
  order_partial <- function(x, k, decreasing = FALSE) {
    order_partial_cpp(x, as.integer(k), isTRUE(decreasing))
  }
} else {
  message("Rcpp not available; using plain R for (partial) order.")
  order_partial <- function(x, k, decreasing = FALSE) {
    order(x, decreasing = decreasing)[seq_len(min(k, length(x)))]
  }
  dinkelbach_mis_xr <- NULL
}
