#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <cstdint>

using namespace std;

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) {
      return static_cast<int>(__popcnt64(x));
  }
#else
  static inline int popcnt64(uint64_t x) {
      return __builtin_popcountll(x);
  }
#endif

using Lit = uint16_t; // 16-bit literal = 2*feat + sign
using PathKey = std::vector<Lit>;
static constexpr int DEFER_PREDICTION = -1;

struct Packed {
    vector<uint64_t> w; // words (64-bit each)
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {} // allocates a vector of nwords many 64-bit words, with all bits off

    inline void clear() { std::fill(w.begin(), w.end(), 0ULL); } // reset the mask to all bits off to reuse the same object without reallocating
    inline bool any() const { // checks if any bits are set to 1
        for (uint64_t t : w) if (t) return true;
        return false;
    }
    inline int count() const { // how many bits are 1 in total? for each word, queries popcount for 1 bits in that word.
        int s = 0;
        for (uint64_t t : w) s += popcnt64(t);
        return s;
    }
};

// scramble a 64-bit value
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

// hash the array of words into a 64-bit hash value: many-to-one in theory, but with our expected amount of pruning, something like 54k total keys for a reasonably sized rashomon set computation, which yields something like 10^-11 probability of having a collision somewhere.
// https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
static inline uint64_t hash_mask64(const uint64_t* w, int n_words, uint64_t tail_mask) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;
        uint64_t m = mix64(x);
        h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// 2*feat + sign
static inline Lit enc_lit(int feat, int sign01) {
    // requires feat <= 32767 so that (feat<<1 | sign) fits in 16 bits
    return (Lit)((feat << 1) | (sign01 & 1));
}

static inline const PathKey& empty_pk() {
    static const PathKey k;
    return k;
}

// insert literal into PathKey, maintaining sorted canonical order
static inline void pk_insert_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.insert(it, lit);
}

// remove literal from PathKey (must exist)
static inline void pk_erase_sorted(PathKey& pk, Lit lit) {
    auto it = std::lower_bound(pk.begin(), pk.end(), lit);
    pk.erase(it);
}

struct PackedPredMulti {
    std::vector<Packed> by_class; // size = num_classes, each is n_words over eval rows
    Packed deferred; // rows deferred by this tree
};

// bucket by objective
struct ObjBucketMulti {
    int obj;
    std::vector<PackedPredMulti> preds; // one entry per tree at this objective
};

struct PredPackWithObj {
    int obj;
    PackedPredMulti pred;
};

// used for exact, non-probabilistic keyks at the expense of more memory. we intern the exact bytes of a mask/bitvector and assign a small integer ID.
// first unique mask id 0, second unique mask id 1 and so on.
class MaskIdTable {
public:
    uint32_t intern(const Packed& mask, int n_words, uint64_t tail_mask) {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t); // constant across the dataset, how many words needed * 64 bit length
        string key;
        key.resize(bytes);
        // uint64_t* out = reinterpret_cast<uint64_t*>(&key[0]); // pointer to the start of key
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[i];
            if (i == n_words - 1) x &= tail_mask; // the last word may have padding bits, tail_mask zeroes out the unused bits.
            // out[i] = x; // the byte representation of mask.w - we need to convert to use as a key in the unordered map
            std::memcpy(&key[i * sizeof(uint64_t)], &x, sizeof(uint64_t));
        }
        auto it = table.find(key); // have we seen this bitmask before?
        if (it != table.end()) return it->second; // return the previously assigned id if it points to the entry, meaning we have it already
        uint32_t id = (uint32_t)pool_size++; // use the value, then increment it
        table.emplace(std::move(key), id); // store without copying
        return id;
    }

    size_t size() const { return pool_size; }

private:
    unordered_map<string, uint32_t> table;
    size_t pool_size = 0;
};

class LitIdTable {
public:
    uint32_t intern(const std::vector<Lit>& lits) {
        const size_t bytes = lits.size() * sizeof(Lit);
        std::string key;
        key.resize(bytes);
        if (bytes) std::memcpy(&key[0], lits.data(), bytes);

        auto it = table.find(key);
        if (it != table.end()) return it->second;
        uint32_t id = (uint32_t)pool_size++;
        table.emplace(std::move(key), id);
        return id;
    }


    size_t size() const { return pool_size; }

private:
    std::unordered_map<std::string, uint32_t> table;
    size_t pool_size = 0;
};


// two structures to define the key type used in hash maps
// K2: for greedy and lickety cache (subproblem, depth)
// K3: tries (subproblem, depth, budget)
// define equality with operator== and how to hash the keys for unordered_map

struct K2 {
    uint64_t k; // hash or interned-id
    int depth;
    bool operator==(const K2& o) const { return k == o.k && depth == o.depth; } // element-wise equality
    struct Hash { // custom hash
        size_t operator()(const K2& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct K3 {
    uint64_t k; // hash or interned-id
    int depth;
    int budget;
    bool operator==(const K3& o) const { return k == o.k && depth == o.depth && budget == o.budget; }
    struct Hash {
        size_t operator()(const K3& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t b = (size_t)x.budget;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= b + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct KLA {
    uint64_t k;
    int depth;
    int la; // lookahead used for this call
    bool operator==(const KLA& o) const { return k == o.k && depth == o.depth && la == o.la; }
    struct Hash {
        size_t operator()(const KLA& x) const noexcept {
            size_t h = (size_t)x.k;
            size_t d = (size_t)x.depth;
            size_t a = (size_t)x.la;
            h ^= d + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            return h;
        }
    };
};

struct HistEntry {
    int obj;
    uint64_t cnt;
};
static inline bool hist_less(const HistEntry& a, const HistEntry& b){ return a.obj < b.obj; } // helper for sorting

struct TreeTrieNode; // fwd

struct LeafNode {
    // 0..C-1 means predict class.
    // DEFER_PREDICTION means defer to black-box.
    int prediction;

    //   predict leaf: gamma + mistakes
    //   defer leaf:   gamma + round(eta_defer * n_leaf) + bb_mistakes
    int loss;
};

struct SplitNode {
    int feature = -1;
    shared_ptr<TreeTrieNode> left;
    shared_ptr<TreeTrieNode> right;
    uint64_t num_valid_trees = 0; // trees contributed by this split under parent's budget
};

struct TreeTrieNode {
    int budget = 0;
    int min_objective = numeric_limits<int>::max();
    vector<LeafNode> leaves; // (prediction,loss) for as many [<=2 in binary classification] if within budget
    vector<SplitNode> splits; // stores splitnodes which have the feature they split on and left and right trienodes
    vector<HistEntry> hist; // sorted ascending by obj; counts aggregated (obj, count) are elements
    bool hist_built = false; // wait until the end to build the histograms because we don't know if they'll be used in the final trie because of multipass

    uint64_t count_trees() const {
        ensure_hist_built();
        uint64_t s = 0;
        for (const auto& e : hist) s += e.cnt;
        return s;
    }

    uint64_t count_leq(int objective) const {
        ensure_hist_built();
        if (hist.empty()) return 0ULL;
        uint64_t total = 0ULL;
        for (const auto& e : hist) {
            if (e.obj > objective) break; // hist is sorted ascending by obj
            total += e.cnt;
        }
        return total;
    }

    void add_hist(int obj, uint64_t add_cnt = 1) {
        auto it = lower_bound(hist.begin(), hist.end(), HistEntry{obj,0}, hist_less); // find the first poisition in hist where obj could be inserted without breaking sort order
        if (it != hist.end() && it->obj == obj) it->cnt += add_cnt; // if it already exists, just increment
        else hist.insert(it, HistEntry{obj, add_cnt}); // otherwise, add
        if (obj < min_objective) min_objective = obj; // keep min_objective fresh
    }
    
    void add_leaf(int prediction, int loss) {
        leaves.push_back(LeafNode{prediction, loss});
        if (loss < min_objective) min_objective = loss;
    }
    
    void add_leaf_and_build(int prediction, int loss) { // assumes you call within budget
        leaves.push_back(LeafNode{prediction, loss});
        add_hist(loss, 1);
    }

    void add_split(int feat,
               const shared_ptr<TreeTrieNode>& L,
               const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;
        s.num_valid_trees = 0; // will be filled in post-processing

        if (L && R) {
            int min_sum = (L->min_objective == numeric_limits<int>::max() ||
                        R->min_objective == numeric_limits<int>::max())
                        ? numeric_limits<int>::max()
                        : (L->min_objective + R->min_objective);
            if (min_sum < min_objective) min_objective = min_sum;
        }
        splits.push_back(std::move(s));
    }

    void add_split_and_build(int feat,
                   const shared_ptr<TreeTrieNode>& L,
                   const shared_ptr<TreeTrieNode>& R) {
        SplitNode s;
        s.feature = feat;
        s.left  = L;
        s.right = R;

        int min_sum = L->min_objective + R->min_objective;
        if (min_sum < min_objective)
            min_objective = min_sum;

        if ((L && !L->hist.empty()) && (R && !R->hist.empty())) {
            unordered_map<int, uint64_t> sum_counts; // make a temporary map to map obj -> count until we know how they distribute in full to then transfer to the vector-based histogram
            sum_counts.reserve(L->hist.size() * 2); // 2x is a good starting estimate

            uint64_t valid = 0;
            vector<int> R_objs; R_objs.reserve(R->hist.size()); // split (obj, cnt) into two parallel ararys, just for R due to the binary search needs in the future
            vector<uint64_t> R_cnts; R_cnts.reserve(R->hist.size());
            for (auto &e : R->hist) { R_objs.push_back(e.obj); R_cnts.push_back(e.cnt); }

            for (const auto& le : L->hist) {
                if (le.obj > budget) break; // should never happen by invariant but if we do lossy caching/more heuristics
                int rem = budget - le.obj; // R cannot exceed
                auto it_end = upper_bound(R_objs.begin(), R_objs.end(), rem); // find the first index strictly greater than rim
                int idx_end = (int)distance(R_objs.begin(), it_end); // gets the index of it_end (it_end is an iterator)
                for (int j = 0; j < idx_end; ++j) { // go until the last index that doesn't exceed rem
                    int tot = le.obj + R_objs[j];
                    uint64_t addc = le.cnt * R_cnts[j];
                    sum_counts[tot] += addc;
                    valid += addc;
                }
            }
            // we've updated our temporary sum_counts map, now we must merge it into the existing histogram
            if (!sum_counts.empty()) {
                vector<HistEntry> tmp; tmp.reserve(sum_counts.size());
                for (auto &kv : sum_counts) tmp.push_back(HistEntry{kv.first, kv.second}); // back the (obj, count) format and sorting
                sort(tmp.begin(), tmp.end(), hist_less);

                // now, we have to aggregate this into the histogram for all splits at that node
                vector<HistEntry> merged; merged.reserve(hist.size() + tmp.size());
                // simply merge two sorted lists into a new list and swap it in
                size_t i=0, j=0;
                while (i < hist.size() && j < tmp.size()) {
                    if (hist[i].obj < tmp[j].obj) merged.push_back(hist[i++]);
                    else if (tmp[j].obj < hist[i].obj) merged.push_back(tmp[j++]);
                    else { merged.push_back(HistEntry{hist[i].obj, hist[i].cnt + tmp[j].cnt}); ++i; ++j; }
                }
                while (i < hist.size()) merged.push_back(hist[i++]);
                while (j < tmp.size()) merged.push_back(tmp[j++]);
                hist.swap(merged);
            }
            s.num_valid_trees = valid;
        }

        splits.push_back(std::move(s)); // adding this split information to the trienode
    }

    // post-process the trie to build per-node histograms using the existing helpers.
    // assumes leaves/splits/min_objective/budget are already set by construct_trie.
    static void build_histograms_post(TreeTrieNode* node) {
        if (node->hist_built) return;

        // ensure children are processed first (post-order)
        for (auto &s : node->splits) {
            if (s.left)  build_histograms_post(s.left.get());
            if (s.right) build_histograms_post(s.right.get());
        }

        // rebuild this node's histogram from scratch
        std::vector<SplitNode> saved = std::move(node->splits);
        node->splits.clear();
        node->hist.clear();
        node->hist_built = false; // (will set true at end)

        // add leaf contributions
        for (const auto &leaf : node->leaves) {
            node->add_hist(leaf.loss, 1); // could call add_leaf_and_build but that is overkill here
        }

        // re-add splits, letting add_split_and_build do the heavy lifting:
        // merges L/R histograms into node->hist
        // computes s.num_valid_trees
        // refreshes min_objective though that isn't needed
        for (auto &s : saved) {
            node->add_split_and_build(s.feature, s.left, s.right);
        }

        node->hist_built = true;
    }

    void ensure_hist_built() const {
        if (!hist_built) {
            TreeTrieNode::build_histograms_post(const_cast<TreeTrieNode*>(this));
        }
    }

};

struct ExportLeafNode {
    int id = -1;
    int parent_trie_id = -1;
    int prediction = -1;

    // gamma + misclassification count for this leaf option
    int loss = 0;

    // same as parent OR-node subproblem size
    int subproblem_size = 0;
};

struct ExportSplitNode {
    int id = -1;
    int parent_trie_id = -1;
    int feature = -1;
    int left_trie_id = -1;
    int right_trie_id = -1;

    // best completion objective if this split is chosen:
    // left.min_objective + right.min_objective
    int min_objective = std::numeric_limits<int>::max();
};

struct ExportTreeTrieNode {
    int id = -1;

    // remaining budget at this OR node
    int budget = 0;

    // best possible completion objective from this OR node
    int min_objective = std::numeric_limits<int>::max();

    // number of active samples in this subproblem
    int subproblem_size = 0;

    std::vector<int> leaf_ids;
    std::vector<int> split_ids;
};

struct ExportANDORGraph {
    int root_trie_id = -1;
    std::vector<ExportTreeTrieNode> trie_nodes;
    std::vector<ExportSplitNode> split_nodes;
    std::vector<ExportLeafNode> leaf_nodes;
};

struct PredNode {
    int feature;  // -1 for leaf
    int prediction; // only meaningful if feature == -1
    shared_ptr<PredNode> left;
    shared_ptr<PredNode> right;
};

// for joint rashomon set prediction / rid
// struct ObjBucket {
//     int obj;
//     std::vector<Packed> preds; // each is a prediction bitvector for one tree at this obj. predictions for all trees with an objective.
// };

struct EvalCtx {
    int n_eval = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;
    std::vector<Packed> X_bits_eval; // everything needed for evaluation dataset
};

struct TrieNodeCountStats {
    std::size_t total_trie_nodes = 0;

    std::size_t distinct_subproblem_depth = 0;

    std::size_t distinct_subproblem_depth_budget = 0;
};


class PRAXIS {
public:
    enum class KeyMode { HASH64, EXACT, LITS_EXACT };

private:
    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL; // to clear high bits in last word
    int gamma = 0;
    int8_t trained_depth_budget = -1; 

    int best_objective = 0;
    int obj_bound = 0;

    double multiplicative_slack = 0.0;

    vector<Packed> X_bits; // vector of Packed, each Packed is a feature column. packed is a sequence of 64-bit words where each bit corresponds to the row value for the column
    // Packed Ypos; // each bit of a word is the label for the row
    int num_classes = 0;
    std::vector<Packed> Y_bits; // vector is size size num_classes; Y_bits[c] has 1s where y==c

    // deferral support.
    bool use_deferral = false;
    double eta_defer = 0.0;

    // training rows where black-box is wrong: bb_pred[i] != y[i].
    // used to compute defer leaf objective.
    Packed BBwrong;


    KeyMode key_mode = KeyMode::HASH64; // will change later in fit
    bool trie_cache_enabled = false;
    bool proxy_caching_enabled = true;
    mutable MaskIdTable mask_ids; // used only if in exact mode
    mutable LitIdTable lit_ids; // for itemset mode

    int lookahead_init = 1; // will be changed later
    bool use_multipass = true; // sim
    bool rule_list_mode = false;
    bool majority_leaf_only = false;
    bool cache_cheap_subproblems = false;
    bool stronger_rollout = false;
    int greedy_split_mode = 1;
    int num_proxy_features = -1; // <=0 means use all feature. positive for feature selection

    
    int proxy_style = 0; // 0=constant k (current), 1=cyclic (recursively applying split), 2=cyclic-consistent, 3=split without postprocessing, 4=split with postprocessing
    std::vector<int> k_at_depth; // size depth_budget (only used for style 2)

    unordered_map<K2, int, K2::Hash> greedy_cache;
    unordered_map<K2,  int, K2::Hash>  lickety_cache_k2; // used when lookahead_init <= 1
    unordered_map<KLA, int, KLA::Hash>  lickety_cache_kla; // used when lookahead_init > 1

    unordered_map<K3, shared_ptr<TreeTrieNode>, K3::Hash> trie_cache; // if trie_cache_enabled is on

    // bitwise and of the bitvectors represented as lists of words. makes a sparser list of words
    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        const int m = n_words;
        for (int i = 0; i < m; ++i) out.w[i] = a.w[i] & b.w[i]; //.w[i] is a word, doing bitwise ands on words for all words
        out.w[m-1] &= tail_mask; // bitwise and out.w[m-1] & tail_mask. tail_mask is 1 iff valid, so it is setting invalid bits to 0.
    }
    // marks the samples active in a but not in b.
    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        const int m = n_words;
        for (int i = 0; i < m; ++i) out.w[i] = a.w[i] & ~b.w[i];
        out.w[m-1] &= tail_mask;
    }
    // return the number of 1 bits in the intersection a & b
    inline int popcount_and(const Packed& a, const Packed& b) const {
        int s = 0;
        for (int i = 0; i < n_words; ++i) s += popcnt64(a.w[i] & b.w[i]);
        return s;
    }

    inline uint64_t key_of_mask(const Packed& mask) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            throw std::runtime_error("key_of_mask called in LITS_EXACT mode; use key_of_state(mask, pk)");
        }
        return key_of_state(mask, empty_pk());
    }



    inline uint64_t key_of_state(const Packed& mask, const PathKey& pk) const {
        switch (key_mode) {
            case KeyMode::HASH64:
                return hash_mask64(mask.w.data(), n_words, tail_mask);
            case KeyMode::EXACT:
                return (uint64_t)mask_ids.intern(mask, n_words, tail_mask); // cast 32->64
            case KeyMode::LITS_EXACT:
                return (uint64_t)lit_ids.intern(pk);
        }
        return 0;
    }

    inline uint64_t key_of_subproblem(const Packed& mask, const PathKey& pk) const {
        if (key_mode == KeyMode::LITS_EXACT) {
            return key_of_state(mask, pk); // interns pk
        } else {
            return key_of_mask(mask); 
        }
    }

    inline int proxy_feat_count_() const {
        return (num_proxy_features > 0) ? std::min(num_proxy_features, n_features) : n_features;
    }


    inline bool use_kla_cache() const { return lookahead_init > 1; }
    
    inline int count_total(const Packed& mask) const { return mask.count(); } // number of active samples
    // inline int count_pos(const Packed& mask) const { return popcount_and(mask, Ypos); } // number of active samples that are positive

    inline void count_per_class(const Packed& mask, std::vector<int>& counts) const {
        counts.assign((size_t)num_classes, 0);
        for (int c = 0; c < num_classes; ++c) {
            counts[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
        }
    }

    inline int count_class(const Packed& mask, int c) const {
        return popcount_and(mask, Y_bits[(size_t)c]);
    }


    static inline double entropy(double p) {
        const double eps = 1e-12;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    static inline double entropy_multiclass(const std::vector<int>& cnts, int n) {
        if (n <= 0) return 0.0;
        const double invn = 1.0 / (double)n;
        double H = 0.0;
        for (int c = 0; c < (int)cnts.size(); ++c) {
            const int k = cnts[(size_t)c];
            if (k <= 0) continue;
            const double p = (double)k * invn;
            H -= p * log2(p);
        }
        return H;
    }


    // count the distinct subproblems/bitvectors/literals/fingerprints, not considering depth or greedy/lickety
    size_t count_distinct_subproblems_union() const {
        std::unordered_set<uint64_t> seen;

        const size_t lsz = use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size();
        const size_t approx = greedy_cache.size() + lsz;
        seen.reserve(approx * 2 + 16);

        // greedy: key is (k, depth). we ignore depth by inserting only k.
        for (const auto& kv : greedy_cache) {
            seen.insert(kv.first.k);
        }

        // lickety: either (k, depth) or (k, depth, la). ignore depth/la by inserting only k.
        if (use_kla_cache()) {
            for (const auto& kv : lickety_cache_kla) {
                seen.insert(kv.first.k);
            }
        } else {
            for (const auto& kv : lickety_cache_k2) {
                seen.insert(kv.first.k);
            }
        }

        return seen.size();
    }

    

public:
    shared_ptr<TreeTrieNode> result;

    void set_key_mode(KeyMode m) { key_mode = m; }
    void set_trie_cache_enabled(bool on) { trie_cache_enabled = on; }
    void set_multiplicative_slack(double s) { multiplicative_slack = s; }
    void set_use_multipass(bool on) { use_multipass = on; }
    void set_rule_list_mode(bool on) { rule_list_mode = on; }
    void set_majority_leaf_only(bool on) { majority_leaf_only = on; }
    void set_cache_cheap_subproblems(bool on) { cache_cheap_subproblems = on; }
    void set_greedy_split_mode(int m) { greedy_split_mode = m; }
    void set_proxy_caching_enabled(bool on) { proxy_caching_enabled = on; }
    void set_stronger_rollout(bool on) { stronger_rollout = on; }

    void fit(const std::vector<std::vector<bool>>& X_col_major,
             const std::vector<int>& y,
             double lambda,
             int8_t depth_budget,
             double rashomon_mult,
             int8_t lookahead_k,
             int root_budget,
             bool use_multipass_flag,
             bool rule_list_mode_flag,
             int proxy_style_in,
             bool majority_leaf_only_flag,
             bool cache_cheap_subproblems_flag,
             bool proxy_caching_flag,
             int num_proxy_features_in,
             bool rashomon_mode,
             bool stronger_rollout_flag = false,
             bool use_deferral_flag = false,
             double eta_defer_in = 0.0,
             const std::vector<int>& bb_pred = {}
            ) {
        n_features = (int)X_col_major.size();
        n_samples  = (int)X_col_major[0].size();
        n_words = (n_samples + 63) / 64; // 64 -> 1, 65 -> 2
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL; // if multiple of 64, all 1s. otherwise, n_samples % 64 1s followed by 0s.
        gamma = (int)llround(lambda * (double)n_samples);
        use_deferral = use_deferral_flag;
        eta_defer = eta_defer_in;

        if (use_deferral) {
            if ((int)bb_pred.size() != (int)y.size()) {
                throw std::runtime_error("fit: use_deferral=true requires bb_pred with same length as y.");
            }
            if (!std::isfinite(eta_defer) || eta_defer < 0.0) {
                throw std::runtime_error("fit: eta_defer must be finite and nonnegative.");
            }
        }
        trained_depth_budget = depth_budget;
        lookahead_init = lookahead_k;
        use_multipass = use_multipass_flag;
        rule_list_mode = rule_list_mode_flag;
        majority_leaf_only = majority_leaf_only_flag;
        cache_cheap_subproblems = cache_cheap_subproblems_flag;
        proxy_style = proxy_style_in;
        proxy_caching_enabled = proxy_caching_flag;
        stronger_rollout = stronger_rollout_flag;
        if (!rashomon_mode) { // force proxy caching on in single-tree mode - required for this codebase
            proxy_caching_enabled = true;
            cache_cheap_subproblems = true;
        } 
        if (num_proxy_features_in <= 0) num_proxy_features = n_features;
        else num_proxy_features = std::min(num_proxy_features_in, n_features);

        X_bits.assign(n_features, Packed(n_words)); // length n_features with entries of Packed, initialized to all 0s bits, we will set below.
        for (int f = 0; f < n_features; ++f) {
            auto &col = X_bits[f].w; // reference to the array of 64-bit words for that feature
            for (int i = 0; i < n_samples; ++i) {
                if (X_col_major[f][i]) col[i>>6] |= (1ULL << (i & 63)); // i>>6 integer division by 64 to answer what word are we in. then i & 63 = i % 64 to get index within word. (1ULL << (i & 63)) creates a 64-bit mask with exactly one bit = 1 at the position you need, then do bitwise or with ol[i >> 6] to set the position to 1 if it is not already set.
                // if feature f is true for sample i, set the bit corresponding to row i to true in the packed column by doing bitwise or with the current column and a 64 bit word with exactly one 1.
            }
            col[n_words-1] &= tail_mask; // bitwise and to 0 out invalid
        }

        // so the format is now a contiguous array of 64 bit words. if our remainder is 5 (and we are in word x), then our bit is the 5th least significant one in the word.
        // so increasing row index increases the bit position within a word.
        // last word is padded with 0 (at the front, most significant bits)

        // Ypos = Packed(n_words);
        // for (int i = 0; i < n_samples; ++i) {
        //     if (y[i]) Ypos.w[i>>6] |= (1ULL << (i & 63));
        // }
        // Ypos.w[n_words-1] &= tail_mask;
        int y_max = 0;
        for (int i = 0; i < n_samples; ++i) y_max = std::max(y_max, y[i]);
        num_classes = y_max + 1;

        Y_bits.assign((size_t)num_classes, Packed(n_words));
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].clear();
        }

        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[i];
            Y_bits[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }

        BBwrong = Packed((size_t)n_words);
        BBwrong.clear();

        if (use_deferral) {
            for (int i = 0; i < n_samples; ++i) {
                const int bi = bb_pred[(size_t)i];

                if (bi < 0 || bi >= num_classes) {
                    throw std::runtime_error(
                        "fit: bb_pred values must be in the same class range as y."
                    );
                }

                if (bi != y[(size_t)i]) {
                    BBwrong.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }

            BBwrong.w[(size_t)(n_words - 1)] &= tail_mask;
        }

        Packed root(n_words);
        for (int i = 0; i < n_words-1; ++i) root.w[i] = ~0ULL; // not 0, so all 1.
        root.w[n_words-1] = tail_mask; // enforce 0s for out of scope

        const PathKey& root_pk = empty_pk();

        // SINGLE TREE SUPPORT
        if (!rashomon_mode) {
            int single_obj = 0;
            if (lookahead_init <= 0) {
                single_obj = train_greedy(root, depth_budget, root_pk);
            } else {
                if (proxy_style == 4) {
                    single_obj = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    single_obj = generalized_lickety_split(root, depth_budget, lookahead_init, root_pk);
                }
            }
            
            std::cout << "Single-tree objective: " << single_obj
                    << " (" << (double)single_obj / (double)n_samples << ")\n";
            return; // done; do NOT build trie or do rashomon things
        }

        if (root_budget >= 0) {
            // user-specified bound: skip reference solution
            obj_bound = root_budget;
            std::cout << "Objective bound (user-set): " << obj_bound << "\n";

        } else {
            if (lookahead_init <= 0) { // set based on greedy even if our proxy is a leaf
                best_objective = train_greedy(root, depth_budget, root_pk);
            } else {
                if (proxy_style == 4) {
                    best_objective = split_algorithm(root, depth_budget, lookahead_init, root_pk);
                } else {
                    best_objective = generalized_lickety_split(root, depth_budget, lookahead_init, root_pk);
                }
            }
            cout << "Best objective: " << best_objective
                << " (" << (double)best_objective / (double)n_samples << ")\n";

            obj_bound = (int)llround(best_objective * (1.0 + rashomon_mult) * (1.0 + multiplicative_slack));

            cout << "Objective bound: " << obj_bound << "\n";
        }

        if (proxy_style == 2) { // we define to be depth_budget+1 size but depth 0 doesn't actually matter
            k_at_depth.assign(depth_budget + 1, 1);
            int K = lookahead_init;
            int kk = K;
            for (int d = depth_budget; d >= 0; --d) {
                k_at_depth[d] = std::min(d, kk);
                kk = (kk > 1) ? (kk - 1) : K; // increment down then wrap
            }
        } else {
            k_at_depth.clear();
        }


        result = construct_trie(root, depth_budget, obj_bound, root_pk);

        // cout << "Found " << result->count_trees() << " trees\n"; // we'll let the user compute this query if they want it because it is somewhat expensive
        cout << "Minimum objective: " << result->min_objective << "\n";
        cout << "Cache sizes - Greedy: " << greedy_cache.size()
            << ", Lickety: " << (use_kla_cache() ? lickety_cache_kla.size() : lickety_cache_k2.size())
            << ", Trie: " << trie_cache.size();
        // cout << ", Distinct subproblems (greedy U lickety): " << count_distinct_subproblems_union();
        // if (key_mode == KeyMode::EXACT) {
        //     cout << ", Unique masks: " << mask_ids.size();
        // }
        // if (key_mode == KeyMode::LITS_EXACT) {
        //     cout << ", Unique literal subproblems: " << lit_ids.size();
        // }
        cout << ", Trie cache: " << (trie_cache_enabled ? "ON" : "OFF");
        cout << "\n";
    }

    // predict using the i-th tree in the Rashomon set: X_row_major: binary [n_samples][n_features]
    std::vector<uint8_t> get_predictions(
        uint64_t tree_index,
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& bb_pred_row = {},
        int defer_placeholder = 99
    ) const {
        const std::size_t n_samples = X_row_major.size();
        if (n_samples == 0) return {};

        const std::size_t n_features = X_row_major[0].size();
        const int8_t depth_budget = trained_depth_budget;

        if ((int)n_features != this->n_features) {
            throw std::runtime_error("Prediction X has different number of features than training.");
        }

        std::shared_ptr<PredNode> tree;

        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::runtime_error("Single-tree mode only supports tree_index == 0.");
            }
            if (depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            Packed root((size_t)n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[(size_t)i] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            tree = build_best_tree_from_caches(root, depth_budget, root_pk);
        } 
        // standard rashomon mode
        else {
            tree = get_ith_tree(tree_index);
        }

        std::vector<uint8_t> out(n_samples, 0);

        std::vector<int> idx(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i) {
            idx[i] = static_cast<int>(i);
        }

        // predict_tree_recursive(tree.get(), X_row_major, out, idx);
        const bool use_placeholder = bb_pred_row.empty();

        if (!use_placeholder && bb_pred_row.size() != n_samples) {
            throw std::runtime_error("Prediction bb_pred_row has different number of rows than X.");
        }

        predict_tree_recursive(
            tree.get(),
            X_row_major,
            bb_pred_row,
            out,
            idx,
            use_placeholder,
            defer_placeholder
        );
        return out;
    }


    // get predictions from all trees in the rashomon set, as a vector of prediction vectors (one per tree).
    std::vector<std::vector<uint8_t>> get_all_predictions(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& bb_pred_row = {},
        int defer_placeholder = 99
    ) const {
        uint64_t total = result ? result->count_trees() : 0ULL;
        std::vector<std::vector<uint8_t>> all;
        all.reserve(static_cast<std::size_t>(total));

        for (uint64_t i = 0; i < total; ++i) {
            all.push_back(
                get_predictions(i, X_row_major, bb_pred_row, defer_placeholder)
            );
        }

        return all;
    }

    std::pair<std::vector<std::vector<int>>, std::vector<int>>
    get_tree_paths(std::uint64_t tree_index) const {
        // single tree mode
        if (!result) {
            if (tree_index != 0) {
                throw std::out_of_range("Single-tree mode only supports tree_index == 0.");
            }

            // root mask - all samples active
            Packed root(n_words);
            for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
            root.w[n_words - 1] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t depth_budget = trained_depth_budget;

            auto tree = build_best_tree_from_caches(root, depth_budget, root_pk);

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;
            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }

        // standard rashomon mode
        auto tree = get_ith_tree(tree_index);
        std::vector<std::vector<int>> paths;
        std::vector<int> preds;
        std::vector<int> current;

        collect_paths(tree.get(), current, paths, preds);
        return {paths, preds};
    }


    // for individual decision tree algorithm
    std::pair<std::vector<std::vector<int>>, std::vector<int>>
        get_tree_paths_from_tree(const std::shared_ptr<PredNode>& tree) const {
            if (!tree) {
                throw std::runtime_error("Null tree passed to get_tree_paths_from_tree.");
            }

            std::vector<std::vector<int>> paths;
            std::vector<int> preds;
            std::vector<int> current;

            collect_paths(tree.get(), current, paths, preds);
            return {paths, preds};
        }


    // return (unnormalized_objective, normalized_objective) for the ith tree
    std::pair<int, double> get_ith_tree_objective(std::uint64_t i) const {
        // if (!result) {
        //     throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        // }
        if (!result) {
            if (i != 0) throw std::out_of_range("Single-tree mode only supports i==0.");
            if (!proxy_caching_enabled) {
                throw std::runtime_error("Single-tree objective requires proxy_caching_enabled, or recompute objective.");
            }
            if (trained_depth_budget < 0) {
                throw std::runtime_error("trained_depth_budget not set. Call fit() first.");
            }

            // root mask
            Packed root((size_t)n_words);
            for (int w = 0; w < n_words - 1; ++w) root.w[(size_t)w] = ~0ULL;
            root.w[(size_t)(n_words - 1)] = tail_mask;

            const PathKey& root_pk = empty_pk();
            const int8_t d = trained_depth_budget;

            const uint64_t km = key_of_subproblem(root, root_pk);

            int best = std::numeric_limits<int>::max(); // it suffices to return the minimum among the root caches

            // greedy
            if (auto itg = greedy_cache.find(K2{km, d}); itg != greedy_cache.end())
                best = std::min(best, itg->second);

            // lickety
            if (use_kla_cache()) {
                for (int kk = 0; kk <= (int)(d - 1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                if (auto it = lickety_cache_k2.find(K2{km, d}); it != lickety_cache_k2.end())
                    best = std::min(best, it->second);
            }

            if (best == std::numeric_limits<int>::max()) {
                throw std::runtime_error("Root objective not found in caches (greedy/lickety).");
            }

            double normalized = (double)best / (double)n_samples;
            return {best, normalized};
        }

        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        std::uint64_t total = result->count_trees();
        if (i >= total) {
            throw std::out_of_range("Tree index out of range in get_ith_tree_objective");
        }

        std::uint64_t cum = 0;
        int target_obj = -1;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                break;
            }
            cum += e.cnt;
        }

        if (target_obj < 0) {
            throw std::runtime_error("Failed to locate objective bucket in get_ith_tree_objective");
        }

        double normalized = (n_samples > 0)
            ? static_cast<double>(target_obj) / static_cast<double>(n_samples)
            : 0.0;

        return {target_obj, normalized};
    }

    // root LicketySPLIT objective with lookahead=1 so the user can compare frontier cuts to the reference solution
    int root_lickety_objective_lookahead1(int depth_budget) {
        if (n_samples == 0) {
            throw std::runtime_error("Model not fitted.");
        }

        Packed root(n_words);
        for (int i = 0; i < n_words - 1; ++i) root.w[i] = ~0ULL;
        root.w[n_words - 1] = tail_mask;

        PathKey root_pk;
        return generalized_lickety_split(root, depth_budget, /*k=*/1, root_pk);
    }

    TrieNodeCountStats count_reconstructed_trie_node_stats() const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit(..., rashomon_mode=true) first."
            );
        }
        if (trained_depth_budget < 0) {
            throw std::runtime_error(
                "trained_depth_budget not set. Call fit() first."
            );
        }

        TrieNodeCountStats stats;

        // unique reachable treetrienodes objects, by pointer identity.
        std::unordered_set<const TreeTrieNode*> processed;

        // distinct semantic keys reconstructed from active training masks.
        std::unordered_set<K2, K2::Hash> seen_k2;
        std::unordered_set<K3, K3::Hash> seen_k3;

        struct StackItem {
            std::shared_ptr<TreeTrieNode> node;
            Packed mask;
            int depth;
        };

        // root mask: all training samples active.
        Packed root_mask((size_t)n_words);
        for (int i = 0; i < n_words - 1; ++i) {
            root_mask.w[(size_t)i] = ~0ULL;
        }
        root_mask.w[(size_t)(n_words - 1)] = tail_mask;

        std::vector<StackItem> stack;
        stack.push_back(StackItem{
            result,
            std::move(root_mask),
            static_cast<int>(trained_depth_budget)
        });

        while (!stack.empty()) {
            StackItem item = std::move(stack.back());
            stack.pop_back();

            if (!item.node) continue;

            const TreeTrieNode* cur = item.node.get();

            // reconstruct the subproblem id from the active mask.
            const uint64_t subproblem_key =
                hash_mask64(item.mask.w.data(), n_words, tail_mask);

            seen_k2.insert(K2{subproblem_key, item.depth});
            seen_k3.insert(K3{subproblem_key, item.depth, item.node->budget});

            // count / expand each unique treetrienode object once.
            if (processed.find(cur) != processed.end()) {
                continue;
            }
            processed.insert(cur);

            // reconstruct child masks from the current mask and split feature.
            for (const SplitNode& split : cur->splits) {
                if (!split.left || !split.right) continue;

                Packed L((size_t)n_words);
                Packed R((size_t)n_words);

                and_bits(item.mask, X_bits[(size_t)split.feature], L);
                andnot_bits(item.mask, X_bits[(size_t)split.feature], R);

                stack.push_back(StackItem{
                    split.left,
                    std::move(L),
                    item.depth - 1
                });

                stack.push_back(StackItem{
                    split.right,
                    std::move(R),
                    item.depth - 1
                });
            }
        }

        stats.total_trie_nodes = processed.size();
        stats.distinct_subproblem_depth = seen_k2.size();
        stats.distinct_subproblem_depth_budget = seen_k3.size();

        return stats;
    }

    ExportANDORGraph export_andor_graph(
        std::size_t max_trie_nodes = 2000000000000ULL,
        std::size_t max_split_nodes = 500000000000ULL,
        std::size_t max_leaf_nodes = 500000000000ULL
    ) const {
        if (!result) {
            throw std::runtime_error(
                "No Rashomon trie has been constructed. Call fit(..., rashomon_mode=true) first."
            );
        }

        ExportANDORGraph out;

        std::unordered_map<const TreeTrieNode*, int> trie_id;
        std::unordered_set<const TreeTrieNode*> processed;

        auto finite_min_sum = [](int a, int b) -> int {
            if (a == std::numeric_limits<int>::max() ||
                b == std::numeric_limits<int>::max()) {
                return std::numeric_limits<int>::max();
            }
            return a + b;
        };

        auto get_trie_id = [&](const std::shared_ptr<TreeTrieNode>& p) -> int {
            if (!p) return -1;

            const TreeTrieNode* raw = p.get();
            auto it = trie_id.find(raw);
            if (it != trie_id.end()) return it->second;

            if (out.trie_nodes.size() >= max_trie_nodes) {
                throw std::runtime_error(
                    "export_andor_graph exceeded max_trie_nodes. "
                    "Increase the limit or fit a smaller Rashomon set."
                );
            }

            int id = static_cast<int>(out.trie_nodes.size());
            trie_id.emplace(raw, id);

            ExportTreeTrieNode node;
            node.id = id;
            node.budget = raw->budget;
            node.min_objective = raw->min_objective;
            node.subproblem_size = 0;  // filled when processed with its active mask

            out.trie_nodes.push_back(std::move(node));
            return id;
        };

        // Root mask: all training samples active.
        Packed root_mask(n_words);
        for (int i = 0; i < n_words - 1; ++i) {
            root_mask.w[(size_t)i] = ~0ULL;
        }
        root_mask.w[(size_t)(n_words - 1)] = tail_mask;

        out.root_trie_id = get_trie_id(result);

        struct StackItem {
            std::shared_ptr<TreeTrieNode> node;
            Packed mask;
        };

        std::vector<StackItem> stack;
        stack.push_back(StackItem{result, root_mask});

        while (!stack.empty()) {
            StackItem item = std::move(stack.back());
            stack.pop_back();

            std::shared_ptr<TreeTrieNode> cur_ptr = item.node;
            if (!cur_ptr) continue;

            const TreeTrieNode* cur = cur_ptr.get();
            if (processed.find(cur) != processed.end()) continue;
            processed.insert(cur);

            const int cur_id = trie_id.at(cur);
            const int cur_size = item.mask.count();

            // Fill OR-node metadata.
            out.trie_nodes[cur_id].budget = cur->budget;
            out.trie_nodes[cur_id].min_objective = cur->min_objective;
            out.trie_nodes[cur_id].subproblem_size = cur_size;

            // Export leaf alternatives.
            for (const LeafNode& leaf : cur->leaves) {
                if (out.leaf_nodes.size() >= max_leaf_nodes) {
                    throw std::runtime_error(
                        "export_andor_graph exceeded max_leaf_nodes. "
                        "Increase the limit or fit a smaller Rashomon set."
                    );
                }

                ExportLeafNode e;
                e.id = static_cast<int>(out.leaf_nodes.size());
                e.parent_trie_id = cur_id;
                e.prediction = leaf.prediction;
                e.loss = leaf.loss;
                e.subproblem_size = cur_size;

                out.trie_nodes[cur_id].leaf_ids.push_back(e.id);
                out.leaf_nodes.push_back(std::move(e));
            }

            // Export split / AND alternatives.
            for (const SplitNode& split : cur->splits) {
                if (!split.left || !split.right) continue;

                if (out.split_nodes.size() >= max_split_nodes) {
                    throw std::runtime_error(
                        "export_andor_graph exceeded max_split_nodes. "
                        "Increase the limit or fit a smaller Rashomon set."
                    );
                }

                // Compute child masks only during export.
                Packed L(n_words), R(n_words);
                and_bits(item.mask, X_bits[(size_t)split.feature], L);
                andnot_bits(item.mask, X_bits[(size_t)split.feature], R);

                const int left_id = get_trie_id(split.left);
                const int right_id = get_trie_id(split.right);

                ExportSplitNode e;
                e.id = static_cast<int>(out.split_nodes.size());
                e.parent_trie_id = cur_id;
                e.feature = split.feature;
                e.left_trie_id = left_id;
                e.right_trie_id = right_id;
                e.min_objective = finite_min_sum(
                    split.left->min_objective,
                    split.right->min_objective
                );

                out.trie_nodes[cur_id].split_ids.push_back(e.id);
                out.split_nodes.push_back(std::move(e));

                if (processed.find(split.left.get()) == processed.end()) {
                    stack.push_back(StackItem{split.left, std::move(L)});
                }
                if (processed.find(split.right.get()) == processed.end()) {
                    stack.push_back(StackItem{split.right, std::move(R)});
                }
            }
        }

        return out;
    }

private:
    shared_ptr<TreeTrieNode> construct_trie(const Packed& mask, int8_t depth, int budget, const PathKey& pk) {
        const uint64_t k = key_of_subproblem(mask, pk);
        K3 key{k, depth, budget};

        if (trie_cache_enabled) {
            if (auto it = trie_cache.find(key); it != trie_cache.end()) {
                return it->second; // exact lookup for simplicity
            }
        }

        auto node = make_shared<TreeTrieNode>(); // wraps in shared pointer so memory management is automatic
        node->budget = budget;

        int n_sub = 0;

        if (num_classes == 2) {
            int pos = 0;
            int bb_wrong = 0;
            count_total_pos_bbwrong_binary(mask, n_sub, pos, bb_wrong);

            const int neg = n_sub - pos;

            if (!majority_leaf_only) {
                const int cost0 = gamma + pos;
                const int cost1 = gamma + neg;

                if (cost0 <= budget) node->add_leaf(0, cost0);
                if (cost1 <= budget) node->add_leaf(1, cost1);
            } else {
                const int best_c = (pos >= neg) ? 1 : 0;
                const int best_cost = gamma + std::min(pos, neg);

                if (best_cost <= budget) node->add_leaf(best_c, best_cost);
            }

            if (use_deferral && n_sub > 0) {
                const int defer_cost =
                    gamma + defer_penalty_from_count_(n_sub) + bb_wrong;

                if (defer_cost <= budget) {
                    node->add_leaf(DEFER_PREDICTION, defer_cost);
                }
            }
        } else {
            n_sub = count_total(mask);

            std::vector<int> cnts;
            count_per_class(mask, cnts);

            if (!majority_leaf_only) {
                for (int c = 0; c < num_classes; ++c) {
                    const int mis = n_sub - cnts[(size_t)c];
                    const int cost = gamma + mis;
                    if (cost <= budget) node->add_leaf(c, cost);
                }
            } else {
                // choose argmax class
                int best_c = 0;
                int best_cnt = cnts[0];
                for (int c = 1; c < num_classes; ++c) {
                    const int v = cnts[(size_t)c];
                    if (v > best_cnt || (v == best_cnt && c > best_c)) {
                        best_cnt = v;
                        best_c = c;
                    }
                }

                const int mis = n_sub - best_cnt;
                const int best_cost = gamma + mis;
                if (best_cost <= budget) node->add_leaf(best_c, best_cost);
            }

            // defer leaf is an additional leaf option.
            if (use_deferral && n_sub > 0) {
                const int bb_wrong = count_bb_wrong(mask);
                const int defer_cost =
                    gamma
                    + defer_penalty_from_count_(n_sub)
                    + bb_wrong;

                if (defer_cost <= budget) {
                    node->add_leaf(DEFER_PREDICTION, defer_cost);
                }
            }
        }

        if (depth == 0 || budget < 2 * gamma) {
            if (trie_cache_enabled) trie_cache.emplace(key, node);
            return node;
        }

        Packed L(n_words), R(n_words);

        const int8_t k_here = (proxy_style == 2 && depth >= 0 && depth < (int)k_at_depth.size())
            ? k_at_depth[depth-1]
            : lookahead_init;

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);

            if (!L.any() || !R.any()) continue;

            // build child pks (canonical sorted)
            // pk refs default to EMPTY
            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();

            // only build PKs in LITS_EXACT
            PathKey pkL_local, pkR_local;
            if (key_mode == KeyMode::LITS_EXACT) {
                pkL_local = pk;
                pkR_local = pk;
                pk_insert_sorted(pkL_local, enc_lit(f, 1));
                pk_insert_sorted(pkR_local, enc_lit(f, 0));
                pkLp = &pkL_local;
                pkRp = &pkR_local;
            }

            int lossL, lossR;

            // to evaluate whether lossL+lossR is within budget (for non rule list mode), we can first handle an early pruning case
            if (lookahead_init < 0) {
                lossL = leaf_objective(L);
            } else if (lookahead_init == 0) {
                lossL = train_greedy(L, depth - 1, *pkLp);
            } else {
                if (proxy_style == 4) {
                    lossL = split_algorithm(L, depth - 1, k_here, *pkLp);
                } else {
                    lossL = generalized_lickety_split(L, depth - 1, k_here, *pkLp);
                }
            }

            // either L or R would work here, could take larger, but very cheap to just choose one
            if (!rule_list_mode) {
                if (lossL + gamma > budget) continue;
            }

            // now compute R if we need it for more information
            if (lookahead_init < 0) {
                lossR = leaf_objective(R);
            } else if (lookahead_init == 0) {
                lossR = train_greedy(R, depth - 1, *pkRp);
            } else {
                if (proxy_style == 4) {
                    lossR = split_algorithm(R, depth - 1, k_here, *pkRp);
                } else {
                    lossR = generalized_lickety_split(R, depth - 1, k_here, *pkRp);
                }
            }

            // standard pruning logic in paper
            if (!rule_list_mode) {
                if (lossL + lossR > budget) continue; // approximation decision tree rashomon set
            } else {
                if (lossL > budget - gamma && lossR > budget - gamma) continue; // exact rule list rashomon set
            }

            
            // LR has two entries: first and second.
            std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>> LR; 
            if (rule_list_mode) {
                LR = solve_siblings(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            } else if (use_multipass) {
                LR = solve_siblings(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            } else {
                LR = symmetric_single_pass(lossL, lossR, L, R, budget, depth, *pkLp, *pkRp);
            }

            // the left and right TreeTrieNode (OR nodes) to be added to the AND/OR graph being built
            if (!LR.first || !LR.second) continue; // safeguard, especially needed if we allow non-injective keys
            
            node->add_split(f, LR.first, LR.second); // add split with left and right subtries
        }

        if (trie_cache_enabled) trie_cache.emplace(key, node);
        return node;
    }

    // returns left and right treetrienode. the left and right mask are constants, even as you recurse on construct_trie
    pair<shared_ptr<TreeTrieNode>, shared_ptr<TreeTrieNode>>
    solve_siblings(int loss_l, int loss_r,
               const Packed& Lmask, const Packed& Rmask,
               int budget, int8_t depth,
               const PathKey& pkL, const PathKey& pkR) {
        int left_budget  = budget - loss_r;
        shared_ptr<TreeTrieNode> left_node =
            (left_budget >= 0) ? construct_trie(Lmask, depth - 1, left_budget, pkL)
                               : nullptr; // handles some potential issues with non-injective keys
        int min_left = (left_node ? left_node->min_objective : numeric_limits<int>::max());

        int right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
        shared_ptr<TreeTrieNode> right_node =
            (right_budget >= 0) ? construct_trie(Rmask, depth - 1, right_budget, pkR)
                                : nullptr;
        int min_right = (right_node ? right_node->min_objective : numeric_limits<int>::max());

        while (true) {
            bool improved = false;

            int new_left_budget = (min_right == numeric_limits<int>::max()) ? -1 : (budget - min_right);
            if (new_left_budget > left_budget) {
                left_budget = new_left_budget;
                if (left_budget >= 0) {
                    left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
                    int new_min_left = left_node->min_objective;
                    if (new_min_left < min_left) min_left = new_min_left;
                }
            }

            int new_right_budget = (min_left == numeric_limits<int>::max()) ? -1 : (budget - min_left);
            if (new_right_budget > right_budget) {
                right_budget = new_right_budget;
                if (right_budget >= 0) {
                    right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
                    int new_min_right = right_node->min_objective;
                    if (new_min_right < min_right) { min_right = new_min_right; improved = true; }
                }
            }

            if (!improved) break;
        }

        return {left_node, right_node};
    }

    // this is solely for ableation study purposes - if practical we would subtract minobjective from the other side
    std::pair<std::shared_ptr<TreeTrieNode>, std::shared_ptr<TreeTrieNode>>
    symmetric_single_pass(int loss_l, int loss_r,
                 const Packed& Lmask, const Packed& Rmask,
                 int budget, int8_t depth,
                 const PathKey& pkL, const PathKey& pkR) {
        int left_budget  = budget - loss_r;
        int right_budget = budget - loss_l;

        std::shared_ptr<TreeTrieNode> left_node  = nullptr;
        std::shared_ptr<TreeTrieNode> right_node = nullptr;

        if (left_budget >= 0) { // robustness incase we change pruning
            left_node = construct_trie(Lmask, depth - 1, left_budget, pkL);
        }
        if (right_budget >= 0) {
            right_node = construct_trie(Rmask, depth - 1, right_budget, pkR);
        }

        return {left_node, right_node};
    
    }

    inline void count_total_pos_binary(const Packed& mask, int& n, int& pos) const {
        n = 0;
        pos = 0;

        // class 1 is the positive class
        const Packed& Ypos = Y_bits[(size_t)1];

        for (int i = 0; i < n_words; ++i) {
            const uint64_t mw = mask.w[(size_t)i];
            n   += popcnt64(mw);
            pos += popcnt64(mw & Ypos.w[(size_t)i]);
        }
    }

    inline void count_total_pos_bbwrong_binary(
        const Packed& mask,
        int& n,
        int& pos,
        int& bb_wrong
    ) const {
        n = 0;
        pos = 0;
        bb_wrong = 0;

        const Packed& Ypos = Y_bits[(size_t)1];

        if (use_deferral) {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t mw = mask.w[(size_t)i];
                n        += popcnt64(mw);
                pos      += popcnt64(mw & Ypos.w[(size_t)i]);
                bb_wrong += popcnt64(mw & BBwrong.w[(size_t)i]);
            }
        } else {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t mw = mask.w[(size_t)i];
                n   += popcnt64(mw);
                pos += popcnt64(mw & Ypos.w[(size_t)i]);
            }
        }
    }

    // int leaf_objective(const Packed& mask) const {
    //     if (num_classes == 2) {
    //         int n_sub, pos;
    //         count_total_pos_binary(mask, n_sub, pos);

    //         if (n_sub == 0) return 0;

    //         return gamma + std::min(pos, n_sub - pos);
    //     }

    //     const int n_sub = count_total(mask);
    //     if (n_sub == 0) return 0;

    //     int best_cnt = 0;
    //     for (int c = 0; c < num_classes; ++c) {
    //         best_cnt = std::max(best_cnt, count_class(mask, c));
    //     }
    //     const int mis = n_sub - best_cnt;
    //     return gamma + mis;
    // }

    inline void make_child_pks_if_needed_(
        int feat,
        const PathKey& pk,
        const PathKey*& pkLp,
        const PathKey*& pkRp,
        PathKey& pkL_local,
        PathKey& pkR_local
    ) const {
        pkLp = &empty_pk();
        pkRp = &empty_pk();
        if (key_mode == KeyMode::LITS_EXACT) {
            pkL_local = pk;
            pkR_local = pk;
            pk_insert_sorted(pkL_local, enc_lit(feat, 1));
            pk_insert_sorted(pkR_local, enc_lit(feat, 0));
            pkLp = &pkL_local;
            pkRp = &pkR_local;
        }
    }

    // inline int leaf_objective_binary_from_counts(int n_sub, int pos) const {
    //     if (n_sub == 0) return 0;
    //     return gamma + std::min(pos, n_sub - pos);
    // }

    struct BestLeafAction {
        int prediction = 0; // 0..C-1, or DEFER_PREDICTION
        int loss = 0;
    };

    inline int defer_penalty_from_count_(int n_sub) const {
        return (int)llround(eta_defer * (double)n_sub);
    }

    inline int count_bb_wrong(const Packed& mask) const {
        if (!use_deferral) return 0;
        return popcount_and(mask, BBwrong);
    }

    inline int predict_leaf_objective_binary_from_counts(int n_sub, int pos) const {
        if (n_sub == 0) return 0;
        return gamma + std::min(pos, n_sub - pos);
    }

    inline int leaf_objective_binary_from_counts(int n_sub, int pos) const {
        return leaf_objective_binary_from_counts(n_sub, pos, 0);
    }

    inline int leaf_objective_binary_from_counts(
        int n_sub,
        int pos,
        int bb_wrong
    ) const {
        if (n_sub == 0) return 0;

        const int predict_loss = gamma + std::min(pos, n_sub - pos);

        if (!use_deferral) {
            return predict_loss;
        }

        const int defer_loss =
            gamma
            + defer_penalty_from_count_(n_sub)
            + bb_wrong;

        return std::min(predict_loss, defer_loss);
    }

    BestLeafAction best_leaf_action(const Packed& mask) const {
        if (num_classes == 2) {
            int n_sub = 0;
            int pos = 0;
            int bb_wrong = 0;

            count_total_pos_bbwrong_binary(mask, n_sub, pos, bb_wrong);

            if (n_sub == 0) {
                return BestLeafAction{0, 0};
            }

            const int neg = n_sub - pos;

            BestLeafAction best;
            if (pos >= neg) {
                best.prediction = 1;
                best.loss = gamma + neg;
            } else {
                best.prediction = 0;
                best.loss = gamma + pos;
            }

            if (use_deferral) {
                const int defer_loss =
                    gamma
                    + defer_penalty_from_count_(n_sub)
                    + bb_wrong;

                if (defer_loss < best.loss) {
                    best.prediction = DEFER_PREDICTION;
                    best.loss = defer_loss;
                }
            }

            return best;
        }

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            return BestLeafAction{0, 0};
        }

        int best_pred = 0;
        int best_cnt = -1;

        for (int c = 0; c < num_classes; ++c) {
            const int cnt = count_class(mask, c);
            if (cnt > best_cnt || (cnt == best_cnt && c > best_pred)) {
                best_cnt = cnt;
                best_pred = c;
            }
        }

        BestLeafAction best;
        best.prediction = best_pred;
        best.loss = gamma + (n_sub - best_cnt);

        if (use_deferral) {
            const int bb_wrong = count_bb_wrong(mask);
            const int defer_loss =
                gamma
                + defer_penalty_from_count_(n_sub)
                + bb_wrong;

            if (defer_loss < best.loss) {
                best.prediction = DEFER_PREDICTION;
                best.loss = defer_loss;
            }
        }

        return best;
    }

    int leaf_objective(const Packed& mask) const {
        return best_leaf_action(mask).loss;
    }


    int train_greedy(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1 && (greedy_split_mode == 1 || greedy_split_mode == 2)) {
            return depth1_exact_solver_cached(mask, pk); 
        }

        uint64_t kmask = 0;
        K2 key{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key.k = kmask;

            if (auto it = greedy_cache.find(key); it != greedy_cache.end()) {
                return it->second;
            }
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(mask, n_sub, pos, bb_wrong);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos, bb_wrong);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                greedy_cache.emplace(key, leaf_loss);
            }
            return leaf_loss;
        }

        // decide which split-selection heuristic to use
        bool use_entropy;
        if (greedy_split_mode == 0) {
            use_entropy = true;                  // always entropy-driven
        } else if (greedy_split_mode == 1) {
            use_entropy = (depth_budget != 1);   // special depth==1 solver
        } else { // greedy_split_mode == 2
            use_entropy = false;                 // always minimize child leaf objective
        }


        // choose split via entropy gain
        int best_feat;
        if (num_classes == 2) {
            best_feat = find_best_split_binary_known_counts(
                mask,
                n_sub,
                pos,
                bb_wrong,
                use_entropy
            );
        } else {
            best_feat = find_best_split(mask, use_entropy);
        }
        if (best_feat < 0) {
            if (proxy_caching_enabled) greedy_cache.emplace(key, leaf_loss);
            return leaf_loss;
        }
        
        Packed L(n_words), R(n_words);
        and_bits(mask, X_bits[best_feat], L);
        andnot_bits(mask, X_bits[best_feat], R);
       if (!L.any() || !R.any()) {
            throw std::logic_error(
                "Invalid split choice in greedy tree method"
            );
        }
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;
        make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

        int left_obj  = train_greedy(L, depth_budget - 1, *pkLp);
        int right_obj = train_greedy(R, depth_budget - 1, *pkRp);
        int split_obj = left_obj + right_obj;

        int ans = min(leaf_loss, split_obj);
        if (proxy_caching_enabled) greedy_cache.emplace(key, ans);
        return ans;
    }

    int eval_with_lookahead(const Packed& m, int depth, int k, const PathKey& pk) {
        if (k <= 0) return train_greedy(m, depth, pk);
        return generalized_lickety_split(m, depth, k, pk);
    }


    inline int next_k_cycle(int8_t k) const {
        // cycle: ... 3->2->1->K->K-1->...
        return (k > 1) ? (k - 1) : lookahead_init;
    }

    inline void split_bits_count_left(
        const Packed& mask,
        const Packed& Xf,
        Packed& L,
        Packed& R,
        int& left_n
    ) const {
        left_n = 0;
        for (int i = 0; i < n_words; ++i) {
            const uint64_t mw = mask.w[(size_t)i];
            const uint64_t lw = mw & Xf.w[(size_t)i];
            L.w[(size_t)i] = lw;
            R.w[(size_t)i] = mw & ~Xf.w[(size_t)i];
            left_n += popcnt64(lw);
        }
        L.w[(size_t)(n_words - 1)] &= tail_mask;
        R.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline void split_count_left_binary_fast_(
        const Packed& mask,
        const Packed& feat,
        int& left_n,
        int& left_pos,
        int& left_bb_wrong
    ) const {
        left_n = 0;
        left_pos = 0;
        left_bb_wrong = 0;

        const Packed& Ypos = Y_bits[(size_t)1];

        if (use_deferral) {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t lw = mask.w[(size_t)i] & feat.w[(size_t)i];
                left_n        += popcnt64(lw);
                left_pos      += popcnt64(lw & Ypos.w[(size_t)i]);
                left_bb_wrong += popcnt64(lw & BBwrong.w[(size_t)i]);
            }
        } else {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t lw = mask.w[(size_t)i] & feat.w[(size_t)i];
                left_n   += popcnt64(lw);
                left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
            }
        }
    }
        
    inline void split_bits_count_left_binary(
        const Packed& mask,
        const Packed& feat,
        Packed& L,
        Packed& R,
        int& left_n,
        int& left_pos,
        int& left_bb_wrong
    ) const {
        left_n = 0;
        left_pos = 0;
        left_bb_wrong = 0;

        const Packed& Ypos = Y_bits[(size_t)1];

        if (use_deferral) {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t lw = mask.w[(size_t)i] & feat.w[(size_t)i];
                L.w[(size_t)i] = lw;
                R.w[(size_t)i] = mask.w[(size_t)i] & ~feat.w[(size_t)i];

                left_n        += popcnt64(lw);
                left_pos      += popcnt64(lw & Ypos.w[(size_t)i]);
                left_bb_wrong += popcnt64(lw & BBwrong.w[(size_t)i]);
            }
        } else {
            for (int i = 0; i < n_words; ++i) {
                const uint64_t lw = mask.w[(size_t)i] & feat.w[(size_t)i];
                L.w[(size_t)i] = lw;
                R.w[(size_t)i] = mask.w[(size_t)i] & ~feat.w[(size_t)i];

                left_n   += popcnt64(lw);
                left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
            }
        }

        L.w[(size_t)(n_words - 1)] &= tail_mask;
        R.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    int depth1_exact_solver_cached(const Packed& mask, const PathKey& pk) {
        // const uint64_t kmask = key_of_subproblem(mask, pk);
        // constexpr int DEPTH = 1;
        // constexpr int KTAG  = 0;

        // int cached;
        // if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;

        constexpr int8_t DEPTH = 1;
        constexpr int8_t KTAG  = 0;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        if (num_classes == 2) {
            int n_sub = 0;
            int pos_total = 0;
            int bb_wrong_total = 0;
            count_total_pos_bbwrong_binary(mask, n_sub, pos_total, bb_wrong_total);

            const int leaf_loss =
                leaf_objective_binary_from_counts(n_sub, pos_total, bb_wrong_total);

            // only cache cheap subproblems if flag enabled
            if (leaf_loss <= 2 * gamma) {
                if (proxy_caching_enabled) {
                    cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
                }
                return leaf_loss;
            }

            int best_sum = std::numeric_limits<int>::max();

            const Packed& Ypos = Y_bits[(size_t)1];
            const int F = proxy_feat_count_();

            for (int f = 0; f < F; ++f) {
                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                split_count_left_binary_fast_(
                    mask,
                    X_bits[(size_t)f],
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;
                const int right_bb_wrong = bb_wrong_total - left_bb_wrong;

                const int sum =
                    leaf_objective_binary_from_counts(left_n, left_pos, left_bb_wrong)
                    +
                    leaf_objective_binary_from_counts(right_n, right_pos, right_bb_wrong);

                if (sum < best_sum) best_sum = sum;
            }

            int ans = leaf_loss;
            if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

            if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
            return ans;
        }

        const int leaf_loss = leaf_objective(mask);

        // only cache cheap subproblems if flag enabled
        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const int sum = leaf_objective(L) + leaf_objective(R);
            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    int depth2_special_solver_cached(const Packed& mask, const PathKey& pk){
        constexpr int8_t DEPTH = 2;
        constexpr int8_t KTAG  = 1;

        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(mask, n_sub, pos, bb_wrong);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos, bb_wrong);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                split_bits_count_left_binary(
                    mask,
                    X_bits[(size_t)f],
                    L,
                    R,
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

                if (left_n == 0 || left_n == n_sub) continue;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depth1_exact_solver_cached(L, *pkLp); // depth==1 uses k=0 internally
            const int right_best = depth1_exact_solver_cached(R, *pkRp);
            const int sum = left_best + right_best;

            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, DEPTH, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }

    int depthd_exact_solver_cached(const Packed& mask, int8_t depth_budget, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);
        if (depth_budget == 2) return depth2_special_solver_cached(mask, pk);

        const int8_t DEPTH = depth_budget;
        const int8_t KTAG  = depth_budget - 1;


        uint64_t kmask = 0;
        int cached;

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            if (try_get_lickety_cached_(kmask, DEPTH, KTAG, cached)) return cached;
        }

        int n_sub = 0;
        int pos = 0;
        int bb_wrong = 0;
        int leaf_loss = 0;

        if (num_classes == 2) {
            count_total_pos_bbwrong_binary(mask, n_sub, pos, bb_wrong);
            leaf_loss = leaf_objective_binary_from_counts(n_sub, pos, bb_wrong);
        } else {
            n_sub = count_total(mask);
            leaf_loss = leaf_objective(mask);
        }

        if (leaf_loss <= 2 * gamma) {
            if (proxy_caching_enabled) {
                cache_lickety_if_true_(kmask, DEPTH, KTAG, leaf_loss,/*allow_cache=*/cache_cheap_subproblems);
            }
            return leaf_loss;
        }

        int best_sum = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            if (num_classes == 2) {
                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                split_bits_count_left_binary(
                    mask,
                    X_bits[(size_t)f],
                    L,
                    R,
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

                if (left_n == 0 || left_n == n_sub) continue;
            } else {
                and_bits(mask, X_bits[f], L);
                andnot_bits(mask, X_bits[f], R);
                if (!L.any() || !R.any()) continue;
            }

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_best  = depthd_exact_solver_cached(L, depth_budget - 1, *pkLp);
            const int right_best = depthd_exact_solver_cached(R, depth_budget - 1, *pkRp);

            const int sum = left_best + right_best;
            if (sum < best_sum) best_sum = sum;
        }

        int ans = leaf_loss;
        if (best_sum != std::numeric_limits<int>::max()) ans = std::min(ans, best_sum);

        if (proxy_caching_enabled) cache_lickety_if_true_(kmask, depth_budget, KTAG, ans, /*allow_cache=*/true);
        return ans;
    }


    inline void cache_lickety_if_true_(uint64_t kmask, int8_t depth_budget, int8_t k, int val, bool allow_cache) {
        if (!allow_cache) return;
        const bool use_kla = use_kla_cache();
        if (use_kla) lickety_cache_kla.emplace(KLA{kmask, depth_budget, k}, val);
        else         lickety_cache_k2.emplace(K2 {kmask, depth_budget},     val);
    }

    inline bool try_get_lickety_cached_(uint64_t kmask, int8_t depth_budget, int8_t k, int& out_val) const {
        const bool use_kla = use_kla_cache();
        if (use_kla) {
            auto it = lickety_cache_kla.find(KLA{kmask, depth_budget, k});
            if (it == lickety_cache_kla.end()) return false;
            out_val = it->second;
            return true;
        } else {
            auto it = lickety_cache_k2.find(K2{kmask, depth_budget});
            if (it == lickety_cache_k2.end()) return false;
            out_val = it->second;
            return true;
        }
    }

    // our modified lickety_split algorithm that is O(nk^2d^2). 
    int lickety_split_k1(const Packed& mask, int8_t depth_budget, const PathKey& pk)
    {
        if (depth_budget == 0) return leaf_objective(mask);
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);
        if (depth_budget == 2) return depth2_special_solver_cached(mask, pk);
        if (depth_budget == 3) return train_greedy(mask, depth_budget, pk);

        // ---- caching (k=1 => use K2 cache) ----
        uint64_t kmask = 0;
        K2 key2{0, depth_budget};

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                return it->second;
        }

        const int leaf_loss = leaf_objective(mask);

        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                lickety_cache_k2.emplace(key2, leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = std::numeric_limits<int>::max();

        Packed L(n_words), R(n_words);
        Packed bestL(n_words), bestR(n_words);

        const int F = proxy_feat_count_();
        const int8_t child_depth = (int8_t)(depth_budget - 1);

        int best_cached_sum = std::numeric_limits<int>::max();

        // ---- k=1 lookahead: evaluate children via GREEDY (because child_k = 0) ----
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_greedy  = train_greedy(L, child_depth, *pkLp);
            const int right_greedy = train_greedy(R, child_depth, *pkRp);
            const int sum = left_greedy + right_greedy;

            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;
            }

            if (stronger_rollout && proxy_caching_enabled) {
                int left_val = left_greedy;
                int right_val = right_greedy;
                bool have_left = false;
                bool have_right = false;

                const uint64_t kL = key_of_subproblem(L, *pkLp);
                const uint64_t kR = key_of_subproblem(R, *pkRp);

                if (auto itL = lickety_cache_k2.find(K2{kL, child_depth});
                    itL != lickety_cache_k2.end()) {
                    left_val = itL->second;
                    have_left = true;
                }

                if (auto itR = lickety_cache_k2.find(K2{kR, child_depth});
                    itR != lickety_cache_k2.end()) {
                    right_val = itR->second;
                    have_right = true;
                }

                if (have_left || have_right) {
                    best_cached_sum = std::min(best_cached_sum, left_val + right_val);
                }
            }
        }

        int ans = leaf_loss;

         if (best_cached_sum != std::numeric_limits<int>::max()) {
            ans = std::min(ans, best_cached_sum);
        }

        // ---- recurse with constant k=1 (proxy_style=0 behavior) ----
        if (best_feat >= 0) {
            const int8_t next_depth = (int8_t)(depth_budget - 1);

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_loss  = lickety_split_k1(bestL, next_depth, *pkLp);
            const int right_loss = lickety_split_k1(bestR, next_depth, *pkRp);

            ans = std::min(ans, left_loss + right_loss);
        }

        if (proxy_caching_enabled) {
            lickety_cache_k2.emplace(key2, ans);
        }
        return ans;
    }

    // a generalized lickety_split algorithm with a lookahead parameter k. we also support other proxy styles here (such as recursively applying split) that have the same flavor.
    int generalized_lickety_split(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        if (k == 1 && lookahead_init == 1 && proxy_style == 0) {
            return lickety_split_k1(mask, depth_budget, pk);
        }

        if (depth_budget == 0) {
            return leaf_objective(mask);
        }
        if (depth_budget == 1) return depth1_exact_solver_cached(mask, pk);

        if (k > depth_budget - 1) k = depth_budget - 1;

        if (depth_budget == 2 && k == 1) return depth2_special_solver_cached(mask, pk);
        if (k == depth_budget - 1) {
            return depthd_exact_solver_cached(mask, depth_budget, pk);
        }

        uint64_t kmask = 0;
        K2  key2{0, depth_budget};
        KLA keyla{0, depth_budget, k};

        const bool use_kla = use_kla_cache();

        if (proxy_caching_enabled) {
            kmask = key_of_subproblem(mask, pk);
            key2.k = kmask;
            keyla.k = kmask;

            if (use_kla) {
                if (auto it = lickety_cache_kla.find(keyla); it != lickety_cache_kla.end())
                    return it->second;
            } else {
                if (auto it = lickety_cache_k2.find(key2); it != lickety_cache_k2.end())
                    return it->second;
            }
        }

        const int leaf_loss = leaf_objective(mask);
        if (leaf_loss <= 2 * gamma) {
            if (cache_cheap_subproblems && proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, leaf_loss);
                else         lickety_cache_k2.emplace(key2,  leaf_loss);
            }
            return leaf_loss;
        }

        int best_feat = -1;
        int best_sum  = numeric_limits<int>::max();

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);

        const int child_k = k - 1;
        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int sum = eval_with_lookahead(L, depth_budget - 1, child_k, *pkLp) + eval_with_lookahead(R, depth_budget - 1, child_k, *pkRp);
            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                // bestL.w = L.w; // could deep-element copy here but we'll wait until the end
                // bestR.w = R.w;
            }
        }

        if (best_feat >= 0) {
            and_bits(mask, X_bits[best_feat], bestL);
            andnot_bits(mask, X_bits[best_feat], bestR);
        }

        int ans = leaf_loss; 

        int8_t k_recurse;
        if (proxy_style == 0) {
            // style 0: constant k (recursively choosing based on lower tier LicketySPLIT)
            k_recurse = k;
        } else if (proxy_style == 3) {
            // style 3: if we're running SPLIT without postprocessing, we don't need to do further recursive calls (the tree is fully determined).
            ans = std::min(ans, best_sum);
            if (proxy_caching_enabled) {
                if (use_kla) lickety_cache_kla.emplace(keyla, ans);
                else         lickety_cache_k2.emplace(key2,  ans);
            }
            return ans;
        } else {
            // styles 1/2: restart when it hits 0 (recursively applying SPLIT, so we're cycling k, k-1, k-2, ... 2 1 k)
            k_recurse = (child_k == 0) ? lookahead_init : child_k;
        }


        if (best_feat >= 0) {
            const int next_depth = depth_budget - 1;

            int left_loss, right_loss;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);

            left_loss  = generalized_lickety_split(bestL, next_depth, k_recurse, *pkLp);
            right_loss = generalized_lickety_split(bestR, next_depth, k_recurse, *pkRp);
            ans = std::min(ans, left_loss + right_loss); // do licketysplit and take the minimum of it and leaf, even if greedy doesn't perform better than the leaf.

        }
        if (proxy_caching_enabled) {
            if (use_kla) lickety_cache_kla.emplace(keyla, ans);
            else lickety_cache_k2.emplace(key2,  ans);
        }
        
        return ans;
    }

    int split_algorithm(const Packed& mask, int8_t depth_budget, int8_t k, const PathKey& pk) {
        if (depth_budget <= 0) return leaf_objective(mask);
        if (k > depth_budget - 1) k = depth_budget - 1; // same amount of computation is optimal
        if (k == depth_budget - 1) {
            return depthd_exact_solver_cached(mask, depth_budget, pk);
        }

        // if lookahead is exhausted, switch to optimal at this remaining depth
        if (k <= 0) {
            return depthd_exact_solver_cached(mask, depth_budget, pk);
        }

        const int8_t child_d = (int8_t)(depth_budget - 1);
        const int8_t child_k = (int8_t)(k - 1);

        Packed L(n_words), R(n_words);

        int best_feat = -1;
        int best_score = std::numeric_limits<int>::max();

        Packed bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            // choose the best split based on generalized_lickety_split at (d-1, k-1). with the strategy 3/4 (SPLIT algorithm), this is tracing out the splits that the SPLIT algorithm without postprocessing chose.
            int left_score  = generalized_lickety_split(L, child_d, child_k, *pkLp);
            int right_score = generalized_lickety_split(R, child_d, child_k, *pkRp);
            int score = left_score + right_score;

            if (score < best_score) {
                best_score = score;
                best_feat = f;

                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        }

        if (best_feat < 0) {
            return leaf_objective(mask);
        }

        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        }

        // instead of recursing with a greedy tree, we apply postprocessing here.
        // we do optimal on the children because we chose a split that was already best for greedy completion (aka we are tracing out the top k splits of the SPLIT tree without postprocessing to then apply it)
        if (child_k <= 0) {
            int left_cost  = depthd_exact_solver_cached(bestL, child_d, *pkLp);
            int right_cost = depthd_exact_solver_cached(bestR, child_d, *pkRp);
            return left_cost + right_cost;
        }

        // otherwise recurse using split() itself to find the next split it made
        int left_cost  = split_algorithm(bestL, child_d, child_k, *pkLp);
        int right_cost = split_algorithm(bestR, child_d, child_k, *pkRp);
        return left_cost + right_cost;
    }

    int find_best_split_binary_known_counts(
        const Packed& mask,
        int n_sub,
        int pos_total,
        int bb_wrong_total,
        bool use_entropy
    ) const {
        if (n_sub <= 1) return -1;

        const Packed& Ypos = Y_bits[(size_t)1];

        int best_f = -1;
        const int F = proxy_feat_count_();

        if (use_entropy) {
            double best_score = 1e300;

            for (int f = 0; f < F; ++f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double pl = (double)left_pos  / (double)left_n;
                const double pr = (double)right_pos / (double)right_n;

                const double score = wl * entropy(pl) + wr * entropy(pr);

                if (score < best_score) {
                    best_score = score;
                    best_f = f;
                }
            }

            return best_f;
        } else {
            int best_sum = std::numeric_limits<int>::max();

            for (int f = 0; f < F; ++f) {
                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                Packed L(n_words), R(n_words);

                split_bits_count_left_binary(
                    mask,
                    X_bits[(size_t)f],
                    L,
                    R,
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;
                const int right_bb_wrong = bb_wrong_total - left_bb_wrong;

                const int left_loss =
                    leaf_objective_binary_from_counts(left_n, left_pos, left_bb_wrong);

                const int right_loss =
                    leaf_objective_binary_from_counts(right_n, right_pos, right_bb_wrong);

                const int sum = left_loss + right_loss;

                if (sum < best_sum) {
                    best_sum = sum;
                    best_f = f;
                }
            }

            return best_f;
        }
    }

    int find_best_split_binary(const Packed& mask, bool use_entropy) const {
        int n_sub = 0;
        int pos_total = 0;
        int bb_wrong_total = 0;
        count_total_pos_bbwrong_binary(mask, n_sub, pos_total, bb_wrong_total);

        if (n_sub <= 1) return -1;

        const Packed& Ypos = Y_bits[(size_t)1];

        int best_f = -1;
        const int F = proxy_feat_count_();

        if (use_entropy) {
            double best_score = 1e300;

            for (int f = 0; f < F; ++f) {
                int left_n = 0;
                int left_pos = 0;

                const Packed& Xf = X_bits[(size_t)f];

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n   += popcnt64(lw);
                    left_pos += popcnt64(lw & Ypos.w[(size_t)i]);
                }

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double pl = (double)left_pos  / (double)left_n;
                const double pr = (double)right_pos / (double)right_n;

                const double score = wl * entropy(pl) + wr * entropy(pr);

                if (score < best_score) {
                    best_score = score;
                    best_f = f;
                }
            }

            return best_f;
        } else {
            int best_sum = std::numeric_limits<int>::max();

            for (int f = 0; f < F; ++f) {
                int left_n = 0;
                int left_pos = 0;
                int left_bb_wrong = 0;

                split_count_left_binary_fast_(
                    mask,
                    X_bits[(size_t)f],
                    left_n,
                    left_pos,
                    left_bb_wrong
                );

                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;
                const int right_bb_wrong = bb_wrong_total - left_bb_wrong;

                const int left_loss =
                    leaf_objective_binary_from_counts(left_n, left_pos, left_bb_wrong);

                const int right_loss =
                    leaf_objective_binary_from_counts(right_n, right_pos, right_bb_wrong);

                const int sum = left_loss + right_loss;

                if (sum < best_sum) {
                    best_sum = sum;
                    best_f = f;
                }
            }

            return best_f;
        }
    }


    int find_best_split(const Packed& mask, bool use_entropy) const {
        if (num_classes == 2) {
            return find_best_split_binary(mask, use_entropy);
        }
        const int n_sub = count_total(mask);
        if (n_sub <= 1) return -1;

        Packed L(n_words);

        if (use_entropy) {
            // total class counts under mask
            std::vector<int> total_cnt((size_t)num_classes, 0);
            for (int c = 0; c < num_classes; ++c) {
                total_cnt[(size_t)c] = popcount_and(mask, Y_bits[(size_t)c]);
            }

            int best_f = -1;
            double best_score = 1e300;

            std::vector<int> left_cnt((size_t)num_classes, 0);
            std::vector<int> right_cnt((size_t)num_classes, 0);

            const int F = proxy_feat_count_();
            for (int f = 0; f < F; ++f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[(size_t)f].w[i];
                L.w[n_words - 1] &= tail_mask;

                const int left_n  = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                // left class counts: popcount_and(L, Y_bits[c])
                for (int c = 0; c < num_classes; ++c) {
                    left_cnt[(size_t)c] = popcount_and(L, Y_bits[(size_t)c]);
                }

                // right class counts = total - left (no need to build)
                for (int c = 0; c < num_classes; ++c) {
                    right_cnt[(size_t)c] = total_cnt[(size_t)c] - left_cnt[(size_t)c];
                }

                const double wl = (double)left_n  / (double)n_sub;
                const double wr = (double)right_n / (double)n_sub;

                const double Hl = entropy_multiclass(left_cnt,  left_n);
                const double Hr = entropy_multiclass(right_cnt, right_n);

                const double score = (wl * Hl + wr * Hr);
                if (score < best_score) { 
                    best_score = score; 
                    best_f = f; 
                }
            }
            return best_f;

        } else {
            // minimize child leaf objectives: leaf_objective(L)+leaf_objective(R)
            int best_f = -1;
            int best_sum = std::numeric_limits<int>::max();

            Packed R(n_words);
            const int F = proxy_feat_count_();
            for (int f = 0; f < F; ++f) {
                // L = mask & X_bits[f]
                for (int i = 0; i < n_words; ++i) L.w[i] = mask.w[i] & X_bits[f].w[i]; 
                L.w[n_words-1] &= tail_mask;

                const int left_n = L.count();
                const int right_n = n_sub - left_n;
                if (left_n == 0 || right_n == 0) continue;

                // R = mask & ~X_bits[f]
                for (int i = 0; i < n_words; ++i) R.w[i] = mask.w[i] & ~X_bits[f].w[i];
                R.w[n_words-1] &= tail_mask;

                const int sum = leaf_objective(L) + leaf_objective(R);
                if (sum < best_sum) { best_sum = sum; best_f = f; }
            }
            return best_f;
        }
    }

    // not used by PRAXIS Rashomon mode, to support giving single decision tree algorithm results in package.
    shared_ptr<PredNode> build_best_tree_from_caches(const Packed& mask, int8_t depth_budget, const PathKey& pk) const {
        const int INF = std::numeric_limits<int>::max();

        const int n_sub = count_total(mask);
        if (n_sub == 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = 0;
            return t;
        }

        // std::vector<int> cnts;
        // count_per_class(mask, cnts);
        // int best_c = 0;
        // int best_cnt = cnts[0];
        // for (int c = 1; c < num_classes; ++c) {
        //     int v = cnts[(size_t)c];
        //     if (v > best_cnt || (v == best_cnt && c > best_c)) {
        //         best_cnt = v;
        //         best_c = c;
        //     }
        // }
        // const int leaf_pred = best_c;
        // const int mis = n_sub - best_cnt;
        // const int leaf_loss = gamma + mis;

        const BestLeafAction leaf = best_leaf_action(mask);
        const int leaf_pred = leaf.prediction;
        const int leaf_loss = leaf.loss;

        if (depth_budget <= 0) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // helper: lookup min cached objective for (mask, depth, pk) across greedy + lickety caches
        auto best_cached_obj = [&](const Packed& m, int8_t d, const PathKey& pk_child) -> int {
            if (d < 0) return 0;
            if (d==0) return leaf_objective(m);
            if (!proxy_caching_enabled) return INF;

            const uint64_t km = key_of_subproblem(m, pk_child);

            int best = INF;

            // greedy cache: (subproblem, depth)
            {
                auto itg = greedy_cache.find(K2{km, d});
                if (itg != greedy_cache.end()) best = std::min(best, itg->second);
            }

            // lickety cache:
            if (use_kla_cache()) {
                // try all k = 0..d-1
                for (int kk = 0; kk <= (int)(d-1); ++kk) {
                    auto it = lickety_cache_kla.find(KLA{km, d, kk});
                    if (it != lickety_cache_kla.end()) best = std::min(best, it->second);
                }
            } else {
                // K2-form: no k needed
                auto it = lickety_cache_k2.find(K2{km, d});
                if (it != lickety_cache_k2.end()) best = std::min(best, it->second);
            }

            return best;
        };

        // choose best split using cached objectives
        const int8_t child_d = (int8_t)(depth_budget - 1);

        int best_feat = -1;
        int best_sum  = INF;

        Packed L(n_words), R(n_words), bestL(n_words), bestR(n_words);
        PathKey bestPkL, bestPkR;
        bool have_best_pks = false;

        const int F = proxy_feat_count_();
        for (int f = 0; f < F; ++f) {
            and_bits(mask, X_bits[f], L);
            andnot_bits(mask, X_bits[f], R);
            if (!L.any() || !R.any()) continue;

            const PathKey* pkLp = &empty_pk();
            const PathKey* pkRp = &empty_pk();
            PathKey pkL_local, pkR_local;
            make_child_pks_if_needed_(f, pk, pkLp, pkRp, pkL_local, pkR_local);

            const int left_obj  = best_cached_obj(L, child_d, *pkLp);
            const int right_obj = best_cached_obj(R, child_d, *pkRp);
            if (left_obj == INF || right_obj == INF) continue;

            const int sum = left_obj + right_obj;
            if (sum < best_sum) {
                best_sum = sum;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;

                if (key_mode == KeyMode::LITS_EXACT) {
                    bestPkL = *pkLp;
                    bestPkR = *pkRp;
                    have_best_pks = true;
                } else {
                    have_best_pks = false;
                }
            }
        }

        // If no split is yielded or it doesn't beat leaf, return leaf.
        if (best_feat < 0 || best_sum >= leaf_loss) {
            auto t = make_shared<PredNode>();
            t->feature = -1;
            t->prediction = leaf_pred;
            return t;
        }

        // --- recurse on chosen split ---
        const PathKey* pkLp = &empty_pk();
        const PathKey* pkRp = &empty_pk();
        PathKey pkL_local, pkR_local;

        if (key_mode == KeyMode::LITS_EXACT) {
            if (have_best_pks) {
                pkLp = &bestPkL;
                pkRp = &bestPkR;
            } else {
                make_child_pks_if_needed_(best_feat, pk, pkLp, pkRp, pkL_local, pkR_local);
            }
        } else {
            // non-literal mode: pk is ignored by key_of_subproblem anyway, so keep empty_pk()
            pkLp = &empty_pk();
            pkRp = &empty_pk();
        }

        auto left_tree  = build_best_tree_from_caches(bestL, child_d, *pkLp);
        auto right_tree = build_best_tree_from_caches(bestR, child_d, *pkRp);

        auto t = make_shared<PredNode>();
        t->feature = best_feat;
        t->prediction = -1;
        t->left = left_tree;
        t->right = right_tree;
        return t;
    }

    shared_ptr<PredNode> get_ith_tree(uint64_t i) const {
        if (!result) {
            throw runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }
        
        // count_trees will ensure that the histograms are built at the root and every child node (by building them if they are not yet built)
        uint64_t total = result->count_trees();
        if (i >= total) {
            throw out_of_range("Tree index out of range in get_ith_tree");
        }

        uint64_t cum = 0;
        int target_obj = -1;
        uint64_t k_within = 0;

        // hist is sorted by objective ascending
        for (const auto& e : result->hist) {
            if (i < cum + e.cnt) {
                target_obj = e.obj;
                k_within = i - cum;
                break;
            }
            cum += e.cnt;
        }
        if (target_obj < 0) {
            throw runtime_error("Failed to locate objective bucket in get_ith_tree");
        }

        return get_kth_tree_with_objective(result.get(), target_obj, k_within);
    }

    shared_ptr<PredNode> get_kth_tree_with_objective(const TreeTrieNode* node, int target_obj, uint64_t k) const {
        if (!node) {
            throw runtime_error("Null node in get_kth_tree_with_objective");
        }

        // handle leaf-only trees at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss == target_obj) {
                if (k == 0) {
                    auto t = make_shared<PredNode>();
                    t->feature = -1;
                    t->prediction = leaf.prediction;
                    return t;
                }
                --k;
            }
        }

        // handle splits
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();

            // total_here = #trees under this split with exactly target_obj
            uint64_t total_here = 0;

            // for each L histogram entry, we go over it. R is sorted by obj, so we can binary search to find each r_obj that pairs to sum to exactly this target_obj.
            for (const auto& le : L->hist) {
                int l_obj = le.obj;
                uint64_t lc = le.cnt;
                int r_obj = target_obj - l_obj;
                // binary search r_obj in R->hist
                auto it = lower_bound(
                    R->hist.begin(), R->hist.end(),
                    HistEntry{r_obj, 0},
                    hist_less
                );
                if (it != R->hist.end() && it->obj == r_obj) {
                    uint64_t rc = it->cnt;
                    uint64_t pairs = lc * rc;

                    if (total_here + pairs > k) { // k is smaller than the culm amount in this split we've seen so far (for the first time), so we know that we want to recurse on this split (which was already known), with this particular l_obj and r_obj, but we also need what index within each objective to recurse 
                        uint64_t rel = k - total_here; // what index inside this block the tree lives (again, 0 indexed)
                        uint64_t left_idx  = rel / rc; // left contributes lc possibilities, right contributes rc, a cross product without filtering, this indexing scheme works to break ties
                        uint64_t right_idx = rel % rc;

                        auto left_tree  = get_kth_tree_with_objective(L, l_obj,  left_idx); // now we have all the information we need, recurse
                        auto right_tree = get_kth_tree_with_objective(R, r_obj, right_idx);

                        auto t = make_shared<PredNode>();
                        t->feature = split.feature;
                        t->prediction = -1;
                        t->left = left_tree;
                        t->right = right_tree;
                        return t;
                    }

                    total_here += pairs;
                }

            }
            // skip all trees from this split that achieve target_obj
            k -= total_here;
            
        }

        throw out_of_range("Index out of range for given objective in get_kth_tree_with_objective");
    }

    void predict_tree_recursive(
        const PredNode* node,
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& bb_pred_row,
        std::vector<uint8_t>& out,
        const std::vector<int>& idx,
        bool use_placeholder,
        int defer_placeholder
    ) const {
        if (!node) return;

        // leaf: assign prediction to all indices in this subset.
        if (node->feature < 0) {
            if (node->prediction == DEFER_PREDICTION) {
                for (int row : idx) {
                    const int p = use_placeholder
                        ? defer_placeholder
                        : bb_pred_row[(size_t)row];

                    out[(size_t)row] = static_cast<uint8_t>(p);
                }
            } else {
                uint8_t pred = static_cast<uint8_t>(node->prediction);
                for (int row : idx) {
                    out[(size_t)row] = pred;
                }
            }
            return;
        }

        // internal node: split indices by feature
        int f = node->feature;
        std::vector<int> left_idx;
        std::vector<int> right_idx;
        left_idx.reserve(idx.size()); 
        right_idx.reserve(idx.size());

        for (int row : idx) {
            uint8_t v = X_row_major[row][f];
            if (v) left_idx.push_back(row); // 1 is left
            else   right_idx.push_back(row);
        }

        if (!left_idx.empty()) {
            predict_tree_recursive(
                node->left.get(),
                X_row_major,
                bb_pred_row,
                out,
                left_idx,
                use_placeholder,
                defer_placeholder
            );
        }
        if (!right_idx.empty()) {
            predict_tree_recursive(
                node->right.get(),
                X_row_major,
                bb_pred_row,
                out,
                right_idx,
                use_placeholder,
                defer_placeholder
            );
        }
    }

    void collect_paths(const PredNode* node, std::vector<int>& current, std::vector<std::vector<int>>& paths, std::vector<int>& preds) const {
        if (!node) {
            throw std::logic_error("collect_paths: encountered null node");
        }

        // leaf: record this path and prediction
        if (node->feature < 0) {
            paths.push_back(current); // current starts empty and is appended to along the dfs
            preds.push_back(node->prediction);
            return;
        }

        int f = node->feature;
        // IMPORTANT: we have to switch to 1-indexing here so that +- for the 0th (1st) feature means something

        // go left (true) -> +f or rather f+1
        current.push_back(f+1);
        collect_paths(node->left.get(), current, paths, preds);
        current.pop_back(); // backtrack after we complete a path so we have 1 vector that is updated in a nice way throughout this

        // go right (false) -> -f (technically -(f+1))
        current.push_back(-(f+1));
        collect_paths(node->right.get(), current, paths, preds);
        current.pop_back();
    }

// whole trie prediction for RID

private:
    static inline void and_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] & b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline void andnot_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] & ~b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline void or_bits_eval(const Packed& a, const Packed& b, Packed& out, int n_words, uint64_t tail_mask) {
        for (int i = 0; i < n_words; ++i) out.w[i] = a.w[i] | b.w[i];
        out.w[n_words - 1] &= tail_mask;
    }

    static inline bool any_eval(const Packed& a) {
        for (uint64_t t : a.w) if (t) return true;
        return false;
    }

    static inline void clear_eval(Packed& a) {
        std::fill(a.w.begin(), a.w.end(), 0ULL);
    }

    static inline Packed zeros_eval(int n_words) {
        return Packed((size_t)n_words); // zeros words
    }

    static inline Packed copy_eval_mask(const Packed& m, int n_words, uint64_t tail_mask) {
        Packed out((size_t)n_words);
        out.w = m.w; // vector copy
        out.w[n_words - 1] &= tail_mask; // bitwise and
        return out;
    }

    static inline PackedPredMulti zeros_predmulti(int n_words, int num_classes) {
        PackedPredMulti p;
        p.by_class.assign((size_t)num_classes, Packed((size_t)n_words));
        p.deferred = Packed((size_t)n_words);
        return p;
    }

    static inline void clear_predmulti(PackedPredMulti& pm) {
        for (Packed& p : pm.by_class) {
            std::fill(p.w.begin(), p.w.end(), 0ULL);
        }
        std::fill(pm.deferred.w.begin(), pm.deferred.w.end(), 0ULL);
    }

    // OR-combine two multiclass prediction packs into out
    static inline void or_predmulti(
        const PackedPredMulti& a,
        const PackedPredMulti& b,
        PackedPredMulti& out,
        int n_words,
        uint64_t tail_mask
    ) {
        const int C = (int)a.by_class.size();

        for (int c = 0; c < C; ++c) {
            auto& ow = out.by_class[(size_t)c].w;
            const auto& aw = a.by_class[(size_t)c].w;
            const auto& bw = b.by_class[(size_t)c].w;

            for (int w = 0; w < n_words; ++w) {
                ow[(size_t)w] = aw[(size_t)w] | bw[(size_t)w];
            }

            if (n_words > 0) {
                ow[(size_t)(n_words - 1)] &= tail_mask;
            }
        }

        for (int w = 0; w < n_words; ++w) {
            out.deferred.w[(size_t)w] =
                a.deferred.w[(size_t)w] | b.deferred.w[(size_t)w];
        }

        if (n_words > 0) {
            out.deferred.w[(size_t)(n_words - 1)] &= tail_mask;
        }
    }


    // build packed feature columns for EVAL X, turning into column major
    static inline EvalCtx build_eval_ctx_(const std::vector<std::vector<uint8_t>>& X_row_major, int n_features_expected) {
        EvalCtx ctx;

        ctx.n_eval = (int)X_row_major.size();
        if (ctx.n_eval == 0) {
            ctx.n_words = 0;
            ctx.tail_mask = ~0ULL;
            return ctx;
        }

        const int d = (int)X_row_major[0].size();
        if (d != n_features_expected) {
            throw std::runtime_error("Eval X has different number of features than training.");
        }

        ctx.n_words = (ctx.n_eval + 63) / 64;
        ctx.tail_mask = (ctx.n_eval % 64) ? ((1ULL << (ctx.n_eval % 64)) - 1ULL) : ~0ULL;

        ctx.X_bits_eval.assign((size_t)d, Packed((size_t)ctx.n_words));

        for (int f = 0; f < d; ++f) {
            Packed &col = ctx.X_bits_eval[f];
            clear_eval(col);
            for (int i = 0; i < ctx.n_eval; ++i) {
                if (X_row_major[(size_t)i][(size_t)f]) {
                    col.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
                }
            }
            col.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
        }

        return ctx;
    }

    // all 1s bitvector (for the evaluation passed in dataset not train)
    static inline Packed eval_root_mask_(int n_words, uint64_t tail_mask) {
        Packed m((size_t)n_words);
        if (n_words == 0) return m;
        for (int i = 0; i < n_words - 1; ++i) m.w[(size_t)i] = ~0ULL;
        m.w[(size_t)(n_words - 1)] = tail_mask;
        return m;
    }

    static inline std::vector<ObjBucketMulti> to_sorted_buckets_multi_(
        std::unordered_map<int, std::vector<PackedPredMulti>>& acc
    ) {
        std::vector<ObjBucketMulti> out;
        out.reserve(acc.size());
        for (auto &kv : acc) {
            ObjBucketMulti b;
            b.obj = kv.first;
            b.preds = std::move(kv.second);
            out.push_back(std::move(b));
        }
        std::sort(out.begin(), out.end(),
                [](const ObjBucketMulti& a, const ObjBucketMulti& b){ return a.obj < b.obj; });
        return out;
    }


    // core recursion: returns buckets of predictions grouped by objective for ALL trees rooted at node with obj <= budget.
    std::vector<ObjBucketMulti> collect_preds_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask, // does not decrease size, just gets sparser
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        if (node->min_objective == std::numeric_limits<int>::max()) return {};
        if (node->min_objective > budget) return {};

        // accumulate as obj (training) -> list of preds on evaluation (Packed)
        //std::unordered_map<int, std::vector<Packed>> acc;
        std::unordered_map<int, std::vector<PackedPredMulti>> acc;
        // heuristic reserve
        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaves at this node
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            PackedPredMulti pm = zeros_predmulti(ctx.n_words, num_classes);

            if (ctx.n_words > 0) {
                if (leaf.prediction == DEFER_PREDICTION) {
                    pm.deferred.w = eval_mask.w;
                    pm.deferred.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                } else {
                    const int pc = leaf.prediction;
                    if (pc < 0 || pc >= num_classes) {
                        throw std::runtime_error("Leaf prediction is outside valid class range.");
                    }

                    pm.by_class[(size_t)pc].w = eval_mask.w;
                    pm.by_class[(size_t)pc].w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                }
            }

            acc[leaf.loss].push_back(std::move(pm));
            
            // storing the predictions in the map with that objective.
        }

        // splits
        const int INF = std::numeric_limits<int>::max();

        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;
            if (minL == INF || minR == INF) continue;

            // cap child budgets using the other side's min objective so everything found will pair with exactly one subtree on the other side
            int bL = budget - minR;
            int bR = budget - minL;
            if (bL < 0 || bR < 0) continue;

            // also cap by the budgets actually used to build those trie nodes. should never change anything (assuming we do iterative budget refinement, otherwise this is needed for tightening).
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            // evaluation dataset routing masks
            Packed Lmask((size_t)ctx.n_words), Rmask((size_t)ctx.n_words);
            if (ctx.n_words > 0) {
                and_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Lmask, ctx.n_words, ctx.tail_mask);
                andnot_bits_eval(eval_mask, ctx.X_bits_eval[(size_t)split.feature], Rmask, ctx.n_words, ctx.tail_mask);
            }

            // recurse
            auto Lb = collect_preds_by_obj_(L, bL, Lmask, ctx); // these return sorted lists of objective bucket objects
            auto Rb = collect_preds_by_obj_(R, bR, Rmask, ctx);
            if (Lb.empty() || Rb.empty()) continue;

            // for filtering by <= budget, both Lb and Rb are sorted by obj.
            // we'll two-pointer for each left obj to find all right objs <= (budget - l_obj).
            // size_t r_hi = 0; // exclusive upper bound index in Rb
            // for (size_t li = 0; li < Lb.size(); ++li) {
            //     const int lo = Lb[li].obj; // smallest objective initially
            //     if (lo > budget) break;
            //     const int rem = budget - lo; // how far do we have to look

            //     //while (r_hi < Rb.size() && Rb[r_hi].obj <= rem) ++r_hi; // getting the first invalid index. never look past the remainder because RHS is also sorted
            //     // we could either go over Lb in reverse order or just do this 
            //     auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
            //     size_t r_hi = static_cast<size_t>(std::distance(R_objs.begin(), it_end));

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj); // don't need predictions here
            }

            for (size_t li = 0; li < Lb.size(); ++li) {
                const int lo = Lb[li].obj;
                if (lo > budget) break;

                const int rem = budget - lo;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const size_t r_hi = static_cast<size_t>(
                    std::distance(R_objs.begin(), it_end)
                );
                            
                if (r_hi == 0) continue; // no right objs fit

                // cross product (filtered)
                for (size_t ri = 0; ri < r_hi; ++ri) { // r_hi is one past the last valid
                    const int ro = Rb[ri].obj; // objective value for that bucket
                    const int tot = lo + ro; // additivity of objectives

                    // combine each left pred with each right pred (disjoint masks so OR is correct)
                    const auto& Lpreds = Lb[li].preds;
                    const auto& Rpreds = Rb[ri].preds;

                    // reserve some space in this objective bucket to reduce reallocs
                    auto &dest = acc[tot]; // alias for simplicity
                    // rough reserve: only if currently empty
                    if (dest.empty()) {
                            dest.reserve(std::max(Lpreds.size(), Rpreds.size()));
                        }

                    for (const auto& lp : Lpreds) {
                        for (const auto& rp : Rpreds) {
                            PackedPredMulti comb = zeros_predmulti(ctx.n_words, num_classes);
                            if (ctx.n_words > 0) {
                                or_predmulti(lp, rp, comb, ctx.n_words, ctx.tail_mask);
                            }
                            dest.push_back(std::move(comb));
                        }
                    }

                }
            }
        }

        return to_sorted_buckets_multi_(acc);
    }

    struct ObjMistakeBucket {
        int obj;
        std::vector<int> mistakes; // one entry per tree at this objective
    };

    struct MistakesWithObj {
        int obj;
        int mistakes;
    };

    struct ObjDeferralBucket {
        int obj;
        std::vector<int> deferrals;
    };

    struct DeferralsWithObj {
        int obj;
        int deferrals;
    };

    static inline std::vector<ObjDeferralBucket> to_sorted_deferral_buckets_(
        std::unordered_map<int, std::vector<int>>& acc
    ) {
        std::vector<ObjDeferralBucket> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjDeferralBucket b;
            b.obj = kv.first;
            b.deferrals = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(out.begin(), out.end(),
            [](const ObjDeferralBucket& a, const ObjDeferralBucket& b) {
                return a.obj < b.obj;
            });

        return out;
    }

    std::vector<ObjDeferralBucket> collect_deferrals_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask,
        const EvalCtx& ctx
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (node->min_objective == INF) return {};
        if (node->min_objective > budget) return {};

        std::unordered_map<int, std::vector<int>> acc;

        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaf alternatives
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            int n_def = 0;

            if (ctx.n_words > 0 && leaf.prediction == DEFER_PREDICTION) {
                n_def = count_eval_mask_(eval_mask, ctx.n_words);
            }

            acc[leaf.loss].push_back(n_def);
        }

        // split alternatives
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < minL || bR < minR) continue;

            Packed Lmask((size_t)ctx.n_words);
            Packed Rmask((size_t)ctx.n_words);

            if (ctx.n_words > 0) {
                const Packed& Xf = ctx.X_bits_eval[(size_t)split.feature];

                for (int w = 0; w < ctx.n_words; ++w) {
                    const uint64_t mw = eval_mask.w[(size_t)w];
                    const uint64_t xw = Xf.w[(size_t)w];

                    Lmask.w[(size_t)w] = mw & xw;
                    Rmask.w[(size_t)w] = mw & ~xw;
                }

                Lmask.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
                Rmask.w[(size_t)(ctx.n_words - 1)] &= ctx.tail_mask;
            }

            auto Lb = collect_deferrals_by_obj_(L, bL, Lmask, ctx);
            auto Rb = collect_deferrals_by_obj_(R, bR, Rmask, ctx);

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                if (lb.obj > bL) continue;

                const int rem = budget - lb.obj;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const int j_end = (int)std::distance(R_objs.begin(), it_end);

                for (int j = 0; j < j_end; ++j) {
                    const auto& rb = Rb[(size_t)j];

                    const int total_obj = lb.obj + rb.obj;
                    if (total_obj > budget) continue;

                    auto& vec = acc[total_obj];

                    for (int ld : lb.deferrals) {
                        for (int rd : rb.deferrals) {
                            vec.push_back(ld + rd);
                        }
                    }
                }
            }
        }

        return to_sorted_deferral_buckets_(acc);
    }



    static inline std::vector<ObjMistakeBucket> to_sorted_mistake_buckets_(
        std::unordered_map<int, std::vector<int>>& acc
    ) {
        std::vector<ObjMistakeBucket> out;
        out.reserve(acc.size());

        for (auto& kv : acc) {
            ObjMistakeBucket b;
            b.obj = kv.first;
            b.mistakes = std::move(kv.second);
            out.push_back(std::move(b));
        }

        std::sort(out.begin(), out.end(),
            [](const ObjMistakeBucket& a, const ObjMistakeBucket& b) {
                return a.obj < b.obj;
            });

        return out;
    }

    static inline Packed build_eval_bb_wrong_bits_(
        const std::vector<int>& y_eval,
        const std::vector<int>& bb_pred_eval,
        int num_classes,
        int n_words,
        uint64_t tail_mask
    ) {
        if (bb_pred_eval.size() != y_eval.size()) {
            throw std::runtime_error("Eval bb_pred has different number of rows than Eval y.");
        }

        Packed wrong((size_t)n_words);

        for (int i = 0; i < (int)y_eval.size(); ++i) {
            const int yi = y_eval[(size_t)i];
            const int bi = bb_pred_eval[(size_t)i];

            if (yi < 0 || yi >= num_classes) {
                throw std::runtime_error("Eval y contains a class not seen during training.");
            }
            if (bi < 0 || bi >= num_classes) {
                throw std::runtime_error("Eval bb_pred contains a class not seen during training.");
            }

            if (yi != bi) {
                wrong.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
            }
        }

        if (n_words > 0) {
            wrong.w[(size_t)(n_words - 1)] &= tail_mask;
        }

        return wrong;
    }

    static inline std::vector<Packed> build_eval_y_bits_(
        const std::vector<int>& y_eval,
        int num_classes,
        int n_words,
        uint64_t tail_mask
    ) {
        std::vector<Packed> Y_eval((size_t)num_classes, Packed((size_t)n_words));

        for (int c = 0; c < num_classes; ++c) {
            clear_eval(Y_eval[(size_t)c]);
        }

        for (int i = 0; i < (int)y_eval.size(); ++i) {
            const int yi = y_eval[(size_t)i];
            if (yi < 0 || yi >= num_classes) {
                throw std::runtime_error("Eval y contains a class not seen during training.");
            }
            Y_eval[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }

        if (n_words > 0) {
            for (int c = 0; c < num_classes; ++c) {
                Y_eval[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
            }
        }

        return Y_eval;
    }

    static inline int count_eval_mask_(const Packed& mask, int n_words) {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(mask.w[(size_t)i]);
        }
        return s;
    }

    static inline int popcount_and_eval_(
        const Packed& a,
        const Packed& b,
        int n_words
    ) {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(a.w[(size_t)i] & b.w[(size_t)i]);
        }
        return s;
    }

    std::vector<ObjMistakeBucket> collect_mistakes_by_obj_(
        const TreeTrieNode* node,
        int budget,
        const Packed& eval_mask,
        const EvalCtx& ctx,
        const std::vector<Packed>& Y_eval_bits,
        const Packed* BBwrong_eval
    ) const {
        if (!node) return {};
        if (budget < 0) return {};

        constexpr int INF = std::numeric_limits<int>::max();

        if (node->min_objective == INF) return {};
        if (node->min_objective > budget) return {};

        // training objective maps to list of eval misclassification counts,
        // one scalar per tree
        std::unordered_map<int, std::vector<int>> acc;

        const int max_objs = budget - node->min_objective + 1;
        acc.reserve((size_t)std::max(1, max_objs));

        // leaf alternatives
        for (const auto& leaf : node->leaves) {
            if (leaf.loss > budget) continue;

            int mistakes = 0;

            if (ctx.n_words > 0) {
                if (leaf.prediction == DEFER_PREDICTION) {
                    if (!BBwrong_eval) {
                        throw std::runtime_error(
                            "Deferred leaf encountered, but eval bb_pred was not provided."
                        );
                    }

                    mistakes = popcount_and_eval_(
                        eval_mask,
                        *BBwrong_eval,
                        ctx.n_words
                    );
                } else {
                    const int pred_class = leaf.prediction;
                    if (pred_class < 0 || pred_class >= num_classes) {
                        throw std::runtime_error("Leaf prediction is outside valid class range.");
                    }

                    const int n_here = count_eval_mask_(eval_mask, ctx.n_words);
                    const int correct = popcount_and_eval_(
                        eval_mask,
                        Y_eval_bits[(size_t)pred_class],
                        ctx.n_words
                    );

                    mistakes = n_here - correct;
                }
            }

            acc[leaf.loss].push_back(mistakes);
        }

        // split alternatives
        for (const auto& split : node->splits) {
            const TreeTrieNode* L = split.left.get();
            const TreeTrieNode* R = split.right.get();
            if (!L || !R) continue;

            const int minL = L->min_objective;
            const int minR = R->min_objective;

            if (minL == INF || minR == INF) continue;

            int bL = budget - minR;
            int bR = budget - minL;

            if (bL < 0 || bR < 0) continue;

            // safety
            bL = std::min(bL, L->budget);
            bR = std::min(bR, R->budget);

            Packed Lmask((size_t)ctx.n_words);
            Packed Rmask((size_t)ctx.n_words);

            if (ctx.n_words > 0) {
                and_bits_eval(
                    eval_mask,
                    ctx.X_bits_eval[(size_t)split.feature],
                    Lmask,
                    ctx.n_words,
                    ctx.tail_mask
                );

                andnot_bits_eval(
                    eval_mask,
                    ctx.X_bits_eval[(size_t)split.feature],
                    Rmask,
                    ctx.n_words,
                    ctx.tail_mask
                );
            }

            auto Lb = collect_mistakes_by_obj_(L, bL, Lmask, ctx, Y_eval_bits, BBwrong_eval);
            auto Rb = collect_mistakes_by_obj_(R, bR, Rmask, ctx, Y_eval_bits, BBwrong_eval);

            if (Lb.empty() || Rb.empty()) continue;

            std::vector<int> R_objs;
            R_objs.reserve(Rb.size());
            for (const auto& rb : Rb) {
                R_objs.push_back(rb.obj);
            }

            for (const auto& lb : Lb) {
                const int lo = lb.obj;
                if (lo > budget) break;

                const int rem = budget - lo;

                auto it_end = std::upper_bound(R_objs.begin(), R_objs.end(), rem);
                const size_t r_hi = (size_t)std::distance(R_objs.begin(), it_end);

                if (r_hi == 0) continue;

                for (size_t ri = 0; ri < r_hi; ++ri) {
                    const int ro = Rb[ri].obj;
                    const int tot = lo + ro;

                    const auto& Lmis = lb.mistakes;
                    const auto& Rmis = Rb[ri].mistakes;

                    auto& dest = acc[tot];                    

                    for (int lm : Lmis) {
                        for (int rm : Rmis) {
                            dest.push_back(lm + rm);
                        }
                    }
                }
            }
        }

        return to_sorted_mistake_buckets_(acc);
    }

public:

    std::vector<DeferralsWithObj> get_all_deferrals_objs_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        int budget_override = -1
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        auto buckets = collect_deferrals_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx
        );

        std::vector<DeferralsWithObj> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.deferrals.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int d : b.deferrals) {
                out.push_back(DeferralsWithObj{b.obj, d});
            }
        }

        return out;
    }

    std::vector<int> get_all_deferrals_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        int budget_override = -1
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        auto buckets = collect_deferrals_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx
        );

        std::vector<int> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.deferrals.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int d : b.deferrals) {
                out.push_back(d);
            }
        }

        return out;
    }


    // main entry: enumerate ALL valid trees under the trie root (or budget_override if >=0),
    // returning (training_objective, prediction vector on evaluation dataset) for each tree.
    // NOTE: this can be extremely large in memory if the Rashomon set is huge.
    std::vector<PredPackWithObj> get_all_predictions_packed_trie(const std::vector<std::vector<uint8_t>>& X_row_major, int budget_override = -1) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features); // get the evaluation dataset in column major

        // decide budget
        int budget = (budget_override >= 0) ? budget_override : result->budget;

        // root eval mask = all eval rows are 1, with the padding 0s (eval_rows % 64)
        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        // collect grouped by objective
        auto buckets = collect_preds_by_obj_(result.get(), budget, root_mask, ctx);

        // flatten
        std::vector<PredPackWithObj> out;
        // compute total count for reserve
        size_t total = 0;
        for (const auto& b : buckets) total += b.preds.size();
        out.reserve(total);

        for (auto &b : buckets) {
            for (auto &p : b.preds) {
                out.push_back(PredPackWithObj{b.obj, std::move(p)});
            }
        }
        return out;
    }

    std::vector<MistakesWithObj> get_all_misclassifications_objs_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<int>& bb_pred_eval = {}
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        if ((int)y_eval.size() != ctx.n_eval) {
            throw std::runtime_error("Eval y has different number of rows than Eval X.");
        }

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        std::vector<Packed> Y_eval_bits = build_eval_y_bits_(
            y_eval,
            num_classes,
            ctx.n_words,
            ctx.tail_mask
        );

        Packed BBwrong_eval_storage((size_t)ctx.n_words);
        const Packed* BBwrong_eval_ptr = nullptr;

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "get_all_misclassifications_objs_packed_trie: "
                    "deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            BBwrong_eval_storage = build_eval_bb_wrong_bits_(
                y_eval,
                bb_pred_eval,
                num_classes,
                ctx.n_words,
                ctx.tail_mask
            );

            BBwrong_eval_ptr = &BBwrong_eval_storage;
        }



       auto buckets = collect_mistakes_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx,
            Y_eval_bits,
            BBwrong_eval_ptr
        );

        std::vector<MistakesWithObj> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.mistakes.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int m : b.mistakes) {
                out.push_back(MistakesWithObj{b.obj, m});
            }
        }

        return out;
    }

    std::vector<int> get_all_misclassifications_packed_trie(
        const std::vector<std::vector<uint8_t>>& X_row_major,
        const std::vector<int>& y_eval,
        int budget_override = -1,
        const std::vector<int>& bb_pred_eval = {}
    ) const {
        if (!result) {
            throw std::runtime_error("No Rashomon trie has been constructed. Call fit() first.");
        }

        EvalCtx ctx = build_eval_ctx_(X_row_major, this->n_features);

        if ((int)y_eval.size() != ctx.n_eval) {
            throw std::runtime_error("Eval y has different number of rows than Eval X.");
        }

        const int budget = (budget_override >= 0) ? budget_override : result->budget;

        Packed root_mask = eval_root_mask_(ctx.n_words, ctx.tail_mask);

        std::vector<Packed> Y_eval_bits = build_eval_y_bits_(
            y_eval,
            num_classes,
            ctx.n_words,
            ctx.tail_mask
        );

        Packed BBwrong_eval_storage((size_t)ctx.n_words);
        const Packed* BBwrong_eval_ptr = nullptr;

        if (use_deferral) {
            if (bb_pred_eval.empty()) {
                throw std::runtime_error(
                    "get_all_misclassifications_objs_packed_trie: "
                    "deferral was enabled during fit, so bb_pred_eval is required."
                );
            }

            BBwrong_eval_storage = build_eval_bb_wrong_bits_(
                y_eval,
                bb_pred_eval,
                num_classes,
                ctx.n_words,
                ctx.tail_mask
            );

            BBwrong_eval_ptr = &BBwrong_eval_storage;
        }

        auto buckets = collect_mistakes_by_obj_(
            result.get(),
            budget,
            root_mask,
            ctx,
            Y_eval_bits,
            BBwrong_eval_ptr
        );

        std::vector<int> out;

        size_t total = 0;
        for (const auto& b : buckets) {
            total += b.mistakes.size();
        }
        out.reserve(total);

        for (auto& b : buckets) {
            for (int m : b.mistakes) {
                out.push_back(m);
            }
        }

        return out;
    }
};