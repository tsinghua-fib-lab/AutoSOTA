#include "suf/masked_compile.hpp"

#include "suf/validate.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>

namespace suf {

namespace {
struct PredExpansion {
  int a = -1;
  int b = -1;
  int w = -1;
  bool invert = false;
};

struct BoolBuilder {
  std::vector<BoolNode> nodes;

  int add_pred(int pred_index) {
    BoolNode n;
    n.kind = BoolNode::Kind::PRED;
    n.pred_index = pred_index;
    nodes.push_back(n);
    return static_cast<int>(nodes.size() - 1);
  }

  int add_not(int idx) {
    BoolNode n;
    n.kind = BoolNode::Kind::NOT;
    n.lhs = idx;
    nodes.push_back(n);
    return static_cast<int>(nodes.size() - 1);
  }

  int add_bin(BoolNode::Kind kind, int lhs, int rhs) {
    BoolNode n;
    n.kind = kind;
    n.lhs = lhs;
    n.rhs = rhs;
    nodes.push_back(n);
    return static_cast<int>(nodes.size() - 1);
  }
};

static inline u64 mask_bits(int bits) {
  if (bits >= 64) return ~0ULL;
  return (1ULL << bits) - 1ULL;
}

static inline u8 add_carry(u64 a, u64 b, int bits) {
  if (bits >= 64) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
    return static_cast<u8>((sum >> 64) & 1u);
  }
  unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
  return static_cast<u8>((sum >> bits) & 1u);
}

static inline u64 add_mod(u64 a, u64 b, int bits) {
  if (bits >= 64) return a + b;
  const u64 m = mask_bits(bits);
  return (a + b) & m;
}

static u64 sub_mod(u64 a, u64 b, int bits) {
  if (bits >= 64) return a - b;
  const u64 m = mask_bits(bits);
  return (a + (m + 1ULL) - b) & m;
}

static std::size_t interval_index_for(const std::vector<u64>& cuts, u64 x) {
  auto it = std::upper_bound(cuts.begin(), cuts.end(), x);
  if (it == cuts.begin()) return 0;
  return static_cast<std::size_t>(std::distance(cuts.begin(), it) - 1);
}

static void build_masked_partition(const SUFDescriptor& base,
                                   int in_bits,
                                   u64 r_in,
                                   std::vector<u64>& out_cuts,
                                   std::vector<Polynomial>& out_polys) {
  ensure(!base.cuts.empty(), "masked compile: empty cuts");
  const std::size_t m = base.cuts.size();

  std::vector<u64> boundaries;
  boundaries.reserve(m + 1);
  for (std::size_t i = 0; i < m; ++i) {
    boundaries.push_back(add_mod(base.cuts[i], r_in, in_bits));
  }
  std::sort(boundaries.begin(), boundaries.end());

  const bool has_zero = std::binary_search(boundaries.begin(), boundaries.end(), 0ULL);
  if (!has_zero) {
    boundaries.push_back(0ULL);
    std::sort(boundaries.begin(), boundaries.end());
  }

  if (boundaries.size() == m) {
    bool inserted = false;
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
      const u64 start = boundaries[i];
      const u64 end = (i + 1 < boundaries.size()) ? boundaries[i + 1]
                                                  : (in_bits >= 64 ? 0ULL : (1ULL << in_bits));
      unsigned __int128 length = 0;
      if (i + 1 < boundaries.size()) {
        length = static_cast<unsigned __int128>(end) - static_cast<unsigned __int128>(start);
      } else if (in_bits < 64) {
        length = static_cast<unsigned __int128>(1ULL << in_bits) - static_cast<unsigned __int128>(start);
      } else {
        length = (static_cast<unsigned __int128>(1) << 64) - static_cast<unsigned __int128>(start);
      }
      if (length > 1) {
        unsigned __int128 delta = static_cast<unsigned __int128>(start) + (length / 2);
        u64 split = (in_bits >= 64) ? static_cast<u64>(delta)
                                    : static_cast<u64>(delta & mask_bits(in_bits));
        boundaries.push_back(split);
        std::sort(boundaries.begin(), boundaries.end());
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      const bool full_domain = (in_bits < 64) && ((1ULL << in_bits) == boundaries.size());
      ensure(full_domain, "masked compile: cannot insert dummy split (all intervals length 1)");
    }
  }

  out_cuts = boundaries;
  out_polys.resize(out_cuts.size());
  for (std::size_t i = 0; i < out_cuts.size(); ++i) {
    const u64 rep = out_cuts[i];
    const u64 x = sub_mod(rep, r_in, in_bits);
    const std::size_t idx = interval_index_for(base.cuts, x);
    out_polys[i] = base.polys[idx];
  }
}

} // namespace

MaskedGateInstance compile_masked_gate_instance(const SUFDescriptor& base,
                                                int in_bits,
                                                u64 r_in,
                                                int party,
                                                std::mt19937_64& rng) {
  ensure(in_bits > 0 && in_bits <= 64, "masked compile: in_bits 1..64");
  validate_suf(base);

  MaskedGateInstance inst;
  inst.in_bits = in_bits;
  const u64 r = (in_bits >= 64) ? r_in : (r_in & mask_bits(in_bits));
  inst.r_in = r;

  build_masked_partition(base, in_bits, r, inst.desc.cuts, inst.desc.polys);

  auto add_const_bit = [&](u8 bit) -> int {
    Predicate p;
    p.kind = PredKind::CONST;
    p.param = 0; // do not embed secret in public descriptor
    inst.desc.predicates.push_back(p);
    const u8 share = static_cast<u8>(rng() & 1ULL);
    const u8 out = (party == 0) ? share : static_cast<u8>(share ^ (bit & 1u));
    inst.const_pred_bits.push_back(out);
    return static_cast<int>(inst.desc.predicates.size() - 1);
  };

  auto add_cmp_atom = [&](PredKind kind, u64 threshold, u8 f, u64 input_add) -> int {
    Predicate p;
    p.kind = kind;
    p.param = threshold;
    p.gamma = threshold;
    p.f = f;
    p.input_add = input_add;
    inst.desc.predicates.push_back(p);
    inst.const_pred_bits.push_back(0);
    return static_cast<int>(inst.desc.predicates.size() - 1);
  };

  std::vector<PredExpansion> expansions(base.predicates.size());
  for (std::size_t i = 0; i < base.predicates.size(); ++i) {
    const auto& p = base.predicates[i];
    PredExpansion exp;
    switch (p.kind) {
      case PredKind::LT: {
        const u64 theta = add_mod(r, p.param, in_bits);
        const u8 w = add_carry(r, p.param, in_bits);
        exp.a = add_cmp_atom(PredKind::LT, theta, 0, p.input_add);
        exp.b = add_cmp_atom(PredKind::LT, r, 0, p.input_add);
        exp.w = add_const_bit(w);
        break;
      }
      case PredKind::LTLOW: {
        const u8 f = p.f;
        const u64 rf = (f >= 64) ? r : (r & mask_bits(f));
        const u64 theta = add_mod(rf, p.gamma, f);
        const u8 w = add_carry(rf, p.gamma, f);
        exp.a = add_cmp_atom(PredKind::LTLOW, theta, f, p.input_add);
        exp.b = add_cmp_atom(PredKind::LTLOW, rf, f, p.input_add);
        exp.w = add_const_bit(w);
        break;
      }
      case PredKind::MSB: {
        const u64 beta = (in_bits == 64) ? (1ULL << 63) : (1ULL << (in_bits - 1));
        const u64 theta = add_mod(r, beta, in_bits);
        const u8 w = add_carry(r, beta, in_bits);
        exp.a = add_cmp_atom(PredKind::LT, theta, 0, 0);
        exp.b = add_cmp_atom(PredKind::LT, r, 0, 0);
        exp.w = add_const_bit(w);
        exp.invert = true;
        break;
      }
      case PredKind::MSB_ADD: {
        const u64 beta = (in_bits == 64) ? (1ULL << 63) : (1ULL << (in_bits - 1));
        const u64 theta = add_mod(r, beta, in_bits);
        const u8 w = add_carry(r, beta, in_bits);
        exp.a = add_cmp_atom(PredKind::LT, theta, 0, p.param);
        exp.b = add_cmp_atom(PredKind::LT, r, 0, p.param);
        exp.w = add_const_bit(w);
        exp.invert = true;
        break;
      }
      case PredKind::CONST: {
        const int idx = add_const_bit(static_cast<u8>(p.param & 1ULL));
        exp.a = idx;
        exp.b = idx;
        exp.w = idx;
        break;
      }
      default:
        fail("masked compile: unsupported predicate kind");
    }
    expansions[i] = exp;
  }

  inst.desc.helpers.clear();
  inst.desc.helpers.reserve(base.helpers.size());
  for (const auto& h : base.helpers) {
    BoolBuilder b;
    std::unordered_map<int, int> cache;
    std::function<int(int)> rewrite = [&](int idx) -> int {
      const auto& node = h.nodes[static_cast<std::size_t>(idx)];
      switch (node.kind) {
        case BoolNode::Kind::PRED: {
          auto it = cache.find(node.pred_index);
          if (it != cache.end()) return it->second;
          const auto& exp = expansions[static_cast<std::size_t>(node.pred_index)];
          int a = b.add_pred(exp.a);
          int bidx = b.add_pred(exp.b);
          int w = b.add_pred(exp.w);
          int t = b.add_bin(BoolNode::Kind::XOR, a, bidx);
          int root = b.add_bin(BoolNode::Kind::XOR, t, w);
          if (exp.invert) {
            root = b.add_not(root);
          }
          cache[node.pred_index] = root;
          return root;
        }
        case BoolNode::Kind::NOT: {
          int lhs = rewrite(node.lhs);
          return b.add_not(lhs);
        }
        case BoolNode::Kind::AND:
        case BoolNode::Kind::OR:
        case BoolNode::Kind::XOR: {
          int lhs = rewrite(node.lhs);
          int rhs = rewrite(node.rhs);
          return b.add_bin(node.kind, lhs, rhs);
        }
        default:
          return -1;
      }
    };
    BoolExpr out;
    out.root = rewrite(h.root);
    out.nodes = std::move(b.nodes);
    inst.desc.helpers.push_back(std::move(out));
  }

  return inst;
}

} // namespace suf
