#include "suf/ref_eval.hpp"
#include "suf/validate.hpp"

#include <cassert>
#include <iostream>

using namespace suf;

static SUFDescriptor make_toy() {
  SUFDescriptor d;
  d.cuts = {0, 128};
  d.polys.resize(2);
  d.polys[0].coeffs = {0, 1}; // f(x)=x
  d.polys[1].coeffs = {1, 1}; // f(x)=x+1

  Predicate p;
  p.kind = PredKind::LT;
  p.param = 128;
  d.predicates.push_back(p);

  BoolExpr e;
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, 0});
  e.root = 0;
  d.helpers.push_back(e);
  return d;
}

int main() {
  auto d = make_toy();
  validate_suf(d);

  for (int x = 0; x < 256; ++x) {
    auto res = eval_suf_ref(d, static_cast<u64>(x));
    const u64 expected = (x < 128) ? static_cast<u64>(x) : static_cast<u64>(x + 1);
    if (res.arith != expected) {
      std::cerr << "arith mismatch at " << x << " got " << res.arith << " expected " << expected << "\n";
      return 1;
    }
    const u64 helper = res.helpers[0];
    const u64 helper_exp = (x < 128) ? 1 : 0;
    if (helper != helper_exp) {
      std::cerr << "helper mismatch at " << x << " got " << helper << " expected " << helper_exp << "\n";
      return 1;
    }
  }

  std::cout << "ok\n";
  return 0;
}
