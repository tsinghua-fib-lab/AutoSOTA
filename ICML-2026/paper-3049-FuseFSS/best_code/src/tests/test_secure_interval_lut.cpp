#include "suf/secure_program.hpp"
#include "suf/ref_eval.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

using namespace suf;

static SUFDescriptor make_desc() {
  SUFDescriptor d;
  d.cuts = {0, 50, 100, 150};
  d.polys.resize(4);
  d.polys[0].coeffs = {1};
  d.polys[1].coeffs = {2};
  d.polys[2].coeffs = {3};
  d.polys[3].coeffs = {4};
  return d;
}

static SUFDescriptor make_masked_poly_desc() {
  SUFDescriptor d;
  d.cuts = {0, 64, 128};
  d.polys.resize(3);
  d.polys[0].coeffs = {1, 2};
  d.polys[1].coeffs = {10, 3};
  d.polys[2].coeffs = {20, 4};
  return d;
}

static OperatorSpecification make_multi_output_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0, 64, 128};
  spec.pieces.resize(3);
  spec.pieces[0].polys = {Polynomial{{1, 2}}, Polynomial{{7}}};
  spec.pieces[1].polys = {Polynomial{{10, 3}}, Polynomial{{20, 4}}};
  spec.pieces[2].polys = {Polynomial{{20, 4}}, Polynomial{{30, 5}}};
  return spec;
}

static BoolExpr pred_expr_for_test(int pred_index) {
  BoolExpr e;
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, pred_index});
  e.root = 0;
  return e;
}

static OperatorSpecification make_relu_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0, 128};

  Predicate msb;
  msb.kind = PredKind::MSB;
  spec.predicates.push_back(msb);

  spec.pieces.resize(2);
  spec.pieces[0].polys = {Polynomial{{1}}, Polynomial{{3}}};
  spec.pieces[0].aux_words = {11, 12};
  spec.pieces[0].bool_outputs = {pred_expr_for_test(0)};
  spec.pieces[1].polys = {Polynomial{{0}}, Polynomial{{5}}};
  spec.pieces[1].aux_words = {21, 22};
  spec.pieces[1].bool_outputs = {pred_expr_for_test(0)};

  PostprocessArithExpr y;
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::MUL, 0, 1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::ADD, 2, 3, -1, 0});
  y.root = 4;
  spec.postprocess.arith_exprs.push_back(y);
  spec.postprocess.arithmetic_outputs = {0};

  PostprocessBoolExpr z;
  z.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::BOOL_OUT, -1, -1, 0, 0});
  z.root = 0;
  spec.postprocess.bool_exprs.push_back(z);
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

static OperatorSpecification make_b2a_a2b_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0};
  spec.pieces.resize(1);
  spec.pieces[0].polys = {Polynomial{{7}}};

  PostprocessArithExpr x_expr;
  x_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  x_expr.root = 0;
  spec.postprocess.arith_exprs.push_back(x_expr);

  PostprocessBoolExpr low_bit;
  low_bit.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::A2B, -1, -1, 0, 0});
  low_bit.root = 0;
  spec.postprocess.bool_exprs.push_back(low_bit);

  PostprocessArithExpr y;
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::B2A, -1, -1, 0, 0});
  y.root = 0;
  spec.postprocess.arith_exprs.push_back(y);

  spec.postprocess.arithmetic_outputs = {1};
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

static OperatorSpecification make_kappa_a2b_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0};
  spec.pieces.resize(1);
  spec.pieces[0].polys = {Polynomial{{1}}};

  PostprocessArithExpr x_expr;
  x_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  x_expr.root = 0;
  spec.postprocess.arith_exprs.push_back(x_expr);

  PostprocessArithExpr y_expr;
  y_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
  y_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::KAPPA_A, -1, -1, 0, 0});
  y_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::ADD, 0, 1, -1, 0});
  y_expr.root = 2;
  spec.postprocess.arith_exprs.push_back(y_expr);

  PostprocessBoolExpr z_expr;
  z_expr.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::KAPPA_B, -1, -1, 0, 0});
  PostprocessBoolNode a2b3;
  a2b3.op = PostprocessBoolOp::A2B;
  a2b3.index = 0;
  a2b3.bit_index = 3;
  z_expr.nodes.push_back(a2b3);
  z_expr.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::XOR, 0, 1, -1, 0});
  z_expr.root = 2;
  spec.postprocess.bool_exprs.push_back(z_expr);

  spec.postprocess.arithmetic_outputs = {1};
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

static SUFDescriptor make_and_helper_desc() {
  SUFDescriptor d;
  d.cuts = {0, 128};
  d.polys.resize(2);
  d.polys[0].coeffs = {0};
  d.polys[1].coeffs = {1};

  Predicate p0;
  p0.kind = PredKind::LT;
  p0.param = 128;
  d.predicates.push_back(p0);

  Predicate p1;
  p1.kind = PredKind::LTLOW;
  p1.f = 4;
  p1.gamma = 8;
  d.predicates.push_back(p1);

  BoolExpr e;
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, 0});
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, 1});
  e.nodes.push_back(BoolNode{BoolNode::Kind::AND, 0, 1, -1});
  e.root = 2;
  d.helpers.push_back(e);
  return d;
}

int main() {
  {
    const std::size_t N = 256;
    std::vector<u64> h_in(N);
    for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<u64>(i);

    u64* d_in = nullptr;
    u64* d_out0 = nullptr;
    u64* d_out1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_out0, N * sizeof(u64));
    cudaMalloc(&d_out1, N * sizeof(u64));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    auto desc = make_desc();
    GpuSecureSufProgram prog0(desc, 0, 42);
    GpuSecureSufProgram prog1(desc, 1, 42);

    prog0.eval(d_in, N, d_out0, nullptr, 0);
    prog1.eval(d_in, N, d_out1, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_out0(N), h_out1(N);
    cudaMemcpy(h_out0.data(), d_out0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out1.data(), d_out1, N * sizeof(u64), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < N; ++i) {
      auto ref = eval_suf_ref(desc, h_in[i]);
      u64 got = h_out0[i] + h_out1[i];
      if (got != ref.arith) {
        std::cerr << "mismatch at " << i << " got=" << got << " exp=" << ref.arith << "\n";
        return 1;
      }
    }

    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
  }

  {
    const std::size_t N = 256;
    const u64 mask = 37;
    std::vector<u64> h_plain(N);
    std::vector<u64> h_masked(N);
    for (std::size_t i = 0; i < N; ++i) {
      h_plain[i] = static_cast<u64>(i);
      h_masked[i] = (h_plain[i] + mask) & 0xFFULL;
    }

    u64* d_in = nullptr;
    u64* d_out0 = nullptr;
    u64* d_out1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_out0, N * sizeof(u64));
    cudaMalloc(&d_out1, N * sizeof(u64));
    cudaMemcpy(d_in, h_masked.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    auto desc = make_masked_poly_desc();
    GpuSecureSufProgram prog0(desc, 0, 123, 8, true, mask);
    GpuSecureSufProgram prog1(desc, 1, 123, 8, true, mask);

    prog0.eval(d_in, N, d_out0, nullptr, 0);
    prog1.eval(d_in, N, d_out1, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_out0(N), h_out1(N);
    cudaMemcpy(h_out0.data(), d_out0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out1.data(), d_out1, N * sizeof(u64), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < N; ++i) {
      auto ref = eval_suf_ref(desc, h_plain[i]);
      u64 got = h_out0[i] + h_out1[i];
      if (got != ref.arith) {
        std::cerr << "masked poly mismatch at " << i << " got=" << got
                  << " exp=" << ref.arith << " x_hat=" << h_masked[i] << "\n";
        return 1;
      }
    }

    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
  }

  {
    const std::size_t N = 256;
    constexpr int kOutputs = 2;
    const auto spec = make_multi_output_spec();
    std::vector<u64> h_in(N);
    for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<u64>(i);

    u64* d_in = nullptr;
    u64* d_out0 = nullptr;
    u64* d_out1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_out0, kOutputs * N * sizeof(u64));
    cudaMalloc(&d_out1, kOutputs * N * sizeof(u64));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog0(spec, 0, 777);
    GpuSecureSufProgram prog1(spec, 1, 777);
    if (prog0.num_arithmetic_outputs() != kOutputs || prog1.num_arithmetic_outputs() != kOutputs) {
      std::cerr << "unexpected output count\n";
      return 1;
    }

    prog0.eval(d_in, N, d_out0, nullptr, 0);
    prog1.eval(d_in, N, d_out1, nullptr, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_out0(kOutputs * N), h_out1(kOutputs * N);
    cudaMemcpy(h_out0.data(), d_out0, h_out0.size() * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out1.data(), d_out1, h_out1.size() * sizeof(u64), cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < N; ++i) {
      const u64 x = h_in[i];
      const auto piece = interval_index(spec.boundaries, x);
      for (int out = 0; out < kOutputs; ++out) {
        const u64 got = h_out0[static_cast<std::size_t>(out) * N + i] +
                        h_out1[static_cast<std::size_t>(out) * N + i];
        const u64 exp = eval_poly(spec.pieces[piece].polys[static_cast<std::size_t>(out)], x);
        if (got != exp) {
          std::cerr << "multi-output mismatch at " << i << " out=" << out
                    << " got=" << got << " exp=" << exp << "\n";
          return 1;
        }
      }
    }

    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
  }

  {
    constexpr int kOutputs = 2;
    const auto spec = make_multi_output_spec();
    u64* d_in = nullptr;
    u64* d_out0 = nullptr;
    u64* d_out1 = nullptr;
    cudaMalloc(&d_in, sizeof(u64));
    cudaMalloc(&d_out0, kOutputs * sizeof(u64));
    cudaMalloc(&d_out1, kOutputs * sizeof(u64));

    for (std::size_t i = 0; i < 32; ++i) {
      const u64 x = static_cast<u64>((i * 9 + 5) & 0xFFULL);
      const u64 mask = static_cast<u64>((37 + i * 17) & 0xFFULL);
      const u64 x_masked = (x + mask) & 0xFFULL;
      cudaMemcpy(d_in, &x_masked, sizeof(u64), cudaMemcpyHostToDevice);

      GpuSecureSufProgram prog0(spec, 0, 9000 + i, true, mask);
      GpuSecureSufProgram prog1(spec, 1, 9000 + i, true, mask);
      prog0.eval(d_in, 1, d_out0, nullptr, 0);
      prog1.eval(d_in, 1, d_out1, nullptr, 0);
      cudaDeviceSynchronize();

      u64 h_out0[kOutputs] = {};
      u64 h_out1[kOutputs] = {};
      cudaMemcpy(h_out0, d_out0, kOutputs * sizeof(u64), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_out1, d_out1, kOutputs * sizeof(u64), cudaMemcpyDeviceToHost);
      const auto piece = interval_index(spec.boundaries, x);
      for (int out = 0; out < kOutputs; ++out) {
        const u64 got = h_out0[out] + h_out1[out];
        const u64 exp = eval_poly(spec.pieces[piece].polys[static_cast<std::size_t>(out)], x);
        if (got != exp) {
          std::cerr << "fresh-mask multi-output mismatch at i=" << i
                    << " x=" << x << " mask=" << mask << " out=" << out
                    << " got=" << got << " exp=" << exp << "\n";
          return 1;
        }
      }
    }

    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
  }

  {
    const std::size_t N = 256;
    const u64 mask = 37;
    const auto spec = make_relu_phi_spec();
    ReferenceMpcRuntime runtime;
    SecureEvalOptions opts;
    opts.mask_aware = true;
    opts.mask_in = mask;
    opts.mode = SecureEvalMode::PaperStrictSharedX;
    opts.runtime = &runtime;

    std::vector<u64> h_masked(N);
    for (std::size_t i = 0; i < N; ++i) {
      h_masked[i] = (static_cast<u64>(i) + mask) & 0xFFULL;
    }

    u64* d_in = nullptr;
    u64* d_y0 = nullptr;
    u64* d_y1 = nullptr;
    u64* d_z0 = nullptr;
    u64* d_z1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_y0, N * sizeof(u64));
    cudaMalloc(&d_y1, N * sizeof(u64));
    cudaMalloc(&d_z0, N * sizeof(u64));
    cudaMalloc(&d_z1, N * sizeof(u64));
    cudaMemcpy(d_in, h_masked.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog0(spec, 0, 4242, opts);
    GpuSecureSufProgram prog1(spec, 1, 4242, opts);
    if (prog0.num_arithmetic_outputs() != 1 || prog0.num_helpers() != 1) {
      std::cerr << "unexpected ReLU Phi output shape\n";
      return 1;
    }

    prog0.eval(d_in, N, d_y0, d_z0, 0);
    prog1.eval(d_in, N, d_y1, d_z1, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_y0(N), h_y1(N), h_z0(N), h_z1(N);
    cudaMemcpy(h_y0.data(), d_y0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y1.data(), d_y1, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z0.data(), d_z0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z1.data(), d_z1, N * sizeof(u64), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < N; ++i) {
      const u64 x = static_cast<u64>(i);
      const u64 got_y = h_y0[i] + h_y1[i];
      const u64 exp_y = (x < 128) ? (x + 3ULL) : 5ULL;
      const u64 got_z = (h_z0[i] ^ h_z1[i]) & 1ULL;
      const u64 exp_z = (x >= 128) ? 1ULL : 0ULL;
      if (got_y != exp_y || got_z != exp_z) {
        std::cerr << "paper-strict ReLU Phi mismatch at " << i
                  << " got_y=" << got_y << " exp_y=" << exp_y
                  << " got_z=" << got_z << " exp_z=" << exp_z << "\n";
        return 1;
      }
    }

    bool fast_threw = false;
    try {
      GpuSecureSufProgram fast_prog(spec, 0, 4242, true, mask);
      fast_prog.eval(d_in, 1, d_y0, d_z0, 0);
    } catch (const std::runtime_error&) {
      fast_threw = true;
      cudaDeviceSynchronize();
    }
    if (!fast_threw) {
      std::cerr << "expected Phi fast-path fail-fast\n";
      return 1;
    }

    bool no_runtime_threw = false;
    try {
      SecureEvalOptions bad_opts = opts;
      bad_opts.runtime = nullptr;
      GpuSecureSufProgram bad_prog(spec, 0, 4242, bad_opts);
      bad_prog.eval(d_in, 1, d_y0, d_z0, 0);
    } catch (const std::runtime_error&) {
      no_runtime_threw = true;
      cudaDeviceSynchronize();
    }
    if (!no_runtime_threw) {
      std::cerr << "expected paper-strict no-runtime fail-fast\n";
      return 1;
    }

    cudaFree(d_in);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_z0);
    cudaFree(d_z1);
  }

  {
    const std::size_t N = 256;
    const auto spec = make_relu_phi_spec();
    ReferenceMpcRuntime runtime;
    std::vector<u64> h_masks(N);
    std::vector<u64> h_masked(N);
    for (std::size_t i = 0; i < N; ++i) {
      h_masks[i] = static_cast<u64>((17 + i * 73) & 0xFFULL);
      h_masked[i] = (static_cast<u64>(i) + h_masks[i]) & 0xFFULL;
    }

    SecureEvalOptions opts;
    opts.mask_aware = true;
    opts.mask_vector = &h_masks;
    opts.mode = SecureEvalMode::PaperStrictSharedX;
    opts.runtime = &runtime;

    u64* d_in = nullptr;
    u64* d_y0 = nullptr;
    u64* d_y1 = nullptr;
    u64* d_z0 = nullptr;
    u64* d_z1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_y0, N * sizeof(u64));
    cudaMalloc(&d_y1, N * sizeof(u64));
    cudaMalloc(&d_z0, N * sizeof(u64));
    cudaMalloc(&d_z1, N * sizeof(u64));
    cudaMemcpy(d_in, h_masked.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog0(spec, 0, 7777, opts);
    GpuSecureSufProgram prog1(spec, 1, 7777, opts);
    prog0.eval(d_in, N, d_y0, d_z0, 0);
    prog1.eval(d_in, N, d_y1, d_z1, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_y0(N), h_y1(N), h_z0(N), h_z1(N);
    cudaMemcpy(h_y0.data(), d_y0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y1.data(), d_y1, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z0.data(), d_z0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z1.data(), d_z1, N * sizeof(u64), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < N; ++i) {
      const u64 got_y = h_y0[i] + h_y1[i];
      const u64 exp_y = (i < 128) ? (static_cast<u64>(i) + 3ULL) : 5ULL;
      const u64 got_z = (h_z0[i] ^ h_z1[i]) & 1ULL;
      const u64 exp_z = (i >= 128) ? 1ULL : 0ULL;
      if (got_y != exp_y || got_z != exp_z) {
        std::cerr << "per-wire mask ReLU Phi mismatch at " << i
                  << " mask=" << h_masks[i]
                  << " got_y=" << got_y << " exp_y=" << exp_y
                  << " got_z=" << got_z << " exp_z=" << exp_z << "\n";
        return 1;
      }
    }

    bool fast_vector_threw = false;
    try {
      SecureEvalOptions bad_opts = opts;
      bad_opts.mode = SecureEvalMode::FastPublicX;
      GpuSecureSufProgram bad_prog(spec, 0, 7777, bad_opts);
      bad_prog.eval(d_in, 1, d_y0, d_z0, 0);
    } catch (const std::runtime_error&) {
      fast_vector_threw = true;
      cudaDeviceSynchronize();
    }
    if (!fast_vector_threw) {
      std::cerr << "expected per-wire vector mask fast-path fail-fast\n";
      return 1;
    }

    cudaFree(d_in);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_z0);
    cudaFree(d_z1);
  }

  {
    const std::size_t N = 32;
    const auto spec = make_b2a_a2b_phi_spec();
    const auto compiled = compile_operator_spec(spec);
    if (compiled.shape.b2a_conversions != 1 || compiled.shape.a2b_conversions != 1) {
      std::cerr << "unexpected B2A/A2B cost accounting\n";
      return 1;
    }

    ReferenceMpcRuntime runtime;
    std::vector<u64> h_masks(N);
    std::vector<u64> h_masked(N);
    for (std::size_t i = 0; i < N; ++i) {
      h_masks[i] = static_cast<u64>((5 + i * 29) & 0xFFULL);
      h_masked[i] = (static_cast<u64>(i) + h_masks[i]) & 0xFFULL;
    }

    SecureEvalOptions opts;
    opts.mask_aware = true;
    opts.mask_vector = &h_masks;
    opts.mode = SecureEvalMode::PaperStrictSharedX;
    opts.runtime = &runtime;

    u64* d_in = nullptr;
    u64* d_y0 = nullptr;
    u64* d_y1 = nullptr;
    u64* d_z0 = nullptr;
    u64* d_z1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_y0, N * sizeof(u64));
    cudaMalloc(&d_y1, N * sizeof(u64));
    cudaMalloc(&d_z0, N * sizeof(u64));
    cudaMalloc(&d_z1, N * sizeof(u64));
    cudaMemcpy(d_in, h_masked.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog0(spec, 0, 9191, opts);
    GpuSecureSufProgram prog1(spec, 1, 9191, opts);
    prog0.eval(d_in, N, d_y0, d_z0, 0);
    prog1.eval(d_in, N, d_y1, d_z1, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_y0(N), h_y1(N), h_z0(N), h_z1(N);
    cudaMemcpy(h_y0.data(), d_y0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y1.data(), d_y1, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z0.data(), d_z0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z1.data(), d_z1, N * sizeof(u64), cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < N; ++i) {
      const u64 exp = static_cast<u64>(i & 1ULL);
      const u64 got_y = h_y0[i] + h_y1[i];
      const u64 got_z = (h_z0[i] ^ h_z1[i]) & 1ULL;
      if (got_y != exp || got_z != exp) {
        std::cerr << "B2A/A2B Phi mismatch at " << i
                  << " got_y=" << got_y << " got_z=" << got_z
                  << " exp=" << exp << "\n";
        return 1;
      }
    }

    bool no_runtime_threw = false;
    try {
      SecureEvalOptions bad_opts = opts;
      bad_opts.runtime = nullptr;
      GpuSecureSufProgram bad_prog(spec, 0, 9191, bad_opts);
      bad_prog.eval(d_in, 1, d_y0, d_z0, 0);
    } catch (const std::runtime_error&) {
      no_runtime_threw = true;
      cudaDeviceSynchronize();
    }
    if (!no_runtime_threw) {
      std::cerr << "expected B2A/A2B no-runtime fail-fast\n";
      return 1;
    }

    cudaFree(d_in);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_z0);
    cudaFree(d_z1);
  }

  {
    const std::size_t N = 64;
    const auto spec = make_kappa_a2b_phi_spec();
    const auto compiled = compile_operator_spec(spec);
    if (compiled.shape.kappa_a_words != 1 || compiled.shape.kappa_b_bits != 1 ||
        compiled.shape.a2b_conversions != 1) {
      std::cerr << "unexpected kappa/A2B shape accounting\n";
      return 1;
    }

    ReferenceMpcRuntime runtime;
    std::vector<u64> h_masks(N);
    std::vector<u64> h_masked(N);
    std::vector<u64> h_kappa_a(N);
    std::vector<u8> h_kappa_b(N);
    for (std::size_t i = 0; i < N; ++i) {
      h_masks[i] = static_cast<u64>((9 + i * 31) & 0xFFULL);
      h_masked[i] = (static_cast<u64>(i) + h_masks[i]) & 0xFFULL;
      h_kappa_a[i] = static_cast<u64>(100 + i);
      h_kappa_b[i] = static_cast<u8>((i >> 1) & 1U);
    }

    SecureEvalOptions opts;
    opts.mask_aware = true;
    opts.mask_vector = &h_masks;
    opts.mode = SecureEvalMode::PaperStrictSharedX;
    opts.runtime = &runtime;
    opts.kappa_a = &h_kappa_a;
    opts.kappa_a_count = 1;
    opts.kappa_b = &h_kappa_b;
    opts.kappa_b_count = 1;

    u64* d_in = nullptr;
    u64* d_y0 = nullptr;
    u64* d_y1 = nullptr;
    u64* d_z0 = nullptr;
    u64* d_z1 = nullptr;
    cudaMalloc(&d_in, N * sizeof(u64));
    cudaMalloc(&d_y0, N * sizeof(u64));
    cudaMalloc(&d_y1, N * sizeof(u64));
    cudaMalloc(&d_z0, N * sizeof(u64));
    cudaMalloc(&d_z1, N * sizeof(u64));
    cudaMemcpy(d_in, h_masked.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog0(spec, 0, 31337, opts);
    GpuSecureSufProgram prog1(spec, 1, 31337, opts);
    prog0.eval(d_in, N, d_y0, d_z0, 0);
    prog1.eval(d_in, N, d_y1, d_z1, 0);
    cudaDeviceSynchronize();

    std::vector<u64> h_y0(N), h_y1(N), h_z0(N), h_z1(N);
    cudaMemcpy(h_y0.data(), d_y0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y1.data(), d_y1, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z0.data(), d_z0, N * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z1.data(), d_z1, N * sizeof(u64), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < N; ++i) {
      const u64 got_y = h_y0[i] + h_y1[i];
      const u64 exp_y = 1ULL + h_kappa_a[i];
      const u64 got_z = (h_z0[i] ^ h_z1[i]) & 1ULL;
      const u64 exp_z = static_cast<u64>(h_kappa_b[i] ^ ((i >> 3) & 1ULL));
      if (got_y != exp_y || got_z != exp_z) {
        std::cerr << "kappa/A2B bit Phi mismatch at " << i
                  << " got_y=" << got_y << " exp_y=" << exp_y
                  << " got_z=" << got_z << " exp_z=" << exp_z << "\n";
        return 1;
      }
    }

    bool missing_kappa_threw = false;
    try {
      SecureEvalOptions bad_opts = opts;
      bad_opts.kappa_a = nullptr;
      GpuSecureSufProgram bad_prog(spec, 0, 31337, bad_opts);
      bad_prog.eval(d_in, 1, d_y0, d_z0, 0);
    } catch (const std::runtime_error&) {
      missing_kappa_threw = true;
      cudaDeviceSynchronize();
    }
    if (!missing_kappa_threw) {
      std::cerr << "expected missing kappa fail-fast\n";
      return 1;
    }

    cudaFree(d_in);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_z0);
    cudaFree(d_z1);
  }

  {
    u64 h_in = 3;
    u64* d_in = nullptr;
    u64* d_helper = nullptr;
    cudaMalloc(&d_in, sizeof(u64));
    cudaMalloc(&d_helper, sizeof(u64));
    cudaMemcpy(d_in, &h_in, sizeof(u64), cudaMemcpyHostToDevice);

    GpuSecureSufProgram prog(make_and_helper_desc(), 0, 123);
    bool threw = false;
    try {
      prog.eval(d_in, 1, nullptr, d_helper, 0);
    } catch (const std::runtime_error&) {
      threw = true;
      cudaDeviceSynchronize();
    }
    if (!threw) {
      std::cerr << "expected helper AND fail-fast\n";
      return 1;
    }

    cudaFree(d_in);
    cudaFree(d_helper);
  }

  std::cout << "ok\n";
  return 0;
}
