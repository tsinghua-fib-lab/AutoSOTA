#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "sptp_exp_opt.hpp"
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


using namespace torch::autograd;

class FlashTP_Large : public Function<FlashTP_Large> {
public:
    static torch::Tensor forward(
        AutogradContext *ctx, 
        torch::Tensor in1, 
        torch::Tensor in2,
        torch::Tensor weight,

        torch::Tensor per_edge_src,
        torch::Tensor per_edge_dst,

        torch::Tensor t_in1_idxing,
        torch::Tensor t_in1_ival,
        torch::Tensor t_in1_related_path_idx,

        torch::Tensor t_path_array1,
        torch::Tensor t_path_array2,
        torch::Tensor t_per_upath_fiber_start,
        torch::Tensor t_path_weight,
        torch::Tensor t_per_path_weight_pos,

        torch::Tensor t_per_upath_fiber_array,
        torch::Tensor t_unique_cg_val,
        torch::Tensor t_per_exec_info,

        int64_t upath_cnt,
        std::vector<int64_t> per_block_batch,
        int64_t max_ir_dim,
        int64_t out_size
    ) {
        int64_t node_cnt = in1.size(0);

        auto options = torch::TensorOptions().dtype(in1.dtype()).device(in1.device());
        torch::Tensor out = torch::zeros({node_cnt, out_size}, options);

        sptp_linear_fwd_v2_shared_exp(in1,in2,weight, out, per_edge_src, per_edge_dst, t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, t_per_exec_info, upath_cnt, per_block_batch[0], max_ir_dim * 2 + 1);

        ctx->save_for_backward({in1,in2,weight, per_edge_src, per_edge_dst,
        t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, 
        t_per_upath_fiber_array,t_unique_cg_val, t_per_exec_info});

        ctx->saved_data["upath_cnt"] = upath_cnt;
        ctx->saved_data["per_block_batch"] = per_block_batch[1];
        ctx->saved_data["max_ir_dim"] = max_ir_dim;

        return out;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_output) {
        auto saved = ctx->get_saved_variables();
        auto in1 = saved[0];
        auto in2 = saved[1];
        auto weight = saved[2];

        auto per_edge_src = saved[3];
        auto per_edge_dst = saved[4];

        auto t_in1_idxing = saved[5];
        auto t_in1_ival = saved[6];
        auto t_in1_related_path_idx = saved[7];

        auto t_path_array1 = saved[8];
        auto t_path_array2 = saved[9];
        auto t_per_upath_fiber_start = saved[10];
        auto t_path_weight = saved[11];
        auto t_per_path_weight_pos = saved[12];

        auto t_per_upath_fiber_array = saved[13];
        auto t_unique_cg_val = saved[14];
        auto t_per_exec_info = saved[15];

        auto upath_cnt = ctx->saved_data["upath_cnt"].toInt();
        auto per_block_batch = ctx->saved_data["per_block_batch"].toInt();
        auto max_ir_dim = ctx->saved_data["max_ir_dim"].toInt();

        auto node_cnt = in1.size(0);
        auto batch_size = in2.size(0);
        auto in1_size = in1.size(1);
        auto in2_size = in2.size(1);

        auto options = torch::TensorOptions().dtype(in1.dtype()).device(in1.device());
        auto mem_debug = torch::empty({1, 1}, options);
        auto mem_dl_din1 = torch::zeros({node_cnt, in1_size}, options);
        auto mem_dl_din2 = torch::empty({batch_size, in2_size * upath_cnt}, options);
        auto mem_dl_dw = torch::empty_like(weight);

        sptp_linear_bwd_v1_shared_exp(
            in1,in2,weight, grad_output[0], per_edge_src, per_edge_dst, mem_dl_din1, mem_dl_din2, mem_dl_dw, mem_debug,
            t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val,t_per_exec_info, upath_cnt, per_block_batch, max_ir_dim*2+1
        );
        auto mem_dl_din2_summed = mem_dl_din2.reshape({batch_size, upath_cnt, in2_size}).sum(1);
        torch::Tensor _n1;
        torch::Tensor _n2;
        torch::Tensor _n3;
        torch::Tensor _n4;
        torch::Tensor _n5;
        torch::Tensor _n6;
        torch::Tensor _n7;
        torch::Tensor _n8;
        torch::Tensor _n9;
        torch::Tensor _n10;
        torch::Tensor _n11;
        torch::Tensor _n12;
        torch::Tensor _n13;
        torch::Tensor _n14;
        torch::Tensor _n15;
        torch::Tensor _n16;
        torch::Tensor _n17;
        return {
            mem_dl_din1, 
            mem_dl_din2_summed, 
            mem_dl_dw,
            _n1, _n2, 
            _n3, _n4, _n5, 
            _n6, _n7, _n8, _n9, _n10, 
            _n11, _n12, _n13, 
            _n14, _n15, _n16, _n17,
        };
    }
};

torch::Tensor flashtp_large(
        torch::Tensor in1, 
        torch::Tensor in2,
        torch::Tensor weight,

        torch::Tensor per_edge_src,
        torch::Tensor per_edge_dst,

        torch::Tensor t_in1_idxing,
        torch::Tensor t_in1_ival,
        torch::Tensor t_in1_related_path_idx,

        torch::Tensor t_path_array1,
        torch::Tensor t_path_array2,
        torch::Tensor t_per_upath_fiber_start,
        torch::Tensor t_path_weight,
        torch::Tensor t_per_path_weight_pos,

        torch::Tensor t_per_upath_fiber_array,
        torch::Tensor t_unique_cg_val,
        torch::Tensor t_per_exec_info,

        int64_t upath_cnt,
        std::vector<int64_t> per_block_batch,
        int64_t max_ir_dim,
        int64_t out_size) {
    return FlashTP_Large::apply(in1,in2,weight, per_edge_src, per_edge_dst, t_in1_idxing, t_in1_ival, t_in1_related_path_idx, t_path_array1,t_path_array2,t_per_upath_fiber_start, t_path_weight, t_per_path_weight_pos, t_per_upath_fiber_array,t_unique_cg_val, t_per_exec_info, upath_cnt, per_block_batch, max_ir_dim, out_size);
}

PYBIND11_MODULE(flashtp_large_kernel_lammps, m) {
    m.doc() = "FlashTP large-kernel LAMMPS custom ops";
}
static auto registry = torch::RegisterOperators()
    .op("flashtp_large_kernel_lammps::sptp_linear_fwd_v2_shared_exp", &flashtp_large);
