import os
import itertools
from typing import List

import e3nn
import torch
from torch.utils.cpp_extension import load

import flashTP_e3nn.flashtp_extcg_kernel  as flashtp_extcg_kernel

# flashtp_extcg_kernel = None
import logging
logger = logging.getLogger(__name__)

def kernel_init():
    # global flashtp_extcg_kernel
    # if flashtp_extcg_kernel:
    #     return

    # #     os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # flashtp_extcg_kernel = load(
    #     name="flashtp_extcg_kernel",
    #     sources=[
    #         f"{script_dir}/sptp_exp_opt.cpp",
    #         f"{script_dir}/fwd_sptp_linear_v2_shared_exp.cu",
    #         f"{script_dir}/bwd_sptp_linear_shared_exp.cu",
    #         f"{script_dir}/bwd_bwd_sptp_linear_v2_shared_exp.cu",
    #         f"{script_dir}/fwd_sptp_linear_v2_shared_exp_double.cu",
    #         f"{script_dir}/bwd_sptp_linear_shared_exp_double.cu",
    #         f"{script_dir}/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",
    #     ],
    #     extra_cuda_cflags=["-lineinfo"],
    #     verbose=True,
    # )

    

    @torch.library.custom_op(
        "flashtp_extcg_kernel::sptp_linear_fwd_v2_shared_exp",
        mutates_args=(),
        device_types="cuda",
    )

    def sptp_linear_fwd_v2_shared_exp(
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        out_size: int,
        t_cg_idx_array: torch.Tensor,
    ) -> torch.Tensor:
        node_cnt = in1.shape[0]
        batch_size = in2.shape[0]
        out = torch.zeros((node_cnt, out_size), device=in1.device, dtype=in1.dtype)

        assert in2.dtype == in1.dtype
        assert weight.dtype == in1.dtype

        flashtp_extcg_kernel.sptp_linear_fwd_v2_shared_exp(
            in1,
            in2,
            weight,
            out,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
            per_block_batch[0],
            max_ir_dim * 2 + 1,
            t_cg_idx_array,
        )

        # out_reduced = scatter(out, per_edge_dst.to(torch.int64), dim=0, dim_size=node_cnt, reduce="sum")
        # del out

        return out


    @sptp_linear_fwd_v2_shared_exp.register_fake
    def _(
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        out_size: int,
        t_cg_idx_array: torch.Tensor,
    ) -> torch.Tensor:
        # node_cnt = in1.shape[0]
        out = torch.empty((len(in1), out_size), device=in1.device, dtype=in1.dtype)

        # out_reduced = scatter(out, per_edge_dst.to(torch.int64), dim=0, dim_size=node_cnt, reduce="sum")
        # del out

        return out


    def fused_e3nn_setup_fwd_context_exp(ctx, inputs, output):
        (
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
            per_block_batch,
            max_ir_dim,
            out_size,
            t_cg_idx_array,
        ) = inputs
        ctx.save_for_backward(
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            t_cg_idx_array,
        )
        ctx.max_fiber_size = max_fiber_size
        ctx.upath_cnt = upath_cnt
        ctx.per_block_batch = per_block_batch
        ctx.max_ir_dim = max_ir_dim


    def fused_e3nn_bwd_exp(ctx, grad_output):
        (
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            t_cg_idx_array,
        ) = ctx.saved_tensors

        grad_list = torch.ops.flashtp_extcg_kernel.sptp_linear_bwd_v2_shared_exp(
            in1,
            in2,
            weight,
            grad_output,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            ctx.max_fiber_size,
            ctx.upath_cnt,
            ctx.per_block_batch,
            ctx.max_ir_dim,
            t_cg_idx_array,
        )

        return (
            grad_list[0],  # in1_grad
            grad_list[1],  # in2_grad
            grad_list[2],  # weight_grad
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


    @torch.library.custom_op(
        "flashtp_extcg_kernel::sptp_linear_bwd_v2_shared_exp",
        mutates_args=(),
        device_types="cuda",
    )
    def sptp_linear_bwd_v2_shared_exp(
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        grad_output: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        t_cg_idx_array: torch.Tensor,
    ) -> List[torch.Tensor]:
        node_cnt = in1.shape[0]
        batch_size = in2.shape[0]
        in2_size = in2.shape[1]
        in1_size = in1.shape[1]

        mem_debug = torch.empty((1, 1), device=in1.device, dtype=in1.dtype)
        mem_dl_din1 = torch.zeros((node_cnt, in1_size), device=in1.device, dtype=in1.dtype)
        mem_dl_din2 = torch.empty(
            (batch_size, in2_size * upath_cnt), device=in1.device, dtype=in1.dtype
        )
        mem_dl_dw = torch.empty_like(weight)

        assert in2.dtype == in1.dtype
        assert weight.dtype == in1.dtype
        assert grad_output.dtype == in1.dtype

        flashtp_extcg_kernel.sptp_linear_bwd_v1_shared_exp(
            in1,
            in2,
            weight,
            grad_output.contiguous(),
            per_edge_src,
            per_edge_dst,
            mem_dl_din1,
            mem_dl_din2,
            mem_dl_dw,
            mem_debug,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
            per_block_batch[1],
            max_ir_dim * 2 + 1,
            t_cg_idx_array,
        )
        mem_dl_din2_summed = mem_dl_din2.reshape((batch_size, upath_cnt, in2_size)).sum(
            dim=1
        )

        del mem_dl_din2

        return [mem_dl_din1, mem_dl_din2_summed, mem_dl_dw]


    @sptp_linear_bwd_v2_shared_exp.register_fake
    def _(
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        grad_output: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        t_cg_idx_array: torch.Tensor,
    ) -> List[torch.Tensor]:
        dl_din1_reduced = torch.empty_like(in1)
        mem_dl_din2_summed = torch.empty_like(in2)
        mem_dl_dw = torch.empty_like(weight)

        return [dl_din1_reduced, mem_dl_din2_summed, mem_dl_dw]


    @torch.library.custom_op(
        "flashtp_extcg_kernel::sptp_linear_bwd_bwd_v2_shared_exp",
        mutates_args=(),
        device_types="cuda",
    )
    def sptp_linear_bwd_bwd_v2_shared_exp(
        dF_in1: torch.Tensor,
        dF_in2: torch.Tensor,
        dF_dw: torch.Tensor,
        dE_dout: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        t_cg_idx_array: torch.Tensor,
    ) -> List[torch.Tensor]:
        node_cnt = in1.shape[0]
        batch_size = in2.shape[0]
        in1_size = in1.shape[1]
        in2_size = in2.shape[1]
        out_size = dE_dout.shape[1]

        # torch.cuda.memory._dump_snapshot(f"/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/snap.pickle")

        dF_dout = torch.zeros((node_cnt, out_size), device=in1.device, dtype=in1.dtype)
        # dF_dout = torch.empty((batch_size, out_size) , device=in2.device, dtype=in2.dtype)
        dL_din1 = torch.zeros((node_cnt, in1_size), device=in1.device, dtype=in1.dtype)
        dL_din2_duplicate = torch.empty(
            (batch_size, in2_size * upath_cnt), device=in1.device, dtype=in1.dtype
        )
        dL_dw = torch.empty_like(weight)
        mem_debug = torch.empty((1, 1), device=in1.device, dtype=in1.dtype)

        assert dF_in1.dtype == in1.dtype
        assert dF_in2.dtype == in1.dtype
        assert dF_dw.dtype == in1.dtype
        assert dE_dout.dtype == in1.dtype
        assert in2.dtype == in1.dtype
        assert weight.dtype == in1.dtype

        flashtp_extcg_kernel.sptp_linear_bwd_bwd_v2_shared_exp(
            dF_in1,
            dF_in2,
            dF_dw,
            dE_dout,
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            dF_dout,
            dL_dw,
            dL_din1,
            dL_din2_duplicate,
            mem_debug,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
            per_block_batch[2],
            max_ir_dim * 2 + 1,
            t_cg_idx_array,
        )

        dL_din2 = dL_din2_duplicate.reshape((batch_size, upath_cnt, in2_size)).sum(dim=1)

        del dL_din2_duplicate
        # del dF_dout

        return [dL_din1, dL_din2, dL_dw, dF_dout]
        # return [dL_din1_reduced, dL_din2, dL_dw, dF_dout_reduced]


    @sptp_linear_bwd_bwd_v2_shared_exp.register_fake
    def _(
        dF_in1: torch.Tensor,
        dF_in2: torch.Tensor,
        dF_dw: torch.Tensor,
        dE_dout: torch.Tensor,
        in1: torch.Tensor,
        in2: torch.Tensor,
        weight: torch.Tensor,
        per_edge_src: torch.Tensor,
        per_edge_dst: torch.Tensor,
        t_in1_idxing: torch.Tensor,
        t_in1_ival: torch.Tensor,
        t_in1_related_path_idx: torch.Tensor,
        t_path_array1: torch.Tensor,
        t_path_array2: torch.Tensor,
        t_per_upath_fiber_start: torch.Tensor,
        t_path_weight: torch.Tensor,
        t_per_path_weight_pos: torch.Tensor,
        t_per_upath_fiber_array: torch.Tensor,
        t_unique_cg_val: torch.Tensor,
        t_per_exec_info: torch.Tensor,
        t_partial_fiber_start: torch.Tensor,
        t_partial_fiber_end: torch.Tensor,
        max_fiber_size: int,
        upath_cnt: int,
        per_block_batch: List[int],
        max_ir_dim: int,
        t_cg_idx_array: torch.Tensor,
    ) -> List[torch.Tensor]:
        dL_din1_reduced = torch.empty_like(in1)  # Same shape as in1
        dL_din2 = torch.empty_like(in2)  # Same shape as in2
        dL_dw = torch.empty_like(weight)  # Same shape as weight
        dF_dout = torch.empty_like(dE_dout)  # Same shape as dE_dout

        return [dL_din1_reduced, dL_din2, dL_dw, dF_dout]
        # return [dL_din1_reduced, dL_din2, dL_dw, dF_dout_reduced]


    def fused_e3nn_setup_bwd_context_exp(ctx, inputs, output):
        (
            in1,
            in2,
            weight,
            dE_dout,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
            per_block_batch,
            max_ir_dim,
            t_cg_idx_array,
        ) = inputs

        ctx.save_for_backward(
            dE_dout,
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            t_cg_idx_array,
        )
        ctx.max_fiber_size = max_fiber_size
        ctx.upath_cnt = upath_cnt
        ctx.per_block_batch = per_block_batch
        ctx.max_ir_dim = max_ir_dim


    @torch.compiler.allow_in_graph
    def fused_e3nn_bwd_bwd_exp(ctx, grad_output):
        (
            dE_dout,
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            t_cg_idx_array,
        ) = ctx.saved_tensors

        dF_in1 = grad_output[0]
        dF_in2 = grad_output[1]
        dF_w = grad_output[2]

        grad_list = torch.ops.flashtp_extcg_kernel.sptp_linear_bwd_bwd_v2_shared_exp(
            dF_in1,
            dF_in2,
            dF_w,
            dE_dout.detach(),
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            ctx.max_fiber_size,
            ctx.upath_cnt,
            ctx.per_block_batch,
            ctx.max_ir_dim,
            t_cg_idx_array,
        )

        return (
            grad_list[0],
            grad_list[1],
            grad_list[2],  # weight_grad
            grad_list[3],  # mem_dL_dO_grad
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


    torch.library.register_autograd(
        "flashtp_extcg_kernel::sptp_linear_fwd_v2_shared_exp",
        fused_e3nn_bwd_exp,
        setup_context=fused_e3nn_setup_fwd_context_exp,
    )

    torch.library.register_autograd(
        "flashtp_extcg_kernel::sptp_linear_bwd_v2_shared_exp",
        fused_e3nn_bwd_bwd_exp,
        setup_context=fused_e3nn_setup_bwd_context_exp,
    )


def find_optimal_yz(x, max_y=32):
    # Start from the maximum allowed threads per block and decrease
    best_y, best_z = max_y, x // max_y
    for y in range(max_y, 0, -1):
        z = x // y
        if y * z > best_y * best_z and y * z < x:
            best_y, best_z = y, z
    return best_y, best_z


import math


def find_optimal_abc(a, b, c, max_y=32):
    # Start from the maximum allowed threads per block and decrease
    approx_y = math.ceil(a / b)
    max_shmem_size = []
    max_x_size = []

    for y in range(1, approx_y + 1):
        per_thread_shmem = (a - y * c) / y
        x = math.floor(per_thread_shmem / b)
        x = min(max_y, x)
        if x <= max_y and x >= 1:
            max_shmem_size.append(x * y * b + y * c)
            max_x_size.append(x)

    max_x = 0
    max_shmem = 0
    for x, shmem in zip(max_x_size, max_shmem_size):
        # if difference is larger than 10KB
        # if difference is small keep larger x
        if abs(max_shmem - shmem) > 30 * 1024 and shmem > max_shmem:
            max_shmem = shmem
            max_x = x
    return max_x


class fused_uvu_TP_exp_opt_extcg(torch.nn.Module):
    def __init__(
        self,
        i_in1,
        i_in2,
        i_out,
        instructions,
        unique_cg_mat,
        unique_cg_val,
        warpsize=32,
        per_block_batch=None,
        smem_size=None,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        kernel_init()

        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.WARPSIZE = warpsize
        self.device = device
        self.dtype = dtype

        self.per_block_batch = per_block_batch
        self.l_max = max(
            [
                max([x[1].l for x in i_in1]),
                max([x[1].l for x in i_in2]),
                max([x[1].l for x in i_out]),
            ]
        )

        
        self.unique_cg_val = unique_cg_val

        tp_inst_outorder = sorted(instructions, key=lambda x: x.i_out)

        per_path_fiber_start, per_path_fiber_array, per_in1_ir_pathinfo = (
            self.cgmat2fiber(tp_inst_outorder, unique_cg_mat)
        )

        self.metadata_list = self.metadata_gen(
            instructions,
            per_path_fiber_start,
            per_path_fiber_array,
            per_in1_ir_pathinfo,
        )
        self.out_dim = self.i_out.dim

        d_size = 4
        if self.dtype == torch.float64:
            d_size = 8
        max_ir_dim = self.l_max * 2 + 1
        in2_size = i_in2.dim
        WARPSIZE = 32
        perwarp_in2_size = in2_size
        if in2_size % 2 == 0:
            perwarp_in2_size = in2_size + 1

        if(smem_size is None):
            gpu_name = torch.cuda.get_device_name()
            if "A100" in gpu_name:
                SMEM_SIZE = 163
            elif "H100" in gpu_name:
                SMEM_SIZE = 227
            else:
                logger.warning("Need to tune SMEM_SIZE, using safe mode of SMEM_SIZE=48KB")
                SMEM_SIZE = 48
        else:
            SMEM_SIZE = smem_size
        logger.info(f"SMEM_SIZE set to {SMEM_SIZE} KB")
        # print("max_fiber_size", self.metadata_list[-2])
        max_fiber_size = self.metadata_list[-2]

        if (self.per_block_batch == None):

            fwd_per_block_opt_batch = find_optimal_abc(
                SMEM_SIZE * 1024,
                d_size * (WARPSIZE * (max_ir_dim * 3) + perwarp_in2_size),
                max_fiber_size * 6,
            )

            bwd_per_block_opt_batch = find_optimal_abc(
                SMEM_SIZE * 1024,
                d_size * WARPSIZE * (max_ir_dim * 5 + perwarp_in2_size) + in2_size * d_size,
                max_fiber_size * 6,
            )

            bwd_bwd_per_block_opt_batch = find_optimal_abc(
                SMEM_SIZE * 1024,
                d_size * WARPSIZE * (max_ir_dim * 7 + perwarp_in2_size)
                + in2_size * d_size * 2,
                max_fiber_size * 6,
            )
            self.per_block_batch = [fwd_per_block_opt_batch,
                                    bwd_per_block_opt_batch,
                                    bwd_bwd_per_block_opt_batch]



        logger.info(f"Using per_block_batch of {self.per_block_batch}")
        # exit()

    def forward(self, in1, in2, weight, per_edge_src, per_edge_dst):
        # out = torch.empty((batch_size, out_size), device=in1.device, dtype=in1.dtype)
        # sptp_linear.sptp_linear_fwd_v2_shared(
        #     in1,in2,weight, out,
        #     *self.metadata_list, 1, self.l_max*2+1
        # )
        out = torch.ops.flashtp_extcg_kernel.sptp_linear_fwd_v2_shared_exp(
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            *self.metadata_list,
            self.per_block_batch,
            self.l_max,
            # self.i_out.dim
            self.out_dim,
            self.t_cg_idx_array,
        )
        return out


    def cgmat2fiber(self, tp_inst_outorder, unique_cg_mat):
        per_path_fiber_start = [0]
        per_path_fiber_array = []
        per_in1_ir_pathinfo = {}

        for inst in tp_inst_outorder:
            path_cg = unique_cg_mat[
                f"{self.i_in1[inst.i_in1][1].l}_{self.i_in2[inst.i_in2][1].l}_{self.i_out[inst.i_out][1].l}"
            ]
            for i, j, k in path_cg.nonzero():
                cg_idx = self.unique_cg_val.index(path_cg[i, j, k].to(torch.float32))
                per_path_fiber_array.append([i.item(), j.item(), k.item(), cg_idx])
            per_path_fiber_start.append(len(per_path_fiber_array))

            if inst.i_in1 not in per_in1_ir_pathinfo:
                per_in1_ir_pathinfo[inst.i_in1] = []
            per_in1_ir_pathinfo[inst.i_in1].append(
                [inst.i_out, inst.i_in2, inst.path_weight]
            )

        return per_path_fiber_start, per_path_fiber_array, per_in1_ir_pathinfo

    def metadata_gen(
        self,
        instructions,
        per_path_fiber_start,
        per_path_fiber_array,
        per_in1_ir_pathinfo,
    ):
        weight_uv_pair = []
        weight_uv_pair_sorted_chunk = []
        out_order = []
        current = 0

        for inst in instructions:
            weight_uv_pair.append(
                (self.i_in1[inst.i_in1][0], self.i_in2[inst.i_in2][0])
            )
            out_order.append(inst.i_out)
        for u, v in weight_uv_pair:
            weight_uv_pair_sorted_chunk.append(slice(current, current + u * v))
            current += u * v
        out2weight_order = torch.tensor(out_order).argsort()

        in1_idxing = []
        in1_ival = []
        in1_related_path_idx = [0]

        path_array1 = []
        path_array2 = []
        path_weight = []
        per_path_weight_pos = []
        per_exec_info = []

        per_upath_fiber_start = []
        per_upath_fiber_array = []

        in1_slices = self.i_in1.slices()
        in2_slices = self.i_in2.slices()
        out_slices = self.i_out.slices()

        for in1_ir_idx, (mul, ir) in enumerate(self.i_in1):
            assert mul % self.WARPSIZE == 0
            in1_idx_start = in1_slices[in1_ir_idx].start
            i_val = ir.dim
            in1_idxing.append(in1_idx_start)
            in1_ival.append(i_val)
            in1_related_path_idx.append(
                in1_related_path_idx[-1] + len(per_in1_ir_pathinfo[in1_ir_idx])
            )  # end point : current number of path + path related to this in1_ir_idx

            dummy_list = []
            dummy_list2 = []

            for out_ir_idx, in2_ir_idx, pw in per_in1_ir_pathinfo[in1_ir_idx]:
                # start and end nnz for path (in1_ir_idx, in2_ir_idx, out_ir_idx)
                # uses out_ir_idx as each path have unique out_ir_idx
                fiber_start = per_path_fiber_start[out_ir_idx]
                fiber_end = per_path_fiber_start[out_ir_idx + 1]
                # per_upath_fiber_start.append([fiber_start, fiber_end])
                upath_fiber_start = len(per_upath_fiber_array)
                upath_fiber_end = upath_fiber_start + fiber_end - fiber_start
                per_upath_fiber_start.append([upath_fiber_start, upath_fiber_end])

                per_upath_fiber_array += per_path_fiber_array[fiber_start:fiber_end]

                dummy_list.append([out_slices[out_ir_idx].start])
                dummy_list2.append(
                    [
                        self.i_out[out_ir_idx].ir.dim,
                        in2_slices[in2_ir_idx].start,
                        self.i_in2[in2_ir_idx].ir.dim,
                        in2_slices[in2_ir_idx].stop,
                    ]
                )
                path_weight.append(pw)

                per_path_weight_pos.append(
                    weight_uv_pair_sorted_chunk[out2weight_order[out_ir_idx]].start
                )
            path_array1.append(dummy_list)
            path_array2.append(dummy_list2)

            for i in range(mul // self.WARPSIZE):
                per_exec_info.append([in1_ir_idx, i])
       
        cg_idx_array = []
        for d in per_upath_fiber_array:
            cg_idx_array.append(d[-1])
        for d in per_upath_fiber_array:
            d[-1] = 0

        partial_fiber_size = []
        partial_fiber_start = []
        partial_fiber_end = []
        for target_in1 in range(len(in1_related_path_idx) - 1):
            path_idx_start = in1_related_path_idx[target_in1]
            path_idx_end = in1_related_path_idx[target_in1 + 1]

            fiber_min = per_upath_fiber_start[path_idx_start][0]
            fiber_max = per_upath_fiber_start[path_idx_end - 1][1]

            partial_fiber_start.append(fiber_min)
            partial_fiber_end.append(fiber_max)
            partial_fiber_size.append(fiber_max - fiber_min)
       
        max_fiber_size = max(partial_fiber_size)

        t_in1_idxing = torch.tensor(in1_idxing, dtype=torch.int32, device=self.device)
        t_in1_ival = torch.tensor(in1_ival, dtype=torch.int32, device=self.device)
        t_in1_related_path_idx = torch.tensor(
            in1_related_path_idx, dtype=torch.int32, device=self.device
        )

        t_path_array1 = torch.tensor(
            list(itertools.chain.from_iterable(path_array1)),
            dtype=torch.uint32,
            device=self.device,
        )
        t_path_array2 = torch.tensor(
            list(itertools.chain.from_iterable(path_array2)),
            dtype=torch.uint8,
            device=self.device,
        )
        t_path_weight = torch.tensor(path_weight, dtype=self.dtype, device=self.device)
        t_per_path_weight_pos = torch.tensor(
            per_path_weight_pos, dtype=torch.int32, device=self.device
        )
        t_per_upath_fiber_start = torch.tensor(
            per_upath_fiber_start, dtype=torch.uint16, device=self.device
        )
        t_per_upath_fiber_array = torch.tensor(
            per_upath_fiber_array, dtype=torch.uint8, device=self.device
        )
        t_unique_cg_val = torch.tensor(
            self.unique_cg_val, dtype=self.dtype, device=self.device
        )
        t_per_exec_info = torch.tensor(
            list(itertools.chain.from_iterable(per_exec_info)),
            dtype=torch.uint16,
            device=self.device,
        )
        t_partial_fiber_start = torch.tensor(
            partial_fiber_start, dtype=torch.uint16, device=self.device
        )
        t_partial_fiber_end = torch.tensor(
            partial_fiber_end, dtype=torch.uint16, device=self.device
        )
        
        upath_cnt = len(per_exec_info)

        self.t_cg_idx_array = torch.tensor(
            cg_idx_array, dtype=torch.uint16, device=self.device
        )
        # for idx, (start,end) in enumerate(t_per_upath_fiber_start[:-2]):
        #     print(start, end)
        #     print(idx, t_per_upath_fiber_array[start:end,-1].unique())

        return [
            t_in1_idxing,
            t_in1_ival,
            t_in1_related_path_idx,
            t_path_array1,
            t_path_array2,
            t_per_upath_fiber_start,
            t_path_weight,
            t_per_path_weight_pos,
            t_per_upath_fiber_array,
            t_unique_cg_val,
            t_per_exec_info,
            t_partial_fiber_start,
            t_partial_fiber_end,
            max_fiber_size,
            upath_cnt,
        ]


__all__ = ["fused_uvu_TP_exp_opt_extcg"]
