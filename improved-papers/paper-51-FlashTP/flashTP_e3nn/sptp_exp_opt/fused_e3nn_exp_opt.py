import os
import itertools
from typing import List

import e3nn
import torch
from torch.utils.cpp_extension import load
from torch_scatter import scatter
import logging


flashtp_kernel = None
logger = logging.getLogger(__name__)


# lazy compile

# def _init():
#     global flashtp_kernel
#     if not flashtp_kernel:
#         return

# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
script_dir = os.path.dirname(os.path.abspath(__file__))
flashtp_kernel = load(
    name="flashtp_kernel",
    sources=[
        f"{script_dir}/sptp_exp_opt.cpp",
        f"{script_dir}/fwd_sptp_linear_v2_shared_exp.cu",
        f"{script_dir}/bwd_sptp_linear_shared_exp.cu",
        f"{script_dir}/bwd_bwd_sptp_linear_v2_shared_exp.cu",
        f"{script_dir}/fwd_sptp_linear_v2_shared_exp_double.cu",
        f"{script_dir}/bwd_sptp_linear_shared_exp_double.cu",
        f"{script_dir}/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",
    ],
    extra_cuda_cflags=["-lineinfo"],
    verbose=True,
)

@torch.library.custom_op(
    "flashtp_kernel::sptp_linear_fwd_v2_shared_exp",
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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
    out_size: int,
) -> torch.Tensor:
    node_cnt = in1.shape[0]
    batch_size = in2.shape[0]
    out = torch.zeros((node_cnt, out_size), device=in1.device, dtype=in1.dtype)

    assert in2.dtype == in1.dtype
    assert weight.dtype == in1.dtype

    flashtp_kernel.sptp_linear_fwd_v2_shared_exp(
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
        upath_cnt,
        per_block_batch[0],
        max_ir_dim * 2 + 1,
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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
    out_size: int,
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
        upath_cnt,
        per_block_batch,
        max_ir_dim,
        out_size,
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
    )
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
    ) = ctx.saved_tensors

    grad_list = torch.ops.flashtp_kernel.sptp_linear_bwd_v2_shared_exp(
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
        ctx.upath_cnt,
        ctx.per_block_batch,
        ctx.max_ir_dim,
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
    )


@torch.library.custom_op(
    "flashtp_kernel::sptp_linear_bwd_v2_shared_exp",
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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
) -> List[torch.Tensor]:
    node_cnt = in1.shape[0]
    batch_size = in2.shape[0]
    in2_size = in2.shape[1]
    in1_size = in1.shape[1]

    mem_debug = torch.empty((1, 1), device=in1.device, dtype=in1.dtype)
    mem_dl_din1 = torch.empty(
        (batch_size, in1_size), device=in1.device, dtype=in1.dtype
    )
    mem_dl_din2 = torch.empty(
        (batch_size, in2_size * upath_cnt), device=in1.device, dtype=in1.dtype
    )
    mem_dl_dw = torch.empty_like(weight)

    assert in2.dtype == in1.dtype
    assert weight.dtype == in1.dtype
    assert grad_output.dtype == in1.dtype

    flashtp_kernel.sptp_linear_bwd_v1_shared_exp(
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
        upath_cnt,
        per_block_batch[1],
        max_ir_dim * 2 + 1,
    )
    mem_dl_din2_summed = mem_dl_din2.reshape((batch_size, upath_cnt, in2_size)).sum(
        dim=1
    )

    dl_din1_reduced = scatter(
        mem_dl_din1,
        per_edge_src.to(torch.int64),
        dim=0,
        dim_size=node_cnt,
        reduce="sum",
    )

    del mem_dl_din1
    del mem_dl_din2

    return [dl_din1_reduced, mem_dl_din2_summed, mem_dl_dw]


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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
) -> List[torch.Tensor]:
    dl_din1_reduced = torch.empty_like(in1)
    mem_dl_din2_summed = torch.empty_like(in2)
    mem_dl_dw = torch.empty_like(weight)

    return [dl_din1_reduced, mem_dl_din2_summed, mem_dl_dw]


@torch.library.custom_op(
    "flashtp_kernel::sptp_linear_bwd_bwd_v2_shared_exp",
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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
) -> List[torch.Tensor]:
    node_cnt = in1.shape[0]
    batch_size = in2.shape[0]
    in1_size = in1.shape[1]
    in2_size = in2.shape[1]
    out_size = dE_dout.shape[1]

    # torch.cuda.memory._dump_snapshot(f"/home2/lsy/mdsim/fused_e3nn/fused_e3nn_kernel/snap.pickle")

    dF_dout = torch.zeros((node_cnt, out_size), device=in1.device, dtype=in1.dtype)
    # dF_dout = torch.empty((batch_size, out_size) , device=in2.device, dtype=in2.dtype)
    dL_din1 = torch.empty((batch_size, in1_size), device=in1.device, dtype=in1.dtype)
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

    flashtp_kernel.sptp_linear_bwd_bwd_v2_shared_exp(
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
        upath_cnt,
        per_block_batch[2],
        max_ir_dim * 2 + 1,
    )

    dL_din2 = dL_din2_duplicate.reshape((batch_size, upath_cnt, in2_size)).sum(dim=1)
    dL_din1_reduced = scatter(
        dL_din1, per_edge_src.to(torch.int64), dim=0, dim_size=node_cnt, reduce="sum"
    )
    # dF_dout_reduced = scatter(dF_dout, per_edge_dst.to(torch.int64), dim=0, dim_size=node_cnt, reduce="sum")

    del dL_din1
    del dL_din2_duplicate
    # del dF_dout

    return [dL_din1_reduced, dL_din2, dL_dw, dF_dout]
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
    upath_cnt: int,
    per_block_batch: List[int],
    max_ir_dim: int,
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
        upath_cnt,
        per_block_batch,
        max_ir_dim,
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
    )
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
    ) = ctx.saved_tensors

    dF_in1 = grad_output[0]
    dF_in2 = grad_output[1]
    dF_w = grad_output[2]

    grad_list = torch.ops.flashtp_kernel.sptp_linear_bwd_bwd_v2_shared_exp(
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
        ctx.upath_cnt,
        ctx.per_block_batch,
        ctx.max_ir_dim,
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
    )


torch.library.register_autograd(
    "flashtp_kernel::sptp_linear_fwd_v2_shared_exp",
    fused_e3nn_bwd_exp,
    setup_context=fused_e3nn_setup_fwd_context_exp,
)

torch.library.register_autograd(
    "flashtp_kernel::sptp_linear_bwd_v2_shared_exp",
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
# def find_optimal_abc(a,b,c, max_y=32):
#     # Start from the maximum allowed threads per block and decrease
#     approx_y = math.ceil(a / b)
#     max_shmem_size = []
#     max_x_size = []

#     for y in range(1, approx_y+1):
#         per_thread_shmem = (a - y*c) / y
#         x = math.floor(per_thread_shmem / b)
#         if(x <= max_y and x>= 1):
#             max_shmem_size.append(x*y*b + y*c)
#             max_x_size.append(x)

#     max_x = 0
#     max_shmem = 0
#     for x, shmem in zip(max_x_size, max_shmem_size):
#         # if difference is larger than 10KB
#         # if difference is small keep larger x
#         if(abs(max_shmem - shmem) > 10*1024 and shmem > max_shmem):
#             max_shmem = shmem
#             max_x  = x
#     return max_x


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


class fused_uvu_TP_exp_opt(torch.nn.Module):
    def __init__(
        self,
        i_in1,
        i_in2,
        i_out,
        inst_tuple,
        warpsize=32,
        per_block_batch=4,
        device="cuda",
        shared_weights=False,
        internal_weights=False,
        dtype=torch.float32,
    ):
        super().__init__()
        # _init()

        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.inst_tuple = inst_tuple
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

        self.per_block_opt_batch = []

        ## forward

        _irreps_out = [e3nn.o3.Irrep(o.ir.l, o.ir.p) for o in i_out]
        uvuv_tp = e3nn.o3.FullTensorProduct(i_in1, i_in2, filter_ir_out=_irreps_out)
        uvuv_i_out = uvuv_tp.irreps_out
        tp = e3nn.o3.TensorProduct(
            i_in1,
            i_in2,
            i_out,
            inst_tuple,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

        unique_cg, unique_cg_mat = self.extract_cg_info(
            uvuv_i_out, uvuv_tp.instructions
        )
        # print(unique_cg[0].dtype)

        self.unique_cg_val = list(set(unique_cg))
        tp_inst_outorder = sorted(tp.instructions, key=lambda x: x.i_out)

        per_path_fiber_start, per_path_fiber_array, per_in1_ir_pathinfo = (
            self.cgmat2fiber(tp_inst_outorder, unique_cg_mat)
        )

        self.metadata_list = self.metadata_gen(
            tp.instructions,
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
        gpu_name = torch.cuda.get_device_name()
        if "A100" in gpu_name:
            SMEM_SIZE = 163
        elif "H100" in gpu_name:
            SMEM_SIZE = 227
        else:
            logger.warning("Need to tune SMEM_SIZE, using safe mode of SMEM_SIZE=48KB")
            SMEM_SIZE = 48
        logger.info(f"SMEM_SIZE set to {SMEM_SIZE} KB")

        perwarp_in2_size = in2_size
        if in2_size % 2 == 0:
            perwarp_in2_size = in2_size + 1

        fiber_array_length = self.metadata_list[-3].numel()
        fwd_per_block_opt_batch = (SMEM_SIZE * 1024) // (
            d_size * (WARPSIZE * (max_ir_dim * 3) + perwarp_in2_size)
        )

        # if (fwd_per_block_opt_batch > 32):
        #     y,z = find_optimal_yz(fwd_per_block_opt_batch)
        #     print(fwd_per_block_opt_batch, y,z, (d_size * (WARPSIZE * (max_ir_dim*3) + perwarp_in2_size)))
        #     real_size = z*(d_size*y * (WARPSIZE * (max_ir_dim*3) + perwarp_in2_size) + fiber_array_length)
        #     while(real_size > (SMEM_SIZE*1024)):
        #         fwd_per_block_opt_batch-=1
        #         y,z = find_optimal_yz(fwd_per_block_opt_batch)
        #         print(fwd_per_block_opt_batch, y,z, (d_size * (WARPSIZE * (max_ir_dim*3) + perwarp_in2_size)))
        #         real_size = z*(d_size*y * (WARPSIZE * (max_ir_dim*3) + perwarp_in2_size) + fiber_array_length)

        #     self.per_block_opt_batch.append(y)
        # else:
        #     self.per_block_opt_batch.append(fwd_per_block_opt_batch)

        fwd_per_block_opt_batch = find_optimal_abc(
            SMEM_SIZE * 1024,
            d_size * (WARPSIZE * (max_ir_dim * 3) + perwarp_in2_size),
            fiber_array_length,
        )
        self.per_block_opt_batch.append(fwd_per_block_opt_batch)

        bwd_per_block_opt_batch = find_optimal_abc(
            SMEM_SIZE * 1024,
            d_size * WARPSIZE * (max_ir_dim * 5 + perwarp_in2_size) + in2_size * d_size,
            0,
        )
        self.per_block_opt_batch.append(bwd_per_block_opt_batch)

        bwd_bwd_per_block_opt_batch = find_optimal_abc(
            SMEM_SIZE * 1024,
            d_size * WARPSIZE * (max_ir_dim * 7 + perwarp_in2_size)
            + in2_size * d_size * 2,
            0,
        )
        self.per_block_opt_batch.append(bwd_bwd_per_block_opt_batch)

        # bwd_per_block_opt_batch = ((SMEM_SIZE*1024) // (d_size * WARPSIZE * (max_ir_dim*5 + perwarp_in2_size)))
        # if (bwd_per_block_opt_batch > 32):
        #     y,z = find_optimal_yz(bwd_per_block_opt_batch)
        #     self.per_block_opt_batch.append(y)
        # else:
        #     self.per_block_opt_batch.append(bwd_per_block_opt_batch)

        # bwd_bwd_per_block_opt_batch = ((SMEM_SIZE*1024) // (d_size * WARPSIZE * (max_ir_dim*7+perwarp_in2_size)))

        # if (bwd_bwd_per_block_opt_batch > 32):
        #     y,z = find_optimal_yz(bwd_bwd_per_block_opt_batch)
        #     self.per_block_opt_batch.append(y)
        # else:
        #     self.per_block_opt_batch.append(bwd_bwd_per_block_opt_batch)

        logger.info(f"Using per_block_batch of {self.per_block_opt_batch}")
        # exit()

    def forward(self, in1, in2, weight, per_edge_src, per_edge_dst):
        # out = torch.empty((batch_size, out_size), device=in1.device, dtype=in1.dtype)
        # sptp_linear.sptp_linear_fwd_v2_shared(
        #     in1,in2,weight, out,
        #     *self.metadata_list, 1, self.l_max*2+1
        # )
        out = torch.ops.flashtp_kernel.sptp_linear_fwd_v2_shared_exp(
            in1,
            in2,
            weight,
            per_edge_src,
            per_edge_dst,
            *self.metadata_list,
            self.per_block_opt_batch,
            self.l_max,
            # self.i_out.dim
            self.out_dim,
        )
        return out

    def extract_cg_info(self, uvuv_i_out, instructions):
        unique_cg = []
        # asdfasd = []
        unique_cg_mat = {}
        idx = 0
        for inst in instructions:
            i = inst.i_in1
            j = inst.i_in2
            k = inst.i_out

            mul_in1, ir_in1 = self.i_in1[i]
            mul_in2, ir_in2 = self.i_in2[j]
            mul_out, ir_out = uvuv_i_out[k]

            cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)
            # print(idx, cg.unique())
            idx += 1
            # asdfasd.append(cg.unique().tolist())
            # remove small difference in fp64 precision problem
            # by converting it to fp32 then back to target dtype
            cg_fp32 = cg.to(torch.float32)
            unique_cg += cg_fp32.unique().to(self.dtype).tolist()

            partial_mat_cg = torch.zeros(
                self.i_in1[i].dim, self.i_in2[j].dim, uvuv_i_out[k].dim
            )
            # print(cg)
            unique_cg_mat[f"{ir_in1.l}_{ir_in2.l}_{ir_out.l}"] = cg

            ## uvuv
            for u, v in itertools.product(range(mul_in1), range(mul_in2)):
                partial_mat_cg[
                    u * ir_in1.dim : (u + 1) * ir_in1.dim,
                    v * ir_in2.dim : (v + 1) * ir_in2.dim,
                    (u * mul_in2 + v) * ir_out.dim : (u * mul_in2 + v + 1) * ir_out.dim,
                ] = cg

        # print("cg u partition")
        # merged_groups = []
        # set_indices = []

        # # Convert each set to a list to work with mutable groups
        # sets = [set(s) for s in unique_cg]
        # indices = list(range(len(sets)))

        # while sets:
        #     # Sort sets by size to prioritize minimizing unique values
        #     sets, indices = zip(*sorted(zip(sets, indices), key=lambda x: len(x[0]), reverse=True))
        #     sets, indices = list(sets), list(indices)
        #     merged = sets.pop(0)
        #     merged_indices = [indices.pop(0)]

        #     # Avoid modifying list size during iteration by using a copy
        #     removable_indices = []
        #     for i, other_set in enumerate(sets):
        #         union = merged.union(other_set)
        #         if len(union) < 256:
        #             merged = union
        #             merged_indices.append(indices[i])
        #             removable_indices.append(i)

        #     # Remove the marked indices after iteration
        #     for index in sorted(removable_indices, reverse=True):
        #         sets.pop(index)
        #         indices.pop(index)

        #     merged_groups.append(merged)
        #     set_indices.append(merged_indices)

        # # Minimize total unique values by flattening the groups
        # print(merged_groups, set_indices)
        # for e in merged_groups:
        #     print(len(e))

        # print(len(set(asdfasd)))

        # exit()

        return unique_cg, unique_cg_mat

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

        in1_idxing = [0]
        in1_ival = []
        in1_related_path_idx = [0]

        path_array1 = []
        path_array2 = []
        path_weight = []
        per_path_weight_pos = []

        per_upath_fiber_start = []
        per_upath_fiber_array = []

        in1_slices = self.i_in1.slices()
        in2_slices = self.i_in2.slices()
        out_slices = self.i_out.slices()

        for in1_ir_idx, (mul, ir) in enumerate(self.i_in1):
            assert mul % self.WARPSIZE == 0
            in1_idx_start = in1_slices[in1_ir_idx].start
            i_val = ir.dim

            if mul >= self.WARPSIZE:
                original_start = len(per_upath_fiber_array)
                for i in range(mul // self.WARPSIZE):
                    upath_start_pointer = original_start
                    in1_idxing.append(in1_idx_start + self.WARPSIZE * i_val * (i + 1))
                    in1_ival.append(i_val)
                    in1_related_path_idx.append(
                        in1_related_path_idx[-1] + len(per_in1_ir_pathinfo[in1_ir_idx])
                    )

                    dummy_list = []
                    dummy_list2 = []
                    # Bug? TODO:

                    for out_ir_idx, in2_ir_idx, pw in per_in1_ir_pathinfo[in1_ir_idx]:
                        # should be in order
                        fiber_start = per_path_fiber_start[out_ir_idx]
                        fiber_end = per_path_fiber_start[out_ir_idx + 1]

                        # upath_fiber_start = len(per_upath_fiber_array)
                        upath_fiber_start = upath_start_pointer
                        upath_fiber_end = upath_fiber_start + fiber_end - fiber_start

                        # move pointer
                        upath_start_pointer += fiber_end - fiber_start

                        per_upath_fiber_start.append(
                            [upath_fiber_start, upath_fiber_end]
                        )
                        # print(fiber_array_orignal[1:4])
                        # print(fiber_start,fiber_end)
                        # print(fiber_array_orignal[fiber_start:fiber_end])

                        ## push for first chunk
                        if i == 0:
                            per_upath_fiber_array += per_path_fiber_array[
                                fiber_start:fiber_end
                            ]

                        dummy_list.append(
                            [
                                out_slices[out_ir_idx].start
                                + self.WARPSIZE * self.i_out[out_ir_idx].ir.dim * i,
                                out_slices[out_ir_idx].start
                                + self.WARPSIZE
                                * self.i_out[out_ir_idx].ir.dim
                                * (i + 1),
                            ]
                        )
                        dummy_list2.append(
                            [
                                self.i_out[out_ir_idx].ir.dim,
                                in2_slices[in2_ir_idx].start,
                                self.i_in2[in2_ir_idx].ir.dim,
                                in2_slices[in2_ir_idx].stop,
                            ]
                        )
                        path_weight.append(pw)

                        # TODO:??
                        per_path_weight_pos.append(
                            weight_uv_pair_sorted_chunk[
                                out2weight_order[out_ir_idx]
                            ].start
                            + self.WARPSIZE * i
                        )

                    path_array1.append(dummy_list)
                    path_array2.append(dummy_list2)

        t_in1_idxing = torch.tensor(in1_idxing, dtype=torch.int32, device=self.device)
        t_in1_ival = torch.tensor(in1_ival, dtype=torch.int32, device=self.device)
        t_in1_related_path_idx = torch.tensor(
            in1_related_path_idx, dtype=torch.int32, device=self.device
        )

        t_path_array1 = torch.tensor(
            list(itertools.chain.from_iterable(path_array1)),
            dtype=torch.uint16,
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
        upath_cnt = len(in1_idxing) - 1

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
            upath_cnt,
        ]


__all__ = ["fused_uvu_TP_exp"]
