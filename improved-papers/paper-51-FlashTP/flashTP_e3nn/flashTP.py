import torch
import e3nn
import logging

from .sptp_exp_opt_large.fused_e3nn_exp_opt_large import fused_uvu_TP_exp_opt_large
from .sptp_exp_opt_large.fused_e3nn_exp_opt_large_lammps import fused_uvu_TP_exp_opt_large_lammps
from .sptp_exp_opt_extcg.fused_e3nn_exp_opt_extcg import fused_uvu_TP_exp_opt_extcg

logger = logging.getLogger(__name__)

def uvu_TP(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions,
    block_batch_cnt=None,
    smem_size = None,
    device="cuda",
    dtype=torch.float32,
    use_lammps=False
):
    """
    Wrapper function to create and return different fused_uvu_TP instance

    Constructs and returns a `fused_uvu_TP` object for the TensorProduct.

    Parameters
    ----------
    irreps_in1 : e3nn.o3.Irreps
        Representation of the first irreps.
    irreps_in2 : e3nn.o3.Irreps
        Representation of the second irreps.
    irreps_out : e3nn.o3.Irreps
        Representation of the output irreps.
    instructions : List[tuple]
        Path used for the TensorProduct
    block_batch_cnt : Sequence[int], optional
        Three-element list `[B1, B2, B3]` specifying the number of blocks
        processed per batch in each dimension (maximum of 32). If `None`, uses built-in
        heuristics to choose block sizes.
    smem_size : int, optional
        Shared memory size in KiBs. 
        If `None`, it uses built-in value which is 163KiBs (A100), 227KiBs (H100), 48KiBs (Default).
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
        view this link for shared memory size for different GPUs.
    device : str or torch.device, default="cuda"
    dtype : torch.dtype, default=torch.float32
        Data type for all internal tensors and kernels.
    use_lammps : when using lammps it uses .so file that includes C++ side torch registration (only registered forward and backward for inference)
    """

    # Supported TensorProduct options
    # irrep_normalization = "component"
    # path_normalization  = "element"
    # internal_weights = False
    # shared_weights = False


    # create equivalent TP of e3nn
    tp = e3nn.o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights=False,
        internal_weights=False,
    )
    
    unique_cg, unique_cg_mat = extract_cg_info(
        irreps_in1,
        irreps_in2,
        irreps_out, tp.instructions, dtype
    )
    unique_cg_val = list(set(unique_cg))

    # if number of unique cg values is greater than 256,
    # its index can't be stored in uint8 and handled by fused_uvu_TP_exp_opt_extcg
    if len(unique_cg_val) <= 256:
        if use_lammps:
            logger.info("Using fused_uvu_TP_exp_opt_large_lammps")
            return fused_uvu_TP_exp_opt_large_lammps(
                i_in1=irreps_in1,
                i_in2=irreps_in2,
                i_out=irreps_out,
                instructions=tp.instructions,
                unique_cg_mat=unique_cg_mat,
                unique_cg_val = unique_cg_val,
                per_block_batch=block_batch_cnt,
                smem_size=smem_size,
                device=device,
                dtype=dtype
            )
        else:
            logger.info("Using fused_uvu_TP_exp_opt_large")
            return fused_uvu_TP_exp_opt_large(
                i_in1=irreps_in1,
                i_in2=irreps_in2,
                i_out=irreps_out,
                instructions=tp.instructions,
                unique_cg_mat=unique_cg_mat,
                unique_cg_val = unique_cg_val,
                per_block_batch=block_batch_cnt,
                smem_size=smem_size,
                device=device,
                dtype=dtype
            )
    else:
        logger.info("Using fused_uvu_TP_exp_opt_extcg")
        return fused_uvu_TP_exp_opt_extcg(
            i_in1=irreps_in1,
            i_in2=irreps_in2,
            i_out=irreps_out,
            instructions=tp.instructions,
            unique_cg_mat=unique_cg_mat,
            unique_cg_val = unique_cg_val,
            per_block_batch=block_batch_cnt,
            smem_size=smem_size,
            device=device,
            dtype=dtype
        )


def extract_cg_info(i_in1, i_in2, uvuv_i_out, instructions, dtype):
    unique_cg = []
    unique_cg_mat = {}
    for inst in instructions:
        i = inst.i_in1
        j = inst.i_in2
        k = inst.i_out

        mul_in1, ir_in1 = i_in1[i]
        mul_in2, ir_in2 = i_in2[j]
        mul_out, ir_out = uvuv_i_out[k]

        cg = e3nn.o3.wigner_3j(ir_in1.l, ir_in2.l, ir_out.l)

        # For fp64 there are small difference between same values due to floating point problem.
        # This increase the number of unique values unnecessarily.
        # We remove it by converting it to fp32 then back to target datatype
        cg_fp32 = cg.to(torch.float32)
        unique_cg += cg_fp32.unique().to(dtype).tolist()

        unique_cg_mat[f"{ir_in1.l}_{ir_in2.l}_{ir_out.l}"] = cg


    return unique_cg, unique_cg_mat
