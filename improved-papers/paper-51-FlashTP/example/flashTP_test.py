import torch
import e3nn
import os,sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flashTP_e3nn
_torch_scatter_exist = False
try:
    # Please install torch-scatter if you want to 
    # reproduce the e3nn performance evaluation
    from torch_scatter import scatter
    _torch_scatter_exist = True
except ImportError:
    pass

import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =============================================================================
# Command-line arguments (in order):
#   1) filename (str)
#        Path to the TensorProduct file for each model.
#   2) layer_idx (int)
#        Zero-based index of the layer you want to profile in the TensorProduct file.
#   3) target_edge_cnt (int)
#        Number of total edges for the run (e.g., 16384 for 16k, 32768 for 32k).
#   4) used_dtype_str (str)
#        Data type to use, as a string (e.g., 'fp32', 'fp64').
#   5) run_cueq (str)
#        Flag to select implementation:
#          'true'  → run including CuEquivariance  
#          'false' → only run e3nn and FlashTP
#   6) channel_multiplier (int)
#        Multiplier on hidden-channel width of input1 of selected TensorProduct
#        For TensorProduct ("32x0e", "1x0e+1x1o+1x2e+1x3o") 
#        with multiplier 2 => it will be ("64x0e", "1x0e+1x1o+1x2e+1x3o")
# =============================================================================

def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    
    filename = sys.argv[1]
    layer_idx = int(sys.argv[2])
    target_edge_cnt = int(sys.argv[3]) # 16k, 32k
    used_dtype_str = sys.argv[4]
    run_cueq = sys.argv[5]
    channel_multiplier = int(sys.argv[6])

    if run_cueq == "False":
        run_cueq = False
    else:
        run_cueq = True
    
    ## dytpe selection
    used_dtype = torch.float32
    if (used_dtype_str == "fp64"):
        used_dtype = torch.float64
    
    # we generate random edges by selecting 64 neighbours for each node
    max_neighbour = 64
    total_node = target_edge_cnt // max_neighbour
    edge_src, edge_dst = flashTP_e3nn.utils.fixed_generate_edgepair(total_node, max_neighbour)

    batch_size = len(edge_src)
    print("batch_size", batch_size)

    e3nn_config, cueq_config = flashTP_e3nn.utils.load_config_e3nn_cueq(filename,layer_idx, channel_mul= channel_multiplier)
    i_in1, i_in2, i_out, inst_tuple = e3nn_config

    torch.set_default_dtype(used_dtype)
    tp = e3nn.o3.TensorProduct(i_in1,i_in2,i_out,inst_tuple,shared_weights=False, internal_weights=False) # path_normalization="none", normalization="none"
    tp = tp.to(device="cuda")
    flashtp = flashTP_e3nn.uvu_TP(i_in1,i_in2,i_out,inst_tuple, device="cuda", dtype=used_dtype)

    if run_cueq:
        import cuequivariance_torch as cuet
        cuet_tp = cuet.ChannelWiseTensorProduct(*cueq_config[:-1], shared_weights=False,internal_weights=False, device="cuda", layout="ir_mul", dtype=used_dtype)
    
    torch.set_default_dtype(torch.float32)

    t_edge_src = torch.tensor(edge_src, device="cuda", dtype=torch.int32)
    t_edge_dst = torch.tensor(edge_dst, device="cuda", dtype=torch.int32)


    for i in range(1):        
        in1_node = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
        in2 = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True,dtype=used_dtype)
        weight = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True,dtype=used_dtype)

        in1_node_c = in1_node.detach().clone().requires_grad_()
        in2_c = in2.detach().clone().requires_grad_()
        weight_c = weight.detach().clone().requires_grad_()


        if run_cueq:
            cin1_node_cuda = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
            cin2_cuda = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True,dtype=used_dtype)
            weight_cueq = torch.rand(batch_size,tp.weight_numel, device="cuda", requires_grad=True,dtype=used_dtype)

        ## e3nn
        # fwd path
        in1 = in1_node[edge_src]
        out_large_exp = tp(in1,in2,weight)
        if(_torch_scatter_exist):
            out_exp = scatter(out_large_exp, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
        else:
            out_exp = torch.zeros(total_node, out_large_exp.shape[1], dtype=out_large_exp.dtype, device=out_large_exp.device)
            out_exp.scatter_add_(
                dim=0,
                index=t_edge_dst.to(torch.int64).unsqueeze(1).expand(-1, out_large_exp.shape[1]),  # (N,1) -> (N,C)
                src=out_large_exp
            )
            
            
        y = torch.nn.functional.gelu(out_exp).sum()

        # original e3nn
        f_in1, f_in2, f_weight = torch.autograd.grad(y, [in1_node,in2,weight], create_graph=True)

        f_in1_gelu = torch.nn.functional.gelu(f_in1)
        f_in2_gelu = torch.nn.functional.gelu(f_in2)
        f_weight_gelu = torch.nn.functional.gelu(f_weight)
        fake_loss = f_in1_gelu.sum() + f_in2_gelu.sum() + f_weight_gelu.sum()

        fake_loss.backward()


        ## flashTP 
        out_ours = flashtp(in1_node_c,in2_c,weight_c, t_edge_src, t_edge_dst)
        y_ours = torch.nn.functional.gelu(out_ours).sum()
        flashTP_e3nn.utils.compare(out_ours, out_exp)

        # original e3nn
        f_in1_c, f_in2_c, f_weight_c = torch.autograd.grad(y_ours, [in1_node_c,in2_c,weight_c], create_graph=True)

        flashTP_e3nn.utils.compare(f_in1, f_in1_c)
        flashTP_e3nn.utils.compare(f_in2, f_in2_c)
        flashTP_e3nn.utils.compare(f_weight, f_weight_c)

        f_in1_gelu_c = torch.nn.functional.gelu(f_in1_c)
        f_in2_gelu_c = torch.nn.functional.gelu(f_in2_c)
        f_weight_gelu_c = torch.nn.functional.gelu(f_weight_c)
        fake_loss_c = f_in1_gelu_c.sum() + f_in2_gelu_c.sum() + f_weight_gelu_c.sum()
        
        fake_loss_c.backward()

        flashTP_e3nn.utils.compare(in1_node.grad, in1_node_c.grad)
        flashTP_e3nn.utils.compare(in2.grad, in2_c.grad)
        flashTP_e3nn.utils.compare(weight.grad, weight_c.grad)


        ## cueq 
        ## Comparing correctness is possible but challenging as the some layouts are different
        if run_cueq:
            cin1_cuda = cin1_node_cuda[edge_src]
            cuet_out_exp = cuet_tp(cin1_cuda,cin2_cuda,weight_cueq)
            cuet_out = scatter(cuet_out_exp, t_edge_dst.to(torch.int64), dim=0, dim_size=total_node, reduce="sum")
            y_cu = torch.nn.functional.gelu(cuet_out).sum()

            f_in1_cu, f_in2_cu, f_weight_cu = torch.autograd.grad(y_cu, [cin1_node_cuda,cin2_cuda,weight_cueq], create_graph=True)
        
            f_in1_gelu_cu = torch.nn.functional.gelu(f_in1_cu)
            f_in2_gelu_cu = torch.nn.functional.gelu(f_in2_cu)
            f_weight_gelu_cu = torch.nn.functional.gelu(f_weight_cu)
            fake_loss_cu = f_in1_gelu_cu.sum() + f_in2_gelu_cu.sum() + f_weight_gelu_cu.sum()

            fake_loss_cu.backward()
            

if __name__ == "__main__":
    main()