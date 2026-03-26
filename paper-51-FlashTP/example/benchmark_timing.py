"""
Timing benchmark for FlashTP kernel microbenchmark.
Measures forward, backward, and double-backward latency.
"""
import torch
import e3nn
import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flashTP_e3nn
import logging
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

def time_kernel(fn, n_warmup=50, n_repeat=200):
    """Time a function using CUDA events."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_repeat  # ms per iteration


def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    # Parse args: filename layer_idx target_edge_cnt dtype channel_multiplier
    if len(sys.argv) < 6:
        print("Usage: python benchmark_timing.py <config_file> <layer_idx> <edge_cnt> <dtype> <channel_mul>")
        sys.exit(1)

    filename = sys.argv[1]
    layer_idx = int(sys.argv[2])
    target_edge_cnt = int(sys.argv[3])
    used_dtype_str = sys.argv[4]
    channel_multiplier = int(sys.argv[5])

    used_dtype = torch.float32
    if used_dtype_str == "fp64":
        used_dtype = torch.float64

    max_neighbour = 64
    total_node = target_edge_cnt // max_neighbour
    edge_src, edge_dst = flashTP_e3nn.utils.fixed_generate_edgepair(total_node, max_neighbour)

    e3nn_config, cueq_config = flashTP_e3nn.utils.load_config_e3nn_cueq(filename, layer_idx, channel_mul=channel_multiplier)
    i_in1, i_in2, i_out, inst_tuple = e3nn_config

    torch.set_default_dtype(used_dtype)
    flashtp = flashTP_e3nn.uvu_TP(i_in1, i_in2, i_out, inst_tuple, device="cuda", dtype=used_dtype)

    # Get weight_numel from e3nn tp
    tp_ref = e3nn.o3.TensorProduct(i_in1, i_in2, i_out, inst_tuple, shared_weights=False, internal_weights=False)
    tp_ref = tp_ref.to(device="cuda")
    torch.set_default_dtype(torch.float32)

    batch_size = target_edge_cnt
    t_edge_src = torch.tensor(edge_src, device="cuda", dtype=torch.int32)
    t_edge_dst = torch.tensor(edge_dst, device="cuda", dtype=torch.int32)

    print(f"Config: {filename}, layer={layer_idx}, edges={target_edge_cnt}, dtype={used_dtype_str}")
    print(f"  i_in1={i_in1}, i_in2={i_in2}, i_out={i_out}")

    # ---- Forward timing ----
    def make_inputs():
        in1_node = torch.rand(total_node, i_in1.dim, device="cuda", requires_grad=True, dtype=used_dtype)
        in2 = torch.rand(batch_size, i_in2.dim, device="cuda", requires_grad=True, dtype=used_dtype)
        weight = torch.rand(batch_size, tp_ref.weight_numel, device="cuda", requires_grad=True, dtype=used_dtype)
        return in1_node, in2, weight

    # Forward timing
    in1_node, in2, weight = make_inputs()
    def fwd():
        out = flashtp(in1_node, in2, weight, t_edge_src, t_edge_dst)
        return out

    fwd_time = time_kernel(fwd)
    print(f"Forward latency: {fwd_time:.3f} ms")

    # Backward timing
    in1_node, in2, weight = make_inputs()
    out = flashtp(in1_node, in2, weight, t_edge_src, t_edge_dst)
    y = torch.nn.functional.gelu(out).sum()

    def bwd():
        grads = torch.autograd.grad(y, [in1_node, in2, weight], retain_graph=True)
        return grads

    bwd_time = time_kernel(bwd)
    print(f"Backward latency: {bwd_time:.3f} ms")

    # Double-backward timing
    in1_node, in2, weight = make_inputs()
    out = flashtp(in1_node, in2, weight, t_edge_src, t_edge_dst)
    y = torch.nn.functional.gelu(out).sum()
    f_in1, f_in2, f_weight = torch.autograd.grad(y, [in1_node, in2, weight], create_graph=True)
    fake_loss = (torch.nn.functional.gelu(f_in1).sum() +
                 torch.nn.functional.gelu(f_in2).sum() +
                 torch.nn.functional.gelu(f_weight).sum())

    def dbwd():
        grads = torch.autograd.grad(fake_loss, [in1_node, in2, weight], retain_graph=True)
        return grads

    dbwd_time = time_kernel(dbwd)
    print(f"Double-backward latency: {dbwd_time:.3f} ms")

    print(f"\n=== TIMING RESULTS ===")
    print(f"Forward: {fwd_time:.3f} ms")
    print(f"Backward: {bwd_time:.3f} ms")
    print(f"Double-backward: {dbwd_time:.3f} ms")


if __name__ == "__main__":
    main()
