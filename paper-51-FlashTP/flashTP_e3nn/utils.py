import torch
import e3nn.o3
import json
import random

_cueq_ava = False
try:
    import cuequivariance as cue
    _cueq_ava = True
except ImportError:
    pass

def mul_Irreps(mul, i_in):
    dd = []
    for ori_mul, ir in i_in:
        dd.append((ori_mul*mul, (ir.l, ir.p)))
    return e3nn.o3.Irreps(dd)
def compare(a, b):
    isclose = torch.isclose(a, b)
    diff_pos = torch.argwhere(isclose == False)
    anything_bad = False
    for pos in diff_pos:
        pos_t = [x for x in pos]
        if(abs(a[tuple(pos_t)] - b[tuple(pos_t)]) > 1e-4):
            anything_bad = True
            print(pos)
            print(a[tuple(pos_t)] - b[tuple(pos_t)])
    if(not anything_bad):
        print("All Good")

IR_IN1_IDX = 0
IR_IN2_IDX = 1
IR_OUT_IDX = 2
INST_IDX = 3
WARPSIZE = 32

# def load_nequip_config(h, l_max, layer_idx):
#     filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
#     with open(filename, "r") as f:
#         f_in = f.read().split("\n")

#     per_layer_dict = dict()
#     for l_idx, d in enumerate(f_in):
#         if(d == "") : continue
#         dd = json.loads(d)
#         per_layer_dict[l_idx] = dd
#     tp_list = per_layer_dict[layer_idx]["tp"]
#     i_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
#     i_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
#     i_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
#     inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]

#     return i_in1, i_in2, i_out, inst_tuple

def load_config_e3nn_cueq(filename, layer_idx, channel_mul = 1, mul_list=None):
    
    # filename = f"/home2/lsy/mdsim/nequip/benchmark_config/4_{h}_{l_max}_p_sc.txt"
    with open(filename, "r") as f:
        f_in = f.read().split("\n")

    per_layer_dict = dict()
    for l_idx, d in enumerate(f_in):
        if(d == "") : continue
        dd = json.loads(d)
        per_layer_dict[l_idx] = dd
    tp_list = per_layer_dict[layer_idx]["tp"]

    ei_in1 = mul_Irreps(channel_mul, e3nn.o3.Irreps(tp_list[IR_IN1_IDX]))
    ei_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
    ei_out = mul_Irreps(channel_mul, e3nn.o3.Irreps(tp_list[IR_OUT_IDX]))
    inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]


    # changing mul for each ir.l
    new_in1_list = []
    new_out_list = []
    changed_idx = [[],[]]

    if(mul_list is not None):
        for idx, (mul,ir) in enumerate(ei_in1):
            if (ir.l in mul_list):
                new_in1_list.append((mul_list[ir.l], ir))
                for inst in inst_tuple:
                    if(idx == inst[0]):
                        changed_idx[0].append(inst[2])
                        changed_idx[1].append(mul_list[ir.l])
            else:
                new_in1_list.append((mul, ir))

        for idx, (mul,ir) in enumerate(ei_out):
            if (idx in changed_idx[0]):
                new_out_list.append((changed_idx[1][changed_idx[0].index(idx)], ir))
            else:
                new_out_list.append((mul, ir))

        ei_in1 = e3nn.o3.Irreps(new_in1_list)
        ei_out = e3nn.o3.Irreps(new_out_list)

    if not _cueq_ava:
        ci_in1 = ei_in1
        ci_in2 = ei_in2
        ci_out = ei_out
        print("cuEquivariance not installed, using e3nn Irreps directly for cueq config")
        # raise ImportError("cuEquivariance not installed")
    else:
        ci_in1 = cue.Irreps("O3", str(ei_in1))
        ci_in2 = cue.Irreps("O3", tp_list[IR_IN2_IDX])
        ci_out = cue.Irreps("O3", str(ei_out))


    return [ei_in1,ei_in2,ei_out,inst_tuple] , [ci_in1,ci_in2,ci_out,inst_tuple]


# def load_mace_config_e3nn_cueq(name, layer_idx):
#     filename = f"/home2/lsy/mdsim/mace/bench_config/{name}.json"
#     with open(filename, "r") as f:
#         f_in = f.read().split("\n")

#     per_layer_dict = dict()
#     for l_idx, d in enumerate(f_in):
#         if(d == "") : continue
#         dd = json.loads(d)
#         per_layer_dict[l_idx] = dd
#     tp_list = per_layer_dict[layer_idx]["tp"]

#     ei_in1 = e3nn.o3.Irreps(tp_list[IR_IN1_IDX])
#     ei_in2 = e3nn.o3.Irreps(tp_list[IR_IN2_IDX])
#     ei_out = e3nn.o3.Irreps(tp_list[IR_OUT_IDX])
#     inst_tuple = [tuple(x) for x in tp_list[INST_IDX]]


#     # changing mul for each ir.l
#     new_in1_list = []
#     new_out_list = []
#     changed_idx = [[],[]]
#     # mul_list = {}
#     mul_list = {0:128, 1:64}

#     for idx, (mul,ir) in enumerate(ei_in1):
#         if (ir.l in mul_list):
#             new_in1_list.append((mul_list[ir.l], ir))
#             for inst in inst_tuple:
#                 if(idx == inst[0]):
#                     changed_idx[0].append(inst[2])
#                     changed_idx[1].append(mul_list[ir.l])
#         else:
#             new_in1_list.append((mul, ir))

#     for idx, (mul,ir) in enumerate(ei_out):
#         if (idx in changed_idx[0]):
#             new_out_list.append((changed_idx[1][changed_idx[0].index(idx)], ir))
#         else:
#             new_out_list.append((mul, ir))

#     ei_in1 = e3nn.o3.Irreps(new_in1_list)
#     ei_out = e3nn.o3.Irreps(new_out_list)

#     ci_in1 = cue.Irreps("O3", str(ei_in1))
#     ci_in2 = cue.Irreps("O3", tp_list[IR_IN2_IDX])
#     ci_out = cue.Irreps("O3", str(ei_out))


#     return [ei_in1,ei_in2,ei_out,inst_tuple] , [ci_in1,ci_in2,ci_out,inst_tuple]


def generate_edgepair (num_node, max_neighbour):
    edge_src = []
    edge_dst = []
    for i in range(num_node):
        num_neighbour = random.randint(0,max_neighbour)
        for j in range(num_neighbour):
            edge_dst.append(i)
            src_idx = i
            while(src_idx == i):
                src_idx = random.randint(0,num_node-1)
            edge_src.append(src_idx)
    return edge_src, edge_dst 

def fixed_generate_edgepair (num_node, max_neighbour):
    edge_src = []
    edge_dst = []
    for i in range(num_node):
        num_neighbour = max_neighbour
        for j in range(num_neighbour):
            edge_dst.append(i)
            src_idx = i
            while(src_idx == i):
                src_idx = random.randint(0,num_node-1)
            edge_src.append(src_idx)
    return edge_src, edge_dst 
