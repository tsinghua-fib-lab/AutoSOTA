from ..quant_layers.conv import PTQSLQuantConv2d, ChannelwiseBatchingQuantConv2d
from ..quant_layers.linear import PTQSLBatchingQuantLinear, PostGeluPTQSLBatchingQuantLinear
from ..quant_layers.matmul import PTQSLBatchingQuantMatMul, SoSPTQSLBatchingQuantMatMul

no_softmax = False
no_postgelu = False

bit = 8 # the bit of quantization
conv_fc_name_list = ["qconv", "qlinear_qkv", "qlinear_proj", "qlinear_MLP_1", "qlinear_MLP_2", "qlinear_classifier", "qlinear_reduction"]
matmul_name_list = [ "qmatmul_qk", "qmatmul_scorev"]
w_bit = {name: bit for name in conv_fc_name_list}
a_bit = {name: bit for name in conv_fc_name_list}
A_bit = {name: bit for name in matmul_name_list}
B_bit = {name: bit for name in matmul_name_list}

ptqsl_conv2d_kwargs = {
    "metric": "hessian",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    'search_round': 3,
    "n_V": 1,
    "n_H": 1,
}
ptqsl_linear_kwargs = {
    "metric": "hessian",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    'search_round': 3,
    "n_V": 1,
    "n_H": 1,
    "n_a": 1,
    "bias_correction":True # Conventionally I'll not add an actual bias correction in linear
}
ptqsl_matmul_kwargs = {
    "metric": "hessian",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    'search_round': 3,
    "n_G_A": 1,
    "n_V_A": 1,
    "n_H_A": 1,
    "n_G_B": 1,
    "n_V_B": 1,
    "n_H_B": 1,
}


def get_module(module_type, *args, **kwargs):
    if module_type == "qconv":
        kwargs.update(ptqsl_conv2d_kwargs)
        module=ChannelwiseBatchingQuantConv2d(*args,**kwargs,w_bit=w_bit["qconv"],a_bit=32) # turn off activation quantization
        # module=PTQSLQuantConv2d(*args,**kwargs,w_bit=w_bit["qconv"],a_bit=32) # turn off activation quantization
    elif "qlinear" in module_type:
        kwargs.update(ptqsl_linear_kwargs)
        if module_type == "qlinear_qkv":
            kwargs["n_V"] *= 3  # q, k, v
            module=PTQSLBatchingQuantLinear(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type])
        elif module_type == "qlinear_MLP_2":
            if no_postgelu:
                module=PTQSLBatchingQuantLinear(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type])
            else:
                module=PostGeluPTQSLBatchingQuantLinear(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type])
        elif module_type == "qlinear_classifier":
            kwargs["n_V"] = 1
            module=PTQSLBatchingQuantLinear(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type])
        else:
            module=PTQSLBatchingQuantLinear(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type])
    elif "qmatmul" in module_type:
        kwargs.update(ptqsl_matmul_kwargs)
        if module_type == "qmatmul_qk":
            module=PTQSLBatchingQuantMatMul(*args,**kwargs,A_bit=A_bit[module_type],B_bit=B_bit[module_type])
        elif module_type == "qmatmul_scorev":
            if no_softmax:
                module=PTQSLBatchingQuantMatMul(*args,**kwargs,A_bit=A_bit[module_type],B_bit=B_bit[module_type])
            else:
                module=SoSPTQSLBatchingQuantMatMul(*args,**kwargs,A_bit=A_bit[module_type],B_bit=B_bit[module_type])
    return module