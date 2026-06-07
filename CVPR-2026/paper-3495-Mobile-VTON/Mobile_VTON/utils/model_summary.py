import logging
import sys
import os.path as osp
import torch

# Setup logging
logger = logging.getLogger(__name__)

# Setup working directory
WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
logger.debug(f"Working directory: {WORK_DIR}")
if WORK_DIR not in sys.path:
    logger.warning(f"Working directory ({WORK_DIR}) is not in sys.path. Adding it.")
    sys.path.append(WORK_DIR)

try:
    from torchinfo import summary
except Exception as e:
    logger.error("torchinfo not working")
    logger.error(e)
    summary = None

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
except Exception as e:
    logger.error("fvcore not working")
    logger.error(e)
    FlopCountAnalysis = None
    parameter_count = None
    parameter_count_table = None

try:
    from thop import profile, clever_format
except Exception as e:
    logger.error("thop not working")
    logger.error(e)
    profile = None
    clever_format = None

try:
    from calflops import calculate_flops
except Exception as e:
    logger.error("calflops not working")
    logger.error(e)
    calculate_flops = None

try:
    from torch_flops import TorchFLOPsByFX
except Exception as e:
    logger.error("torch_flops not working")
    logger.error(e)
    TorchFLOPsByFX = None

try:
    from ptflops import get_model_complexity_info
except Exception as e:
    logger.error("ptflops not working")
    logger.error(e)
    get_model_complexity_info = None


def get_large_number_str(large_number, unit="B", base=1000, format_str=".3f"):
    if large_number is None:
        return str(None)
    large_number_K = large_number / base
    large_number_M = large_number_K / base
    large_number_B = large_number_M / base
    large_number_T = large_number_B / base
    if unit is None:
        unit = ""
    unit = unit.upper()
    if unit == "AUTO":
        if large_number_T > 1:
            unit = "T"
        elif large_number_B > 1:
            unit = "B"
        elif large_number_M > 1:
            unit = "M"
        elif large_number_K > 1:
            unit = "K"
        else:
            unit = ""

    show_dict = {
        "": large_number,
        "K": large_number_K,
        "M": large_number_M,
        "B": large_number_B,
        "G": large_number_B,
        "T": large_number_T,
    }
    large_number_str = f"{format(show_dict[unit], format_str)}"
    if unit != "":
        large_number_str += f" {unit}"
    return large_number_str


@torch.no_grad()
def calculate_model_params_with_manual_method(model, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: manual START =======")
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info("======= Summary =======")
        params_unit = unit_cfg.get("params_unit", "M")
        params_base = unit_cfg.get("params_base", 1000)
        format_str = unit_cfg.get("format_str", ".3f")
        total_params_str = get_large_number_str(total_params, unit=params_unit, base=params_base, format_str=format_str)
        logger.info(f"Model Params: {total_params_str}")
        logger.info("======= Method: manual  END  =======")
    return total_params


def print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg={}):
    params_unit = unit_cfg.get("params_unit", "M")
    params_base = unit_cfg.get("params_base", 1000)
    macs_unit = unit_cfg.get("macs_unit", "G")
    macs_base = unit_cfg.get("macs_base", 1000)
    flops_unit = unit_cfg.get("flops_unit", "G")
    flops_base = unit_cfg.get("flops_base", 1000)
    format_str = unit_cfg.get("format_str", ".3f")

    total_params_str = get_large_number_str(total_params, unit=params_unit, base=params_base, format_str=format_str)
    logger.info(f"Model Params: {total_params_str}")
    total_macs_str = get_large_number_str(total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    logger.info(f"Model MACs: {total_macs_str}")
    total_flops_str = get_large_number_str(total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    logger.info(f"Model FLOPs: {total_flops_str}")

@torch.no_grad()
def summary_model_with_torchinfo(model, input_size, device, verbose=False, verbose_level=1, unit_cfg={}):
    #! Not accurate
    # 0: quiet, 1: summary, 2: detailed summary
    if verbose:
        logger.info("======= Method: torchinfo START =======")
    m1_verbose = 0 if not verbose else 1
    res = summary(
        model, input_size=input_size, device=device,
        verbose=0 if not verbose else verbose_level,
    )
    total_params = res.total_params
    total_macs = None
    total_flops = None
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: torchinfo  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_fvcore(model, input_size, device, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: fvcore START =======")
        logger.info(parameter_count_table(model))
    res_dict = parameter_count(model)
    macs = FlopCountAnalysis(
        model,
        (torch.zeros(input_size).to(device),),
    )  # 虽然FlopCounter, 返回的是MACs的值
    total_params = res_dict[""]
    total_macs = macs.total()
    total_flops = None
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: fvcore  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_thop(model, input_size, device, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: thop START =======")
    total_macs, total_params = profile(model, inputs=(torch.zeros(input_size).to(device),), verbose=verbose)
    if verbose:
        macs, params = clever_format([total_macs, total_params], "%.3f")
        logger.info(f"Model Params:{params}, Model MACs: {macs}")
    total_flops = None
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: thop  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_manual_method(model, input_size, device, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: manual START =======")
    total_params = sum(p.numel() for p in model.parameters())
    total_macs = None
    total_flops = None
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: manual  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_calflops(model, input_size, device, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: calflops START =======")
    total_flops, total_macs, total_params = calculate_flops(model, input_shape=input_size, print_detailed=False, print_results=verbose, output_as_string=False)
    if verbose:
        logger.info(f"FLOPs: {total_flops}, MACs: {total_macs}, Params: {total_params}")
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: calflops  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_torch_flops(model, input_size, device, verbose=False, unit_cfg={}):
    if TorchFLOPsByFX is None:
        raise Exception("torch_flops not working")
    if verbose:
        logger.info("======= Method: torch_flops START =======")
    # Build the graph of the model. You can specify the operations (listed in `MODULE_FLOPs_MAPPING`, `FUNCTION_FLOPs_MAPPING` and `METHOD_FLOPs_MAPPING` in 'flops_ops.py') to ignore.
    flops_counter = TorchFLOPsByFX(model)
    if verbose:
        # Print the grath (not essential)
        logger.info('*' * 120)
        flops_counter.graph_model.graph.print_tabular()
    # Feed the input tensor
    flops_counter.propagate(torch.zeros(input_size).to(device))
    if verbose:
        # Print the full result table. It also returns the detailed result of each operation in a 2D list.
        result_table = flops_counter.print_result_table()
    # Print FLOPs, execution time and max GPU memory.
    total_params = None
    total_macs = None
    total_flops = flops_counter.print_total_flops(show=verbose)
    if verbose:
        total_time = flops_counter.print_total_time(show=verbose)
        max_memory = flops_counter.print_max_memory(show=verbose)
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: torch_flops  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_with_ptflops(model, input_size, device, verbose=False, unit_cfg={}):
    if verbose:
        logger.info("======= Method: ptflops START =======")
    total_macs, total_params = get_model_complexity_info(
        model, input_size[1:], as_strings=False, backend='aten',
        print_per_layer_stat=True, verbose=verbose
    )
    total_flops = None
    if verbose:
        logger.info("======= Summary =======")
        print_params_macs_and_flops(total_params, total_macs, total_flops, unit_cfg=unit_cfg)
        logger.info("======= Method: ptflops  END  =======")
    return total_params, total_macs, total_flops

@torch.no_grad()
def summary_model_params_macs_and_flops(model, input_size, device, verbose=False, unit_cfg={}):
    model = model.to(device)
    # * Method 1: torchinfo
    try:
        m1_total_params, m1_total_macs, m1_total_flops = summary_model_with_torchinfo(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 1 (TorchInfo) failed: {e}")
        m1_total_params, m1_total_macs, m1_total_flops = None, None, None
    # * Method 2: fvcore
    try:
        m2_total_params, m2_total_macs, m2_total_flops = summary_model_with_fvcore(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 2 (FVCore) failed: {e}")
        m2_total_params, m2_total_macs, m2_total_flops = None, None, None
    # * Method 3: thop
    try:
        m3_total_params, m3_total_macs, m3_total_flops = summary_model_with_thop(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 3 (Thop) failed: {e}")
        m3_total_params, m3_total_macs, m3_total_flops = None, None, None
    # * Method 4: manual
    try:
        m4_total_params, m4_total_macs, m4_total_flops = summary_model_with_manual_method(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 4 (Manual) failed: {e}")
        m4_total_params, m4_total_macs, m4_total_flops = None, None, None
    # * Method 5: calflops
    try:
        m5_total_params, m5_total_macs, m5_total_flops = summary_model_with_calflops(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 5 (Calflops) failed: {e}")
        m5_total_params, m5_total_macs, m5_total_flops = None, None, None
    # * Method 6: torch_flops
    try:
        m6_total_params, m6_total_macs, m6_total_flops = summary_model_with_torch_flops(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 6 (TorchFLOPs) failed: {e}")
        m6_total_params, m6_total_macs, m6_total_flops = None, None, None
    # * Method 7: ptflops
    try:
        m7_total_params, m7_total_macs, m7_total_flops = summary_model_with_ptflops(model, input_size, device, verbose=verbose, unit_cfg=unit_cfg)
    except Exception as e:
        logger.info(f"Method 7 (PTFlops) failed: {e}")
        m7_total_params, m7_total_macs, m7_total_flops = None, None, None

    params_unit = unit_cfg.get("params_unit", "M")
    params_base = unit_cfg.get("params_base", 1000)
    macs_unit = unit_cfg.get("macs_unit", "G")
    macs_base = unit_cfg.get("macs_base", 1000)
    flops_unit = unit_cfg.get("flops_unit", "G")
    flops_base = unit_cfg.get("flops_base", 1000)
    format_str = unit_cfg.get("format_str", ".3f")

    # * Summary
    m1_total_params_str = get_large_number_str(m1_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m2_total_params_str = get_large_number_str(m2_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m3_total_params_str = get_large_number_str(m3_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m4_total_params_str = get_large_number_str(m4_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m5_total_params_str = get_large_number_str(m5_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m6_total_params_str = get_large_number_str(m6_total_params, unit=params_unit, base=params_base, format_str=format_str)
    m7_total_params_str = get_large_number_str(m7_total_params, unit=params_unit, base=params_base, format_str=format_str)

    m1_total_macs_str = get_large_number_str(m1_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m2_total_macs_str = get_large_number_str(m2_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m3_total_macs_str = get_large_number_str(m3_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m4_total_macs_str = get_large_number_str(m4_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m5_total_macs_str = get_large_number_str(m5_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m6_total_macs_str = get_large_number_str(m6_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)
    m7_total_macs_str = get_large_number_str(m7_total_macs, unit=macs_unit, base=macs_base, format_str=format_str)

    m1_total_flops_str = get_large_number_str(m1_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m2_total_flops_str = get_large_number_str(m2_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m3_total_flops_str = get_large_number_str(m3_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m4_total_flops_str = get_large_number_str(m4_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m5_total_flops_str = get_large_number_str(m5_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m6_total_flops_str = get_large_number_str(m6_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)
    m7_total_flops_str = get_large_number_str(m7_total_flops, unit=flops_unit, base=flops_base, format_str=format_str)

    logger.info("======= Summary START =======")
    logger.info(f"Method 1  (TorchInfo): Params: {m1_total_params_str:<10}, MACs: {m1_total_macs_str:<10}, FLOPs: {m1_total_flops_str:<10}")
    logger.info(f"Method 2     (FVCore): Params: {m2_total_params_str:<10}, MACs: {m2_total_macs_str:<10}, FLOPs: {m2_total_flops_str:<10}")
    logger.info(f"Method 3       (Thop): Params: {m3_total_params_str:<10}, MACs: {m3_total_macs_str:<10}, FLOPs: {m3_total_flops_str:<10}")
    logger.info(f"Method 4     (Manual): Params: {m4_total_params_str:<10}, MACs: {m4_total_macs_str:<10}, FLOPs: {m4_total_flops_str:<10}")
    logger.info(f"Method 5   (Calflops): Params: {m5_total_params_str:<10}, MACs: {m5_total_macs_str:<10}, FLOPs: {m5_total_flops_str:<10}")
    logger.info(f"Method 6 (TorchFLOPs): Params: {m6_total_params_str:<10}, MACs: {m6_total_macs_str:<10}, FLOPs: {m6_total_flops_str:<10}")
    logger.info(f"Method 7    (PTFlops): Params: {m7_total_params_str:<10}, MACs: {m7_total_macs_str:<10}, FLOPs: {m7_total_flops_str:<10}")
    logger.info("======= Summary  END  =======")


if __name__ == "__main__":
    logger.warning("Running as a script. This is not recommended.")
