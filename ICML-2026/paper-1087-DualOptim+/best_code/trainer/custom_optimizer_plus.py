from typing import Optional

import numpy as np
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from sympy.physics.quantum.gate import normalized


class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


def adam_decouple_plus_step(
        g, p, state1=None, beta1=0.9, state2=None, beta2=0.999, state1_base=None, beta1_base=0.9, another_beta1_base=0.9,
        state2_base=None, beta2_base=0.95, another_beta2_base=0.95, eps=1e-8, step=1, base_step=1, lr=1e-4, weight_decay=0., enable_base=True
):
    # weight decay
    if weight_decay != 0.:
        p.mul_(1 - lr * weight_decay)

    # update base state
    another_step = base_step - step
    if state1_base is not None:
        # if v5, comment out
        # state1_base.mul_(beta1_base).add_(g, alpha=1 - beta1_base)
        # v4
        # state1_normalized = state1 / (1 - beta1 ** step)
        # state1_base.mul_(beta1_base).add_(g - state1_normalized, alpha=1 - beta1_base)

        # state1_base_normalized = state1_base / (1 - beta1_base ** base_step)
        # v5
        state1_base_normalized = state1_base / (1 - beta1_base ** (base_step-1)) if base_step > 1 else state1_base
        # v5.1
        # state1_base_normalized = state1_base / (1 - beta1_base ** (step - 1) * another_beta1_base ** another_step) if base_step > 1 else state1_base

    if state2_base is not None:
        # if v5, comment out
        # state2_base.mul_(beta2_base).addcmul_(g, g, value=1 - beta2_base)
        # v4
        # state2_normalized = state2 / (1 - beta2 ** step)
        # state2_base.mul_(beta2_base).add_(torch.square(g) - state2_normalized, alpha=1 - beta2_base)

        # state2_base_normalized = state2_base / (1 - beta2_base ** base_step)
        # v5
        state2_base_normalized = state2_base / (1 - beta2_base ** (base_step-1)) if base_step > 1 else state2_base
        # v5.1
        # state2_base_normalized = state2_base / (1 - beta2_base ** (step - 1) * another_beta2_base ** another_step) if base_step > 1 else state2_base

    # update delta state
    # v2, if v7 comment out
    if state1_base is not None and enable_base:
        state1.mul_(beta1).add_(g - state1_base_normalized, alpha=1 - beta1)
    else:
        state1.mul_(beta1).add_(g, alpha=1 - beta1)
    if state2_base is not None and enable_base:
        state2.mul_(beta2).add_(torch.square(g) - state2_base_normalized, alpha=1 - beta2)
        # v6
        # state2.mul_(beta2).add_(torch.square(g - state1_base_normalized), alpha=1 - beta2)
    else:
        state2.mul_(beta2).add_(torch.square(g), alpha=1 - beta2)

    # simulate single Adam
    # state1.copy_(state1_base)
    # state2.copy_(state2_base)

    # correct bias
    state1_normalized = state1 / (1 - beta1 ** step)
    state2_normalized = state2 / (1 - beta2 ** step)


    exp_avg = state1_normalized + state1_base_normalized if state1_base is not None and enable_base else state1_normalized
    denom = (state2_normalized + state2_base_normalized).abs().sqrt().add_(eps) if state2_base is not None and enable_base else state2_normalized.sqrt().add_(eps)

    # simulate single Adam
    # exp_avg = state1_normalized
    # denom = state2_normalized.sqrt().add_(eps)

    p.data.addcdiv_(exp_avg, denom, value=-lr)

    # v5
    if state1_base is not None:
        state1_base.mul_(beta1_base).add_(g, alpha=1 - beta1_base)
    if state2_base is not None:
        state2_base.mul_(beta2_base).addcmul_(g, g, value=1 - beta2_base)
    # v7
    # state1.mul_(beta1).add_(g - state1_base_normalized, alpha=1 - beta1)
    # state2.mul_(beta2).add_(torch.square(g) - state2_base_normalized, alpha=1 - beta2)


def adam_decouple_plus_step_8bit_blockwise(
        g, p, state1=None, beta1=0.9, state2=None, beta2=0.999,  state1_base=None, beta1_base=0.9, another_beta1_base=0.9,
        state2_base=None, beta2_base=0.95, another_beta2_base=0.95, eps=1e-8, step=1, base_step=1, lr=1e-4, weight_decay=0., enable_base=True,
        qmap1=None, qmap2=None, qmap1_base=None, qmap2_base=None, absmax1=None, absmax2=None, absmax1_base=None, absmax2_base=None, blocksize=256
):

    # weight decay
    if weight_decay != 0.:
        p.mul_(1 - lr * weight_decay)

    # dequantize state
    if state1_base is not None:
        if qmap1_base is not None and absmax1_base is not None:
            state1_base_fp32 = F.dequantize_blockwise(state1_base, code=qmap1_base, absmax=absmax1_base, blocksize=blocksize)
        else:
            state1_base_fp32 = state1_base
    if state2_base is not None:
        if qmap2_base is not None and absmax2_base is not None:
            state2_base_fp32 = F.dequantize_blockwise(state2_base, code=qmap2_base, absmax=absmax2_base, blocksize=blocksize)
        else:
            state2_base_fp32 = state2_base
    if state1 is not None:
        if qmap1 is not None and absmax1 is not None:
            state1_fp32 = F.dequantize_blockwise(state1, code=qmap1, absmax=absmax1, blocksize=blocksize)
        else:
            state1_fp32 = state1
    if state2 is not None:
        if qmap2 is not None and absmax2 is not None:
            state2_fp32 = F.dequantize_blockwise(state2, code=qmap2, absmax=absmax2, blocksize=blocksize)
        else:
            state2_fp32 = state2

    # update base state
    another_step = base_step - step
    if state1_base is not None:
        # if v5, comment out
        # state1_base_fp32.mul_(beta1_base).add_(g, alpha=1 - beta1_base)
        # v4
        # state1_fp32_normalized = state1_fp32 / (1 - beta1 ** step)
        # state1_base_fp32.mul_(beta1_base).add_(g - state1_fp32_normalized, alpha=1 - beta1_base)

        # state1_base_fp32_normalized = state1_base_fp32 / (1 - beta1_base ** base_step)
        # v5
        state1_base_fp32_normalized = state1_base_fp32 / (1 - beta1_base ** (base_step-1)) if base_step > 1 else state1_base_fp32
        # v5.1
        # state1_base_fp32_normalized = state1_base_fp32 / (1 - beta1_base ** (step-1) * another_beta1_base ** another_step) if base_step > 1 else state1_base_fp32
    if state2_base is not None:
        # if v5, comment out
        # state2_base_fp32.mul_(beta2_base).addcmul_(g, g, value=1 - beta2_base)
        # v4
        # state2_fp32_normalized = state2_fp32 / (1 - beta2 ** step)
        # state2_base_fp32.mul_(beta2_base).add_(torch.square(g) - state2_fp32_normalized, alpha=1 - beta2_base)
        # state2_base_fp32_normalized = state2_base_fp32 / (1 - beta2_base ** base_step)
        # v5
        state2_base_fp32_normalized = state2_base_fp32 / (1 - beta2_base ** (base_step-1)) if base_step > 1 else state2_base_fp32
        # v5.1
        # state2_base_fp32_normalized = state2_base_fp32 / (1 - beta2_base ** (step-1) * another_beta2_base ** another_step) if base_step > 1 else state2_base_fp32

    # update delta state
    # v2, if v7 comment out
    if state1_base is not None and enable_base:
        state1_fp32.mul_(beta1).add_(g - state1_base_fp32_normalized, alpha=1 - beta1)
    else:
        state1_fp32.mul_(beta1).add_(g, alpha=1 - beta1)
    if state2_base is not None and enable_base:
        state2_fp32.mul_(beta2).add_(torch.square(g) - state2_base_fp32_normalized, alpha=1 - beta2)
    else:
        state2_fp32.mul_(beta2).add_(torch.square(g), alpha=1 - beta2)


    # correct bias
    state1_fp32_normalized = state1_fp32 / (1 - beta1 ** step)
    state2_fp32_normalized = state2_fp32 / (1 - beta2 ** step)

    exp_avg = state1_fp32_normalized + state1_base_fp32_normalized if state1_base is not None and enable_base else state1_fp32_normalized
    denom = (state2_fp32_normalized + state2_base_fp32_normalized).abs().sqrt().add_(eps) if state2_base is not None and enable_base else state2_fp32_normalized.sqrt().add_(eps)

    p.data.addcdiv_(exp_avg, denom, value=-lr)

    # v5
    if state1_base is not None:
        state1_base_fp32.mul_(beta1_base).add_(g, alpha=1 - beta1_base)
    if state2_base is not None:
        state2_base_fp32.mul_(beta2_base).addcmul_(g, g, value=1 - beta2_base)

    # v7
    # state1_fp32.mul_(beta1).add_(g - state1_base_fp32_normalized, alpha=1 - beta1)
    # state2_fp32.mul_(beta2).add_(torch.square(g) - state2_base_fp32_normalized, alpha=1 - beta2)

    # quantize state
    if state1_base is not None and qmap1_base is not None and absmax1_base is not None:
        state1_base, _ = F.quantize_blockwise(state1_base_fp32, code=qmap1_base, absmax=absmax1_base, out=state1_base, blocksize=blocksize)
    if state2_base is not None and qmap2_base is not None and absmax2_base is not None:
        state2_base, _ = F.quantize_blockwise(state2_base_fp32, code=qmap2_base, absmax=absmax2_base, out=state2_base, blocksize=blocksize)
    if state1 is not None and qmap1 is not None and absmax1 is not None:
        state1, _ = F.quantize_blockwise(state1_fp32, code=qmap1, absmax=absmax1, out=state1, blocksize=blocksize)
    if state2 is not None and qmap2 is not None and absmax2 is not None:
        state2, _ = F.quantize_blockwise(state2_fp32, code=qmap2, absmax=absmax2, out=state2, blocksize=blocksize)


class Optimizer2StateDecouplePlus(bnb.optim.optimizer.Optimizer8bit):
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.95, 0.95, 0.999),
        alpha=1.0,
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
        decouple_m=True,
        decouple_v=True,
        quantize_delta=False,
        quantize_base=False,
        lr_ratio_1=1.0,
        lr_ratio_2=1.0,
        switch_freq_1=1,
        switch_freq_2=1
    ):
        """
        Base 2-state update optimizer with decoupled momentum class.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.999)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
            decouple_m(`bool`, defaults to `False`):
                Whether to decouple the momentum.
            decouple_v(`bool`, defaults to `False`):
                Whether to decouple the velocity.
            lr_ratio_1 (`float`, defaults to `1.0`):
                The learning rate ratio for the forget part.
            lr_ratio_2 (`float`, defaults to `1.0`):
                The learning rate ratio for the retain part.
            alpha (`float`, defaults to 0.0):
                The alpha value for the fusing base and delta.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if isinstance(betas, str):
            # format: '(beta1, beta2)'
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if optimizer_name != "adam":
            raise ValueError(f"Unsupported optimizer name: {optimizer_name}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lr_ratio_1=lr_ratio_1,
            lr_ratio_2=lr_ratio_2,
            alpha=alpha
        )

        super().__init__(params, defaults, optim_bits, is_paged)

        self.non_castable_tensor_keys = {
            "qmap1",
            "qmap2",
            "qmap1_1",
            "qmap2_1",
            "qmap1_2",
            "qmap2_2",
            "qmap1_base",
            "qmap2_base",
            "max1",
            "max2",
            "max1_1",
            "max2_1",
            "max1_2",
            "max2_2",
            "max1_base",
            "max2_base",
            "new_max1",
            "new_max2",
            "new_max1_1",
            "new_max2_1",
            "new_max1_2",
            "new_max2_2",
            "new_max1_base",
            "new_max2_base",
            "state1",
            "state2",
            "state1_1",
            "state2_1",
            "state1_2",
            "state2_2",
            "state1_base",
            "state2_base",
            "gnorm_vec",
            "gnorm_vec1",
            "gnorm_vec2",
            "gnorm_vec_base",
            "absmax1",
            "absmax2",
            "absmax1_1",
            "absmax2_1",
            "absmax1_2",
            "absmax2_2",
            "absmax1_base",
            "absmax2_base",
            "unorm_vec",
            "unorm_vec1",
            "unorm_vec2",
            "unorm_vec_base",
        }
        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros
            args["decouple_m"] = decouple_m
            args["decouple_v"] = decouple_v
            args["switch_freq_1"] = switch_freq_1
            args["switch_freq_2"] = switch_freq_2
            args["quantize_delta"] = quantize_delta
            args["quantize_base"] = quantize_base
            self.args = MockArgs(args)
        else:
            self.args = args

        if optim_bits == 8:
            if self.args.decouple_m:
                self.name2qmap["dynamic1_1"] = F.create_dynamic_map(signed=True)
                self.name2qmap["dynamic1_2"] = F.create_dynamic_map(signed=True)
                self.name2qmap["dynamic_base"] = F.create_dynamic_map(signed=True)
            else:
                self.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)

            if self.args.decouple_v:
                self.name2qmap["dynamic2_1"] = F.create_dynamic_map(signed=True)
                self.name2qmap["dynamic2_2"] = F.create_dynamic_map(signed=True)
                self.name2qmap["udynamic_base"] = F.create_dynamic_map(signed=False)
                # v4, v6
                # self.name2qmap["dynamic_base"] = F.create_dynamic_map(signed=True)
            else:
                self.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)

        self.optimizer_name = optimizer_name

    def get_config(self, gindex, pindex, group):
        config = {}
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["lr_ratio_1"] = group["lr_ratio_1"]
        config["lr_ratio_2"] = group["lr_ratio_2"]
        config["alpha"] = group.get("alpha")
        config["t_alpha"] = group.get("t_alpha")
        config["t_beta3"] = group.get("t_beta3")
        config["optim_bits"] = self.args.optim_bits
        config["min_8bit_size"] = self.args.min_8bit_size
        config["percentile_clipping"] = self.args.percentile_clipping
        config["block_wise"] = self.args.block_wise
        config["max_unorm"] = self.args.max_unorm
        config["skip_zeros"] = self.args.skip_zeros
        config["decouple_m"] = self.args.decouple_m
        config["decouple_v"] = self.args.decouple_v
        config["switch_freq_1"] = self.args.switch_freq_1
        config["switch_freq_2"] = self.args.switch_freq_2
        config["quantize_delta"] = self.args.quantize_delta
        config["quantize_base"] = self.args.quantize_base

        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])
        return config

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(
                f'Amount of optimizer bits not supported: {config["optim_bits"]}'
            )

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0
        state["step1"] = 0
        state["step2"] = 0

        if dtype == torch.float32:
            if config["decouple_m"]:
                state["state1_1"] = self.get_state_buffer(p, dtype=torch.float32)
                state["state1_2"] = self.get_state_buffer(p, dtype=torch.float32)
                state["state1_base"] = self.get_state_buffer(p, dtype=torch.float32)
            else:
                state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            if config["decouple_v"]:
                state["state2_1"] = self.get_state_buffer(p, dtype=torch.float32)
                state["state2_2"] = self.get_state_buffer(p, dtype=torch.float32)
                state["state2_base"] = self.get_state_buffer(p, dtype=torch.float32)
            else:
                state["state2"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                if config["decouple_m"]:
                    if config["quantize_delta"]:
                        self.name2qmap["dynamic1_1"] = self.name2qmap["dynamic1_1"].to(p.device)
                        self.name2qmap["dynamic1_2"] = self.name2qmap["dynamic1_2"].to(p.device)
                    if config["quantize_base"]:
                        self.name2qmap["dynamic_base"] = self.name2qmap["dynamic_base"].to(p.device)
                else:
                    self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                if config["decouple_v"]:
                    if config["quantize_delta"]:
                        # delta state 2 could be negative
                        self.name2qmap["dynamic2_1"] = self.name2qmap["dynamic2_1"].to(p.device)
                        self.name2qmap["dynamic2_2"] = self.name2qmap["dynamic2_2"].to(p.device)
                    if config["quantize_base"]:
                        self.name2qmap["udynamic_base"] = self.name2qmap["udynamic_base"].to(p.device)
                        # v4, v6
                        # self.name2qmap["dynamic_base"] = self.name2qmap["dynamic_base"].to(p.device)
                else:
                    self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            if config["decouple_m"]:
                if config["quantize_delta"]:
                    state["state1_1"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap1_1"] = self.name2qmap["dynamic1_1"]
                    state["state1_2"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap1_2"] = self.name2qmap["dynamic1_2"]
                else:
                    state["state1_1"] = self.get_state_buffer(p, dtype=torch.float32)
                    state["state1_2"] = self.get_state_buffer(p, dtype=torch.float32)

                if config["quantize_base"]:
                    state["state1_base"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap1_base"] = self.name2qmap["dynamic_base"]
                else:
                    state["state1_base"] = self.get_state_buffer(p, dtype=torch.float32)
            else:
                state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap1"] = self.name2qmap["dynamic"]

            if config["decouple_v"]:
                if config["quantize_delta"]:
                    state["state2_1"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap2_1"] = self.name2qmap["dynamic2_1"]
                    state["state2_2"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap2_2"] = self.name2qmap["dynamic2_2"]
                else:
                    state["state2_1"] = self.get_state_buffer(p, dtype=torch.float32)
                    state["state2_2"] = self.get_state_buffer(p, dtype=torch.float32)

                if config["quantize_base"]:
                    state["state2_base"] = self.get_state_buffer(p, dtype=torch.uint8)
                    state["qmap2_base"] = self.name2qmap["udynamic_base"]
                    # v4, v6
                    # state["qmap2_base"] = self.name2qmap["dynamic_base"]
                else:
                    state["state2_base"] = self.get_state_buffer(p, dtype=torch.float32)
            else:
                state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap2"] = self.name2qmap["udynamic"]

            if config["block_wise"]:
                blocksize = 256
                n = p.numel()
                blocks = (n // blocksize) + bool(n % blocksize)

                if config["decouple_m"]:
                    if config["quantize_delta"]:
                        state["absmax1_1"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                        state["absmax1_2"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                    if config["quantize_base"]:
                        state["absmax1_base"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                else:
                    state["absmax1"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )

                if config["decouple_v"]:
                    if config["quantize_delta"]:
                        state["absmax2_1"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                        state["absmax2_2"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                    if config["quantize_base"]:
                        state["absmax2_base"] = torch.zeros(
                            (blocks,), dtype=torch.float32, device=p.device
                        )
                else:
                    state["absmax2"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
            else:
                if config["decouple_m"]:
                    if config["quantize_delta"]:
                        state["max1_1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max1_1"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )
                        state["max1_2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max1_2"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )
                    if config["quantize_base"]:
                        state["max1_base"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max1_base"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )
                else:
                    state["max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max1"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )

                if config["decouple_m"]:
                    if config["quantize_delta"]:
                        state["max2_1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max2_1"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )
                        state["max2_2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max2_2"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )

                    if config["quantize_base"]:
                        state["max2_base"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                        state["new_max2_base"] = torch.zeros(
                            (1,), dtype=torch.float32, device=p.device
                        )
                else:
                    state["max2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max2"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )

        if config["percentile_clipping"] < 100:
            if config["decouple_m"] or config["decouple_v"]:
                state["gnorm_vec1"] = torch.zeros((100,), device=p.device)
                state["gnorm_vec2"] = torch.zeros((100,), device=p.device)
                # state["gnorm_vec_base"] = torch.zeros((100,), device=p.device)
            else:
                state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        if config["max_unorm"] > 0.0:
            if config["decouple_m"] or config["decouple_v"]:
                state["unorm_vec1"] = torch.zeros((1,), device=p.device)
                state["unorm_vec2"] = torch.zeros((1,), device=p.device)
                # state["unorm_vec_base"] = torch.zeros((1,), device=p.device)
            else:
                state["unorm_vec"] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        if (
            state["step"] % (config["switch_freq_1"] + config["switch_freq_2"]) < config["switch_freq_1"]
        ):  # switch between mode 1 (forget) and mode 2 (retain)
            mode = 1
        else:
            mode = 2

        is_dual = config["decouple_m"] or config["decouple_v"]
        ratio = config["lr_ratio_1"] if mode == 1 else config["lr_ratio_2"]

        if config["quantize_base"]:
            dtype = (state["state1_base"].dtype
                     if config["decouple_m"]
                     else state["state2_base"].dtype
                     if config["decouple_v"]
                     else state["state1"].dtype)
        else:
            dtype = (
                state["state1"].dtype
                if not config["decouple_m"]
                else state["state1_1"].dtype
                if mode == 1
            else state["state1_2"].dtype
        )
        state["step"] += 1
        base_step = state["step"]
        if is_dual:
            if mode == 1:
                state["step1"] += 1
                step = state["step1"]
            else:
                state["step2"] += 1
                step = state["step2"]
        else:
            step = state["step"]

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"]
                if not is_dual
                else state["gnorm_vec1"]
                if mode == 1
                else state["gnorm_vec2"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if dtype == torch.float32:
            adam_decouple_plus_step(
                g=grad,
                p=p,
                state1=state["state1"]
                    if not config["decouple_m"]
                    else state["state1_1"]
                    if mode == 1
                    else state["state1_2"],
                beta1=config["betas"][0],
                state2=state["state2"]
                    if not config["decouple_v"]
                    else state["state2_1"]
                    if mode == 1
                    else state["state2_2"],
                beta2=config["betas"][1],
                state1_base=state["state1_base"] if config["decouple_m"] else None,
                beta1_base=config["betas"][2],
                # beta1_base=config["betas"][2] if mode == 1 else config["betas"][2]**(config["switch_freq_1"]/config["switch_freq_2"]),  # v5.1
                # another_beta1_base=config["betas"][2] if mode == 2 else config["betas"][2]**(config["switch_freq_1"] / config["switch_freq_2"]),  # v5.1
                state2_base=state["state2_base"] if config["decouple_v"] else None,
                beta2_base=config["betas"][3],
                # beta2_base=config["betas"][3] if mode == 1 else config["betas"][3]**(config["switch_freq_1"]/config["switch_freq_2"]),  # v5.1
                # another_beta2_base=config["betas"][3] if mode == 2 else config["betas"][3]**(config["switch_freq_1"]/config["switch_freq_2"]),  # v5.1
                eps=config["eps"],
                step=step,
                base_step=base_step,
                lr=ratio * config["lr"],
                weight_decay=config["weight_decay"],
                # enable_base=mode == 1,  # v8
            )

            # simulate single Adam
            # state1_f = state["state1_1"] / (1 - config["betas"][0] ** state["step1"])
            # state1_r = state["state1_2"] / (1 - config["betas"][0] ** state["step2"])
            #
            # state2_f = state["state2_1"] / (1 - config["betas"][1] ** state["step1"])
            # state2_r = state["state2_2"] / (1 - config["betas"][1] ** state["step2"])
            #
            # state_f = state1_f / (state2_f.sqrt() + config["eps"])
            # state_r = state1_r / (state2_r.sqrt() + config["eps"])
            #
            # cos_sim_f_r_m = torch.cosine_similarity(state_f, state_r, dim=0)
            # print(f"Step:{state['step']}, param shape: {p.numel()}, gindex: {gindex}, pindex: {pindex},"
            #       f"F-R sim: {cos_sim_f_r_m}")

            # state_f = state["state1_1"] / (1 - config["betas"][0] ** state["step1"])
            # state_r = state["state1_2"] / (1 - config["betas"][0] ** state["step2"])
            # state_base = state["state1_base"] / (1 - config["betas"][2] ** base_step)
            #
            # update1_f = state_f + config["alpha"] * state_base
            # update1_r = state_r + config["alpha"] * state_base
            #
            # state2_f = state["state2_1"] / (1 - config["betas"][1] ** state["step1"])
            # state2_r = state["state2_2"] / (1 - config["betas"][1] ** state["step2"])
            # state2_base = state["state2_base"] / (1 - config["betas"][3] ** base_step)
            #
            # update2_f = (state2_f + config["alpha"] * state2_base).abs().sqrt() + config["eps"]
            # update2_r = (state2_r + config["alpha"] * state2_base).abs().sqrt() + config["eps"]
            #
            # update_f = update1_f / update2_f
            # update_r = update1_r / update2_r


            # cos_sim_df_base_m = torch.cosine_similarity(state_f, state_base, dim=0)
            # cos_sim_dr_base_m = torch.cosine_similarity(state_r, state_base, dim=0)
            # cos_sim_df_dr_m = torch.cosine_similarity(state_f, state_r, dim=0)
            #
            # cos_sim_f_base_m = torch.cosine_similarity(update1_f, state_base, dim=0)
            # cos_sim_r_base_m = torch.cosine_similarity(update1_r, state_base, dim=0)
            # cos_sim_f_r_m = torch.cosine_similarity(update1_f, update1_r, dim=0)
            #
            # cos_sim_df_f_m = torch.cosine_similarity(state_f, update1_f, dim=0)
            # cos_sim_dr_r_m = torch.cosine_similarity(state_r, update1_r, dim=0)
            #
            # print(f"Step:{base_step}, param shape: {p.numel()}, gindex: {gindex}, pindex: {pindex},"
            #       f"dF-Base sim: {cos_sim_df_base_m}, dR-Base sim: {cos_sim_dr_base_m}, dF-dR sim: {cos_sim_df_dr_m},",
            #       f"F-Base sim: {cos_sim_f_base_m}, R-Base sim: {cos_sim_r_base_m}, F-R sim: {cos_sim_f_r_m},",
            #       f"dF-F sim: {cos_sim_df_f_m}, dR-R sim: {cos_sim_dr_r_m}")

            #
            # cos_sim_update_f_base_m = torch.cosine_similarity(update_f, state_base, dim=0)
            # cos_sim_update_r_base_m = torch.cosine_similarity(update_r, state_base, dim=0)
            # cos_sim_update_f_r_m = torch.cosine_similarity(update_f, update_r, dim=0)
            # print(f"Step:{base_step}, param shape: {p.numel()}, gindex: {gindex}, pindex: {pindex},"
            #       f"Update_F-Update_R sim: {cos_sim_update_f_r_m}"
            #       )
            # print(f"Step:{base_step}, param shape: {p.numel()}, gindex: {gindex}, pindex: {pindex},"
            #       f"Delta_F norm: {state_f.norm()}, Delta_R norm: {state_r.norm()}, Base norm: {state_base.norm()}, "
            #       f"Update_F norm: {update_f.norm()}, Update_R norm: {update_r.norm()}, "
            #       f"Delta_F-base sim: {cos_sim_f_base_m}, Delta_R-base sim: {cos_sim_r_base_m}, Delta_F-delta_R sim: {cos_sim_f_r_m}, "
            #       f"Update_F-base sim: {cos_sim_update_f_base_m}, Update_R-base sim: {cos_sim_update_r_base_m}, Update_F-Update_R sim: {cos_sim_update_f_r_m}"
            # )
            # print(f"m update-base sim: {cos_sim_update_base_m}, v update-base sim: {cos_sim_update_base_v}")
            # print(f"m update-delta sim: {cos_sim_update_delta_m}, v update-delta sim: {cos_sim_update_delta_v}")

        elif dtype == torch.uint8 and not config["block_wise"]:
            raise NotImplementedError("8-bit AdamWDecouplePlus without block-wise quantization is not implemented")

        elif dtype == torch.uint8 and config["block_wise"]:
            adam_decouple_plus_step_8bit_blockwise(
                g=grad,
                p=p,
                state1=state["state1"]
                    if not config["decouple_m"]
                    else state["state1_1"]
                    if mode == 1
                    else state["state1_2"],
                beta1=config["betas"][0],
                state2=state["state2"]
                    if not config["decouple_v"]
                    else state["state2_1"]
                    if mode == 1
                    else state["state2_2"],
                beta2=config["betas"][1],
                state1_base=state["state1_base"] if config["decouple_m"] else None,
                beta1_base=config["betas"][2],
                # beta1_base=config["betas"][2] if mode == 1 else config["betas"][2]**(config["switch_freq_1"] / config["switch_freq_2"]),  # v5.1
                # another_beta1_base=config["betas"][2] if mode == 2 else config["betas"][2]**(config["switch_freq_1"] / config["switch_freq_2"]),  # v5.1
                state2_base=state["state2_base"] if config["decouple_v"] else None,
                beta2_base=config["betas"][3],
                # beta2_base=config["betas"][3] if mode == 1 else config["betas"][3]**(config["switch_freq_1"]/config["switch_freq_2"]),
                # another_beta2_base=config["betas"][3] if mode == 2 else config["betas"][3]**(config["switch_freq_1"]/config["switch_freq_2"]),
                eps=config["eps"],
                step=step,
                base_step=base_step,
                lr=ratio * config["lr"],
                weight_decay=config["weight_decay"],
                qmap1=state["qmap1"]
                    if not config["decouple_m"]
                    else state["qmap1_1"]
                    if mode == 1 and config["quantize_delta"]
                    else state["qmap1_2"]
                    if mode == 2 and config["quantize_delta"]
                    else None,
                qmap2=state["qmap2"]
                    if not config["decouple_v"]
                    else state["qmap2_1"]
                    if mode == 1 and config["quantize_delta"]
                    else state["qmap2_2"]
                    if mode == 2 and config["quantize_delta"]
                    else None,
                absmax1=state["absmax1"]
                    if not config["decouple_m"]
                    else state["absmax1_1"]
                    if mode == 1 and config["quantize_delta"]
                    else state["absmax1_2"]
                    if mode == 2 and config["quantize_delta"]
                    else None,
                absmax2=state["absmax2"]
                    if not config["decouple_v"]
                    else state["absmax2_1"]
                    if mode == 1 and config["quantize_delta"]
                    else state["absmax2_2"]
                    if mode == 2 and config["quantize_delta"]
                    else None,
                qmap1_base=state["qmap1_base"]
                    if config["decouple_m"] and config["quantize_base"]
                    else None,
                qmap2_base=state["qmap2_base"]
                    if config["decouple_v"] and config["quantize_base"]
                else None,
                absmax1_base=state["absmax1_base"]
                    if config["decouple_m"] and config["quantize_base"]
                    else None,
                absmax2_base=state["absmax2_base"]
                    if config["decouple_v"] and config["quantize_base"]
                    else None,
                # enable_base=mode == 1,  # v8
            )

        # assert torch.isnan(p.data).sum() == 0, f"NaN in parameter with shape {p.shape}"

    def prefetch_state(self, p):
        if self.is_paged:
            state = self.state[p]
            s1 = state["state1"] if not self.args.decouple_m else state["state1_1"]

            is_paged = getattr(s1, "is_paged", False)
            if is_paged:
                if "state1" in state:
                    F.prefetch_tensor(state["state1"])
                if "state2" in state:
                    F.prefetch_tensor(state["state2"])
                if "state1_1" in state:
                    F.prefetch_tensor(state["state1_1"])
                if "state2_1" in state:
                    F.prefetch_tensor(state["state2_1"])
                if "state1_2" in state:
                    F.prefetch_tensor(state["state1_2"])
                if "state2_2" in state:
                    F.prefetch_tensor(state["state2_2"])
                if "state1_base" in state:
                    F.prefetch_tensor(state["state1_base"])
                if "state2_base" in state:
                    F.prefetch_tensor(state["state2_base"])


class AdamWDecouplePlus(Optimizer2StateDecouplePlus):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95, 0.9, 0.95),
        alpha=1.0,
        eps=1e-8,
        weight_decay=1e-2,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        decouple_m=True,
        decouple_v=True,
        lr_ratio_1=1.0,
        lr_ratio_2=1.0,
        switch_freq_1=1,
        switch_freq_2=1,
    ):
        """
        8-bit AdamWDecouple optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            alpha,
            eps,
            weight_decay,
            32,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
            lr_ratio_1=lr_ratio_1,
            lr_ratio_2=lr_ratio_2,
            switch_freq_1=switch_freq_1,
            switch_freq_2=switch_freq_2
        )


class AdamWDecouplePlus8bit(Optimizer2StateDecouplePlus):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95, 0.9, 0.95),
        alpha=1.0,
        eps=1e-8,
        weight_decay=1e-2,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        decouple_m=True,
        decouple_v=True,
        lr_ratio_1=1.0,
        lr_ratio_2=1.0,
        switch_freq_1=1,
        switch_freq_2=1,
        quantize_delta=True,
        quantize_base=True,
    ):
        """
        8-bit AdamWDecouple optimizer.

        Arguments:
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            amsgrad (`bool`, defaults to `False`):
                Whether to use the [AMSGrad](https://hf.co/papers/1904.09237) variant of Adam that uses the maximum of past squared gradients instead.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            alpha,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
            lr_ratio_1=lr_ratio_1,
            lr_ratio_2=lr_ratio_2,
            switch_freq_1=switch_freq_1,
            switch_freq_2=switch_freq_2,
            quantize_delta=quantize_delta,
            quantize_base=quantize_base,
        )
