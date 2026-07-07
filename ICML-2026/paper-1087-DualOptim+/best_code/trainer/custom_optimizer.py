from typing import Optional
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F


class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


class Optimizer2StateDecouple(bnb.optim.optimizer.Optimizer8bit):
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
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
        decouple_m=False,
        decouple_v=False,
        lr_ratio_1=1.0,
        lr_ratio_2=1.0,
        switch_freq_1=1,
        switch_freq_2=1,
        alpha=0.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
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
                The alpha value for the AdEMAMix optimizer.
            t_alpha (`Optional[int]`, defaults to `None`):
                Number of iterations for alpha scheduling with AdEMAMix.
            t_beta3 (`Optional[int]`, defaults to `None`):
                Number of iterations for beta scheduling with AdEMAMix.

        """
        # TODO: quantize forget state to 8 bit
        # TODO: quantize delta forget state to 8 bit, forget state = retain state + delta forget state

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
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
        )

        super().__init__(params, defaults, optim_bits, is_paged)

        self.non_castable_tensor_keys = {
            "qmap1",
            "qmap2",
            "qmap1_1",
            "qmap2_1",
            "qmap1_2",
            "qmap2_2",
            "max1",
            "max2",
            "max1_1",
            "max2_1",
            "max1_2",
            "max2_2",
            "new_max1",
            "new_max2",
            "new_max1_1",
            "new_max2_1",
            "new_max1_2",
            "new_max2_2",
            "state1",
            "state2",
            "state1_1",
            "state2_1",
            "state1_2",
            "state2_2",
            "gnorm_vec",
            "gnorm_vec1",
            "gnorm_vec2",
            "absmax1",
            "absmax2",
            "absmax1_1",
            "absmax2_1",
            "absmax1_2",
            "absmax2_2",
            "unorm_vec",
            "unorm_vec1",
            "unorm_vec2",
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
            self.args = MockArgs(args)
        else:
            self.args = args

        if optim_bits == 8:
            if self.args.decouple_m:
                self.name2qmap["dynamic1"] = F.create_dynamic_map(signed=True)
                self.name2qmap["dynamic2"] = F.create_dynamic_map(signed=True)
            else:
                self.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)

            if self.args.decouple_v:
                self.name2qmap["udynamic1"] = F.create_dynamic_map(signed=False)
                self.name2qmap["udynamic2"] = F.create_dynamic_map(signed=False)
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
            else:
                state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            if config["decouple_v"]:
                state["state2_1"] = self.get_state_buffer(p, dtype=torch.float32)
                state["state2_2"] = self.get_state_buffer(p, dtype=torch.float32)
            else:
                state["state2"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                if config["decouple_m"]:
                    self.name2qmap["dynamic1"] = self.name2qmap["dynamic1"].to(p.device)
                    self.name2qmap["dynamic2"] = self.name2qmap["dynamic2"].to(p.device)
                else:
                    self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                if config["decouple_v"]:
                    self.name2qmap["udynamic1"] = self.name2qmap["udynamic1"].to(
                        p.device
                    )
                    self.name2qmap["udynamic2"] = self.name2qmap["udynamic2"].to(
                        p.device
                    )
                else:
                    self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            if config["decouple_m"]:
                state["state1_1"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap1_1"] = self.name2qmap["dynamic1"]

                state["state1_2"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap1_2"] = self.name2qmap["dynamic2"]
            else:
                state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap1"] = self.name2qmap["dynamic"]

            if config["decouple_v"]:
                state["state2_1"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap2_1"] = self.name2qmap["udynamic1"]

                state["state2_2"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap2_2"] = self.name2qmap["udynamic2"]
            else:
                state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
                state["qmap2"] = self.name2qmap["udynamic"]

            if config["block_wise"]:
                n = p.numel()
                blocks = n // 256
                blocks += 1 if n % 256 > 0 else 0

                if config["decouple_m"]:
                    state["absmax1_1"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
                    state["absmax1_2"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
                else:
                    state["absmax1"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )

                if config["decouple_v"]:
                    state["absmax2_1"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
                    state["absmax2_2"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
                else:
                    state["absmax2"] = torch.zeros(
                        (blocks,), dtype=torch.float32, device=p.device
                    )
            else:
                if config["decouple_m"]:
                    state["max1_1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max1_1"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )

                    state["max1_2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max1_2"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )
                else:
                    state["max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max1"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )

                if config["decouple_m"]:
                    state["max2_1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max2_1"] = torch.zeros(
                        (1,), dtype=torch.float32, device=p.device
                    )

                    state["max2_2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    state["new_max2_2"] = torch.zeros(
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
            else:
                state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        if config["max_unorm"] > 0.0:
            if config["decouple_m"] or config["decouple_v"]:
                state["unorm_vec1"] = torch.zeros((1,), device=p.device)
                state["unorm_vec2"] = torch.zeros((1,), device=p.device)
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
        dtype = (
            state["state1"].dtype
            if not config["decouple_m"]
            else state["state1_1"].dtype
            if mode == 1
            else state["state1_2"].dtype
        )

        state["step"] += 1
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
            F.optimizer_update_32bit(
                self.optimizer_name,
                g=grad,
                p=p,
                state1=state["state1"]
                    if not config["decouple_m"]
                    else state["state1_1"]
                    if mode == 1
                    else state["state1_2"],
                beta1=config["betas"][0],
                eps=config["eps"],
                step=step,
                lr=ratio * config["lr"],
                state2=state["state2"]
                    if not config["decouple_v"]
                    else state["state2_1"]
                    if mode == 1
                    else state["state2_2"],
                beta2=config["betas"][1],
                beta3=config["betas"][2] if len(config["betas"]) >= 3 else 0.0,  # no beta3 argument in 0.42.0
                alpha=config["alpha"],  # no alpha argument in 0.42.0
                weight_decay=config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"]
                    if config["max_unorm"] > 0.0 and not is_dual
                    else state["unorm_vec1"]
                    if config["max_unorm"] > 0.0 and mode == 1
                    else state["unorm_vec2"]
                    if config["max_unorm"] > 0.0 and mode == 2
                    else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

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

        elif dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                g=grad,
                p=p,
                state1=state["state1"]
                    if not config["decouple_m"]
                    else state["state1_1"]
                    if mode == 1
                    else state["state1_2"],
                state2=state["state2"]
                    if not config["decouple_v"]
                    else state["state2_1"]
                    if mode == 1
                    else state["state2_2"],
                beta1=config["betas"][0],
                beta2=config["betas"][1],
                eps=config["eps"],
                step=step,
                lr=ratio * config["lr"],
                qmap1=state["qmap1"]
                    if not config["decouple_m"]
                    else state["qmap1_1"]
                    if mode == 1
                    else state["qmap1_2"],
                qmap2=state["qmap2"]
                    if not config["decouple_v"]
                    else state["qmap2_1"]
                    if mode == 1
                    else state["qmap2_2"],
                max1=state["max1"]
                    if not config["decouple_m"]
                    else state["max1_1"]
                    if mode == 1
                    else state["max1_2"],
                max2=state["max2"]
                    if not config["decouple_v"]
                    else state["max2_1"]
                    if mode == 1
                    else state["max2_2"],
                new_max1=state["new_max1"]
                    if not config["decouple_m"]
                    else state["new_max1_1"]
                    if mode == 1
                    else state["new_max1_2"],
                new_max2=state["new_max2"]
                    if not config["decouple_v"]
                    else state["new_max2_1"]
                    if mode == 1
                    else state["new_max2_2"],
                weight_decay=config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"]
                    if config["max_unorm"] > 0.0 and not is_dual
                    else state["unorm_vec1"]
                    if config["max_unorm"] > 0.0 and mode == 1
                    else state["unorm_vec2"]
                    if config["max_unorm"] > 0.0 and mode == 2
                    else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            if config["decouple_m"] and mode == 1:
                state["max1_1"], state["new_max1_1"] = (
                    state["new_max1_1"],
                    state["max1_1"],
                )
            elif config["decouple_m"] and mode == 2:
                state["max1_2"], state["new_max1_2"] = (
                    state["new_max1_2"],
                    state["max1_2"],
                )
            else:
                state["max1"], state["new_max1"] = state["new_max1"], state["max1"]

            if config["decouple_v"] and mode == 1:
                state["max2_1"], state["new_max2_1"] = (
                    state["new_max2_1"],
                    state["max2_1"],
                )
            elif config["decouple_v"] and mode == 2:
                state["max2_2"], state["new_max2_2"] = (
                    state["new_max2_2"],
                    state["max2_2"],
                )
            else:
                state["max2"], state["new_max2"] = state["new_max2"], state["max2"]

        elif dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                g=grad,
                p=p,
                state1=state["state1"]
                    if not config["decouple_m"]
                    else state["state1_1"]
                    if mode == 1
                    else state["state1_2"],
                state2=state["state2"]
                    if not config["decouple_v"]
                    else state["state2_1"]
                    if mode == 1
                    else state["state2_2"],
                beta1=config["betas"][0],
                beta2=config["betas"][1],
                beta3=config["betas"][2] if len(config["betas"]) >= 3 else 0.0,  # no beta3 argument in 0.42.0
                alpha=config["alpha"],  # no alpha argument in 0.42.0
                eps=config["eps"],
                step=step,
                lr=ratio * config["lr"],
                qmap1=state["qmap1"]
                    if not config["decouple_m"]
                    else state["qmap1_1"]
                    if mode == 1
                    else state["qmap1_2"],
                qmap2=state["qmap2"]
                    if not config["decouple_v"]
                    else state["qmap2_1"]
                    if mode == 1
                    else state["qmap2_2"],
                absmax1=state["absmax1"]
                    if not config["decouple_m"]
                    else state["absmax1_1"]
                    if mode == 1
                    else state["absmax1_2"],
                absmax2=state["absmax2"]
                    if not config["decouple_v"]
                    else state["absmax2_1"]
                    if mode == 1
                    else state["absmax2_2"],
                weight_decay=config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

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


class AdamWDecoupleNormal(Optimizer2StateDecouple):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
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
        switch_freq_2=1
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


class AdamWDecouple8bit(Optimizer2StateDecouple):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
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
        switch_freq_2=1
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
            switch_freq_2=switch_freq_2
        )
