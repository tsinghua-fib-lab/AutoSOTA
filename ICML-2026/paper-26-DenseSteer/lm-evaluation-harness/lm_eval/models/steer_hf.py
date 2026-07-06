from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Optional, List, Union

# 引入 steering_vectors 库
try:
    from steering_vectors import SteeringVector
except ImportError:
    raise ImportError("Please install steering-vectors: pip install steering-vectors")

# lm-eval registry
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

@register_model("steer_hf")
class SteerHFLM(HFLM):
    def __init__(self, **kwargs):
        raw_ma = kwargs.get("model_args", {}) or {}
        ma = self._parse_model_args(raw_ma)
        # 消费自定义参数
        def _consume(key, caster=None, default=None):
            if key in kwargs: val = kwargs.pop(key)
            elif key in ma: val = ma.pop(key)
            else: val = default
            return caster(val) if (caster and val is not None) else val

        self.steer_lambda = _consume("steer_lambda", float, 0.0)
        self.steer_vec_path = _consume("steer_vec_path", str, None)
        self.steer_min_token = _consume("steer_min_token", int, 0)

        # ★ 关键参数：指定只干预哪一层 (single layer, backward compatible)
        self.steer_layer = _consume("steer_layer", int, None)

        # ★ NEW: multi-layer support — colon-separated list like "10:17:25"
        # Uses ':' instead of ',' because model_args is already comma-separated
        _steer_layers_raw = _consume("steer_layers", str, None)
        self.steer_layers: Optional[List[int]] = None
        if _steer_layers_raw is not None:
            try:
                self.steer_layers = [int(x.strip()) for x in _steer_layers_raw.split(":") if x.strip()]
            except ValueError:
                raise ValueError(f"steer_layers must be colon-separated integers, got: {_steer_layers_raw}")

        # ★ NEW: L2 normalization for cross-layer comparability (IDEA-09)
        self.steer_normalize = _consume("steer_normalize", str, "false").lower() in ("true", "1", "yes")

        # ★ NEW: lambda decay schedule (IDEA-04) — format: "linear:start:end:decay_steps"
        self.steer_lambda_schedule = _consume("steer_lambda_schedule", str, None)

        # ★ NEW: bidirectional steering (IDEA-06) — second vector with negative lambda
        self.steer_vec_path_neg = _consume("steer_vec_path_neg", str, None)
        self.steer_lambda_neg = _consume("steer_lambda_neg", float, 0.0)

        super().__init__(**kwargs)

        self.steering_vector = None
        self._enabled = (self.steer_lambda != 0.0 and self.steer_vec_path is not None)

        if self._enabled:
            print(f"[SteerHFLM] Loading vector from: {self.steer_vec_path}")
            try:
                # 1. 加载完整对象 (可能包含多个层)
                full_vector = torch.load(self.steer_vec_path, map_location="cpu")

                # 兼容性检查：确保加载的是 SteeringVector 对象
                if isinstance(full_vector, dict) and not isinstance(full_vector, SteeringVector):
                    full_vector = SteeringVector(layer_activations=full_vector)

                # Determine which layers to keep
                layers_to_keep = None
                if self.steer_layers is not None:
                    # Multi-layer mode (IDEA-02): keep all specified layers
                    layers_to_keep = self.steer_layers
                elif self.steer_layer is not None:
                    # Single-layer mode (backward compatible): keep only one layer
                    layers_to_keep = [self.steer_layer]

                if layers_to_keep is not None:
                    # Validate all requested layers exist
                    available = list(full_vector.layer_activations.keys())
                    for l in layers_to_keep:
                        if l not in full_vector.layer_activations:
                            raise ValueError(
                                f"Requested layer {l} not found in vector file. Available: {available}"
                            )

                    # Optional L2 normalization (IDEA-09)
                    selected_data = {}
                    for l in layers_to_keep:
                        vec = full_vector.layer_activations[l].clone()
                        if self.steer_normalize:
                            vec = vec / vec.norm()
                        selected_data[l] = vec

                    print(f"[SteerHFLM] Filtering vector: keeping layers {list(selected_data.keys())}"
                          f"{' (L2-normalized)' if self.steer_normalize else ''}")
                    self.steering_vector = SteeringVector(layer_activations=selected_data)
                else:
                    # 如果没指定层，就默认应用文件里所有的层
                    self.steering_vector = full_vector
                    print(f"[SteerHFLM] No specific layer requested. Applying to ALL layers found:"
                          f" {list(self.steering_vector.layer_activations.keys())}")

                # 3. 移动到正确的设备 (GPU)
                self.steering_vector = self.steering_vector.to(
                    device=self.model.device,
                    dtype=self.model.dtype
                )

            except Exception as e:
                raise ValueError(f"Failed to load or slice steering vector: {e}")

        # ★ IDEA-06: Load second (negative) steering vector for bidirectional control
        self.steering_vector_neg = None
        self._neg_enabled = (self.steer_lambda_neg != 0.0 and self.steer_vec_path_neg is not None)
        if self._neg_enabled:
            print(f"[SteerHFLM] Loading negative vector from: {self.steer_vec_path_neg}")
            try:
                neg_vec = torch.load(self.steer_vec_path_neg, map_location="cpu")
                if isinstance(neg_vec, dict) and not isinstance(neg_vec, SteeringVector):
                    neg_vec = SteeringVector(layer_activations=neg_vec)

                # Use same layer selection as positive vector
                layers_to_keep = None
                if self.steer_layers is not None:
                    layers_to_keep = self.steer_layers
                elif self.steer_layer is not None:
                    layers_to_keep = [self.steer_layer]

                if layers_to_keep is not None:
                    selected_neg = {}
                    for l in layers_to_keep:
                        if l not in neg_vec.layer_activations:
                            available = list(neg_vec.layer_activations.keys())
                            raise ValueError(f"Layer {l} not found in negative vector. Available: {available}")
                        vec = neg_vec.layer_activations[l].clone()
                        if self.steer_normalize:
                            vec = vec / vec.norm()
                        selected_neg[l] = vec
                    self.steering_vector_neg = SteeringVector(layer_activations=selected_neg)
                else:
                    self.steering_vector_neg = neg_vec

                self.steering_vector_neg = self.steering_vector_neg.to(
                    device=self.model.device, dtype=self.model.dtype
                )
                print(f"[SteerHFLM] Negative vector loaded: layers {list(self.steering_vector_neg.layer_activations.keys())}")
            except Exception as e:
                raise ValueError(f"Failed to load negative steering vector: {e}")

    def _parse_model_args(self, ma):
        """Standard lm-eval arg parser"""
        if isinstance(ma, dict):
            return dict(ma)
        if isinstance(ma, str):
            out = {}
            s = ma.strip()
            if not s: return out
            parts = [p for p in s.split(",") if p.strip() != ""]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    out[k.strip()] = self._coerce(v.strip())
                else:
                    out[p.strip()] = True
            return out
        return {}

    def _coerce(self, v: str):
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]
        vl = v.lower()
        if vl == "true": return True
        if vl == "false": return False
        try:
            if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                return int(v)
            return float(v)
        except:
            return v

    def _register_decay_hooks(self):
        """Register custom hooks with per-step lambda decay (IDEA-04).

        Uses a step counter that increments on each forward pass. Early tokens
        (prompt + initial generation) get strong steering; later tokens get weaker
        steering to avoid distorting arithmetic/computation steps.

        Schedule format: "linear:start:end:decay_steps"
        - linear: linearly decay from start to end over decay_steps forward passes
        """
        from steering_vectors import guess_and_enhance_layer_config
        from steering_vectors.layer_matching import collect_matching_layers
        from steering_vectors.steering_vector import get_module

        parts = self.steer_lambda_schedule.split(":")
        schedule_type = parts[0]
        lambda_start = float(parts[1])
        lambda_end = float(parts[2])
        decay_steps = int(parts[3])

        self._steer_step = 0  # Reset per generate_until call

        layer_config = guess_and_enhance_layer_config(
            self.model, None, self.steering_vector.layer_type
        )
        matcher = layer_config[self.steering_vector.layer_type]
        matching_layers = collect_matching_layers(self.model, matcher)

        hooks = []
        for layer_num, vec in self.steering_vector.layer_activations.items():
            layer_name = matching_layers[layer_num]
            module = get_module(self.model, layer_name)
            vec_reshaped = vec.reshape(1, 1, -1).clone()
            steer_min = self.steer_min_token  # capture for closure

            def make_hook(layer_vec, lname):
                def hook(module, input, output):
                    step = self._steer_step
                    self._steer_step += 1

                    # Linear decay schedule
                    progress = min(step / max(decay_steps, 1), 1.0)
                    if schedule_type == "linear":
                        current_mult = lambda_start - (lambda_start - lambda_end) * progress
                    elif schedule_type == "exp":
                        # Exponential decay: start * (end/start)^progress
                        ratio = lambda_end / max(lambda_start, 1e-8)
                        current_mult = lambda_start * (ratio ** progress)
                    else:
                        current_mult = lambda_start  # fallback: no decay

                    # Apply to token positions >= steer_min
                    output[0][:, steer_min:, :] += current_mult * layer_vec.to(
                        device=output[0].device, dtype=output[0].dtype
                    )
                    return output
                return hook

            handle = module.register_forward_hook(make_hook(vec, layer_name))
            hooks.append(handle)

        return hooks

    @contextmanager
    def _steering_ctx(self):
        """
        Apply steering vectors. Supports: static lambda, lambda decay, and
        bidirectional steering (IDEA-06) with nested context managers.
        """
        if not self._enabled or self.steering_vector is None:
            yield
            return

        if self.steer_lambda_schedule is not None:
            # Lambda decay mode (IDEA-04): custom hooks with per-step multiplier
            decay_hooks = self._register_decay_hooks()
            try:
                yield
            finally:
                for h in decay_hooks:
                    h.remove()
        elif self._neg_enabled and self.steering_vector_neg is not None:
            # Bidirectional steering (IDEA-06): positive + negative vectors
            with self.steering_vector.apply(self.model, multiplier=self.steer_lambda):
                with self.steering_vector_neg.apply(self.model, multiplier=self.steer_lambda_neg):
                    yield
        else:
            # Standard single-vector steering
            with self.steering_vector.apply(self.model, multiplier=self.steer_lambda):
                yield

    # ---- 覆盖 lm-eval 的核心生成/评估方法 ----

    def generate_until(self, requests):
        with self._steering_ctx():
            return super().generate_until(requests)

    def loglikelihood(self, requests):
        with self._steering_ctx():
            return super().loglikelihood(requests)

    def loglikelihood_rolling(self, requests):
        with self._steering_ctx():
            return super().loglikelihood_rolling(requests)
