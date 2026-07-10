from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn


def get_weights_dict(T, N, type='S', value=0.001):
    weights_dict = {}
    if type == 'S':
        for i in range(N):
            weights_dict[f'layer_{i}'] = [value] * T
            idx = N - i - 1
            if idx < T:
                weights_dict[f'layer_{i}'][idx] = 1
    elif type == 'D':
        def generate_sequence_log(x, k=1.0):
            indices = np.arange(x)
            raw_sequence = np.exp(-k * indices)
            normalized_sequence = raw_sequence / 100 * np.sum(raw_sequence)
            return normalized_sequence
        for i in range(N):
            seq = [0.0] * T
            log_seq = generate_sequence_log(T - (N - i - 1))
            seq[N - i - 1:] = log_seq.tolist()
            weights_dict[f'layer_{i}'] = seq
            weights_dict[f'layer_{i}'][N - i - 1] = 1
    elif type == 'PC' or type == 'BP':
        for i in range(N):
            weights_dict[f'layer_{i}'] = [1.0] * T
    return weights_dict


# ============================================================
# VGG Models with BatchNorm
# ============================================================

VGG_types = {
    "VGG9": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, "E"],
    "VGG10": [64, 128, 128, "M", 128, 256, "M", 256, 256, "M", 256, 512, "M"],
    "VGG11": [64, 128, 128, "M", 128, 256, "M", 256, 256, "M", 256, 512, "M", 512, "E"],
    "VGG13": [128, 128, 128, "M", 128, 256, "M", 256, 256, "M", 256, 512, "M", 512, 512, "M", 512, "E"],
    "VGG15": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", 4096],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _bn_forward(layer, x, inference, init_step):
    old_state = layer.state.get()
    leaves, treedef = jax.tree_util.tree_flatten(old_state)
    old_clone = jax.tree_util.tree_unflatten(treedef, leaves)
    x = layer(x, inference=inference)
    new_state = jax.lax.cond(init_step, lambda: old_clone, lambda: layer.state.get())
    layer.state.set(new_state)
    return x


class VGGNetBN(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        in_height: int,
        in_width: int,
        in_channels: int,
        model_type: str,
        T,
        N,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
        alpha=0.001,
        precision_type='S',
    ) -> None:
        super().__init__()

        weights_dict = get_weights_dict(T, N, type=precision_type, value=alpha)
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.se_flag = se_flag

        self.vodes = {}
        self.feature_layers, num_vodes_feature = self._init_convs(
            VGG_types[model_type], in_channels, in_height, in_width, weights_dict
        )
        self.classifier_layers = self._init_fcs(
            VGG_types[model_type], in_height, in_width, self.nm_classes.get(), num_vodes_feature, weights_dict
        )
        self.layers = self.feature_layers + self.classifier_layers

        last_idx = N - 1
        self.vodes[f'layer_{last_idx}'].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[f'layer_{i}'].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array = None, out_key='h', ind=0,
                 init_step: bool = False, inference: bool = False):
        if not ind == 0:
            x = self.vodes[f'layer_{ind - 1}'].get("h")
        if ind < len(self.feature_layers):
            for block in self.feature_layers[ind:]:
                for layer in block[:1]:
                    x = layer(x)
                for layer in block[1:2]:
                    x = _bn_forward(layer, x, inference, init_step)
                for layer in block[2:]:
                    x = layer(x)
                x = self.vodes[f'layer_{ind}'](x, output=out_key)
                ind += 1

        x = x.flatten()
        t = ind - len(self.feature_layers)
        for block in self.classifier_layers[t:]:
            for layer in block:
                x = layer(x)
            x = self.vodes[f'layer_{ind}'](x, output=out_key)
            ind += 1

        if y is not None:
            self.vodes[f'layer_{ind - 1}'].set("h", y)

        return self.vodes[f'layer_{ind - 1}'].get("u")

    def _init_convs(self, architecture, in_channels, in_height, in_width, weights_dict):
        layers = []
        num_vodes = 0
        for i in range(len(architecture) - 1):
            x = architecture[i]
            next_x = architecture[i + 1]
            if type(x) == int:
                out_channel = x
                if type(next_x) == int or next_x == "E":
                    layers.append((
                        pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        pxnn.BatchNorm(out_channel, axis_name="batch", momentum=0.1, eps=1e-5),
                        self.act_fn,
                    ))
                    self.vodes[f'layer_{num_vodes}'] = pxc.Vode(name=f'layer_{num_vodes}', weight_dict=weights_dict)
                    num_vodes += 1
                elif next_x == "M":
                    layers.append((
                        pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        pxnn.BatchNorm(out_channel, axis_name="batch", momentum=0.1, eps=1e-5),
                        self.act_fn,
                        pxnn.MaxPool2d(kernel_size=2, stride=2),
                    ))
                    in_height = in_height // 2
                    in_width = in_width // 2
                    self.vodes[f'layer_{num_vodes}'] = pxc.Vode(name=f'layer_{num_vodes}', weight_dict=weights_dict)
                    num_vodes += 1
                else:
                    raise ValueError("Error in architecture definition")
                in_channels = x
        return layers, num_vodes

    def _init_fcs(self, architecture, in_height, in_width, nm_classes, num_vodes_feature, weights_dict):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (in_height % factor) + (in_width % factor) != 0:
            raise ValueError(f"`in_height` and `in_width` must be multiples of {factor}")
        out_height = in_height // factor
        out_width = in_width // factor
        last_out_channels = next(x for x in architecture[:-1][::-1] if type(x) == int)

        if type(architecture[-1]) == int:
            layers = [
                (pxnn.Linear(last_out_channels * out_height * out_width, architecture[-1]), self.act_fn),
                (pxnn.Linear(architecture[-1], nm_classes),),
            ]
            self.vodes[f'layer_{num_vodes_feature}'] = pxc.Vode(
                name=f'layer_{num_vodes_feature}', weight_dict=weights_dict
            )
            self.vodes[f'layer_{num_vodes_feature + 1}'] = pxc.Vode(
                name=f'layer_{num_vodes_feature + 1}', weight_dict=weights_dict,
                energy_fn=pxc.se_energy if self.se_flag else pxc.ce_energy,
            )
        else:
            layers = [
                (pxnn.Linear(last_out_channels * out_height * out_width, nm_classes),),
            ]
            self.vodes[f'layer_{num_vodes_feature}'] = pxc.Vode(
                name=f'layer_{num_vodes_feature}', weight_dict=weights_dict,
                energy_fn=pxc.se_energy if self.se_flag else pxc.ce_energy,
            )
        return layers


class VGG5BN(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
        T,
        N,
        alpha=0.001,
        precision_type='S',
    ) -> None:
        super().__init__()

        weights_dict = get_weights_dict(T, N, type=precision_type, value=alpha)
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), pxnn.BatchNorm(128, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)), pxnn.BatchNorm(256, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)), pxnn.BatchNorm(512, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)), pxnn.BatchNorm(512, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
        ]
        self.classifier_layers = [
            (pxnn.Linear(512 * (input_size // 16) * (input_size // 16), self.nm_classes.get()),),
        ]

        self.vodes = {}
        for l in range(len(self.feature_layers)):
            self.vodes[f'layer_{l}'] = pxc.Vode(name=f'layer_{l}', weight_dict=weights_dict)
        self.vodes['layer_4'] = pxc.Vode(
            name='layer_4', weight_dict=weights_dict,
            energy_fn=pxc.se_energy if se_flag else pxc.ce_energy,
        )
        self.vodes['layer_4'].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[f'layer_{i}'].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array = None, out_key="h", ind=0,
                 init_step: bool = False, inference: bool = False):
        if not ind == 0:
            x = self.vodes[f'layer_{ind - 1}'].get("h")
        if ind < len(self.feature_layers):
            for block in self.feature_layers[ind:]:
                for layer in block[:1]:
                    x = layer(x)
                for layer in block[1:2]:
                    x = _bn_forward(layer, x, inference, init_step)
                for layer in block[2:]:
                    x = layer(x)
                x = self.vodes[f'layer_{ind}'](x, output=out_key)
                ind += 1

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)
            x = self.vodes[f'layer_{ind}'](x, output=out_key)
            ind += 1

        if y is not None:
            self.vodes[f'layer_{ind - 1}'].set("h", y)

        return self.vodes[f'layer_{ind - 1}'].get("u")


class VGG7BN(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
        T,
        N,
        alpha=0.001,
        precision_type='S',
    ) -> None:
        super().__init__()

        weights_dict = get_weights_dict(T, N, type=precision_type, value=alpha)
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), pxnn.BatchNorm(128, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), pxnn.BatchNorm(128, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn),
            (pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)), pxnn.BatchNorm(256, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(0, 0)), pxnn.BatchNorm(256, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn),
            (pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)), pxnn.BatchNorm(512, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn, pxnn.MaxPool2d(kernel_size=2, stride=2)),
            (pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 0)), pxnn.BatchNorm(512, axis_name="batch", momentum=0.1, eps=1e-5), self.act_fn),
        ]
        self.classifier_layers = [
            (pxnn.Linear(512 * ((input_size // 4 - 2) // 2 - 2) * ((input_size // 4 - 2) // 2 - 2), self.nm_classes.get()),),
        ]

        self.vodes = {}
        for l in range(len(self.feature_layers)):
            self.vodes[f'layer_{l}'] = pxc.Vode(name=f'layer_{l}', weight_dict=weights_dict)
        self.vodes['layer_6'] = pxc.Vode(
            name='layer_6', weight_dict=weights_dict,
            energy_fn=pxc.se_energy if se_flag else pxc.ce_energy,
        )
        self.vodes['layer_6'].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[f'layer_{i}'].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array = None, out_key='h', ind=0,
                 init_step: bool = False, inference: bool = False):
        if not ind == 0:
            x = self.vodes[f'layer_{ind - 1}'].get("h")
        if ind < len(self.feature_layers):
            for block in self.feature_layers[ind:]:
                for layer in block[:1]:
                    x = layer(x)
                for layer in block[1:2]:
                    x = _bn_forward(layer, x, inference, init_step)
                for layer in block[2:]:
                    x = layer(x)
                x = self.vodes[f'layer_{ind}'](x, output=out_key)
                ind += 1

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)
            x = self.vodes[f'layer_{ind}'](x, output=out_key)
            ind += 1

        if y is not None:
            self.vodes[f'layer_{ind - 1}'].set("h", y)

        return self.vodes[f'layer_{ind - 1}'].get("u")


# ============================================================
# ResNet Models with BatchNorm
# ============================================================

class DownsampleBN(pxc.EnergyModule):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = pxnn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class BasicBlockBN(pxc.EnergyModule):
    def __init__(self, in_channels, out_channels, stride=1, act_fn=Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()

        self.conv1 = pxnn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, use_bias=False)
        self.bn1 = pxnn.BatchNorm(out_channels, axis_name="batch", momentum=0.1, eps=1e-5)
        self.act_fn = px.static(act_fn)
        self.conv2 = pxnn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, use_bias=False)
        self.bn2 = pxnn.BatchNorm(out_channels, axis_name="batch", momentum=0.1, eps=1e-5)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownsampleBN(in_channels, out_channels, stride)
        else:
            self.downsample = None

    def __call__(self, x: jax.Array, vodes1, vodes2, out_key='h', c=0,
                 inference=False, init_step=False) -> jax.Array:
        if c == 0:
            out = self.act_fn(x)
            out = self.conv1(out)
            out = _bn_forward(self.bn1, out, inference, init_step)
            out = vodes1[0](out, output=out_key)

            out = self.act_fn(out)
            out = self.conv2(out)
            out = _bn_forward(self.bn2, out, inference, init_step)
            out = vodes2[0](out, output=out_key)

            if self.downsample is not None:
                x = self.downsample(x)

            x = vodes1[1](x, output=out_key)
            x = vodes2[1](x, output=out_key)
        else:
            out = self.act_fn(vodes1[0].get(out_key))
            out = self.conv2(out)
            out = _bn_forward(self.bn2, out, inference, init_step)
            out = vodes2[0](out, output=out_key)

            x = vodes1[1].get(out_key)
            x = vodes2[1](x, output=out_key)

        x = out + x
        return x


class ResNetBN(pxc.EnergyModule):
    def __init__(
        self, block, layers, lr_h, T, N, nm_classes=1000, se_flag=True,
        act_fn=Callable[[jax.Array], jax.Array], alpha=0.001, precision_type='S',
    ) -> None:
        super().__init__()

        weights_dict = get_weights_dict(T, N, type=precision_type, value=alpha)
        self.in_channels = 64
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.initial_conv = pxnn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.vodes = {}
        self.vodes['layer_0'] = [pxc.Vode(name='layer_0', weight_dict=weights_dict)]

        layer1, idx = self._make_layer(block, 64, layers[0], stride=1, idx=1, lr_h=lr_h, weights_dict=weights_dict)
        layer2, idx = self._make_layer(block, 128, layers[1], stride=2, idx=idx, lr_h=lr_h, weights_dict=weights_dict)
        layer3, idx = self._make_layer(block, 256, layers[2], stride=2, idx=idx, lr_h=lr_h, weights_dict=weights_dict)
        layer4, idx = self._make_layer(block, 512, layers[3], stride=2, idx=idx, lr_h=lr_h, weights_dict=weights_dict)
        self.layers = layer1 + layer2 + layer3 + layer4

        self.fc = pxnn.Linear(512, nm_classes)

        self.vodes[f'layer_{idx}'] = [
            pxc.Vode(pxc.se_energy if se_flag else pxc.ce_energy, name=f'layer_{idx}', weight_dict=weights_dict)
        ]

        for key in self.vodes.keys():
            for v in self.vodes[key]:
                v.h0.frozen = True
        self.vodes[f'layer_{idx}'][0].h.frozen = True

    def _make_layer(self, block, out_channels, num_blocks, stride=1, idx=0, lr_h=1.0, weights_dict=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride, act_fn=self.act_fn))
            self.vodes[f'layer_{idx}'] = [pxc.Vode(name=f"layer_{idx}", weight_dict=weights_dict)]
            self.vodes[f'layer_{idx + 1}'] = [pxc.Vode(name=f"layer_{idx + 1}", weight_dict=weights_dict)]
            if stride != 1 or self.in_channels != out_channels:
                self.vodes[f'layer_{idx}'].append(pxc.Vode(name=f"layer_{idx}", weight_dict=weights_dict))
                self.vodes[f'layer_{idx + 1}'].append(
                    pxc.Vode(name=f"layer_{idx + 1}", weight=1 / lr_h, weight_dict=weights_dict)
                )
            else:
                self.vodes[f'layer_{idx}'].append(
                    pxc.Vode(name=f"layer_{idx}", weight=1 / lr_h, weight_dict=weights_dict)
                )
                self.vodes[f'layer_{idx + 1}'].append(
                    pxc.Vode(name=f"layer_{idx + 1}", weight=1 / lr_h, weight_dict=weights_dict)
                )
            self.in_channels = out_channels
            idx += 2

        return layers, idx

    def _forward_layer(self, x, layers, idx, out_key='h', c=0, inference=False, init_step=False):
        for layer in layers:
            x = layer(x, self.vodes[f'layer_{idx}'], self.vodes[f'layer_{idx + 1}'],
                      out_key=out_key, c=c, inference=inference, init_step=init_step)
            idx += 2
            c = 0
        return x, idx

    def __call__(self, x: jax.Array, y: jax.Array | None = None, beta: float = 1.0,
                 inference=False, init_step=False, out_key='h', ind=0) -> jax.Array:
        if ind == 0:
            x = self.initial_conv(x)
            x = self.vodes['layer_0'][0](x, output=out_key)
            x, idx = self._forward_layer(x, self.layers, idx=1, out_key=out_key,
                                         inference=inference, init_step=init_step)
        else:
            if ind == 1:
                x = self.vodes[f'layer_0'][0].get(out_key)
            else:
                x = self.vodes[f'layer_{ind - 1}'][0].get(out_key) + self.vodes[f'layer_{ind - 1}'][1].get(out_key)
            c = (ind - 1) % 2
            x, idx = self._forward_layer(x, self.layers[(ind - 1) // 2:], idx=ind - c, out_key=out_key,
                                         c=c, inference=inference, init_step=init_step)

        x4 = self.act_fn(x)
        x4 = jnp.mean(x4, axis=(1, 2))
        x4 = self.fc(x4)
        x4 = self.vodes[f'layer_{idx}'][0](x4, output=out_key)

        if y is not None:
            self.vodes[f'layer_{idx}'][0].set("h", y)

        return self.vodes[f'layer_{idx}'][0].get("u")


# ============================================================
# Unified Model Factory
# ============================================================

def get_model(
    model_name: str,
    nm_classes: int,
    act_fn: Callable[[jax.Array], jax.Array],
    input_size: int = 32,
    se_flag: bool = True,
    T: int = 5,
    alpha: float = 0.001,
    precision_type: str = 'S',
    lr_h: float = 1.0,
):
    if model_name == "ResNet18":
        return ResNetBN(BasicBlockBN, [2, 2, 2, 2], nm_classes=nm_classes, se_flag=se_flag,
                        act_fn=act_fn, lr_h=lr_h, T=T, N=18, alpha=alpha, precision_type=precision_type)
    elif model_name == "ResNet10":
        return ResNetBN(BasicBlockBN, [1, 1, 1, 1], nm_classes=nm_classes, se_flag=se_flag,
                        act_fn=act_fn, lr_h=lr_h, T=T, N=10, alpha=alpha, precision_type=precision_type)
    elif model_name == "VGG5":
        return VGG5BN(nm_classes=nm_classes, input_size=input_size, act_fn=act_fn,
                      se_flag=se_flag, T=T, N=5, alpha=alpha, precision_type=precision_type)
    elif model_name == "VGG7":
        return VGG7BN(nm_classes=nm_classes, input_size=input_size, act_fn=act_fn,
                      se_flag=se_flag, T=T, N=7, alpha=alpha, precision_type=precision_type)
    elif model_name in VGG_types:
        N = int(model_name.split("VGG")[1])
        return VGGNetBN(nm_classes=nm_classes, in_height=input_size, in_width=input_size,
                        in_channels=3, model_type=model_name, T=T, N=N, act_fn=act_fn,
                        se_flag=se_flag, alpha=alpha, precision_type=precision_type)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
