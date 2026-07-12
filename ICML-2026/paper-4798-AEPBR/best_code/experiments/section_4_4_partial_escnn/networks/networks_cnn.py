from typing import Tuple
from escnn import group, gspaces, nn

import sys
import os
from itertools import count

sys.path.append("..")
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to PATH
sys.path.append(parent_dir)
from escnn2.r2convolution import R2Conv
from escnn2.r3convolution import R3Conv
from escnn2 import gspaces
import torch
from functools import partial
import numpy as np

# from .networks_cnn2 import SteerableCNN3DResnet


_MODEL_LAYER_ID_COUNTER = count()
_MODEL_LAYER_ID_STRIDE = 10_000


def _next_model_layer_id_offset(iteration: int) -> int:
    # Learnable-equivariance kernels cache internal objects by layer_id.
    # Make layer_id namespaces unique per constructed model so benchmarking
    # multiple checkpoints in one process does not alias those caches.
    return next(_MODEL_LAYER_ID_COUNTER) * _MODEL_LAYER_ID_STRIDE + 100 * iteration

class SteerableCNN3D(nn.EquivariantModule):
    def __init__(
        self,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        f=True,
        N=-1,
        restrict=False,
        learn_eq=False,
        normalise_basis=True,
        one_eq=False,
        channels=6,
        iteration=0,
        dropout=0,
        L_in=2,
        L_out=None,
        invariant=True,
    ):
        super(SteerableCNN3D, self).__init__()

        self.learn_eq = (learn_eq,)
        self.normalise_basis = (normalise_basis,)

        self._L_in, self._L_out = L_in, L_out

        self._id_offset = _next_model_layer_id_offset(iteration)

        self._c_fac = [channels]
        splits = 1

        assert not (
            last_tensor and restrict
        ), "last tensor and restrict cannot both be on"

        if N == -1:
            if f:
                self.L = 2
                self.act_r2 = gspaces.flipRot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = (None, 1)
                self._group_name = "O3" if N == -1 else f"D{N}"
            else:
                self.L = 3
                self.act_r2 = gspaces.rot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = 1
                self._group_name = "SO3" if N == -1 else f"C{N}"

        elif N > 0:
            raise "not supported yet"

        else:
            self.L = 0
            if f:
                raise "not supported yet"
            else:
                self.act_r2 = gspaces.trivialOnR3()
                self.restrict_id = 1
                self._group_name = "C1"

        self._f = f

        self._N = N

        self.activation_fn = activation

        self._activation_name = activation.__name__

        self.last_tensor = last_tensor

        self.n_classes = n_classes

        self._n_channels = n_channels

        self.mnist_type = mnist_type

        self.restrict = restrict

        self._one_eq = one_eq

        self._splits = splits

        self.dropout = dropout

        self.invariant = invariant

        self._init_layers(splits)

    def _init_layers(self, splits):
        if self.mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif self.mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)

        id_generator = (
            lambda layer, channel: (channel * 10 + layer * (not self._one_eq))
            + self._id_offset
        )

        self.in_type = nn.FieldType(
            self.act_r2, [self.act_r2.trivial_repr for _ in range(self._n_channels)]
        )

        self.upsample = nn.R3Upsampling(self.in_type, size=(h, w, d))

        self.mask = nn.MaskModule(self.in_type, h, margin=1)

        self.layers_eq = torch.nn.ModuleList()

        dropout = self.dropout

        for channel in range(splits):
            # Block 1
            activation_1, out_type = self._activation_and_out_type(
                self._c_fac[channel], min(2, self.L)
            )
            block_1 = nn.SequentialModule(
                R3Conv(
                    self.in_type,
                    out_type,
                    kernel_size=7,
                    padding=2,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(0, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_1,
            )

            # Block 2
            activation_2, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 2, min(3, self.L)
            )
            block_2 = nn.SequentialModule(
                R3Conv(
                    block_1.out_type,
                    out_type,
                    kernel_size=5,
                    padding=2,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(1, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_2,
            )

            # Pooling
            pool_1 = nn.PointwiseAvgPoolAntialiased3D(
                block_2.out_type, sigma=0.66, stride=2, padding=1
            )

            # Block 3
            activation_3, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 4
            )
            block_3 = nn.SequentialModule(
                R3Conv(
                    block_2.out_type,
                    out_type,
                    kernel_size=3,
                    stride=2,
                    padding=padding_3,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(2, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                nn.FieldDropout(out_type, 0.1 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_3,
            )

            # Pooling
            pool_2 = nn.PointwiseAvgPoolAntialiased3D(
                block_3.out_type, sigma=0.66, stride=2, padding=1
            )

            # Block 4
            activation_4, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6
            )
            conv_4 = R3Conv(
                block_3.out_type,
                out_type,
                kernel_size=3,
                stride=2,
                padding=padding_4,
                learnable_eq=self.learn_eq,
                normalise_basis=self.normalise_basis,
                layer_id=id_generator(3, channel),
                L=self._L_in,
                L_out=self._L_out,
            )

            block_4 = nn.SequentialModule(
                conv_4,
                nn.IIDBatchNorm3d(activation_4.in_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_4,
            )

            if self.restrict:
                restriction_4, block_5_in = self._restrict_layer(
                    block_4.out_type, self.restrict_id
                )
            else:
                restriction_4 = lambda x: x
                block_5_in = block_4.out_type

            # Block 5
            activation_5, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6, min(2, self.L)
            )
            conv_5 = R3Conv(
                block_5_in,
                out_type,
                kernel_size=3,
                stride=1,
                padding=1,
                learnable_eq=self.learn_eq,
                normalise_basis=self.normalise_basis,
                layer_id=id_generator(4, channel),
                L=self._L_in,
                L_out=self._L_out,
            )

            block_5 = nn.SequentialModule(
                conv_5,
                nn.IIDBatchNorm3d(out_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_5,
            )

            pool_3 = nn.PointwiseAvgPoolAntialiased3D(
                block_5.out_type, sigma=0.66, stride=1, padding=1
            )

            if self.last_tensor:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.irreps[3] + self.act_r2.irreps[2]]
                    * (self._c_fac[channel] * 8),
                )
                tensor_out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * (self._c_fac * 8),
                )
                tensor_act = nn.TensorProductModule(out_type, tensor_out_type)

            elif not self.invariant:
                irreps = [
                    self.act_r2.fibergroup.irrep(*irr)
                    for irr in self.act_r2.fibergroup.bl_irreps(self.L)
                ]

                out_type = nn.FieldType(
                    self.act_r2,
                    irreps * int(self._c_fac[channel] * 2),
                )
            else:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * int(self._c_fac[channel] * 8),
                )

            block_6 = nn.SequentialModule(
                R3Conv(
                    block_5.out_type,
                    out_type,
                    kernel_size=1,
                    bias=False,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(5, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                )
            )

            layers = [
                self.upsample,
                # self.mask,
                block_1,
                block_2,
                pool_1,
                block_3,
                pool_2,
                block_4,
                restriction_4,
                block_5,
                pool_3,
                block_6,
            ]

            if not self.restrict:
                del layers[-4]

            if self.last_tensor:
                layers.append(tensor_act)

            self.layers_eq.append(nn.SequentialModule(*layers))
        nr_features = int(sum(self._c_fac) * 8)
        self.full_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nr_features),
            torch.nn.ELU(),
            torch.nn.Linear(nr_features, self.n_classes),
        )

    def _restrict_layer(self, in_type, id):
        layers = list()
        layers.append(nn.RestrictionModule(in_type, id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self.act_r2 = layers[-1].out_type.gspace
        self.L = 0

        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer, layers[-1].out_type

    def forward(self, x):
        x = self.in_type(x)
        outs = [layers(x).tensor for layers in self.layers_eq]

        x = torch.cat(outs, axis=1)
        x = self.full_net(x.reshape(x.shape[0], -1))

        return x

    def _activation_and_out_type(self, channels, L=None):
        channels = int(channels)
        if L is None:
            L = self.L
        irreps = self.act_r2.fibergroup.bl_irreps(L)

        if self.activation_fn in {nn.FourierELU, nn.FourierPointwise}:
            L = 2
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )

            out_type = activation.in_type

        elif self.activation_fn in {nn.NormNonLinearity}:
            c = 2
            irreps = (
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])]
            )
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearityUniform:
            c = 1

            irreps = channels * [
                group.directsum(
                    [self.act_r2.trivial_repr for _ in range((len(irreps)))]
                    + [self.act_r2.irrep(*id) for id in irreps]
                )
            ]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearity1:
            c = 1
            irreps = channels * [self.act_r2.trivial_repr] + channels * [
                group.directsum([self.act_r2.irrep(*id) for id in irreps])
            ]

            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(
                out_type, gates=channels * ["gate"] + channels * ["gated"]
            )

        elif self.activation_fn == nn.TensorProductModule:
            c = 2
            out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            tensor_out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            activation = self.activation_fn(out_type, tensor_out_type)

        return activation, out_type

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 4, shape
        assert shape[1] == self.in_type.size, shape
        return [shape[0], self.n_classes]

    @classmethod
    def from_group(
        cls,
        group,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        restrict=False,
        learn_eq=False,
        normalise_basis=True,
        one_eq=True,
        channels=6,
        iteration=0,
        L_in=2,
        L_out=4,
        invariant=True,
    ):
        try:
            if group == "SO3":
                N = -1
                f = False
            elif group == "O3":
                N = -1
                f = True
            # elif group[0] == "D" and int(group[1:]) in {0, 1, 2, 4, 6, 8, 12, 16}:
            #     nr_rots = int(group[1:])
            #     f = True
            #     N = nr_rots if nr_rots > 1 else 0
            # elif group[0] == "C" and int(group[1:]) in {0, 1, 2, 4, 6, 8, 12, 16}:
            #     nr_rots = int(group[1:])
            #     f = False
            #     N = nr_rots if nr_rots > 1 else 0
            elif group == "trivial":
                f = False
                N = 0
            else:
                raise AssertionError("invalid group")
        except Exception as e:
            raise AssertionError(f"invalid group, found exception: {e}")
        return cls(
            f=f,
            N=N,
            activation=activation,
            last_tensor=last_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            mnist_type=mnist_type,
            restrict=restrict,
            learn_eq=learn_eq,
            normalise_basis=normalise_basis,
            one_eq=one_eq,
            channels=channels,
            iteration=iteration,
            L_in=L_in,
            L_out=L_out,
            invariant=invariant,
        )

    @staticmethod
    def supported_activations():
        return {
            nn.NormNonLinearity,
            nn.GatedNonLinearity1,
            nn.GatedNonLinearityUniform,
            nn.FourierPointwise,
            nn.FourierELU,
            nn.TensorProductModule,
        }

    @property
    def network_name(self):
        return f"{'Learnable ' if self.learn_eq else ''}{'Restricted ' if self.restrict else ''}{self._group_name}{self.__class__.__name__}"


def _sequentialmodule_children(module):
    return list(module._modules.values())


def _steerable_3d_main_path_widths(reference_model):
    if isinstance(reference_model, SteerableCNN3D):
        widths = []
        for module in _sequentialmodule_children(reference_model.layers_eq[0]):
            if not isinstance(module, nn.SequentialModule):
                continue
            children = _sequentialmodule_children(module)
            # Restriction layers alter the representation but do not have a plain CNN analogue.
            if len(children) == 2 and isinstance(children[0], nn.RestrictionModule):
                continue
            widths.append(module.out_type.size)
        if len(widths) != 6:
            raise RuntimeError(
                f"Expected 6 main-path widths for SteerableCNN3D, found {len(widths)}."
            )
        return tuple(int(width) for width in widths)

    if isinstance(reference_model, SteerableCNN3DResnet):
        widths = []
        for module in _sequentialmodule_children(reference_model.layers_eq[0]):
            if not isinstance(module, EqResBlock):
                continue
            block_children = _sequentialmodule_children(module.block)
            widths.append(int(block_children[0].out_type.size))
            widths.append(int(block_children[-1].out_type.size))
        if len(widths) != 6:
            raise RuntimeError(
                f"Expected 6 main-path widths for SteerableCNN3DResnet, found {len(widths)}."
            )
        return tuple(widths)

    raise TypeError(f"Unsupported reference model type: {type(reference_model)}")


def _build_steerable_3d_reference(
    reference_group,
    activation,
    resnet,
    n_classes,
    n_channels,
    mnist_type,
    restrict,
    learn_eq,
    normalise_basis,
    one_eq,
    channels,
    iteration,
    L_in,
    L_out,
    invariant,
):
    reference_cls = SteerableCNN3DResnet if resnet else SteerableCNN3D
    return reference_cls.from_group(
        reference_group,
        activation=activation,
        n_classes=n_classes,
        n_channels=n_channels,
        mnist_type=mnist_type,
        restrict=restrict,
        learn_eq=learn_eq,
        normalise_basis=normalise_basis,
        one_eq=one_eq,
        channels=channels,
        iteration=iteration,
        L_in=L_in,
        L_out=L_out,
        invariant=invariant,
    )


class CNN3D(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, mnist_type="single", c=6):
        super(CNN3D, self).__init__()

        if mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)
        c = c

        self.upsample = torch.nn.Upsample(size=(h, w, d))

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, c, 7, stride=1, padding=2),
            torch.nn.BatchNorm3d(c),
            torch.nn.ELU(),
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c, 2 * c, 5, stride=1, padding=2),
            torch.nn.BatchNorm3d(2 * c),
            torch.nn.ELU(),
        )

        self.pool_1 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
            torch.nn.BatchNorm3d(4 * c),
            torch.nn.ELU(),
        )

        self.pool_2 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv3d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv3d(6 * c, 6 * c, 3, stride=1, padding=1),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        self.pool_3 = torch.nn.AvgPool3d(3, stride=1, padding=0)

        self.block_6 = torch.nn.Conv3d(6 * c, 8 * c, 1)

        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8 * c),
            torch.nn.ELU(),
            torch.nn.Linear(8 * c, n_classes),
        )

        self.in_type = lambda x: x

    def forward(self, x):
        x = self.upsample(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.pool_1(x)
        x = self.block_3(x)
        x = self.pool_2(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.pool_3(x)
        x = self.block_6(x)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

    @property
    def network_name(self):
        return "CNN"


class EqResBlock(nn.EquivariantModule):
    def __init__(self, block, skip):
        super(EqResBlock, self).__init__()
        self.in_type = block.in_type
        self.out_type = block.out_type
        self.block = block
        self.skip = skip

    def forward(self, x):
        return self.block(x) + self.skip(x)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.block.evaluate_output_shape(input_shape)


class ResBlock(torch.nn.Module):
    def __init__(self, block, skip):
        super(ResBlock, self).__init__()
        self.block = block
        self.skip = skip

    def forward(self, x):
        return self.block(x) + self.skip(x)


class SteerableCNN3DResnet(nn.EquivariantModule):
    def __init__(
        self,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        f=True,
        N=-1,
        restrict=False,
        learn_eq=False,
        normalise_basis=True,
        one_eq=False,
        channels=6,
        iteration=0,
        dropout=0,
        L_in=2,
        L_out=None,
        invariant=True,
    ):
        super(SteerableCNN3DResnet, self).__init__()

        self._L_in, self._L_out = L_in, L_out

        self._id_offset = _next_model_layer_id_offset(iteration)

        # if split is None:
        self._c_fac = [channels]
        splits = 1

        assert not (
            last_tensor and restrict
        ), "last tensor and restrict cannot both be on"

        if N == -1:
            if f:
                self.L = 2
                self.act_r2 = gspaces.flipRot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = (None, 1)
                self._group_name = "O3" if N == -1 else f"D{N}"
            else:
                self.L = 3
                self.act_r2 = gspaces.rot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = 1
                self._group_name = "SO3" if N == -1 else f"C{N}"

        elif N > 0:
            raise "not supported yet"

        else:
            self.L = 0
            if f:
                # self.act_r2 = gspaces.flipRot3dOnR3(N=1)
                # self.restrict_id = 1
                # self._group_name = "D1"
                raise "not supported yet"
            else:
                self.act_r2 = gspaces.trivialOnR3()
                self.restrict_id = 1
                self._group_name = "C1"

        self.normalise_basis = normalise_basis
        self.learn_eq = learn_eq

        self._f = f

        self._N = N

        self.activation_fn = activation

        self._activation_name = activation.__name__

        self.last_tensor = last_tensor

        self.n_classes = n_classes

        self._n_channels = n_channels

        self.mnist_type = mnist_type

        self.restrict = restrict

        self._one_eq = one_eq

        self._splits = splits

        self.dropout = dropout

        self.invariant = invariant

        self._init_layers(splits)

    def _init_layers(self, splits):
        if self.mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif self.mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)

        id_generator = (
            lambda layer, channel: (channel * 10 + layer * (not self._one_eq))
            + self._id_offset
        )

        self.in_type = nn.FieldType(
            self.act_r2, [self.act_r2.trivial_repr for _ in range(self._n_channels)]
        )
        self.upsample = nn.R3Upsampling(self.in_type, size=(h, w, d))

        # self.mask = nn.MaskModule(self.in_type, h, margin=1)

        self.layers_eq = torch.nn.ModuleList()

        dropout = self.dropout

        for channel in range(splits):
            # Block 1
            activation_1, out_type = self._activation_and_out_type(
                self._c_fac[channel], min(2, self.L)
            )
            block_1 = nn.SequentialModule(
                R3Conv(
                    self.in_type,
                    out_type,
                    kernel_size=7,
                    padding=2,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_1,
            )

            # Block 2
            activation_2, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 2, min(3, self.L)
            )
            block_2 = nn.SequentialModule(
                R3Conv(
                    block_1.out_type,
                    out_type,
                    kernel_size=5,
                    padding=2,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(0, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_2,
            )
            # bl_1 = block_1.evaluate_output_shape((1, 1, 29, 29, 29))
            # bl_2 = block_2.evaluate_output_shape(bl_1)

            skip_1 = nn.SequentialModule(
                R3Conv(
                    block_1.in_type,
                    out_type,
                    kernel_size=7,
                    padding=2,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(0, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                activation_2,
            )

            # x = block_1.in_type(torch.randn(1, 1, 29, 29, 29))
            resblock_1 = EqResBlock(nn.SequentialModule(block_1, block_2), skip_1)

            # Pooling
            pool_1 = nn.PointwiseAvgPoolAntialiased3D(
                block_2.out_type, sigma=0.66, stride=2, padding=1
            )

            # Block 3
            activation_3, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 4
            )
            block_3 = nn.SequentialModule(
                R3Conv(
                    block_2.out_type,
                    out_type,
                    kernel_size=3,
                    stride=2,
                    padding=padding_3,
                ),
                nn.FieldDropout(out_type, 0.1 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_3,
            )

            # Pooling
            pool_2 = nn.PointwiseAvgPoolAntialiased3D(
                block_3.out_type, sigma=0.66, stride=2, padding=1
            )

            # Block 4
            activation_4, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6
            )
            conv_4 = R3Conv(
                block_3.out_type,
                out_type,
                kernel_size=3,
                stride=2,
                padding=padding_4,
                learnable_eq=self.learn_eq,
                normalise_basis=self.normalise_basis,
                layer_id=id_generator(1, channel),
                L=self._L_in,
                L_out=self._L_out,
            )

            block_4 = nn.SequentialModule(
                conv_4,
                nn.IIDBatchNorm3d(activation_4.in_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_4,
            )

            skip_2 = nn.SequentialModule(
                nn.PointwiseAvgPoolAntialiased3D(
                    pool_1.out_type, sigma=0.66, stride=2, padding=0
                ),
                R3Conv(
                    block_3.in_type,
                    out_type,
                    kernel_size=3,
                    padding=0,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(1, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
                activation_4,
            )

            resblock_2 = EqResBlock(
                nn.SequentialModule(block_3, pool_2, block_4), skip_2
            )

            # thing = block_3.in_type(torch.randn(pl_1))
            # out = block_3(thing)
            # bl_3 = out.shape
            # out = pool_2(out)
            # pl_2 = out.shape
            # out = block_4(out)
            # bl_4 = out.shape

            if self.restrict:
                restriction_4, block_5_in = self._restrict_layer(
                    block_4.out_type, self.restrict_id
                )
            else:
                restriction_4 = lambda x: x
                block_5_in = block_4.out_type

            # Block 5
            activation_5, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6, min(2, self.L)
            )
            conv_5 = R3Conv(
                block_5_in,
                out_type,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            block_5 = nn.SequentialModule(
                conv_5,
                nn.IIDBatchNorm3d(out_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_5,
            )

            pool_3 = nn.PointwiseAvgPoolAntialiased3D(
                block_5.out_type, sigma=0.66, stride=1, padding=1
            )

            if self.last_tensor:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.irreps[3] + self.act_r2.irreps[2]]
                    * (self._c_fac[channel] * 8),
                )
                tensor_out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * (self._c_fac * 8),
                )
                tensor_act = nn.TensorProductModule(out_type, tensor_out_type)
            elif not self.invariant:
                irreps = [
                    self.act_r2.fibergroup.irrep(*irr)
                    for irr in self.act_r2.fibergroup.bl_irreps(self.L)
                ]

                out_type = nn.FieldType(
                    self.act_r2,
                    irreps * int(self._c_fac[channel] * 2),
                )
            else:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * int(self._c_fac[channel] * 8),
                )

            block_6 = nn.SequentialModule(
                R3Conv(
                    block_5.out_type,
                    out_type,
                    kernel_size=1,
                    bias=False,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(2, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                )
            )

            skip_3 = nn.SequentialModule(
                R3Conv(
                    block_5_in,
                    out_type,
                    kernel_size=3,
                    padding=0,
                    learnable_eq=self.learn_eq,
                    normalise_basis=self.normalise_basis,
                    layer_id=id_generator(2, channel),
                    L=self._L_in,
                    L_out=self._L_out,
                ),
            )
            # out_skip_3 = skip_3(skip_3.in_type(torch.randn_like(out.tensor))).shape
            # out = block_5(out)

            # bl_5 = out.shape
            # out = pool_3(out)
            # pl_3 = out.shape
            # out = block_6(out)
            # print(out.shape)
            # print(out_skip_3)

            resblock_3 = EqResBlock(
                nn.SequentialModule(block_5, pool_3, block_6), skip_3
            )

            # layers = [
            #     self.upsample,
            #     # self.mask,
            #     block_1,
            #     block_2,
            #     pool_1,
            #     block_3,
            #     pool_2,
            #     block_4,
            #     restriction_4,
            #     block_5,
            #     pool_3,
            #     block_6,
            # ]

            layers = [
                self.upsample,
                # self.mask,
                resblock_1,
                pool_1,
                resblock_2,
                restriction_4,
                resblock_3,
            ]

            if not self.restrict:
                del layers[-2]

            if self.last_tensor:
                layers.append(tensor_act)

            self.layers_eq.append(nn.SequentialModule(*layers))
        nr_features = int(splits * out_type.size)
        self.full_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nr_features),
            torch.nn.ELU(),
            torch.nn.Linear(nr_features, self.n_classes),
        )

    def _restrict_layer(self, in_type, id):
        layers = list()
        layers.append(nn.RestrictionModule(in_type, id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self.act_r2 = layers[-1].out_type.gspace
        self.L = 0

        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer, layers[-1].out_type

    def forward(self, x):
        x = self.in_type(x)
        outs = [layers(x).tensor for layers in self.layers_eq]
        x = torch.cat(outs, axis=1)
        x = self.full_net(x.reshape(x.shape[0], -1))

        return x

    def _activation_and_out_type(self, channels, L=None):
        channels = int(channels)
        if L is None:
            L = self.L
        irreps = self.act_r2.fibergroup.bl_irreps(L)

        if self.activation_fn in {nn.FourierELU, nn.FourierPointwise}:
            L = 2
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )

            out_type = activation.in_type
        elif self.activation_fn in {nn.QuotientFourierELU, nn.QuotientFourierPointwise}:
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                subgroup_id=(False, -1) if not self._f else (False, True, -1),
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )

            out_type = activation.in_type
        elif self.activation_fn in {nn.NormNonLinearity}:
            c = 2
            irreps = (
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])]
            )
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearityUniform:
            c = 1

            irreps = channels * [
                group.directsum(
                    [self.act_r2.trivial_repr for _ in range((len(irreps)))]
                    + [self.act_r2.irrep(*id) for id in irreps]
                )
            ]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearity1:
            c = 1
            irreps = channels * [self.act_r2.trivial_repr] + channels * [
                group.directsum([self.act_r2.irrep(*id) for id in irreps])
            ]

            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(
                out_type, gates=channels * ["gate"] + channels * ["gated"]
            )

        elif self.activation_fn == nn.TensorProductModule:
            c = 2
            out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            tensor_out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            activation = self.activation_fn(out_type, tensor_out_type)

        return activation, out_type

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 4, shape
        assert shape[1] == self.in_type.size, shape
        return [shape[0], self.n_classes]

    @classmethod
    def from_group(
        cls,
        group,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        restrict=False,
        learn_eq=False,
        normalise_basis=True,
        one_eq=True,
        channels=6,
        iteration=0,
        L_in=2,
        L_out=4,
        invariant=True,
    ):
        try:
            if group == "SO3":
                N = -1
                f = False
            elif group == "O3":
                N = -1
                f = True
            elif group == "trivial":
                f = False
                N = 0
            else:
                raise AssertionError("invalid group")
        except Exception as e:
            raise AssertionError(f"invalid group, found exception: {e}")
        return cls(
            f=f,
            N=N,
            activation=activation,
            last_tensor=last_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            mnist_type=mnist_type,
            restrict=restrict,
            learn_eq=learn_eq,
            normalise_basis=normalise_basis,
            one_eq=one_eq,
            channels=channels,
            iteration=iteration,
            L_in=L_in,
            L_out=L_out,
            invariant=invariant,
        )

    @staticmethod
    def supported_activations():
        return {
            nn.NormNonLinearity,
            nn.GatedNonLinearity1,
            nn.GatedNonLinearityUniform,
            nn.FourierPointwise,
            nn.FourierELU,
            nn.TensorProductModule,
        }

    @property
    def network_name(self):
        return f"{'Learnable ' if self.learn_eq else ''}{'Restricted ' if self.restrict else ''}{self._group_name}{self.__class__.__name__}"


class CNN3DResnet(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, mnist_type="single", c=6):
        super(CNN3DResnet, self).__init__()

        if mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)
        c = c

        self.upsample = torch.nn.Upsample(size=(h, w, d))

        block_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, c, 7, stride=1, padding=2),
            torch.nn.BatchNorm3d(c),
            torch.nn.ELU(),
        )

        block_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c, 2 * c, 5, stride=1, padding=2),
            torch.nn.BatchNorm3d(2 * c),
            torch.nn.ELU(),
        )

        skip_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, 2 * c, 7, padding=2)
        )

        self.resblock_1 = ResBlock(torch.nn.Sequential(block_1, block_2), skip_1)

        self.pool_1 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        block_3 = torch.nn.Sequential(
            torch.nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
            torch.nn.BatchNorm3d(4 * c),
            torch.nn.ELU(),
        )

        pool_2 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        block_4 = torch.nn.Sequential(
            torch.nn.Conv3d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        skip_2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(5, stride=2, padding=0), torch.nn.Conv3d(2 * c, 6 * c, 3)
        )

        self.resblock_2 = ResBlock(
            torch.nn.Sequential(block_3, pool_2, block_4), skip_2
        )

        block_5 = torch.nn.Sequential(
            torch.nn.Conv3d(6 * c, 6 * c, 3, stride=1, padding=1),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        pool_3 = torch.nn.AvgPool3d(3, stride=1, padding=0)

        block_6 = torch.nn.Conv3d(6 * c, 8 * c, 1)

        skip_3 = torch.nn.Sequential(torch.nn.Conv3d(6 * c, 8 * c, 3))

        self.resblock_3 = ResBlock(
            torch.nn.Sequential(block_5, pool_3, block_6), skip_3
        )

        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8 * c),
            torch.nn.ELU(),
            torch.nn.Linear(8 * c, n_classes),
        )

        self.in_type = lambda x: x

    def forward(self, x):
        x = self.upsample(x)
        # x = self.block_1(x)
        # x = self.block_2(x)
        x = self.resblock_1(x)
        x = self.pool_1(x)
        # x = self.block_3(x)
        # x = self.pool_2(x)
        # x = self.block_4(x)
        x = self.resblock_2(x)
        # x = self.block_5(x)
        # x = self.pool_3(x)
        # x = self.block_6(x)
        x = self.resblock_3(x)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

    @property
    def network_name(self):
        return "CNN"


class RPPBlock(nn.EquivariantModule):
    def __init__(self, conveq, conv, layer_id):
        super(RPPBlock, self).__init__()
        self.conveq = conveq
        self.conv = conv
        self.out_type = conveq.out_type
        self.in_type = conveq.in_type
        self.layer_id = layer_id

    def forward(self, x):
        conv_out = self.conv(x.tensor)
        eq_out = self.conveq(x)
        return eq_out + self.out_type(conv_out)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return 0


class SteerableRPP3DResnet(torch.nn.Module):
    def __init__(
        self,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        f=True,
        N=-1,
        restrict=False,
        one_eq=False,
        channels=6,
        iteration=0,
        dropout=0,
        L_in=2,
        L_out=None,
        invariant=True,
        learn_eq=None
    ):
        super(SteerableRPP3DResnet, self).__init__()

        self._L_in, self._L_out = L_in, L_out

        self._id_offset = _next_model_layer_id_offset(iteration)

        # if split is None:
        self._c_fac = [channels]
        splits = 1

        assert not (
            last_tensor and restrict
        ), "last tensor and restrict cannot both be on"

        if N == -1:
            if f:
                self.L = 2
                self.act_r2 = gspaces.flipRot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = (None, 1)
                self._group_name = "O3" if N == -1 else f"D{N}"
            else:
                self.L = 3
                self.act_r2 = gspaces.rot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = 1
                self._group_name = "SO3" if N == -1 else f"C{N}"

        elif N > 0:
            raise "not supported yet"

        else:
            self.L = 0
            if f:
                raise "not supported yet"
            else:
                self.act_r2 = gspaces.trivialOnR3()
                self.restrict_id = 1
                self._group_name = "C1"

        self._f = f

        self._N = N

        self.activation_fn = activation

        self._activation_name = activation.__name__

        self.last_tensor = last_tensor

        self.n_classes = n_classes

        self._n_channels = n_channels

        self.mnist_type = mnist_type

        self.restrict = restrict

        self._one_eq = one_eq

        self._splits = splits

        self.dropout = dropout

        self.invariant = invariant

        self._init_layers(splits)

    def _init_layers(self, splits):
        if self.mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif self.mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)

        id_generator = (
            lambda layer, channel: (channel * 10 + layer * (not self._one_eq))
            + self._id_offset
        )

        self.in_type = nn.FieldType(
            self.act_r2, [self.act_r2.trivial_repr for _ in range(self._n_channels)]
        )
        self.upsample = nn.R3Upsampling(self.in_type, size=(h, w, d))

        self.mask = nn.MaskModule(self.in_type, h, margin=1)

        self.layers_eq = torch.nn.ModuleList()

        dropout = self.dropout

        for channel in range(splits):
            # Block 1
            activation_1, out_type = self._activation_and_out_type(
                self._c_fac[channel], min(2, self.L)
            )
            block_1 = nn.SequentialModule(
                R3Conv(
                    self.in_type,
                    out_type,
                    kernel_size=7,
                    padding=2,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_1,
            )

            block_1_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.in_type.size,
                    activation_1.out_type.size,
                    kernel_size=7,
                    padding=2,
                ),
                torch.nn.Dropout(0.05 * dropout),
                torch.nn.BatchNorm3d(activation_1.out_type.size),
                torch.nn.ELU(),
            )

            # Block 2
            activation_2, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 2, min(3, self.L)
            )
            block_2 = nn.SequentialModule(
                R3Conv(
                    block_1.out_type,
                    out_type,
                    kernel_size=5,
                    padding=2,
                ),
                nn.FieldDropout(out_type, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_2,
            )

            block_2_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_1.out_type.size,
                    activation_2.out_type.size,
                    kernel_size=5,
                    padding=2,
                ),
                torch.nn.Dropout(0.05 * dropout),
                torch.nn.BatchNorm3d(activation_2.out_type.size),
                torch.nn.ELU(),
            )
            # bl_1 = block_1.evaluate_output_shape((1, 1, 29, 29, 29))
            # bl_2 = block_2.evaluate_output_shape(bl_1)

            skip_1 = nn.SequentialModule(
                R3Conv(
                    block_1.in_type,
                    out_type,
                    kernel_size=7,
                    padding=2,
                ),
                activation_2,
            )

            skip_1_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_1.in_type.size,
                    activation_2.out_type.size,
                    kernel_size=7,
                    padding=2,
                ),
                torch.nn.ELU(),
            )

            # x = block_1.in_type(torch.randn(1, 1, 29, 29, 29))
            resblock_eq_1 = EqResBlock(nn.SequentialModule(block_1, block_2), skip_1)

            resblock_conv_1 = ResBlock(
                torch.nn.Sequential(block_1_conv, block_2_conv), skip_1_conv
            )

            resblock_1 = RPPBlock(resblock_eq_1, resblock_conv_1, layer_id=0)

            # Pooling
            pool_1 = nn.PointwiseAvgPoolAntialiased3D(
                block_2.out_type, sigma=0.66, stride=2, padding=1
            )

            # Block 3
            activation_3, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 4
            )
            block_3 = nn.SequentialModule(
                R3Conv(
                    block_2.out_type,
                    out_type,
                    kernel_size=3,
                    stride=2,
                    padding=padding_3,
                ),
                nn.FieldDropout(out_type, 0.1 * dropout),
                nn.IIDBatchNorm3d(out_type),
                activation_3,
            )

            block_3_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_2.out_type.size,
                    activation_3.out_type.size,
                    kernel_size=3,
                    stride=2,
                    padding=padding_3,
                ),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.BatchNorm3d(activation_3.out_type.size),
                torch.nn.ELU(),
            )

            # Pooling
            pool_2 = nn.PointwiseAvgPoolAntialiased3D(
                block_3.out_type, sigma=0.66, stride=2, padding=1
            )

            pool_2_conv = torch.nn.AvgPool3d(5, stride=2, padding=1)

            # Block 4
            activation_4, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6
            )
            conv_4 = R3Conv(
                block_3.out_type,
                out_type,
                kernel_size=3,
                stride=2,
                padding=padding_4,
            )

            block_4 = nn.SequentialModule(
                conv_4,
                nn.IIDBatchNorm3d(activation_4.in_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_4,
            )

            block_4_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_3.out_type.size,
                    activation_4.out_type.size,
                    kernel_size=3,
                    stride=2,
                    padding=padding_4,
                ),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.BatchNorm3d(activation_4.out_type.size),
                torch.nn.ELU(),
            )

            skip_2 = nn.SequentialModule(
                nn.PointwiseAvgPoolAntialiased3D(
                    pool_1.out_type, sigma=0.66, stride=2, padding=0
                ),
                R3Conv(
                    block_3.in_type,
                    out_type,
                    kernel_size=3,
                    padding=0,
                ),
                activation_4,
            )

            skip_2_conv = torch.nn.Sequential(
                torch.nn.AvgPool3d(5, stride=2, padding=0),
                torch.nn.Conv3d(
                    block_3.in_type.size,
                    activation_4.out_type.size,
                    kernel_size=3,
                    padding=0,
                ),
                torch.nn.ELU(),
            )

            resblock_eq_2 = EqResBlock(
                nn.SequentialModule(block_3, pool_2, block_4), skip_2
            )

            resblock_conv_2 = ResBlock(
                torch.nn.Sequential(block_3_conv, pool_2_conv, block_4_conv),
                skip_2_conv,
            )

            resblock_2 = RPPBlock(resblock_eq_2, resblock_conv_2, 1)

            # TODO

            # thing = block_3.in_type(torch.randn(pl_1))
            # out = block_3(thing)
            # bl_3 = out.shape
            # out = pool_2(out)
            # pl_2 = out.shape
            # out = block_4(out)
            # bl_4 = out.shape

            if self.restrict:
                restriction_4, block_5_in = self._restrict_layer(
                    block_4.out_type, self.restrict_id
                )
            else:
                restriction_4 = lambda x: x
                block_5_in = block_4.out_type

            # Block 5
            activation_5, out_type = self._activation_and_out_type(
                self._c_fac[channel] * 6, min(2, self.L)
            )
            conv_5 = R3Conv(
                block_5_in,
                out_type,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            block_5 = nn.SequentialModule(
                conv_5,
                nn.IIDBatchNorm3d(out_type),
                nn.FieldDropout(out_type, 0.1 * dropout),
                activation_5,
            )

            block_5_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_5_in.size,
                    activation_5.out_type.size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.BatchNorm3d(activation_5.out_type.size),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.ELU(),
            )

            pool_3 = nn.PointwiseAvgPoolAntialiased3D(
                block_5.out_type, sigma=0.66, stride=1, padding=1
            )

            pool_3_conv = torch.nn.AvgPool3d(3, stride=1, padding=0)

            if self.last_tensor:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.irreps[3] + self.act_r2.irreps[2]]
                    * (self._c_fac[channel] * 8),
                )
                tensor_out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * (self._c_fac * 8),
                )
                tensor_act = nn.TensorProductModule(out_type, tensor_out_type)
            elif not self.invariant:
                irreps = [
                    self.act_r2.fibergroup.irrep(*irr)
                    for irr in self.act_r2.fibergroup.bl_irreps(self.L)
                ]

                out_type = nn.FieldType(
                    self.act_r2,
                    irreps * int(self._c_fac[channel] * 2),
                )
            else:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * int(self._c_fac[channel] * 8),
                )

            block_6 = nn.SequentialModule(
                R3Conv(
                    block_5.out_type,
                    out_type,
                    kernel_size=1,
                    bias=False,
                )
            )

            block_6_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_5.out_type.size,
                    out_type.size,
                    kernel_size=1,
                )
            )

            skip_3 = nn.SequentialModule(
                R3Conv(
                    block_5_in,
                    out_type,
                    kernel_size=3,
                    padding=0,
                ),
            )

            skip_3_conv = torch.nn.Sequential(
                torch.nn.Conv3d(
                    block_5_in.size,
                    out_type.size,
                    kernel_size=3,
                    padding=0,
                ),
            )
            # out_skip_3 = skip_3(skip_3.in_type(torch.randn_like(out.tensor))).shape
            # out = block_5(out)

            # bl_5 = out.shape
            # out = pool_3(out)
            # pl_3 = out.shape
            # out = block_6(out)
            # print(out.shape)
            # print(out_skip_3)

            resblock_eq_3 = EqResBlock(
                nn.SequentialModule(block_5, pool_3, block_6), skip_3
            )

            resblock_conv_3 = ResBlock(
                torch.nn.Sequential(block_5_conv, pool_3_conv, block_6_conv),
                skip_3_conv,
            )

            resblock_3 = RPPBlock(resblock_eq_3, resblock_conv_3, 2)

            # layers = [
            #     self.upsample,
            #     # self.mask,
            #     block_1,
            #     block_2,
            #     pool_1,
            #     block_3,
            #     pool_2,
            #     block_4,
            #     restriction_4,
            #     block_5,
            #     pool_3,
            #     block_6,
            # ]

            layers = [
                self.upsample,
                # self.mask,
                resblock_1,
                pool_1,
                resblock_2,
                restriction_4,
                resblock_3,
            ]

            if not self.restrict:
                del layers[-2]

            if self.last_tensor:
                layers.append(tensor_act)

            self.layers_eq.append(nn.SequentialModule(*layers))
        nr_features = int(splits * out_type.size)
        self.full_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nr_features),
            torch.nn.ELU(),
            torch.nn.Linear(nr_features, self.n_classes),
        )

    def _activation_and_out_type(self, channels, L=None):
        channels = int(channels)
        if L is None:
            L = self.L
        irreps = self.act_r2.fibergroup.bl_irreps(L)

        if self.activation_fn in {nn.FourierELU, nn.FourierPointwise}:
            L = 2
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )

            out_type = activation.in_type
        elif self.activation_fn in {nn.QuotientFourierELU, nn.QuotientFourierPointwise}:
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                subgroup_id=(False, -1) if not self._f else (False, True, -1),
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )

            out_type = activation.in_type
        elif self.activation_fn in {nn.NormNonLinearity}:
            c = 2
            irreps = (
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])]
            )
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearityUniform:
            c = 1

            irreps = channels * [
                group.directsum(
                    [self.act_r2.trivial_repr for _ in range((len(irreps)))]
                    + [self.act_r2.irrep(*id) for id in irreps]
                )
            ]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearity1:
            c = 1
            irreps = channels * [self.act_r2.trivial_repr] + channels * [
                group.directsum([self.act_r2.irrep(*id) for id in irreps])
            ]

            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(
                out_type, gates=channels * ["gate"] + channels * ["gated"]
            )

        elif self.activation_fn == nn.TensorProductModule:
            c = 2
            out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            tensor_out_type = nn.FieldType(
                self.act_r2,
                c
                * channels
                * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            activation = self.activation_fn(out_type, tensor_out_type)

        return activation, out_type

    def forward(self, x):
        x = self.in_type(x)
        outs = [layers(x).tensor for layers in self.layers_eq]
        x = torch.cat(outs, axis=1)
        x = self.full_net(x.reshape(x.shape[0], -1))

        return x

    @classmethod
    def from_group(
        cls,
        group,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        restrict=False,
        one_eq=True,
        channels=6,
        iteration=0,
        L_in=2,
        L_out=4,
        invariant=True,
        learn_eq=None,
        normalise_basis=None,
    ):
        try:
            if group == "SO3":
                N = -1
                f = False
            elif group == "O3":
                N = -1
                f = True
            # elif group[0] == "D" and int(group[1:]) in {0, 1, 2, 4, 6, 8, 12, 16}:
            #     nr_rots = int(group[1:])
            #     f = True
            #     N = nr_rots if nr_rots > 1 else 0
            # elif group[0] == "C" and int(group[1:]) in {0, 1, 2, 4, 6, 8, 12, 16}:
            #     nr_rots = int(group[1:])
            #     f = False
            #     N = nr_rots if nr_rots > 1 else 0
            elif group == "trivial":
                f = False
                N = 0
            else:
                raise AssertionError("invalid group")
        except Exception as e:
            raise AssertionError(f"invalid group, found exception: {e}")
        return cls(
            f=f,
            N=N,
            activation=activation,
            last_tensor=last_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            mnist_type=mnist_type,
            restrict=restrict,
            one_eq=one_eq,
            channels=channels,
            iteration=iteration,
            L_in=L_in,
            L_out=L_out,
            invariant=invariant,
        )

    @staticmethod
    def supported_activations():
        return {
            nn.NormNonLinearity,
            nn.GatedNonLinearity1,
            nn.GatedNonLinearityUniform,
            nn.FourierPointwise,
            nn.FourierELU,
            nn.TensorProductModule,
        }

    @property
    def network_name(self):
        return f"{self._group_name}{self.__class__.__name__}"


if __name__ == "__main__":
    # network = SteerableCNN3DResnet(
    #     mnist_type="single",
    #     basisexpansion="learn_eq_norm",
    #     channels=1,
    #     L_in=1,
    #     L_out=1,
    #     f=False,
    # )

    # network = CNN3DResnet(10, 1, "single")
    # thing = torch.randn(10, 1, 28, 28, 28)

    network = SteerableRPP3DResnet()

    thing = torch.randn(10, 1, 28, 28, 28)
    network(thing)
