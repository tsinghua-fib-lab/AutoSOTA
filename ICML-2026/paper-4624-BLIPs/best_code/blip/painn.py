# Code Adapted from: https://github.com/learningmatter-mit/UQ_singleNN
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch.nn.functional import softmax
from torch import nn
from itertools import repeat
from torch.nn.init import constant_, xavier_uniform_
from functools import partial
from torch.autograd import grad


zeros_initializer = partial(constant_, val=0.0)


class ScaleShift(nn.Module):

    r"""Scale and shift layer for standardization.
    .. math::
       y = x \times \sigma + \mu
    Args:
        means (dict): dictionary of mean values
        stddev (dict): dictionary of standard deviations
    """

    def __init__(self,
                 means=None,
                 stddevs=None):
        super(ScaleShift, self).__init__()

        means = means if (means is not None) else {}
        stddevs = stddevs if (stddevs is not None) else {}
        self.means = means
        self.stddevs = stddevs

    def forward(self, inp, key):
        """Compute layer output.
        Args:
            inp (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """

        stddev = self.stddevs.get(key, 1.0)
        mean = self.means.get(key, 0.0)
        out = inp * stddev + mean

        return out
    
class ShiftedSoftplus(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.softplus(input) - np.log(2.0)


layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "shifted_softplus": ShiftedSoftplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    "swish": torch.nn.SiLU,
}


class Dense(nn.Module):
    """Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=0.,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        super().__init__()
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        self.to(inputs.device)
        y = self.linear(inputs)

        # kept for compatibility with earlier versions of nff
        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):

    src, out, index, dim = gen(
        src=src, index=index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value
    )
    output = out.scatter_add_(dim, index, src)

    return output


def compute_grad(inputs, output, allow_unused=False):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output

    Returns:
        torch.Tensor: gradients with respect to each input component
    """

    assert inputs.requires_grad

    (gradspred,) = grad(
        output,
        inputs,
        grad_outputs=output.data.new(output.shape).fill_(1),
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused,
    )

    return gradspred


def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed


def get_offsets(batch, offset_key, nbr_key="nbr_list"):
    nxyz = batch["nxyz"]
    zero = torch.Tensor([0]).to(nxyz.device)
    offsets = batch.get(offset_key, zero)
    if isinstance(offsets, torch.Tensor) and offsets.is_sparse:
        offsets = offsets.to_dense()
    return offsets


def get_rij(xyz, batch, nbrs, cutoff):

    offsets = get_offsets(batch, "offsets")
    # + offsets not - offsets because it's r_j - r_i,
    # whereas for schnet we've coded it as r_i - r_j
    r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]] + offsets

    # originally, nbrs given is directed, so r_ij computation
    # is more expensive. Since r_ij for a two-way directed
    # nbr is the same, concatenating rij and -rij is the same
    # as supplying directed nbr list. This would save a bit of
    # calculation time.
    nbrs, directed = make_directed(nbrs)
    if not directed:
        r_ij = torch.cat([r_ij, -r_ij], dim=0)

    # remove nbr skin (extra distance added to cutoff
    # to catch atoms that become neighbors between nbr
    # list updates)
    dist = (r_ij.detach() ** 2).sum(-1) ** 0.5

    if type(cutoff) == torch.Tensor:
        dist = dist.to(cutoff.device)
    use_nbrs = dist <= cutoff

    r_ij = r_ij[use_nbrs]
    nbrs = nbrs[use_nbrs]

    return r_ij, nbrs


def add_stress(batch, all_results, nbrs, r_ij):
    """
    Add stress as output. Needs to be divided by lattice volume to get actual stress.
    For batching for loop seemed unavoidable. will change later.
    stress considers both for crystal and molecules.
    For crystals need to divide by lattice volume.
    r_ij considers offsets which is different for molecules and crystals.
    """
    Z = compute_grad(output=all_results["energy"], inputs=r_ij)
    if batch["num_atoms"].shape[0] == 1:
        all_results["stress_volume"] = torch.matmul(Z.t(), r_ij)
    else:
        allstress = []
        for j in range(batch["nxyz"].shape[0]):
            allstress.append(
                torch.matmul(
                    Z[torch.where(nbrs[:, 0] == j)].t(),
                    r_ij[torch.where(nbrs[:, 0] == j)],
                )
            )
        allstress = torch.stack(allstress)
        N = batch["num_atoms"].detach().cpu().tolist()
        split_val = torch.split(allstress, N)
        all_results["stress_volume"] = torch.stack([i.sum(0) for i in split_val])
    return all_results


def sum_and_grad(
    batch, xyz, r_ij, nbrs, atomwise_output, grad_keys, out_keys=None, mean=False
):

    N = batch["num_atoms"].detach().cpu().tolist()
    results = {}
    if out_keys is None:
        out_keys = list(atomwise_output.keys())

    for key, val in atomwise_output.items():
        if key not in out_keys:
            continue

        mol_idx = (
            torch.arange(len(N)).repeat_interleave(torch.LongTensor(N)).to(val.device)
        )
        dim_size = mol_idx.max() + 1

        if val.reshape(-1).shape[0] == mol_idx.shape[0]:
            use_val = val.reshape(-1)

        # summed atom features
        elif val.shape[0] == mol_idx.shape[0]:
            use_val = val.sum(-1)

        else:
            raise Exception(
                (
                    "Don't know how to handle val shape "
                    "{} for key {}".format(val.shape, key)
                )
            )

        pooled_result = scatter_add(use_val, mol_idx, dim_size=dim_size)
        if mean:
            pooled_result = pooled_result / torch.Tensor(N).to(val.device)

        results[key] = pooled_result

    # compute gradients
    for key in grad_keys:

        # pooling has already been done to add to total props for each system
        # but batch still contains multiple systems
        # so need to be careful to do things in batched fashion
        if key == "stress":
            output = results["energy"]
            grad_ = compute_grad(output=output, inputs=r_ij)
            allstress = []
            for i in range(batch["nxyz"].shape[0]):
                allstress.append(
                    torch.matmul(
                        grad_[torch.where(nbrs[:, 0] == i)].t(),
                        r_ij[torch.where(nbrs[:, 0] == i)],
                    )
                )
            allstress = torch.stack(allstress)
            split_val = torch.split(allstress, N)
            grad_ = torch.stack([i.sum(0) for i in split_val])
            if "cell" in batch.keys():
                cell = torch.stack(torch.split(batch["cell"], 3, dim=0))
            elif "lattice" in batch.keys():
                cell = torch.stack(torch.split(batch["lattice"], 3, dim=0))
            volume = torch.Tensor(np.abs(np.linalg.det(cell.cpu().numpy()))).to(
                grad_.get_device()
            )
            grad = grad_ * (1 / volume[:, None, None])
            grad = torch.flatten(grad, start_dim=0, end_dim=1)

        else:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output, inputs=xyz)

        results[key] = grad

    return results


class SumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, batch, xyz, r_ij, nbrs, atomwise_output, grad_keys, out_keys=None
    ):
        results = sum_and_grad(
            batch=batch,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            atomwise_output=atomwise_output,
            grad_keys=grad_keys,
            out_keys=out_keys,
        )
        return results


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, xyz, atomwise_output, grad_keys, out_keys=None):
        results = sum_and_grad(
            batch=batch,
            xyz=xyz,
            atomwise_output=atomwise_output,
            grad_keys=grad_keys,
            out_keys=out_keys,
            mean=True,
        )
        return results


def att_readout_probs(name):
    if name.lower() == "softmax":

        def func(output):
            weights = softmax(output, dim=0)
            return weights

    elif name.lower() == "square":

        def func(output):
            weights = output**2 / (output**2).sum()
            return weights

    else:
        raise NotImplementedError

    return func


class AttentionPool(nn.Module):
    """
    Compute output quantities using attention, rather than a sum over
    atomic quantities. There are two methods to do this:
    (1): "atomwise": Learn the attention weights from atomic fingerprints,
    get atomwise quantities from a network applied to the fingeprints,
    and sum them with attention weights.
    (2) "mol_fp": Learn the attention weights from atomic fingerprints,
    multiply the fingerprints by these weights, add the fingerprints
    together to get a molecular fingerprint, and put the molecular
    fingerprint through a network that predicts the output.

    This one uses `mol_fp`, since it seems more expressive (?)
    """

    def __init__(
        self,
        prob_func,
        feat_dim,
        att_act,
        mol_fp_act,
        num_out_layers,
        out_dim,
        **kwargs,
    ):
        """ """
        super().__init__()

        self.w_mat = Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        self.att_weight = torch.nn.Parameter(torch.rand(1, feat_dim))
        nn.init.xavier_uniform_(self.att_weight, gain=1.414)
        self.prob_func = att_readout_probs(prob_func)
        self.att_act = layer_types[att_act]()

        # reduce the number of features by the same factor in each layer
        feat_num = [int(feat_dim / num_out_layers**m) for m in range(num_out_layers)]

        # make layers followed by an activation for all but the last
        # layer
        mol_fp_layers = [
            Dense(
                in_features=feat_num[i],
                out_features=feat_num[i + 1],
                activation=layer_types[mol_fp_act](),
            )
            for i in range(num_out_layers - 1)
        ]

        # use no activation for the last layer
        mol_fp_layers.append(
            Dense(in_features=feat_num[-1], out_features=out_dim, activation=None)
        )

        # put together in readout network
        self.mol_fp_nn = Sequential(*mol_fp_layers)

    def forward(self, batch, xyz, atomwise_output, grad_keys, out_keys):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key in out_keys:

            batched_feats = atomwise_output["features"]

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                weights = self.prob_func(
                    self.att_act((self.att_weight * self.w_mat(feats)).sum(-1))
                )

                mol_fp = (weights.reshape(-1, 1) * self.w_mat(feats)).sum(0)

                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output, inputs=xyz)
            results[key] = grad

        return results


class MolFpPool(nn.Module):
    def __init__(self, feat_dim, mol_fp_act, num_out_layers, out_dim, **kwargs):

        super().__init__()

        # reduce the number of features by the same factor in each layer
        feat_num = [int(feat_dim / num_out_layers**m) for m in range(num_out_layers)]

        # make layers followed by an activation for all but the last
        # layer
        mol_fp_layers = [
            Dense(
                in_features=feat_num[i],
                out_features=feat_num[i + 1],
                activation=layer_types[mol_fp_act](),
            )
            for i in range(num_out_layers - 1)
        ]

        # use no activation for the last layer
        mol_fp_layers.append(
            Dense(in_features=feat_num[-1], out_features=out_dim, activation=None)
        )

        # put together in readout network
        self.mol_fp_nn = Sequential(*mol_fp_layers)

    def forward(self, batch, xyz, atomwise_output, grad_keys, out_keys):
        """
        Args:
            feats (torch.Tensor): n_atom x feat_dim atomic features,
                after convolutions are finished.
        """

        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key in out_keys:

            batched_feats = atomwise_output["features"]

            # split the outputs into those of each molecule
            split_feats = torch.split(batched_feats, N)
            # sum the results for each molecule

            all_outputs = []
            learned_feats = []

            for feats in split_feats:
                mol_fp = feats.sum(0)
                output = self.mol_fp_nn(mol_fp)
                all_outputs.append(output)
                learned_feats.append(mol_fp)

            results[key] = torch.stack(all_outputs).reshape(-1)
            results[f"{key}_features"] = torch.stack(learned_feats)

        for key in grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output, inputs=xyz)
            results[key] = grad

        return results


POOL_DIC = {
    "sum": SumPool,
    "mean": MeanPool,
    "attention": AttentionPool,
    "mol_fp": MolFpPool,
}


def norm(vec, eps=1e-15):
    result = ((vec ** 2 + eps).sum(-1)) ** 0.5
    return result


def preprocess_r(r_ij):
    """
    r_ij (n_nbrs x 3): tensor of interatomic vectors (r_j - r_i)
    """

    dist = norm(r_ij)
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit


def to_module(activation):
    return layer_types[activation]()


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output


class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_gaussians,
                 cutoff,
                 trainable_gauss):
        super().__init__()

        self.n = torch.arange(1, n_gaussians + 1).float()
        if trainable_gauss:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function
        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device, dtype=shape_d.dtype),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef.to(shape_d.dtype),
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device, dtype=shape_d.dtype),
                             num / denom)

        return output


class InvariantDense(nn.Module):
    def __init__(self,
                 dim,
                 dropout,
                 activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(Dense(in_features=dim,
                                          out_features=dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=dim,
                                          out_features=3 * dim,
                                          bias=True,
                                          dropout_rate=dropout))

    def forward(self, s_j):
        output = self.layers(s_j)
        return output


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_gaussians,
                 cutoff,
                 n_atom_basis,
                 trainable_gauss,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_gaussians=n_gaussians,
                               cutoff=cutoff,
                               trainable_gauss=trainable_gauss)

        dense = Dense(in_features=n_gaussians,
                      out_features=3 * n_atom_basis,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 dropout):
        super().__init__()

        self.inv_dense = InvariantDense(dim=n_atom_basis,
                                        activation=activation,
                                        dropout=dropout)
        self.dist_embed = DistanceEmbed(n_gaussians=n_gaussians,
                                        cutoff=cutoff,
                                        n_atom_basis=n_atom_basis,
                                        trainable_gauss=trainable_gauss,
                                        dropout=dropout)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x n_atom_basis

        n_atom_basis = s_j.shape[-1]
        out_reshape = output.reshape(output.shape[0], 3, n_atom_basis)

        return out_reshape


class MessageBase(nn.Module):

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class MessageBlock(MessageBase):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 dropout,
                 **kwargs):
        super().__init__()
        self.inv_message = InvariantMessage(n_atom_basis=n_atom_basis,
                                            activation=activation,
                                            n_gaussians=n_gaussians,
                                            cutoff=cutoff,
                                            trainable_gauss=trainable_gauss,
                                            dropout=dropout)

    def forward(self,
                s_j,
                v_j,
                r_ij,
                nbrs,
                **kwargs):

        dist, unit = preprocess_r(r_ij)
        inv_out = self.inv_message(s_j=s_j,
                                   dist=dist,
                                   nbrs=nbrs)

        split_0 = inv_out[:, 0, :].unsqueeze(-1)
        split_1 = inv_out[:, 1, :]
        split_2 = inv_out[:, 2, :].unsqueeze(-1)

        unit_add = split_2 * unit.unsqueeze(1)
        delta_v_ij = unit_add + split_0 * v_j[nbrs[:, 1]]
        delta_s_ij = split_1

        # add results from neighbors of each node

        graph_size = s_j.shape[0]
        delta_v_i = scatter_add(src=delta_v_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        delta_s_i = scatter_add(src=delta_s_ij,
                                index=nbrs[:, 0],
                                dim=0,
                                dim_size=graph_size)

        return delta_s_i, delta_v_i


class UpdateBlock(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 activation,
                 dropout):
        super().__init__()
        self.u_mat = Dense(in_features=n_atom_basis,
                           out_features=n_atom_basis,
                           bias=False)
        self.v_mat = Dense(in_features=n_atom_basis,
                           out_features=n_atom_basis,
                           bias=False)
        self.s_dense = nn.Sequential(Dense(in_features=2*n_atom_basis,
                                           out_features=n_atom_basis,
                                           bias=True,
                                           dropout_rate=dropout,
                                           activation=to_module(activation)),
                                     Dense(in_features=n_atom_basis,
                                           out_features=3*n_atom_basis,
                                           bias=True,
                                           dropout_rate=dropout))

    def forward(self,
                s_i,
                v_i):

        # v_i = (num_atoms, num_feats, 3)
        # v_i.transpose(1, 2).reshape(-1, v_i.shape[1])
        # = (num_atoms, 3, num_feats).reshape(-1, num_feats)
        # = (num_atoms * 3, num_feats)
        # -> So the same u gets applied to each atom
        # and for each of the three dimensions, but differently
        # for the different feature dimensions

        v_tranpose = v_i.transpose(1, 2).reshape(-1, v_i.shape[1])

        # now reshape it to (num_atoms, 3, num_feats) and transpose
        # to get (num_atoms, num_feats, 3)

        num_feats = v_i.shape[1]
        u_v = (self.u_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))
        v_v = (self.v_mat(v_tranpose).reshape(-1, 3, num_feats)
               .transpose(1, 2))

        v_v_norm = norm(v_v)
        s_stack = torch.cat([s_i, v_v_norm], dim=-1)

        split = (self.s_dense(s_stack)
                 .reshape(s_i.shape[0], 3, -1))

        # delta v update
        a_vv = split[:, 0, :].unsqueeze(-1)
        delta_v_i = u_v * a_vv

        # delta s update
        a_sv = split[:, 1, :]
        a_ss = split[:, 2, :]

        inner = (u_v * v_v).sum(-1)
        delta_s_i = inner * a_sv + a_ss

        return delta_s_i, delta_v_i


class EmbeddingBlock(nn.Module):
    def __init__(self,
                 n_atom_basis):

        super().__init__()
        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.n_atom_basis = n_atom_basis

    def forward(self,
                z_number,
                **kwargs):

        num_atoms = z_number.shape[0]
        s_i = self.atom_embed(z_number)
        v_i = (torch.zeros(num_atoms, self.n_atom_basis, 3)
               .to(s_i.device))

        return s_i, v_i


class ReadoutBlock(nn.Module):
    def __init__(self,
                 n_atom_basis,
                 output_keys,
                 activation,
                 dropout,
                 means=None,
                 stddevs=None):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {key: nn.Sequential(
                Dense(in_features=n_atom_basis,
                      out_features=n_atom_basis//2,
                      bias=True,
                      dropout_rate=dropout,
                      activation=to_module(activation)),
                Dense(in_features=n_atom_basis//2,
                      out_features=1,
                      bias=True,
                      dropout_rate=dropout))
             for key in output_keys}
        )

        self.scale_shift = ScaleShift(means=means,
                                      stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            output = self.scale_shift(output, key)
            results[key] = output

        return results
    
class PaiNN(nn.Module):
    def __init__(self, model_params):
        """
        Args:
            model_params (dict): dictionary of model parameters
        """

        super().__init__()

        n_atom_basis = model_params["n_atom_basis"]
        activation = model_params["activation"]
        n_gaussians = model_params["n_gaussians"]
        cutoff = model_params["cutoff"]
        n_convolutions = model_params["n_convolutions"]
        output_keys = model_params["output_keys"]
        trainable_gauss = model_params.get("trainable_gauss", False)
        dropout_rate = model_params.get("dropout_rate", 0)
        means = model_params.get("means")
        stddevs = model_params.get("stddevs")
        pool_dic = model_params.get("pool_dic")

        self.excl_vol = model_params.get("excl_vol", False)
        if self.excl_vol:
            self.power = model_params["V_ex_power"]
            self.sigma = model_params["V_ex_sigma"]

        self.grad_keys = model_params["grad_keys"]
        self.embed_block = EmbeddingBlock(n_atom_basis=n_atom_basis)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    n_atom_basis=n_atom_basis,
                    activation=activation,
                    n_gaussians=n_gaussians,
                    cutoff=cutoff,
                    trainable_gauss=trainable_gauss,
                    dropout=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(
                    n_atom_basis=n_atom_basis,
                    activation=activation,
                    dropout=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = model_params.get(
            "skip_connection", {key: False for key in self.output_keys}
        )

        num_readouts = n_convolutions if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                ReadoutBlock(
                    n_atom_basis=n_atom_basis,
                    output_keys=output_keys,
                    activation=activation,
                    dropout=dropout_rate,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(num_readouts)
            ]
        )

        if pool_dic is None:
            self.pool_dic = {key: SumPool() for key in self.output_keys}
        else:
            self.pool_dic = nn.ModuleDict({})
            for out_key, sub_dic in pool_dic.items():
                if out_key not in self.output_keys:
                    continue
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                self.pool_dic[out_key] = pool_class(**kwargs)

        self.compute_delta = model_params.get("compute_delta", False)
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs = batch["nbr_list"]
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            if not xyz.requires_grad:
                xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["embedding"] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out, xyz, r_ij, nbrs, inference=False):

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0].readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key in self.output_keys:
            if key not in self.pool_dic.keys():
                all_results[key] = atomwise_out[key]
            else:
                pool_obj = self.pool_dic[key]
                grad_key = f"{key}_grad"
                grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
                if "stress" in self.grad_keys and "stress" not in all_results:
                    grad_keys.append("stress")
                results = pool_obj(
                    batch=batch,
                    xyz=xyz,
                    r_ij=r_ij,
                    nbrs=nbrs,
                    atomwise_output=atomwise_out,
                    grad_keys=grad_keys,
                    out_keys=[key],
                )
                all_results.update(results)

        return all_results, xyz

    def add_delta(self, all_results):
        for i, e_i in enumerate(self.output_keys):
            if i == 0:
                continue
            e_j = self.output_keys[i - 1]
            key = f"{e_i}_{e_j}_delta"
            all_results[key] = all_results[e_i] - all_results[e_j]
        return all_results

    def activation(self, results):
        activation = nn.Softplus()
        for key, value in results.items():
            if key in ["energy", "energy_grad", "stress", "embedding"]:
                continue
            results[key] = activation(value)
            if key == "alpha":
                results[key] = results[key] + 1

        return results

    def V_ex(self, r_ij, nbr_list, xyz):

        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = (dist.reciprocal() * self.sigma).pow(self.power)

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def run(self, batch, xyz=None, requires_stress=False, inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            atomwise_out["energy"] += r_ex

        all_results, xyz = self.pool(
            batch=batch,
            atomwise_out=atomwise_out,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            inference=inference,
        )

        if requires_stress:
            all_results = add_stress(
                batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij
            )

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        all_results = self.activation(all_results)

        return all_results, xyz

    def forward(
        self, batch, xyz=None, requires_stress=False, inference=False, **kwargs
    ):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(
            batch=batch, xyz=xyz, requires_stress=requires_stress, inference=inference
        )

        return results
