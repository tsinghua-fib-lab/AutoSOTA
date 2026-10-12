import os
import warnings

import torch
from torch import nn

from torchdrug import core, models
from torchdrug.core import Registry as R
from torchdrug.layers import functional


@R.register("models.MyESM")
class MyESM(models.EvolutionaryScaleModeling):

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_type
        input = self.mapping[input]
        input[input == -1] = graph.residue_type[input == -1]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]

        output = self.model(input, repr_layers=[self.repr_layer])
        residue_feature = output["representations"][self.repr_layer]
        logits = output["logits"]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        logits = functional.padded_to_variadic(logits, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        logits = logits[mask]
        residue_type_index = torch.arange(20, dtype=torch.long, device=logits.device)
        logits = logits[:, self.mapping[residue_type_index]]
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
            "logits": logits
        }
    

@R.register("models.ESM1vEnsemble")
class ESM1vEnsemble(nn.Module, core.Configurable):
    """IDEA-608: the five released ESM-1v models as one scoring model.

    ESM-1v was published with five independently pretrained models whose
    masked marginals are meant to be averaged (log-mean of the probabilities).
    This wrapper reuses one full MyESM instance (model 1) for input
    preprocessing and runs all five transformers, returning the log-mean
    distribution as "logits". Only ever used for scoring (masked + WT
    branches), never trained or checkpointed.
    """

    def __init__(self, path, model="ESM-1v"):
        super(ESM1vEnsemble, self).__init__()
        import esm as esm_lib
        self.template = MyESM(path, model)
        self.members = nn.ModuleList([self.template.model])
        for k in range(2, 6):
            fname = "esm1v_t33_650M_UR90S_%d.pt" % k
            data = torch.load(os.path.join(os.path.expanduser(path), fname),
                              map_location="cpu")
            member, _ = esm_lib.pretrained.load_model_and_alphabet_core(
                fname, data)
            self.members.append(member)

    def forward(self, graph, input, all_loss=None, metric=None):
        import math
        logp = None
        for k in range(len(self.members)):
            self.template.model = self.members[k]
            out = self.template(graph, input, all_loss, metric)
            member_logp = torch.log_softmax(out["logits"], dim=-1)
            logp = (member_logp if logp is None
                    else torch.logaddexp(logp, member_logp))
        logp = logp - math.log(len(self.members))
        out["logits"] = logp
        return out


@R.register("models.FusionNetwork")
class FusionNetwork(nn.Module, core.Configurable):

    def __init__(self, sequence_model, structure_model):
        super(FusionNetwork, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.output_dim = structure_model.output_dim

    def forward(self, graph, input, all_loss=None, metric=None, batch=None):
        # Sequence model
        output1 = self.sequence_model(graph, input, all_loss, metric)
        if "logits" in output1:
            logits = output1["logits"]
        else:
            logits = None
        node_output1 = output1["residue_feature"]

        # Structure model
        output2 = self.structure_model(graph, node_output1, all_loss, metric)
        node_feature = output2["node_feature"]
        graph_feature = output2["graph_feature"]

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "sequence_logits": logits,
            # IDEA-401: expose the frozen ESM residue features so snapshot
            # ensemble members can share one sequence-model forward pass.
            "esm_feature": node_output1,
        }