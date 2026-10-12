import torch
from torch import nn

from torchdrug import core, tasks
from torchdrug.core import Registry as R

from s3f import gvp
    

@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):

    def __init__(self, model, mask_rate=0.15, dropout=0.5, graph_construction_model=None, plddt_threshold=None, esm_weight=0.0, wt_esm_weight=0.0):
        super(ResidueTypePrediction, self).__init__(model, mask_rate=mask_rate, num_mlp_layer=1, graph_construction_model=graph_construction_model)
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        num_label = 20
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_output_dim, num_label)
        self.plddt_threshold = plddt_threshold
        # IDEA-303: inference-time fusion weight of the frozen ESM logits into
        # the scoring distribution (convex combination in log-prob space).
        # 0.0 = pure S2F (default, unchanged behavior).
        self.esm_weight = esm_weight
        # IDEA-404: weight of the unmasked wild-type ESM marginal as a third
        # scoring expert (0.0 = off, unchanged behavior). The wt distribution
        # does not depend on the mutation, so the batch carries an unmasked
        # clone of each windowed sequence ("wt_graph") for the extra forward.
        self.wt_esm_weight = wt_esm_weight
        # IDEA-401: optional prediction-level ensemble members. The ESM branch
        # is frozen and shared, so a member is just its GVP encoder + head
        # weights from another checkpoint of the same architecture.
        self.ens_models = nn.ModuleList()
        self.ens_heads = nn.ModuleList()
        self.ens_weights = None
        # IDEA-602: when True, inference() additionally returns the per-residue
        # logits of the three fused components (S2F head, masked-ESM, WT-ESM)
        # so evaluate.py can dump exact per-row component scores for offline
        # weight-simplex search. Costs one extra sequence-model forward on the
        # batch's unmasked clones; the fused prediction is unchanged.
        self.dump_components = False
        # IDEA-606: optional scoring-only sequence backbone (assigned by
        # evaluate.py from S2F_SCORING_ESM). When set, the masked-ESM and
        # WT-ESM scoring branches use ITS logits (ESM-1v / ESM-2-3B) while the
        # GVP's node features still come from the trained ESM-2-650M inside
        # FusionNetwork. None = champion behavior.
        self.scoring_esm = None

    def add_ensemble_member(self, structure_model, head):
        """Register another GVP+head snapshot; log-probs are averaged over
        members in inference()."""
        self.ens_models.append(structure_model)
        self.ens_heads.append(head)

    def set_ensemble_weights(self, weights):
        """Convex weights over [main model] + members (normalized to sum 1).
        None (default) = uniform average."""
        self.ens_weights = weights

    def preprocess(self, train_set, valid_set, test_set):
        return

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        # Random select residues to be masked
        num_nodes = graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        node_index = node_index.clamp(max=num_cum_nodes[-1]-1)

        target = graph.residue_type[node_index]
        mask_id = self.model.sequence_model.alphabet.get_idx("<mask>")
        with graph.residue():
            graph.residue_feature[node_index] = 0
            graph.residue_type[node_index] = mask_id

        # 80% mask, 10% replace, 10% unchange
        replace_prob = torch.rand_like(target.float())
        replace_mask = replace_prob < 0.1
        mutant_residue_type = torch.randint_like(replace_mask.long(), 20)
        with graph.residue():
            graph.residue_feature[node_index[replace_mask], mutant_residue_type[replace_mask]] = 1
            graph.residue_type[node_index[replace_mask]] = mutant_residue_type[replace_mask]
        unchanged_mask = (replace_prob < 0.2) & (replace_prob >= 0.1)
        with graph.residue():
            graph.residue_feature[node_index[unchanged_mask], target[unchanged_mask]] = 1
            graph.residue_type[node_index[unchanged_mask]] = target[unchanged_mask]

        if self.graph_construction_model:
           graph = self.graph_construction_model.apply_edge_layer(graph)
        input = graph.residue_feature.float()

        if isinstance(self.model.structure_model, gvp.SurfGVP):
            output = self.model(graph, input, batch["surf_graph"], all_loss, metric)
        else:
            output = self.model(graph, input, all_loss, metric)
        node_feature = output["node_feature"][node_index]

        node_feature = self.dropout(node_feature)
        pred = self.linear(node_feature)

        return pred, target

    def inference(self, batch):
        graph = batch["graph"]

        if self.graph_construction_model:
           graph = self.graph_construction_model(graph)
        input = graph.residue_feature.float()

        if isinstance(self.model.structure_model, gvp.SurfGVP):
            output = self.model(graph, input, batch["surf_graph"])
        else:
            output = self.model(graph, input)
        node_feature = output["node_feature"]
        pred = self.linear(node_feature)

        if len(self.ens_models):
            # IDEA-401: snapshot ensembling at the prediction level. Average the
            # per-residue log-softmax distributions over members (the shared
            # ESM branch runs once; members only add a GVP + head forward).
            # log_softmax is applied before the average so each member's
            # distribution enters with equal weight regardless of scale.
            esm_feature = output.get("esm_feature")
            logps = [torch.log_softmax(pred, dim=-1)]
            for member_model, member_head in zip(self.ens_models, self.ens_heads):
                member_output = member_model(graph, esm_feature)
                member_pred = member_head(member_output["node_feature"])
                logps.append(torch.log_softmax(member_pred, dim=-1))
            logps = torch.stack(logps, dim=0)
            if self.ens_weights is None:
                pred = logps.mean(dim=0)
            else:
                weights = torch.tensor(self.ens_weights, device=logps.device,
                                       dtype=logps.dtype)
                weights = weights / weights.sum()
                pred = (logps * weights.view(-1, 1, 1)).sum(dim=0)

        if self.plddt_threshold:
            # for AlphaFold2-predicted structure pdb files, plddt is saved as b_factor
            plddt_mask = batch["graph"].b_factor < self.plddt_threshold
            pred[plddt_mask] = output["sequence_logits"][plddt_mask]

        comps = None
        if self.dump_components:
            # IDEA-602: raw logits of the three fused components, captured
            # BEFORE any fusion. The downstream row scoring applies
            # log_softmax + mutant-minus-wild-type per component, which is
            # linear, so any (w, c) fusion is an exact offline recombination.
            comps = {"s2f": pred}
            if self.scoring_esm is not None:
                comps["masked"] = self.scoring_esm(graph, None)["logits"]
            elif output.get("sequence_logits") is not None:
                comps["masked"] = output["sequence_logits"]
            wt_graph = batch.get("wt_graph")
            if wt_graph is not None:
                wt_output = (self.scoring_esm(wt_graph, None)
                             if self.scoring_esm is not None
                             else self.model.sequence_model(wt_graph, None))
                comps["wt"] = wt_output["logits"]

        if self.esm_weight:
            # IDEA-303: geometric-mixture fusion of the S2F head distribution
            # with the frozen ESM logits that FusionNetwork already returns
            # (currently discarded when plddt_threshold is falsy).
            # logp = (1 - w) * log_softmax(S2F_logits) + w * log_softmax(ESM_logits).
            # The downstream scoring takes a per-residue mutant-minus-wild-type
            # difference of log-softmax rows, so the row-wise normalizing
            # constant of this mixture cancels exactly.
            # IDEA-606: with a swapped scoring backbone, the masked-ESM logits
            # come from scoring_esm instead of the FusionNetwork ESM-2-650M.
            if self.scoring_esm is not None:
                sequence_logits = self.scoring_esm(graph, None)["logits"]
            else:
                sequence_logits = output.get("sequence_logits")
            if sequence_logits is not None:
                s2f_logp = torch.log_softmax(pred, dim=-1)
                esm_logp = torch.log_softmax(sequence_logits, dim=-1)
                pred = (1 - self.esm_weight) * s2f_logp + self.esm_weight * esm_logp

        if self.wt_esm_weight:
            # IDEA-404: blend in the unmasked wild-type ESM marginal as a third
            # expert. pred is already the (S2F, masked-ESM) mixture in log-prob
            # space; the wt-conditional distribution is mutation-independent, so
            # one extra sequence-model forward on the batch's unmasked clones
            # suffices. Final weights: a=(1-c)(1-w), b=(1-c)w, c=wt_esm_weight
            # (sums to 1, a:b ratio fixed at the known-good 0.8:0.2).
            wt_graph = batch.get("wt_graph")
            if wt_graph is not None:
                if not self.esm_weight:
                    pred = torch.log_softmax(pred, dim=-1)
                if self.scoring_esm is not None:
                    wt_output = self.scoring_esm(wt_graph, None)
                else:
                    wt_output = self.model.sequence_model(wt_graph, None)
                wt_logp = torch.log_softmax(wt_output["logits"], dim=-1)
                pred = (1 - self.wt_esm_weight) * pred + self.wt_esm_weight * wt_logp

        return pred, graph.num_residues, comps