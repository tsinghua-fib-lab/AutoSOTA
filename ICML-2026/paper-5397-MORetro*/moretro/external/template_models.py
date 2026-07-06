# type: ignore

import logging
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Any
from collections import defaultdict, OrderedDict
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

from rdchiral.main import rdchiralRun, rdchiralRunText
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

logger = logging.getLogger(__name__)


def get_activation(name: str) -> nn.Module:
    _activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    return _activations[name]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_act: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden_act(self.linear(x))


class TemplateModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _run_templates(self):
        pass

    def _smis_to_fp(
        self, smiles: str | list[str], fp_size: int = 2048, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Convert a list of SMILES string to a fingerprint tensor.
        """
        fps = []
        if isinstance(smiles, str):
            smiles = [smiles]

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            fp = GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=fp_size, useChirality=True
            )
            fp = torch.tensor(fp, dtype=torch.float, device=device)
            fps.append(fp)

        if len(fps) == 1:
            return fps[0]
        return torch.stack(fps)


class TemplRel(TemplateModel):
    def __init__(self, args):
        super().__init__()
        if isinstance(args.hidden_sizes, str):
            self.hidden_sizes = [int(size) for size in args.hidden_sizes.split(",")]

        self.args = args
        self.layers = self._build_layers(args)
        self.output_layer = nn.Linear(
            self.hidden_sizes[-1], args.n_templates, bias=True
        )

        self.dropout = nn.Dropout(args.dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    def _build_layers(self, args) -> nn.ModuleList:
        hidden_act = get_activation(args.hidden_activation)
        # input projection layer; no skip connection here
        layers = nn.ModuleList(
            [Dense(args.fp_size, self.hidden_sizes[0], hidden_act=hidden_act)]
        )

        for layer_i in range(len(self.hidden_sizes) - 1):
            in_features = self.hidden_sizes[layer_i]
            out_features = self.hidden_sizes[layer_i + 1]

            if args.skip_connection == "none":
                layer = Dense(in_features, out_features, hidden_act=hidden_act)
            else:
                raise ValueError(f"Unsupported skip_connection: {args.skip_connection}")

            layers.append(layer)

        return layers

    def predict(
        self, products: str | list[str], top_n: int, templates: dict
    ) -> list[list[dict[str, Any]]]:
        # Handle both single product and list of products
        if isinstance(products, str):
            products = [products]
            single_input = True
        else:
            single_input = False

        target_fp = self._smis_to_fp(products, device=next(self.parameters()).device)

        # Get fingerprints and rdchiral reactants for each product
        target_rds = []
        for prod in products:
            target_rds.append(rdchiralReactants(prod))

        with torch.no_grad():
            # Explicitly set model to eval mode to ensure no dropout
            self.eval()
            output = self(target_fp)

        # Process all products with the same logic
        all_predictions = []
        probs = torch.softmax(output, dim=1 if len(products) > 1 else 0).cpu()

        # Handle dimension for single vs multiple products
        if single_input:
            probs = probs.unsqueeze(0)  # Add batch dimension for consistency

        for idx, (prod, target_rd) in enumerate(zip(products, target_rds)):
            top_scores, top_indices = torch.topk(probs[idx], top_n)
            top_scores = top_scores.detach().numpy()
            top_indices = top_indices.detach().numpy()

            predictions = []
            for i in range(top_n):
                template = templates[top_indices[i]]
                pred_reactants = self._run_templates(target_rd, template)
                if len(pred_reactants) > 0:
                    for output_react in pred_reactants:
                        predictions.append(
                            {
                                "score": top_scores[i],
                                "reactants": output_react,
                                "template": template,
                            }
                        )

            predictions = self._postprocessing(predictions, prod)
            all_predictions.append(predictions)

        return all_predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        logits = self.output_layer(x)  # returning *unnormalized* logits
        return logits

    def _run_templates(self, product: rdchiralReactants, template: str) -> list[str]:
        """
        Run the template given product and corresponding template

        Args:
            product: The product to run the template on.
            template: The template to use for the product.

        Returns:
            The generated output after applying the template to the product.

        """
        reactants = template.split(">>")[0].split(".")
        if len(reactants) > 1:
            template = "(" + template.replace(">>", ")>>")
        template_rd = rdchiralReaction(template)
        try:
            output = rdchiralRun(template_rd, product)
        except Exception as e:
            logger.error(f"Error occurred while running template: {e}")
            return []
        result = []
        for out in output:
            result.append(out.split("."))
        return result

    def _postprocessing(self, predictions: list[dict], product: str) -> list[dict]:
        """
        Only retain unique reactants, templates and scores are added together
        """
        prec_to_score = {}
        prec_to_template = {}
        for i in range(len(predictions)):
            prec = frozenset(predictions[i]["reactants"])
            if prec in prec_to_score:
                prec_to_score[prec] += predictions[i]["score"]
                prec_to_template[prec].append(predictions[i]["template"])
            else:
                prec_to_score[prec] = predictions[i]["score"]
                prec_to_template[prec] = [predictions[i]["template"]]

        # Renormalize scores
        total_score = sum(prec_to_score.values())
        for prec in prec_to_score:
            prec_to_score[prec] /= total_score
        final_predictions = []
        for prec in sorted(prec_to_score.keys(), key=lambda x: sorted(list(x))):
            final_predictions.append(
                {
                    "rxn_smiles": ".".join(prec) + ">>" + product,
                    "score": prec_to_score[prec],
                    "template": prec_to_template[prec],
                    "reactants": list(prec),
                }
            )

        return final_predictions


class PDVN(TemplateModel):
    def __init__(
        self,
        trained_model: Path,
        template_path: Path,
        device: str = "cuda",
        fp_dim: int = 2048,
        realistic_filter: bool = False,
    ):
        super(PDVN, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = self.load_model(trained_model, template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device == "cuda":
            self.net.to(device)

        self.realistic_filter = realistic_filter

        self.reference_net, _ = self.load_model(
            template_path.parent / "saved_rollout_state_1_2048.ckpt",
            template_path,
            fp_dim,
        )
        self.reference_net.eval()
        if device == "cuda":
            self.reference_net.to(device)

    def forward(
        self, arr: torch.Tensor, topk: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.net(arr)
        preds_reference = self.reference_net(arr).detach()

        if self.realistic_filter:
            preds_reference, idx_topk = torch.topk(preds_reference, k=topk)
            preds = preds.gather(1, idx_topk)
        else:
            preds, idx_topk = torch.topk(preds, k=topk)
            preds_reference = preds_reference.gather(1, idx_topk)

        return preds, preds_reference, idx_topk

    def predict(
        self,
        products: str | list[str],
        topk: int,
        templates=None,
        backward: bool = True,
    ) -> list[list[dict[str, Any]]]:
        if isinstance(products, str):
            products = [products]

        arr = self._smis_to_fp(products, self.fp_dim, device=self.device)
        if len(products) == 1 and arr.dim() == 1:
            arr = arr.unsqueeze(0)

        if self.device == "cuda":
            arr = arr.to(self.device)

        preds, preds_reference, idx = self.forward(arr, topk=topk)
        probs = F.softmax(preds, dim=1)
        probs_reference = F.softmax(preds_reference, dim=1)
        preds, preds_reference, idx, probs, probs_reference = (
            preds.cpu(),
            preds_reference.cpu(),
            idx.cpu(),
            probs.cpu(),
            probs_reference.cpu(),
        )

        all_predictions = []
        for i, product in enumerate(products):
            rule_k = [self.idx2rules[ids] for ids in idx[i].numpy().tolist()]
            reactants = []
            scores = []
            scores_reference = []
            templates = []
            templates_idx = []

            for j, rule in enumerate(rule_k):
                out1 = []
                try:
                    if backward:
                        out1 = rdchiralRunText(rule, product)
                    else:
                        rxn_prod, rxn_agent, rxn_react = rule.split(">")
                        reversed_rule = (
                            "(" + rxn_react + ")>" + rxn_agent + ">" + rxn_prod[1:-1]
                        )
                        out1 = rdchiralRunText(reversed_rule, product)
                    if len(out1) == 0:
                        continue
                    out1 = sorted(out1)
                    for reactant in out1:
                        reactants.append(reactant)
                        scores.append(probs[i][j].item() / len(out1))
                        scores_reference.append(
                            probs_reference[i][j].item() / len(out1)
                        )
                        templates.append(rule)
                        templates_idx.append(
                            j if self.realistic_filter else idx[i][j].item()
                        )
                except (ValueError, RuntimeError) as e:
                    """
                    RuntimeError: Pre-condition Violation
                    Stereo atoms should be specified before specifying CIS/TRANS bond stereochemistry
                    Violation occurred on line 288 in file Code/GraphMol/Bond.h
                    Failed Expression: what <= STEREOE || getStereoAtoms().size() == 2
                    RDKIT: 2020.09.1
                    BOOST: 1_73
                    """
                except (IndexError, KeyError) as e:
                    """
                    rdchiral bug during function call rdchiralRunText(rule, mol)
                    This error can be reprobuced by the following code:
                    mol = 'C[C@H](OC(=O)C=O)C(=O)O'
                    rule = '([#8:1]-[C:2](=[O;D1;H0:3])-[CH;D2;+0:4]=[O;H0;D1;+0:5])>>[#8:1]-[C:2](=[O;D1;H0:3])-[C@@H;D3;+0:4](-[OH;D1;+0:5])-[C@H;D3;+0:4](-[OH;D1;+0:5])-[C:2](-[#8:1])=[O;D1;H0:3]'
                    out1 = rdchiralRunText(rule, mol)
                    """

            if len(reactants) == 0:
                all_predictions.append([])
                continue

            reactants_d = defaultdict(list)
            for r, s, sr, t, id in zip(
                reactants, scores, scores_reference, templates, templates_idx
            ):
                if "." in r:
                    str_list = sorted(r.strip().split("."))
                    reactants_d[".".join(str_list)].append((s, sr, t, id))
                else:
                    reactants_d[r].append((s, sr, t, id))

            reactants, scores, _, templates, _ = self._merge(reactants_d)
            total = sum(scores)
            scores = [s / total for s in scores]

            final_predictions = []
            for k in range(len(reactants)):
                final_predictions.append(
                    {
                        "rxn_smiles": reactants[k] + ">>" + product,
                        "score": scores[k],
                        "template": [templates[k]],
                        "reactants": reactants[k].split("."),
                    }
                )

            all_predictions.append(final_predictions)

        return all_predictions

    def _merge(self, reactant_d: defaultdict) -> tuple:
        ret = []
        for reactant, l in reactant_d.items():
            ss, srs, ts, ids = zip(*l)
            ret.append((reactant, sum(ss), sum(srs), list(ts)[0], list(ids)[0]))
        reactants, scores, scores_reference, templates, templates_idx = zip(
            *sorted(ret, key=lambda item: item[1], reverse=True)
        )
        return (
            list(reactants),
            list(scores),
            list(scores_reference),
            list(templates),
            list(templates_idx),
        )

    def load_model(self, state_path: Path, template_rule_path: Path, fp_dim: int):
        template_rules = {}
        with open(template_rule_path, "r") as f:
            for i, l in tqdm(enumerate(f), desc="template rules"):
                rule = l.strip()
                template_rules[rule] = i
        idx2rule = {}
        for rule, idx in template_rules.items():
            idx2rule[idx] = rule
        rollout = RolloutPolicyNet(len(template_rules), fp_dim=fp_dim)
        state_dict = torch.load(state_path, map_location="cpu")

        # Create new OrderedDict that does not contain 'module.'
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # remove `module.`
            new_state_dict[name] = v
        rollout.load_state_dict(new_state_dict)
        return rollout, idx2rule


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim, n_rules)

    def forward(self, x, y=None, loss_fn=nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        if y is not None:
            return loss_fn(x, y)
        else:
            return x


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).parent.parent / "models" / "pdvn"
    state_path = path / "rollout_model.ckpt"
    template_path = path / "template_rules_1.dat"
    pdvn = PDVN(
        state_path, template_path, device=-1, fp_dim=2048, realistic_filter=True
    )
    test_smiles = ["CCOC(=O)C1=CC=CC=C1", "CCN1C(=O)C=CC1=O", "CCOC(=O)C=CC=C"]
    predictions = pdvn.predict(test_smiles, topk=20, backward=True)
