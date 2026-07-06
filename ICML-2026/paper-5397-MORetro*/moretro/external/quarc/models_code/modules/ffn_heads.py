import torch
from torch import Tensor, nn
from lightning.pytorch.core.mixins import HyperparametersMixin

from quarc.models_code.modules.blocks import FFNBase

MAX_NUM_REACTANTS = 5


class FFNBaseHead(HyperparametersMixin, nn.Module):
    """Base predictor for fingerprint models."""

    def __init__(
        self,
        fp_dim: int,  # FP_length
        agent_input_dim: int,  # num_classes
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
        additional_input_dim: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.ffn = FFNBase(
            input_size=fp_dim * 2 + agent_input_dim + additional_input_dim,
            hidden_size=hidden_dim,
            output_size=output_dim,
            n_blocks=n_blocks,
            activation=activation,
        )

        self._criterion = None

    @property
    def criterion(self):
        return self._criterion

    def forward(self, rxn_fp: Tensor, agent_input: Tensor) -> Tensor:
        x = torch.cat([rxn_fp, agent_input], dim=1)
        return self.ffn(x)


class FFNAgentHead(FFNBaseHead):
    """Predictor for agent identity: multilabel classification w/ CEloss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._criterion = nn.CrossEntropyLoss(reduction="none")


class FFNAgentHeadWithRxnClass(FFNBaseHead):
    """Predictor for agent identity: multilabel classification w/ CEloss."""

    def __init__(
        self,
        fp_dim: int,  # FP_length
        agent_input_dim: int,  # num_classes
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
    ):
        rxn_class_dim = 2272
        super().__init__(
            fp_dim=fp_dim,
            agent_input_dim=agent_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            additional_input_dim=rxn_class_dim,
        )

        self._criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, rxn_fp: Tensor, agent_input: Tensor, rxn_class: Tensor) -> Tensor:
        x = torch.cat([rxn_fp, agent_input, rxn_class], dim=1)
        return self.ffn(x)


class FFNTemperatureHead(FFNBaseHead):
    """Predictor for binned temperature: multiclass classifcation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._criterion = nn.CrossEntropyLoss()


class FFNReactantAmountHead(FFNBaseHead):
    """Predictor for binned reactant amounts: multiclass classification with shared MLP."""

    def __init__(
        self,
        fp_dim: int,
        agent_input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(
            fp_dim=fp_dim,
            agent_input_dim=agent_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            **kwargs,
        )

        self.ffn = FFNBase(
            input_size=fp_dim * 3 + agent_input_dim,
            hidden_size=hidden_dim,
            output_size=output_dim,
            n_blocks=n_blocks,
            activation=activation,
        )
        self.output_size = output_dim
        self._criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    def forward(self, FP_inputs: Tensor, a_inputs: Tensor, FP_reactants: Tensor) -> Tensor:
        reaction_context = torch.cat([FP_inputs, a_inputs], dim=1)
        reaction_context = reaction_context.unsqueeze(1).expand(
            -1, MAX_NUM_REACTANTS, -1
        )  # b x MAX_NUM_REACTANTS x (4096 + num_classes)

        # concat MLP inputs and split each reactant as a separate data point
        x = torch.cat(
            [reaction_context, FP_reactants], dim=2
        )  # b x MAX_NUM_REACTANTS x (4096 + num_classes + 2048)
        x = x.view(-1, x.shape[-1])  # (b * MAX_NUM_REACTANTS) x (4096 + num_classes + 2048)

        outputs = self.ffn(x)  # (b * MAX_NUM_REACTANTS) x num_binned_classes
        outputs = outputs.view(
            -1, MAX_NUM_REACTANTS, self.output_size
        )  # b x MAX_NUM_REACTANTS x num_binned_classes

        return outputs


class FFNAgentAmountHead(FFNBaseHead):
    """Predictor for binned agent amounts."""

    def __init__(
        self,
        fp_dim: int,
        agent_input_dim: int,
        output_dim: int,  # num_bins only
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
        **kwargs,
    ):
        mlp_out_size = agent_input_dim * output_dim  # num_classes x num_bins
        super().__init__(
            fp_dim=fp_dim,
            agent_input_dim=agent_input_dim,
            output_dim=mlp_out_size,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            **kwargs,
        )
        self._criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.num_classes = agent_input_dim
        self.num_bins = output_dim

    def forward(self, rxn_fp: Tensor, agent_input: Tensor) -> Tensor:
        x = torch.cat([rxn_fp, agent_input], dim=1)
        o = self.ffn(x)
        o = o.view(-1, self.num_classes, self.num_bins)  # b x num_classes x num_bins
        return o
