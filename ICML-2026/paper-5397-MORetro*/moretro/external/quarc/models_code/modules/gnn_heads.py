import torch
from torch import Tensor, nn
from lightning.pytorch.core.mixins import HyperparametersMixin
from chemprop.nn.metrics import CrossEntropyMetric

from quarc.models_code.modules.blocks import FFNBase

MAX_NUM_REACTANTS = 5


class GNNBaseHead(HyperparametersMixin, nn.Module):
    """Base predictor for GNN models."""

    def __init__(
        self,
        graph_input_dim: int,
        agent_input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.save_hyperparameters()

        agent_hidden_dim = 512
        self.agent_projector = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.LayerNorm(agent_hidden_dim),
            nn.ReLU(),
        )

        self.ffn = FFNBase(
            input_size=graph_input_dim + agent_hidden_dim,
            hidden_size=hidden_dim,
            output_size=output_dim,
            n_blocks=n_blocks,
            activation=activation,
        )
        self.output_transform = nn.Identity()

    def forward(self, graph_embedding: Tensor, agent_input: Tensor) -> Tensor:
        learned_agent = self.agent_projector(agent_input)
        combined = torch.cat([graph_embedding, learned_agent], dim=1)
        return self.ffn(combined)

    def encode(self, Z: Tensor, i: int) -> Tensor:
        pass


class GNNAgentHead(GNNBaseHead):
    """Predictor for agent identity: multilabel classification w/ CEloss."""

    _T_default_metric = CrossEntropyMetric

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction="none")


class GNNAgentHeadWithRxnClass(HyperparametersMixin, nn.Module):
    def __init__(
        self,
        graph_input_dim: int,  # Hidden dimension from message passing
        agent_input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.save_hyperparameters()

        agent_hidden_dim = 512
        self.agent_projector = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.LayerNorm(agent_hidden_dim),
            nn.ReLU(),
        )

        rxn_hidden_dim = 512
        self.reaction_projector = nn.Sequential(
            nn.Linear(2272, rxn_hidden_dim),  # FIXME: hardcoded
            nn.LayerNorm(rxn_hidden_dim),
            nn.ReLU(),
        )

        self.ffn = FFNBase(
            input_size=graph_input_dim + agent_hidden_dim + rxn_hidden_dim,
            hidden_size=hidden_dim,
            output_size=output_dim,
            n_blocks=n_blocks,
            activation=activation,
        )
        self.output_transform = nn.Identity()
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, graph_embedding: Tensor, agent_input: Tensor, rxn_class_onehot: Tensor
    ) -> Tensor:
        learned_agent = self.agent_projector(agent_input)
        learned_rxn = self.reaction_projector(rxn_class_onehot)
        combined = torch.cat([graph_embedding, learned_agent, learned_rxn], dim=1)
        return self.ffn(combined)


class GNNTemperatureHead(GNNBaseHead):
    """Predictor for binned temperature: multiclass classification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()


class GNNReactantAmountHead(GNNBaseHead):
    """Predictor that uses a shared MLP for each reactant amount prediction.

    This model processes input data in three main stages:

    1. Reaction-level processing:
        - Obtains `graph_embedding` from the GNN.
        - Transforms `agent_input` through a linear layer to produce `agent_embedding`.
        - Concatenates `graph_embedding` and `agent_embedding` to form `reaction_context`.

    2. Reactant-level processing:
        - For each reactant fingerprint (FP):
            - Transforms `reactant_FP` through a small MLP to obtain `learned_reactant_embedding`.

    3. Final prediction:
        - For each reactant:
            - Concatenates `reaction_context` and `learned_reactant_embedding` and passes it through a shared MLP to predict the amount.
    """

    def __init__(
        self,
        graph_input_dim: int,
        agent_input_dim: int,
        output_dim: int,  # number of bins
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
        **kwargs,
    ):
        super().__init__(
            graph_input_dim=graph_input_dim,
            agent_input_dim=agent_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            **kwargs,
        )

        agent_hidden_dim = 512
        self.agent_projector = nn.Sequential(
            nn.Linear(agent_input_dim, agent_hidden_dim),
            nn.LayerNorm(agent_hidden_dim),
            nn.ReLU(),
        )

        reactant_hidden_dim = 512
        self.reactant_encoder = nn.Sequential(
            nn.Linear(2048, reactant_hidden_dim),
            nn.LayerNorm(reactant_hidden_dim),
            nn.ReLU(),
        )

        # shared MLP
        reaction_context_size = graph_input_dim + agent_hidden_dim
        self.ffn = FFNBase(
            input_size=reaction_context_size
            + reactant_hidden_dim,  # reactant_graph_dim + agent_hidden_dim + reactantFP_hidden_dim
            hidden_size=hidden_dim,
            output_size=output_dim,  # number of bins
            n_blocks=n_blocks,
            activation=activation,
        )

        self.output_size = output_dim
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.output_transform = nn.Identity()

    def forward(self, graph_fp: Tensor, agent_input: Tensor, FP_reactants: Tensor) -> Tensor:
        """Forward pass for the GNN model.

        Args:
            graph_fp (Tensor): Graph fingerprint from GNN of shape (b, graph_fp_dim).
            agent_input (Tensor): Agent inputs, multi-hot encoded of shape (b, num_classes).
            FP_reactants (Tensor): Fingerprints for individual reactants of shape (b, MAX_NUM_REACTANTS, 2048).

        Returns:
            Tensor: Logits for each reactant amount bin of shape (b, MAX_NUM_REACTANTS, num_bins).
        """
        # create reaction context
        agent_embedding = self.agent_projector(agent_input)  # b x hidden_dim
        reaction_context = torch.cat([graph_fp, agent_embedding], dim=1)
        reaction_context = reaction_context.unsqueeze(1).expand(
            -1, MAX_NUM_REACTANTS, -1
        )  # b x MAX_NUM_REACTANTS x (graph_fp_dim + hidden_dim)

        # reactant embeddings
        batch_size = FP_reactants.shape[0]
        FP_flat = FP_reactants.view(-1, 2048)  # (b * MAX_NUM_REACTANTS) x 2048
        learned_reactants = self.reactant_encoder(FP_flat)  # (b * MAX_NUM_REACTANTS) x hidden_dim
        learned_reactants = learned_reactants.view(
            batch_size, MAX_NUM_REACTANTS, -1
        )  # b x MAX_NUM_REACTANTS x hidden_dim

        # combine and predict
        x = torch.cat(
            [reaction_context, learned_reactants], dim=2
        )  # b x MAX_NUM_REACTANTS x (graph_fp_dim + hidden_dim_agent + hidden_dim_reactant)
        x = x.view(
            -1, x.shape[-1]
        )  # (b * MAX_NUM_REACTANTS) x (graph_fp_dim + hidden_dim_agent + hidden_dim_reactant)

        shared_outputs = self.ffn(x)  # (b * MAX_NUM_REACTANTS) x num_bins
        outputs = shared_outputs.view(
            -1, MAX_NUM_REACTANTS, self.output_size
        )  # b x MAX_NUM_REACTANTS x num_bins

        return outputs


class GNNAgentAmountHead(GNNBaseHead):
    """Predictor for binned agent amounts: multiclass classification (oneshot)"""

    def __init__(
        self,
        graph_input_dim: int,
        agent_input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_blocks: int = 3,
        activation: str = "ReLU",
        **kwargs,
    ):
        mlp_out_size = agent_input_dim * output_dim
        super().__init__(
            graph_input_dim=graph_input_dim,
            agent_input_dim=agent_input_dim,
            output_dim=mlp_out_size,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            activation=activation,
            **kwargs,
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.num_classes = agent_input_dim
        self.num_bins = output_dim

    def forward(self, graph_embedding: Tensor, agent_input: Tensor) -> Tensor:
        o = super().forward(graph_embedding, agent_input)
        o = o.view(-1, self.num_classes, self.num_bins)
        return o
