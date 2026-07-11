from __future__ import annotations

import tomllib

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from taming_the_ito_lyon.config.config_options import (
    AdjointType,
    ControlInterpolationType,
    Datasets,
    ExtrapolationSchemeType,
    HiddenStateMode,
    LossType,
    ManifoldType,
    ModelType,
    Optimizer,
    RoughSolution,
    SolverType,
    StepsizeControllerType,
    TrainingMode,
)


class ExperimentConfig(BaseModel):
    """Aggregated non-model experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    # Model selection
    model_type: ModelType = Field(description="Which model architecture to use")

    # Dataset
    dataset_name: Datasets = Field(
        description="Dataset name key from config_options.Datasets"
    )

    @field_validator("dataset_name", mode="before")
    @classmethod
    def coerce_dataset_name(cls, v: object) -> object:
        # Allow TOML strings like "ou_process" (matching enum member names).
        if isinstance(v, str):
            key = v.strip()
            try:
                return Datasets[key.upper()]
            except KeyError:
                return v
        return v

    train_fraction: PositiveFloat = Field(
        default=0.8, le=1.0, description="Fraction of data for training"
    )
    val_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for validation"
    )
    test_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for testing"
    )
    synthetic_gbm_dim: PositiveInt | None = Field(
        default=None,
        description=(
            "Optional channel dimension override for the in-memory synthetic GBM "
            "benchmark dataset."
        ),
    )

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> ExperimentConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"Sum of train, val, and test fractions ({total:.4f}) cannot exceed 1.0"
            )
        return self

    # Extrapolation settings
    extrapolation_scheme: ExtrapolationSchemeType | None = Field(
        default=None, description="Extrapolation scheme"
    )
    n_recon: PositiveInt | None = Field(
        default=None,
        description="Number of reconstruction points for extrapolation (None for standard mode)",
    )

    @model_validator(mode="after")
    def validate_extrapolation_params(self) -> ExperimentConfig:
        if self.extrapolation_scheme is not None:
            # Extrapolation requires the selected model type to support an
            # `extrapolation_scheme` interface.
            if self.model_type not in (
                ModelType.NCDE,
                ModelType.NRDE,
                ModelType.BNRDE,
                ModelType.GRU,
                ModelType.LSTM,
                ModelType.XLSTM,
                ModelType.STACKED_XLSTM,
            ):
                raise ValueError(
                    "extrapolation_scheme is only supported for model_type in "
                    "{ncde, nrde, bnrde, gru, lstm, xlstm, stacked_xlstm}."
                )
        return self

    @property
    def uses_flat_so3_driver(self) -> bool:
        return (
            self.model_type == ModelType.M_ODE
            or self.extrapolation_scheme == ExtrapolationSchemeType.SO3_SG
        )

    # Optimizer
    optimizer: Optimizer = Field(description="Optimizer name")
    learning_rate: PositiveFloat = Field(le=1.0, description="Optimizer learning rate")
    weight_decay: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Optimizer weight decay"
    )
    max_grad_norm: PositiveFloat | None = Field(
        default=None,
        description="Maximum gradient norm for clipping (None for no clipping)",
    )

    # Training
    loss: LossType = Field(description="Loss function to use")
    seed: PositiveInt = Field(description="PRNG seed")
    batch_size: PositiveInt = Field(description="Batch size")
    epochs: PositiveInt = Field(description="Number of epochs")
    early_stopping_patience: PositiveInt = Field(
        default=25, description="Epochs with no val improvement before stopping"
    )

    ito_level2_mmd_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight for the Itô / branched level-2 distribution-matching MMD loss "
            "(only applied to matching branched-signature experiments)."
        ),
    )

    @model_validator(mode="after")
    def validate_early_stopping_patience(self) -> ExperimentConfig:
        if self.early_stopping_patience > self.epochs:
            raise ValueError(
                "early_stopping_patience must be <= epochs "
                f"(got patience={self.early_stopping_patience}, epochs={self.epochs})"
            )
        return self

    manifold: ManifoldType = Field(description="Manifold to use")
    hidden_state_mode: HiddenStateMode = Field(
        description="Hidden state space to use",
        validation_alias=AliasChoices("hidden_state_mode", "hidden_manifold"),
    )

    evolving_out: bool = Field(description="Whether to evolve the output")

    # Training mode (conditional vs unconditional generator)
    training_mode: TrainingMode = Field(
        default=TrainingMode.CONDITIONAL,
        description="Training mode: 'conditional' uses dataset controls; 'unconditional' samples a driver internally.",
    )

    # Performance / logging
    tqdm_update_interval: PositiveInt = Field(
        default=10,
        description=(
            "Update tqdm postfix every N steps. Note: updating postfix typically "
            "requires syncing JAX device arrays to host, so larger values improve "
            "training throughput."
        ),
    )

    @model_validator(mode="after")
    def validate_training_mode(self) -> ExperimentConfig:
        match self.training_mode:
            case TrainingMode.CONDITIONAL:
                if self.dataset_name in (
                    Datasets.BLACK_SCHOLES,
                    Datasets.BERGOMI,
                    Datasets.ROUGH_BERGOMI,
                    Datasets.SIMPLE_RBERGOMI,
                ):
                    raise ValueError(
                        "Rough volatility datasets do not support conditional training"
                    )
            case TrainingMode.UNCONDITIONAL:
                if self.extrapolation_scheme is not None:
                    raise ValueError(
                        "extrapolation_scheme must be None when training_mode='unconditional'"
                    )
                if self.model_type == ModelType.M_ODE:
                    raise ValueError(
                        "model_type='m_ode' only supports conditional training"
                    )
            case _:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
        return self


class SolverConfig(BaseModel):
    """Solver configuration."""

    stepsize_controller: StepsizeControllerType = Field(
        description="Stepsize controller to use"
    )
    solver: SolverType = Field(description="ODE solver to use")
    adjoint: AdjointType = Field(
        default=AdjointType.RECURSIVE_CHECKPOINT,
        description="Adjoint method for backpropagation through the ODE solve",
    )

    # Solver tolerances
    rtol: PositiveFloat = Field(description="Relative tolerance for solver")
    atol: PositiveFloat = Field(description="Absolute tolerance for solver")
    dtmin: PositiveFloat = Field(description="Minimum time step for solver")
    dt0: PositiveFloat = Field(default=0.01, description="Initial step size for solver")

    @field_validator("solver", mode="before")
    @classmethod
    def coerce_solver_name(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "CG2":
            return SolverType.CG2
        return v

    @model_validator(mode="after")
    def validate_solver_tolerances(self) -> SolverConfig:
        if not (0.0 < float(self.rtol) < 1.0):
            raise ValueError(f"rtol must be in (0, 1), got {self.rtol}")
        if not (0.0 < float(self.atol) < 1.0):
            raise ValueError(f"atol must be in (0, 1), got {self.atol}")
        if not (0.0 < float(self.dtmin) <= 1.0):
            raise ValueError(f"dtmin must be in (0, 1], got {self.dtmin}")
        if not (0.0 < float(self.dt0) <= 1.0):
            raise ValueError(f"dt0 must be in (0, 1], got {self.dt0}")
        return self


class NCDEConfig(BaseModel):
    """Top-level NCDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

    # Model params
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP hidden state dimension"
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")
    control_interpolation: ControlInterpolationType = Field(
        default=ControlInterpolationType.HERMITE_CUBIC,
        description="Interpolation scheme used for the control path",
    )


class NRDEConfig(BaseModel):
    """Top-level NRDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

    # Model params
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP hidden state dimension"
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")

    # Log-signature window size in data steps (polyline uses window_size+1 points)
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )


class BNRDEConfig(BaseModel):
    """Top-level BNRDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

    # Model params
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP hidden state dimension"
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    hidden_size: PositiveInt | None = Field(
        default=None,
        description="Euclidean hidden state dimension for BNRDE",
    )
    initial_state_param_dim: PositiveInt | None = Field(
        default=None,
        description="Initial-state parameter dimension for problem-manifold BNRDE",
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )
    rough_solution: RoughSolution = Field(description="roughrax solution convention")


class GRUConfig(BaseModel):
    """Top-level GRU configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

    # Model params
    gru_state_dim: PositiveInt = Field(description="GRU hidden state dimension")
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP width (controls hidden layer width)",
        validation_alias=AliasChoices("init_hidden_dim", "mlp_hidden_dim"),
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")


class LSTMConfig(BaseModel):
    """Top-level stacked LSTM configuration."""

    model_config = ConfigDict(extra="forbid")

    lstm_state_dim: PositiveInt = Field(description="LSTM hidden state dimension")
    num_layers: PositiveInt = Field(
        default=2, description="Number of stacked recurrent layers"
    )
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP width (controls hidden layer width)",
        validation_alias=AliasChoices("init_hidden_dim", "mlp_hidden_dim"),
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")


class XLSTMConfig(BaseModel):
    """Top-level single-layer xLSTM configuration."""

    model_config = ConfigDict(extra="forbid")

    d_model: PositiveInt = Field(description="Model width")
    num_heads: PositiveInt = Field(description="Number of mLSTM heads")
    d_conv: PositiveInt = Field(default=4, description="Depthwise convolution width")
    xlstm_expand: PositiveInt = Field(default=2, description="Inner expansion factor")
    ffn_expand: PositiveInt = Field(default=2, description="FFN expansion factor")
    use_ffn: bool = Field(default=True, description="Whether to include the FFN block")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")


class StackedXLSTMConfig(BaseModel):
    """Top-level stacked xLSTM configuration."""

    model_config = ConfigDict(extra="forbid")

    d_model: PositiveInt = Field(description="Model width")
    num_heads: PositiveInt = Field(description="Number of mLSTM heads")
    num_layers: PositiveInt = Field(description="Number of stacked xLSTM layers")
    d_conv: PositiveInt = Field(default=4, description="Depthwise convolution width")
    xlstm_expand: PositiveInt = Field(default=2, description="Inner expansion factor")
    ffn_expand: PositiveInt = Field(default=2, description="FFN expansion factor")
    use_ffn: bool = Field(default=True, description="Whether to include the FFN block")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")


class MODEConfig(BaseModel):
    """Top-level manifold Neural ODE configuration."""

    model_config = ConfigDict(extra="forbid")

    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    output_scale: PositiveFloat = Field(
        default=1.0,
        description="Bound on local coordinate velocity after tanh",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_config: ExperimentConfig
    solver_config: SolverConfig
    ncde_config: NCDEConfig | None = None
    nrde_config: NRDEConfig | None = None
    bnrde_config: BNRDEConfig | None = None
    m_ode_config: MODEConfig | None = None
    gru_config: GRUConfig | None = None
    lstm_config: LSTMConfig | None = None
    xlstm_config: XLSTMConfig | None = None
    stacked_xlstm_config: StackedXLSTMConfig | None = None

    @model_validator(mode="after")
    def validate_model_config_exists(self) -> "Config":
        """Ensure the correct config section exists for the chosen model_type."""
        model_type = self.experiment_config.model_type

        config_map = {
            ModelType.NCDE: self.ncde_config,
            ModelType.NRDE: self.nrde_config,
            ModelType.BNRDE: self.bnrde_config,
            ModelType.M_ODE: self.m_ode_config,
            ModelType.GRU: self.gru_config,
            ModelType.LSTM: self.lstm_config,
            ModelType.XLSTM: self.xlstm_config,
            ModelType.STACKED_XLSTM: self.stacked_xlstm_config,
        }

        active_config = config_map.get(model_type)
        if active_config is None:
            raise ValueError(
                f"Model type '{model_type}' requires a '{model_type}_config' section in the TOML"
            )

        # Ensure no extra configs are provided
        all_configs = [
            self.ncde_config,
            self.nrde_config,
            self.bnrde_config,
            self.m_ode_config,
            self.gru_config,
            self.lstm_config,
            self.xlstm_config,
            self.stacked_xlstm_config,
        ]
        num_provided = sum(c is not None for c in all_configs)
        if num_provided > 1:
            raise ValueError("Only one model config section should be provided")

        if self.experiment_config.hidden_state_mode == HiddenStateMode.PROBLEM_MANIFOLD:
            if model_type != ModelType.BNRDE:
                raise ValueError(
                    "hidden_state_mode='problem_manifold' is currently supported "
                    "only for model_type='bnrde'."
                )
            assert self.bnrde_config is not None
            if self.bnrde_config.initial_state_param_dim is None:
                raise ValueError(
                    "hidden_state_mode='problem_manifold' requires "
                    "bnrde_config.initial_state_param_dim."
                )
            if self.bnrde_config.hidden_size is not None:
                raise ValueError(
                    "Use bnrde_config.initial_state_param_dim instead of "
                    "hidden_size when hidden_state_mode='problem_manifold'."
                )
            if self.solver_config.solver not in (SolverType.CFEES25,):
                raise ValueError(
                    "hidden_state_mode='problem_manifold' requires solver to be "
                    "'cfees25'."
                )
        elif model_type == ModelType.BNRDE:
            assert self.bnrde_config is not None
            if self.bnrde_config.hidden_size is None:
                raise ValueError(
                    "bnrde_config.hidden_size is required when "
                    "hidden_state_mode='euclidean'."
                )

        return self

    @property
    def nn_config(
        self,
    ) -> (
        NCDEConfig
        | NRDEConfig
        | BNRDEConfig
        | MODEConfig
        | GRUConfig
        | LSTMConfig
        | XLSTMConfig
        | StackedXLSTMConfig
    ):
        """Get the active model configuration based on model_type."""
        model_type = self.experiment_config.model_type

        if model_type == ModelType.NCDE:
            assert self.ncde_config is not None
            return self.ncde_config
        elif model_type == ModelType.NRDE:
            assert self.nrde_config is not None
            return self.nrde_config
        elif model_type == ModelType.BNRDE:
            assert self.bnrde_config is not None
            return self.bnrde_config
        elif model_type == ModelType.M_ODE:
            assert self.m_ode_config is not None
            return self.m_ode_config
        elif model_type == ModelType.GRU:
            assert self.gru_config is not None
            return self.gru_config
        elif model_type == ModelType.LSTM:
            assert self.lstm_config is not None
            return self.lstm_config
        elif model_type == ModelType.XLSTM:
            assert self.xlstm_config is not None
            return self.xlstm_config
        elif model_type == ModelType.STACKED_XLSTM:
            assert self.stacked_xlstm_config is not None
            return self.stacked_xlstm_config
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def load_toml_config(toml_path: str) -> Config:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)
