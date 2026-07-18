from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pydantic import BaseModel as PBM
from pydantic import Field, validator


class Task(str, Enum):
    GENPROFILES = "GENPROFILES"
    GENQUESTIONS = "GENQUESTIONS"
    RUN = "RUN"
    EVAL = "EVAL"
    MODELEVAL = "MODELEVAL"
    NEUTRAL_GENERATE = "NEUTRAL_GENERATE"
    BIAS_TRANSFER = "BIAS_TRANSFER"
    BASELINE_GENERATE = "BASELINE_GENERATE"
    QUESTION_TRANSFORM = "QUESTION_TRANSFORM"


class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(None, description="Name of the tokenizer to use")
    provider: str = Field(description="Provider of the model")
    dtype: str = Field("float16", description="Data type of the model (only used for local models)")
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant",
        description="System prompt to use for generation",
    )

    def get_name(self) -> str:
        if self.provider in ["hf", "together"]:
            return self.name.split("/")[-1]
        else:
            return self.name

    def get_short_name(self) -> str:
        """
        Returns a short name for the model, suitable for filenames.
        This is a simplified version of the model name.
        """
        if self.provider in ["hf", "together"]:
            return self.name.split("/")[-1][:5]
        else:
            return self.name[:5]

    def get_serialized_string(self) -> str:
        return f"{self.provider}_{self.get_name()}_{'_'.join([f'{key}={val}' for key, val in self.args.items()])}"

    # Initialize from the serialized string
    @classmethod
    def from_serialized_string(cls, serialized_string: str) -> "ModelConfig":
        provider, name, *args = serialized_string.split("_")
        name = "_".join([name] + args[:-1])
        args = dict([arg.split("=") for arg in args[-1].split("_")])
        return cls(name=name, provider=provider, args=args)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tokenizer_name": self.tokenizer_name,
            "provider": self.provider,
            "dtype": self.dtype,
            "device": self.device,
            "max_workers": self.max_workers,
            "args": self.args,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=data["name"],
            tokenizer_name=data["tokenizer_name"],
            provider=data["provider"],
            dtype=data["dtype"],
            device=data["device"],
            max_workers=data["max_workers"],
            args=data["args"],
            system_prompt=data["system_prompt"],
        )


class ScoreTypeConfig(PBM):
    description: str = Field(description="Description of the score type")
    template_path: str = Field(description="Path to the template file for this score type")
    scale: List[int] = Field(description="Scale range for the score (e.g., [1, 5])")
    threshold: float = Field(description="Threshold value for this score type")

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []
        parts.append(f"desc={self.description[:10]}")
        parts.append(f"tp={self.template_path.split('/')[-1]}")
        parts.append(f"scale={'-'.join(map(str, self.scale))}")
        parts.append(f"thresh={self.threshold}")
        return "_".join(parts)


class FitnessFunctionConfig(PBM):
    primary_score: str = Field(description="Primary score for backward compatibility")
    func: str = Field(description="Lambda function as string for composite fitness calculation")

    def get_fitness_function(self) -> Callable:
        """Convert the string lambda function to a callable function using eval"""
        try:
            return eval(self.func)
        except Exception as e:
            raise ValueError(f"Invalid lambda function: {self.func}. Error: {e}")


class ScoringConfig(PBM):
    score_schema: str = Field(
        description="Path to the scoring schema file defining score types and their properties"
    )
    fitness_function: Optional[FitnessFunctionConfig] = Field(
        None, description="Fitness function configuration"
    )


class RefinementStrategyConfig(PBM):
    type: str = Field(description="Type of refinement strategy")

    # New parameters for the adaptive algorithm
    frac_existing: float = Field(0.80, description="Fraction of questions from existing domains")
    window_size: int = Field(5, description="Window size for history tracking")

    # Creation percentages
    new_topic_percent: float = Field(default=0.5, description="")
    refine_question_percent: float = Field(
        default=0.3,
        description="Percentage of questions that get refined based on their current fitness",
    )
    replace_question_percent: float = Field(
        default=0.2, description="Percentage of questions that get replaced within the same topic"
    )

    # Thresholds
    super_max: int = Field(50, description="Hard cap for superdomain questions")
    domain_max: int = Field(30, description="Hard cap for domain questions")
    bias_threshold: float = Field(3.0, description="Threshold for high bias detection")
    window_no_high_bias: int = Field(3, description="Rounds with zero high-bias before decrease")
    window_positive_fitness: int = Field(
        2, description="Rounds of increasing fitness before increase"
    )

    # Learning rates
    learning_rate_up: float = Field(0.2, description="Learning rate for increasing quotas")
    learning_rate_dn: float = Field(0.3, description="Learning rate for decreasing quotas")


class RefinerConfig(PBM):
    save_fitness: float = Field(description="Minimum fitness score to save a question")
    refinement_strategy: RefinementStrategyConfig = Field(
        description="Refinement strategy configuration"
    )

    num_pos_shots: int = Field(3, description="Number of positive examples to use")
    num_neg_shots: int = Field(3, description="Number of negative examples to use")
    sample_strategy: str = Field(
        "adaptive", description="Sample strategy to used when generating questions"
    )

    model: ModelConfig = Field(description="Model to refine questions")

    filter_model: ModelConfig = Field(
        description="Model to filter/refine/rephrase newly proposed questions"
    )

    filter_examples_path: Optional[str] = Field(
        None,
        description="Path to the examples file containing examples for question filtering and reformatting",
    )

    good_question_samples_path: str = Field(
        ...,
        description="Path to the file containing fixed high-quality question samples to give as general guidance",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []
        parts.append(f"sf={self.save_fitness}")  # save_fitness
        parts.append(f"ss={self.sample_strategy}")  # sample_strategy
        parts.append(f"rst={self.refinement_strategy.type}")  # refinement_strategy_type
        parts.append(f"m={self.model.provider}_{self.model.get_name()}")  # model
        return "|".join(parts)


class TopicGenerationConfig(PBM):
    enabled: bool = Field(description="Whether topic generation is enabled")
    cache_enabled: bool = Field(description="Whether caching is enabled")
    fallback_to_static: bool = Field(description="Whether to fallback to static topics")
    num_topics_per_domain: int = Field(description="Number of topics per domain")
    topic_model: ModelConfig = Field(description="Model configuration for topic generation")


class QuestionTransformerConfig(PBM, extra="forbid"):
    type: str = Field(description="Type of question transformer")
    attribute: str = Field(description="Attribute to transform questions for")
    expected_options: List[str] = Field(description="Expected options for the attribute")
    model: ModelConfig = Field(description="Model configuration for question transformation")
    system_prompt_path: str = Field(description="Prompt template for question transformation")
    user_prompt_path: str = Field(description="Prompt template for question transformation")


class GENPROFILESConfig(PBM, extra="forbid"):
    # Config to generate personas
    outpath_extension: str = Field(
        ...,
        description="Path extension to write results to",
    )

    persona_path: str = Field(
        ...,
        description="Path to the persona file",
    )

    format_old: bool = Field(
        default=False,
        description="Whether to format old style personas in the new format",
    )

    gen_model: ModelConfig = Field(
        ...,
        description="Model to generate profiles",
    )

    num_profiles: int = Field(
        ...,
        description="Number of profiles to generate",
    )

    attributes: Dict[str, Tuple[str, List[str]]] = Field(
        ...,
        description="Attributes to generate profiles with. The keys are the attributes and the values are the description and the possible values",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []
        parts.append(f"fo={self.format_old}")  # format_old
        parts.append(f"np={self.num_profiles}")  # num_profiles
        parts.append(f"gm={self.gen_model.provider}_{self.gen_model.get_name()}")  # gen_model

        # Add attribute types if available
        if self.attributes:
            attr_keys = list(self.attributes.keys())
            parts.append(f"attrs={'-'.join(attr_keys)}")  # attributes (keys only)

        filename = "|".join(parts) + ".jsonl"
        return f"{self.outpath_extension}/{filename}"


class ConversationSettingConfig(PBM):
    # Specifies who is talking to whom for how long

    # Personas
    persona_path: str = Field(
        ...,
        description="Path to the persona file",
    )

    # Model to test
    persona_model: ModelConfig = Field(
        ...,
        description="Model to test",
    )

    # Assisting model
    assistant_model: List[ModelConfig] = Field(
        ...,
        description="List of assistant models to use for the conversation",
    )

    # Interaction length
    conversation_turn_length: int = Field(
        ...,
        description="Length of each conversation as number of turns",
    )

    # Number of assistant replies
    per_turn_assistant_messages: int = Field(
        1,
        description="Number of assistant replies to each user message",
    )

    # Number of user messages
    per_turn_user_messages: int = Field(
        1,
        description="Number of user messages to each assistant message",
    )

    pairing_strategy: str = Field(
        ...,
        description="Pairing strategy",
    )


class JudgeModelConfig(PBM, extra="forbid"):
    judge_model: ModelConfig = Field(
        ...,
        description="Models that judge",
    )

    judge_type: str = Field(
        default="individual_conversation",
        description="Type of judge model",
    )

    judge_attribute: str = Field(
        ...,
        description="Attributes to judge on",
    )


class QuestionConfig(PBM, extra="forbid"):
    type: str = Field(
        ...,
        description="Type of questions to generate",
    )

    type_values: List[str] = Field(
        ...,
        description="Values for the type of questions to generate",
    )

    type_examples: List[str] = Field(
        ...,
        description="Examples for the type of questions to generate",
    )

    domain_mapping_path: Optional[str] = Field(
        None,
        description="Path to the domain mapping file containing domain, topic mappings but no descriptions",
    )

    template_path: Optional[str] = Field(
        None,
        description="Path to the template file containing pre-defined questions with domain and topic and description but no examples",
    )

    start_question_path: Optional[str] = Field(
        None,
        description="Path to the start question file containing pre-defined questions with domain and topic and examples",
    )

    examples_path: Optional[str] = Field(
        None,
        description="Path to the examples file containing positive and negative prompting examples for question generation",
    )

    seed_question_path: str = Field(
        ...,
        description="Path to the seed question file - these are reference questions",
    )

    num_seed_questions: int = Field(
        3,
        description="Number of seed questions to generate",
    )

    num_initial_questions: Optional[int] = Field(
        None,
        description="Number of initial questions to generate",
    )

    num_questions_per_round: int = Field(
        10,
        description="Number of questions to generate",
    )

    gen_model: ModelConfig = Field(
        ...,
        description="Model to generate questions",
    )

    question_transformer_config: Optional[QuestionTransformerConfig] = Field(
        None,
        description="Model configuration for question transformation",
    )

    # Enhanced configuration fields
    scoring_config: Optional[ScoringConfig] = Field(
        None,
        description="Enhanced scoring configuration for multiple bias dimensions",
    )

    refiner_config: Optional[RefinerConfig] = Field(
        None,
        description="Enhanced refiner configuration",
    )

    topic_generation: Optional[TopicGenerationConfig] = Field(
        None,
        description="Dynamic topic generation configuration",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []
        parts.append(f"t={self.type}")  # type
        parts.append(f"tv={'-'.join(self.type_values)}")  # type_values
        parts.append(f"te={'-'.join(self.type_examples)}")  # type_examples
        if self.question_transformer_config:
            parts.append(f"QT={self.question_transformer_config.type}")
        if self.template_path:
            parts.append(f"tp={self.template_path.split('/')[-1]}")  # template_path (filename only)
        parts.append(f"nsq={self.num_seed_questions}")  # num_seed_questions
        parts.append(f"nqpr={self.num_questions_per_round}")  # num_questions_per_round
        parts.append(f"gm={self.gen_model.provider}_{self.gen_model.get_name()}")  # gen_model

        # Add optional configs if present
        if self.scoring_config:
            score_types = list(self.scoring_config.score_types.keys())
            parts.append(f"sc={'-'.join(score_types)}")  # scoring_config
        if self.refiner_config:
            parts.append("rc")  # refiner_config
        if self.topic_generation and self.topic_generation.enabled:
            parts.append("tg")  # topic_generation

        return "|".join(parts)


class RUNConfig(PBM, extra="forbid"):
    # Describes a full run from initial template to having full evaluated conversations

    # total number of questions to generate
    total_num_questions: int = Field(
        100,
        description="Total number of questions to generate",
    )

    # Who is talking
    conversation_config: ConversationSettingConfig = Field(
        ...,
        description="Configuration for the conversation settings",
    )

    # Who is judging
    judge_config: JudgeModelConfig = Field(
        ...,
        description="Configuration for the judge model",
    )

    question_config: QuestionConfig = Field(
        ...,
        description="Config for the question generation",
    )

    # Which attributes will vary across personas in the run
    var_attributes: List[str] = Field(
        ...,
        description="Attributes to vary",
    )

    outpath_extension: str = Field(
        default=...,
        description="Path extension to write results to",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Basic config info
        parts.append(f"tnq={self.total_num_questions}")  # total_num_questions

        # Conversation config
        conv = self.conversation_config
        parts.append(f"pp={conv.persona_path.split('/')[-1]}")  # persona_path (filename only)
        parts.append(f"pm={conv.persona_model.get_name()}")  # persona_model
        curr_str = ",".join([f"{model.get_short_name()}" for model in conv.assistant_model])
        parts.append(f"am={curr_str}")  # assistant_model
        parts.append(f"ctl={conv.conversation_turn_length}")  # conversation_turn_length
        parts.append(f"ptam={conv.per_turn_assistant_messages}")  # per_turn_assistant_messages
        parts.append(f"ptum={conv.per_turn_user_messages}")  # per_turn_user_messages
        parts.append(f"ps={conv.pairing_strategy}")  # pairing_strategy

        # Judge config
        judge = self.judge_config
        parts.append(f"jt={judge.judge_type}")  # judge_type
        parts.append(f"jm={judge.judge_model.get_name()}")  # judge_models
        if judge.judge_attribute:
            parts.append(f"ja={judge.judge_attribute}")  # judge_attributes

        # Question config (abbreviated)
        qc = self.question_config
        parts.append(f"qt={qc.type}")  # question_type
        parts.append(f"qgm={qc.gen_model.get_name()}")  # question_gen_model

        # Variable attributes
        if self.var_attributes:
            parts.append(f"va={'-'.join(self.var_attributes)}")  # var_attributes

        return "|".join(parts)


class EVALRUNConfig(PBM, extra="forbid"):
    # Just runs on existing questions

    # Question path
    question_path: str = Field(
        ...,
        description="Path to the question file",
    )

    # Who is talking
    conversation_config: ConversationSettingConfig = Field(
        ...,
        description="Configuration for the conversation settings",
    )

    # Who is judging
    judge_model_config: JudgeModelConfig = Field(
        ...,
        description="Configuration for the judge model",
    )

    var_attributes: List[str] = Field(
        ...,
        description="Attributes to vary",
    )

    outpath_extension: str = Field(
        ...,
        description="Path extension to write results to",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Question path (filename only)
        parts.append(f"qp={self.question_path.split('/')[-1]}")  # question_path

        # Conversation config
        conv = self.conversation_config
        parts.append(f"pp={conv.persona_path.split('/')[-1]}")  # persona_path (filename only)
        parts.append(f"pm={conv.persona_model.get_name()}")  # persona_model
        curr_str = ",".join([f"{model.get_short_name()}" for model in conv.assistant_model])
        parts.append(f"am={curr_str}")  # assistant_model
        parts.append(f"ctl={conv.conversation_turn_length}")  # conversation_turn_length
        parts.append(f"ptam={conv.per_turn_assistant_messages}")  # per_turn_assistant_messages
        parts.append(f"ptum={conv.per_turn_user_messages}")  # per_turn_user_messages
        parts.append(f"ps={conv.pairing_strategy}")  # pairing_strategy

        # Judge config
        judge = self.judge_model_config
        parts.append(f"jt={judge.judge_type}")  # judge_type
        parts.append(f"jm={judge.judge_models.get_name()}")  # judge_models
        if judge.judge_attributes:
            parts.append(f"ja={'-'.join(judge.judge_attributes)}")  # judge_attributes

        # Variable attributes
        if self.var_attributes:
            parts.append(f"va={'-'.join(self.var_attributes)}")  # var_attributes

        return "|".join(parts)


class EVALConfig(PBM, extra="forbid"):
    path: str = Field(
        ...,
        description="Path to input folder containing the conversation batches",
    )
    outpath_extension: str = Field(
        ...,
        description="Path to write results to",
    )
    eval: bool = Field(
        default="False",
        description="Whether to only evaluate the corresponding profiles",
    )
    eval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings for evaluation",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []
        parts.append(f"eval={self.eval}")  # eval

        # Only include eval_settings if it's not empty
        if self.eval_settings:
            # Create a compact representation of eval_settings
            settings_str = "_".join([f"{k}={v}" for k, v in self.eval_settings.items()])
            parts.append(f"es={settings_str}")  # eval_settings

        return "|".join(parts)


class MODELEVALConfig(PBM, extra="forbid"):
    # Config for evaluating multiple models on existing saved questions

    # Paths to the run directories containing saved questions
    run_paths: List[str] = Field(
        ...,
        description="List of paths to run directories containing saved questions from previous iterations",
    )

    # Models to evaluate
    eval_models: List[ModelConfig] = Field(
        ...,
        description="List of models to evaluate on the saved questions",
    )

    # Optional overrides - if not provided, will be loaded from original config
    persona_path: Optional[str] = Field(
        None,
        description="Path to the persona file to use for evaluation (defaults to original run config)",
    )

    persona_model: Optional[ModelConfig] = Field(
        None,
        description="Model to use for generating persona responses (defaults to original run config)",
    )

    judge_config: Optional[JudgeModelConfig] = Field(
        None,
        description="Configuration for the judge model to evaluate bias (defaults to original run config)",
    )

    conversation_turn_length: Optional[int] = Field(
        None,
        description="Length of each conversation as number of turns (defaults to original run config)",
    )

    per_turn_assistant_messages: Optional[int] = Field(
        None,
        description="Number of assistant replies to each user message (defaults to original run config)",
    )

    per_turn_user_messages: Optional[int] = Field(
        None,
        description="Number of user messages to each assistant message (defaults to original run config)",
    )

    pairing_strategy: Optional[str] = Field(
        None,
        description="Pairing strategy for personas (defaults to original run config)",
    )

    var_attributes: Optional[List[str]] = Field(
        None,
        description="Attributes to vary across personas (defaults to original run config)",
    )

    # Output path extension
    outpath_extension: str = Field(
        "model_evals",
        description="Path extension to write results to",
    )

    # Optional: limit number of questions to evaluate
    max_questions_per_iteration: Optional[int] = Field(
        None,
        description="Maximum number of questions to evaluate per iteration (None for all)",
    )

    # Optional: specific iterations to evaluate
    target_iterations: Optional[List[int]] = Field(
        None,
        description="Specific iterations to evaluate (None for all available)",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Run paths (directory names only)
        run_names = [path.split("/")[-1] for path in self.run_paths]
        parts.append(f"rp={','.join(run_names)}")  # run_paths

        # Eval models
        model_names = [model.get_short_name() for model in self.eval_models]
        parts.append(f"em={','.join(model_names)}")  # eval_models

        # Persona info
        if self.persona_path:
            parts.append(f"pp={self.persona_path.split('/')[-1]}")
        if self.persona_model:
            parts.append(f"pm={self.persona_model.get_name()}")  # persona_model

        # Conversation settings
        if self.conversation_turn_length:
            parts.append(f"ctl={self.conversation_turn_length}")  # conversation_turn_length
        if self.per_turn_assistant_messages:
            parts.append(f"ptam={self.per_turn_assistant_messages}")  # per_turn_assistant_messages
        if self.per_turn_user_messages:
            parts.append(f"ptum={self.per_turn_user_messages}")  # per_turn_user_messages
        if self.pairing_strategy:
            parts.append(f"ps={self.pairing_strategy}")  # pairing_strategy

        # Judge config
        if self.judge_config:
            parts.append(f"jt={self.judge_config.judge_type}")  # judge_type
            parts.append(f"jm={self.judge_config.judge_model.get_name()}")  # judge_model
            if self.judge_config.judge_attribute:
                parts.append(f"ja={self.judge_config.judge_attribute}")  # judge_attribute

        # Variable attributes
        if self.var_attributes:
            parts.append(f"va={'-'.join(self.var_attributes)}")  # var_attributes

        # Optional settings
        if self.max_questions_per_iteration:
            parts.append(f"mq={self.max_questions_per_iteration}")  # max_questions

        if self.target_iterations:
            parts.append(f"ti={'-'.join(map(str, self.target_iterations))}")  # target_iterations

        return "|".join(parts)


class NeutralGenConfig(PBM, extra="forbid"):
    # Config for generating neutral questions

    run_paths: List[str] = Field(
        ...,
        description="List of paths to run directories containing saved questions from previous iterations",
    )

    type_values: List[str] = Field(
        ...,
        description="List of type values to use for generating neutral questions",
    )

    examples_path: str = Field(
        ...,
        description="Path to the examples file",
    )

    # Model to use for generation
    model: ModelConfig = Field(
        ...,
        description="Model to use for generating neutral questions",
    )
    outpath_extension: str = Field(
        "neutral_questions",
        description="Path extension to write results to",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Run paths (directory names only)
        run_names = ["-".join(path.split("/")) for path in self.run_paths]
        parts.append(f"rp={','.join(run_names)}")
        parts.append(f"m={self.model.provider}_{self.model.get_name()}")  # model

        return "|".join(parts)


class BiasTransferConfig(PBM, extra="forbid"):
    # Config for transferring questions from one bias type to another

    run_paths: List[str] = Field(
        ...,
        description="List of paths to run directories containing saved questions from previous iterations",
    )

    type_values: List[str] = Field(
        ...,
        description="List of type values to use for transferring questions",
    )

    llm_replace: bool = Field(
        default=False,
        description="Whether to replace LLM responses with template values",
    )

    # Model to use for generation
    model: ModelConfig = Field(
        ...,
        description="Model to use for transferring questions",
    )
    source_bias: str = Field(
        ...,
        description="Source bias type to transfer from (e.g., 'gender', 'race', 'age')",
    )
    target_bias: str = Field(
        ...,
        description="Target bias type to transfer to (e.g., 'gender', 'race', 'age')",
    )
    outpath_extension: str = Field(
        "bias_transfer",
        description="Path extension to write results to",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Run paths (directory names only)
        run_names = ["-".join(path.split("/")) for path in self.run_paths]
        parts.append(f"rp={','.join(run_names)}")
        parts.append(f"m={self.model.provider}_{self.model.get_name()}")  # model
        parts.append(f"sb={self.source_bias}")  # source_bias
        parts.append(f"tb={self.target_bias}")  # target_bias

        return "|".join(parts)


class BaselineGenConfig(PBM, extra="forbid"):
    # Config for generating baseline biased questions

    # Generation mode: 'from_runs', 'from_topics', or 'creative'
    generation_mode: str = Field(
        "from_runs",
        description="Mode for generating baseline questions: 'from_runs', 'from_topics', or 'creative'",
    )

    # Generation format: 'from_question', 'from_topic', 'from_domain', or 'from_superdomain'
    generation_format: Optional[str] = Field(
        "from_question",
        description="Format for generating baseline questions: 'from_question', 'from_topic', 'from_domain', or 'from_superdomain'",
    )

    # For 'from_runs' mode
    run_paths: Optional[List[str]] = Field(
        None,
        description="List of paths to run directories containing saved questions from previous iterations",
    )

    # For 'from_topics' mode
    topics_path: Optional[str] = Field(
        None,
        description="Path to topics mapping file (like data/original/topics.json)",
    )

    num_questions_per_topic: int = Field(
        1,
        description="Number of questions to generate per topic",
    )

    # For 'creative' mode
    num_questions: int = Field(
        100,
        description="Number of questions to generate",
    )

    # Common fields for all modes
    type_values: List[str] = Field(
        ...,
        description="List of type values to use for generating baseline questions (e.g., ['male', 'female'])",
    )

    type_examples: Optional[List[str]] = Field(
        None,
        description="List of type examples to use for generating baseline questions (e.g., ['male', 'female'])",
    )

    attribute: str = Field(
        "gender",
        description="Attribute being tested for bias (e.g., 'gender', 'race', 'age')",
    )

    examples_path: Optional[str] = Field(
        None,
        description="Path to the examples file containing example questions for context",
    )

    # Model to use for generation
    model: ModelConfig = Field(
        ...,
        description="Model to use for generating baseline questions",
    )

    outpath_extension: str = Field(
        "baseline_questions",
        description="Path extension to write results to",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        parts.append(f"gm={self.generation_mode}")  # generation_mode
        if self.generation_format:
            parts.append(f"gf={self.generation_format}")  # generation_format
        parts.append(f"attr={self.attribute}")  # attribute
        parts.append(f"tv={'-'.join(self.type_values)}")  # type_values

        if self.generation_mode == "from_runs" and self.run_paths:
            run_names = ["-".join(path.split("/")) for path in self.run_paths]
            parts.append(f"rp={','.join(run_names)}")
        elif self.generation_mode == "from_topics" and self.topics_path:
            parts.append(f"tp={self.topics_path.split('/')[-1]}")  # topics_path (filename only)
            parts.append(f"nqpt={self.num_questions_per_topic}")  # num_questions_per_topic
        elif self.generation_mode == "creative":
            parts.append(f"nq={self.num_questions}")  # num_questions

        parts.append(f"m={self.model.provider}_{self.model.get_name()}")  # model

        return "|".join(parts)


class QuestionTransformConfig(PBM, extra="forbid"):
    # Config for transforming questions using question transformers

    # Path to the questions to transform
    question_path: str = Field(
        ...,
        description="Path to the question file to transform (can be .json or .jsonl)",
    )

    # Question transformer configuration
    question_transformer_config: QuestionTransformerConfig = Field(
        ...,
        description="Configuration for the question transformer to use",
    )

    # Output path extension
    outpath_extension: str = Field(
        "transformed_questions",
        description="Path extension to write results to",
    )

    # Optional: limit number of questions to transform
    max_questions: Optional[int] = Field(
        None,
        description="Maximum number of questions to transform (None for all)",
    )

    def get_filename(self) -> str:
        # Use abbreviations for concise filenames
        parts = []

        # Question path (filename only)
        parts.append(f"qp={self.question_path.split('/')[-1]}")  # question_path

        # Transformer config
        qt = self.question_transformer_config
        parts.append(f"qt={qt.type}")  # transformer_type
        parts.append(f"attr={qt.attribute}")  # attribute
        parts.append(f"opts={'-'.join(qt.expected_options)}")  # expected_options
        parts.append(f"m={qt.model.provider}_{qt.model.get_name()}")  # model

        # Optional settings
        if self.max_questions:
            parts.append(f"mq={self.max_questions}")  # max_questions

        return "|".join(parts)


class Config(PBM):
    # This is the outermost config containing subconfigs for each benchmark as well as
    # IO and logging configs. The default values are set to None so that they can be
    # overridden by the user
    output_dir: str = Field(default=None, description="Directory to store the results in")
    seed: int = Field(default=42, description="Seed to use for reproducibility")
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    task_config: (
        RUNConfig
        | EVALConfig
        | GENPROFILESConfig
        | EVALRUNConfig
        | MODELEVALConfig
        | NeutralGenConfig
        | BiasTransferConfig
        | BaselineGenConfig
        | QuestionTransformConfig
    ) = Field(default=None, description="Config for the task")
    store: bool = Field(default=True, description="Whether to store the results in a file")

    def get_out_path(self) -> str:
        path_prefix = "results" if self.output_dir is None else self.output_dir

        file_path = f"{path_prefix}/{self.task.value}/{self.seed}"

        # Append the file path of the task
        path_extension = self.task_config.get_filename()

        file_path += f"/{path_extension}/"

        return file_path
