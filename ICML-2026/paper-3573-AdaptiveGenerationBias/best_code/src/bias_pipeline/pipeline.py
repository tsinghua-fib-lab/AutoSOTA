from typing import List, Tuple, Optional, Dict
import os
from dataclasses import dataclass

from src.bias_pipeline.question_transfomer.question_transfomer import get_question_transformer
from src.personas import Persona, load_personas
from src.configs import RUNConfig
from src.bias_pipeline.data_types.data_types import (
    Message,
    RootMessage,
    MessageNode,
    Thread,
)
from src.bias_pipeline.data_types.conversation import (
    Conversation,
    ConversationBatch,
    load_conversations,
)
from src.models import BaseModel, get_model, run_parallel_advanced
from src.bias_pipeline.evaluators.evaluator_factory import get_evaluator
from src.bias_pipeline.questionaires.questionaire import (
    load_questionnaire,
    BiasQuestionnaire,
    Question,
)
from src.bias_pipeline.questionaires.generator import QuestionaireGenerator
from src.bias_pipeline.history_state import HistoryState, load_history_state
from src.bias_pipeline.questionaires.filter import quality_filter_conversations
from src.bias_pipeline.refiner.enhanced_refiner import EnhancedRefiner
from src.utils.cost_tracker import get_global_cost_tracker


@dataclass
class PipelineState:
    """
    Unified state container for the bias detection pipeline.
    Contains all data needed to resume pipeline execution.
    """

    personas: Optional[List[Persona]] = None
    conversations: Optional[Dict[int, List[ConversationBatch]]] = None
    questions: Optional[BiasQuestionnaire] = None
    saved_questions: Optional[Dict[int, BiasQuestionnaire]] = None
    history_state: Optional[HistoryState] = None
    iteration: Optional[int] = None
    step: Optional[int] = None

    def __post_init__(self):
        """Initialize empty containers if None."""
        if self.personas is None:
            self.personas = []
        if self.conversations is None:
            self.conversations = {}
        if self.questions is None:
            self.questions = BiasQuestionnaire()
        if self.saved_questions is None:
            self.saved_questions = {}
        if self.history_state is None:
            self.history_state = HistoryState()
        if self.iteration is None:
            self.iteration = 0
        if self.step is None:
            self.step = 0

    def is_empty(self) -> bool:
        """Check if the state is effectively empty."""
        return (
            len(self.personas) == 0
            and len(self.conversations) == 0
            and len(self.questions) == 0
            and len(self.saved_questions) == 0
        )

    def get_total_saved_questions(self, ignore: List[int] = []) -> int:
        """Get the total number of saved questions across all iterations."""
        return sum(len(q) for k, q in self.saved_questions.items() if k not in ignore)

    def get_current_conversations(self) -> List[ConversationBatch]:
        """Get conversations for the current iteration."""
        return self.conversations.get(self.iteration, [])

    def set_current_conversations(self, conversations: List[ConversationBatch]) -> None:
        """Set conversations for the current iteration."""
        if self.iteration is not None:
            self.conversations[self.iteration] = conversations

    def get_conversations_for_iteration(self, iteration: int) -> List[ConversationBatch]:
        """Get conversations for a specific iteration."""
        return self.conversations.get(iteration, [])

    def set_conversations_for_iteration(
        self, iteration: int, conversations: List[ConversationBatch]
    ) -> None:
        """Set conversations for a specific iteration."""
        self.conversations[iteration] = conversations

    def set_state(
        self,
        personas: Optional[List[Persona]] = None,
        conversations: Optional[List[ConversationBatch]] = None,
        questions: Optional[BiasQuestionnaire] = None,
        saved_questions: Optional[Dict[int, BiasQuestionnaire]] = None,
        history_state: Optional[HistoryState] = None,
    ) -> None:
        if personas is not None:
            self.personas = personas
        if conversations is not None:
            # For backward compatibility, set conversations for current iteration
            self.set_current_conversations(conversations)
        if questions is not None:
            self.questions = questions
        if saved_questions is not None:
            self.saved_questions = saved_questions
        if history_state is not None:
            self.history_state = history_state


class BiasDetectionPipeline:
    """
    Orchestrates the entire bias detection workflow:
      1. Persona Construction
      2. Bias-Inducing Questionnaires
      3. Generating Responses for Each Persona
      4. Evaluating Bias
      5. Benchmark Creation
    """

    def __init__(self, config: RUNConfig) -> None:
        self.config = config

        # Conversation Settings
        self.persona_model = get_model(config.conversation_config.persona_model)

        if config.question_config.question_transformer_config is None:
            self.question_transformer = None
        else:
            self.question_transformer = get_question_transformer(
                config.question_config.question_transformer_config,
            )

        self.assistant_models: List[BaseModel] = []

        for model_cfg in config.conversation_config.assistant_model:
            self.assistant_models.append(get_model(model_cfg))

        # Judgement Settings
        self.bias_evaluator = get_evaluator(config=config.judge_config)

        # Question Generation Settings
        # Extract fitness function from scoring config if available
        fitness_function = None
        if (
            hasattr(config.question_config, "scoring_config")
            and config.question_config.scoring_config
            and hasattr(config.question_config.scoring_config, "fitness_function")
            and config.question_config.scoring_config.fitness_function
        ):
            try:
                fitness_function = (
                    config.question_config.scoring_config.fitness_function.get_fitness_function()
                )
            except Exception as e:
                print(f"Warning: Failed to load custom fitness function: {e}. Using default.")
                fitness_function = None

        self.refiner = EnhancedRefiner(
            config.question_config.refiner_config,
            config.var_attributes[0],
            fitness_function,
            config.question_config,
        )

        # State tracking
        self.question_history = None  # Will be initialized in run_pipeline

    def _load_state(
        self,
        outpath: str,
        init_personas: List[Persona],
        init_conversations: List[ConversationBatch],
    ) -> PipelineState:
        """
        Unified method to load the complete pipeline state from the output directory.
        Loads personas, conversations, questions, saved questions, and history state.

        Args:
            outpath (str): The output path where pipeline state is stored
            init_personas (List[Persona]): Initial personas for fallback
            init_conversations (List[ConversationBatch]): Initial conversations for fallback

        Returns:
            PipelineState: Complete pipeline state
        """
        state = PipelineState()

        # Load seed questions as fallback
        try:
            seed_questionnaire = load_questionnaire(self.config.question_config.seed_question_path)
            state.saved_questions[-1] = seed_questionnaire
        except Exception as e:
            print(f"Warning: Could not load seed questions: {e}")

        # If no outpath exists, return initial state
        if not os.path.exists(outpath):
            print("No existing state found, using initial data")
            state.personas = init_personas.copy()
            # For the new structure, put initial conversations in iteration 0
            if init_conversations:
                state.conversations[0] = init_conversations.copy()
            if hasattr(self.config, "question_config") and hasattr(
                self.config.question_config, "seed_question_path"
            ):
                try:
                    state.questions = load_questionnaire(
                        self.config.question_config.seed_question_path
                    )
                except:
                    pass
            return state

        # Get all iteration folders and sort them in reverse order (newest first)
        iteration_folders = []
        for item in os.listdir(outpath):
            item_path = os.path.join(outpath, item)
            if os.path.isdir(item_path) and item.startswith("iteration_"):
                try:
                    iteration_num = int(item.split("_")[1])
                    iteration_folders.append((iteration_num, item_path))
                except (IndexError, ValueError):
                    continue

        iteration_folders.sort(reverse=True)  # Sort by iteration number, newest first

        # Track what we've loaded
        loaded_personas = {}
        loaded_conversations_by_iteration = {}  # Dict[int, List[ConversationBatch]]
        loaded_questions = None
        loaded_saved_questions = {}
        loaded_history_state = None

        latest_iteration_conversations = 0
        latest_step = 0

        # Load from all iterations
        for iteration_num, iteration_path in iteration_folders:
            # Check steps in reverse order: 2, 1, 0
            for step in [2, 1, 0]:
                step_path = os.path.join(iteration_path, f"sb_{step}")

                if not os.path.exists(step_path):
                    continue

                print(f"Checking step {step} in iteration {iteration_num} at {step_path}")

                # Load personas (check both old and new filenames)
                if not loaded_personas:
                    personas_file = None
                    if os.path.exists(os.path.join(step_path, "personals.jsonl")):
                        personas_file = os.path.join(step_path, "personals.jsonl")
                    elif os.path.exists(os.path.join(step_path, "personas.jsonl")):
                        personas_file = os.path.join(step_path, "personas.jsonl")

                    if personas_file:
                        try:
                            personas_list = load_personas(personas_file)
                            for persona in personas_list:
                                loaded_personas[persona.id] = persona
                            print(f"Loaded {len(personas_list)} personas from {personas_file}")
                        except Exception as e:
                            print(f"Warning: Could not load personas from {personas_file}: {e}")

                # Load conversations for this iteration
                if iteration_num not in loaded_conversations_by_iteration:
                    conversations_file = os.path.join(step_path, "conversations.jsonl")
                    if os.path.exists(conversations_file):
                        try:
                            conversations_list = load_conversations(conversations_file)
                            loaded_conversations_by_iteration[iteration_num] = conversations_list
                            print(
                                f"Loaded {len(conversations_list)} conversations from {conversations_file} for iteration {iteration_num}"
                            )
                            latest_iteration_conversations = max(
                                latest_iteration_conversations, iteration_num
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not load conversations from {conversations_file}: {e}"
                            )

                # Load questions
                if loaded_questions is None:
                    questions_file = os.path.join(step_path, "questions.jsonl")
                    if os.path.exists(questions_file):
                        try:
                            loaded_questions = load_questionnaire(questions_file)
                            print(f"Loaded {len(loaded_questions)} questions from {questions_file}")
                        except Exception as e:
                            print(f"Warning: Could not load questions from {questions_file}: {e}")

                # Load saved questions
                if iteration_num not in loaded_saved_questions:
                    saved_questions_file = os.path.join(step_path, "saved_questions.jsonl")
                    if os.path.exists(saved_questions_file):
                        try:
                            saved_q = load_questionnaire(saved_questions_file)
                            loaded_saved_questions[iteration_num] = saved_q
                            print(
                                f"Loaded {len(saved_q)} saved questions from {saved_questions_file}"
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not load saved questions from {saved_questions_file}: {e}"
                            )

                # Load history state
                if loaded_history_state is None:
                    history_file = os.path.join(step_path, "history_state.json")
                    if os.path.exists(history_file):
                        try:
                            loaded_history_state = load_history_state(history_file)
                            print(f"Loaded history state from {history_file}")
                        except Exception as e:
                            print(f"Warning: Could not load history state from {history_file}: {e}")

        # Find highest step under the latest iteration
        if latest_iteration_conversations >= 0:
            for step in [2, 1, 0]:
                step_path = os.path.join(
                    outpath, f"iteration_{latest_iteration_conversations}", f"sb_{step}"
                )
                if os.path.exists(step_path):
                    latest_step = step
                    break

        # Merge loaded data with initial data
        # Personas: use loaded if available, otherwise use initial
        final_personas = {}
        for persona in init_personas:
            final_personas[persona.id] = persona
        final_personas.update(loaded_personas)
        state.personas = list(final_personas.values())

        # Conversations: merge loaded conversations by iteration with initial conversations
        state.conversations = {}
        # Add initial conversations to iteration 0 if no conversations loaded for iteration 0
        if 0 not in loaded_conversations_by_iteration and init_conversations:
            state.conversations[0] = init_conversations

        # Add all loaded conversations
        for iteration_num, conversations_list in loaded_conversations_by_iteration.items():
            state.conversations[iteration_num] = conversations_list

        # Questions, saved questions, and history state
        if loaded_questions is not None:
            state.questions = loaded_questions
        if loaded_saved_questions:
            state.saved_questions.update(loaded_saved_questions)
        if loaded_history_state is not None:
            state.history_state = loaded_history_state

        # Consistency checks
        # Each question has to correspond to a root message in conversations
        # All personas used in conversations must be in loaded personas
        # We have history up until the current iteration

        latest_iteration_saved = max([int(k) for k in state.saved_questions.keys()] + [0])
        # assert latest_iteration_saved == latest_iteration_conversations, (
        #     f"Latest saved questions iteration {latest_iteration_saved} does not match latest conversation iteration {latest_iteration_conversations}"
        # )

        state.iteration = latest_iteration_conversations
        state.step = latest_step

        self._perform_consistency_checks(state)

        total_conversations = sum(len(convs) for convs in state.conversations.values())
        print(
            f"Final state: {len(state.personas)} personas, {total_conversations} conversations across {len(state.conversations)} iterations, "
            f"{len(state.questions)} questions, {len(state.saved_questions)} saved question sets"
        )

        return state

    def _perform_consistency_checks(self, state: PipelineState) -> None:
        """
        Perform consistency checks on the pipeline state.

        Args:
            state (PipelineState): The pipeline state to check
            iteration (int): Current iteration number

        Raises:
            AssertionError: If any consistency check fails
        """
        print(
            f"Performing consistency checks for state in iteration {state.iteration} and step {state.step}..."
        )

        # Check 1: Each question has to correspond to a root message in conversations - In the current iteration
        if state.questions and state.conversations:
            question_ids = set()
            for question in state.questions.to_list():
                question_ids.add(question.get_id())

            conversation_question_ids = set()
            # Iterate through all conversations across all iterations
            for iteration_conversations in state.conversations.values():
                for conversation_batch in iteration_conversations:
                    conversation_question_ids.add(conversation_batch.root_message.id)

            # Check that all conversation questions exist in the question set
            missing_questions = conversation_question_ids - question_ids
            if missing_questions:
                print(
                    f"Warning: Found {len(missing_questions)} conversation questions not in question set: {missing_questions}"
                )

            # Check that all questions have corresponding conversations (optional - may not always be true)
            unused_questions = question_ids - conversation_question_ids
            if unused_questions:
                print(
                    f"Info: Found {len(unused_questions)} questions without conversations: {unused_questions}"
                )

        # Check 2: All personas used in conversations must be in loaded personas
        if state.personas and state.conversations:
            persona_ids = set()
            for persona in state.personas:
                persona_ids.add(persona.id)

            conversation_persona_ids = set()
            # Iterate through all conversations across all iterations
            for iteration_conversations in state.conversations.values():
                for conversation_batch in iteration_conversations:
                    for conversation in conversation_batch.conversations:
                        conversation_persona_ids.add(conversation.persona.id)

            # Check that all conversation personas exist in the persona set
            missing_personas = conversation_persona_ids - persona_ids
            if missing_personas:
                raise AssertionError(
                    f"Found {len(missing_personas)} conversation personas not in persona set: {missing_personas}"
                )

            print(
                f"✓ All {len(conversation_persona_ids)} conversation personas found in persona set"
            )

        # Check 3: We have history up until the current iteration
        if state.history_state:
            available_iterations = state.history_state.get_available_iterations()
            expected_iterations = list(
                range(state.iteration + 1)
            )  # 0 to current iteration inclusive

            missing_iterations = set(expected_iterations) - set(available_iterations)
            if missing_iterations:
                print(f"Warning: Missing history for iterations: {missing_iterations}")

            extra_iterations = set(available_iterations) - set(expected_iterations)
            if extra_iterations:
                print(f"Info: Found history for future iterations: {extra_iterations}")

            print(f"✓ History state covers iterations: {sorted(available_iterations)}")

        print("Consistency checks completed.")

    def _initialize_conversations_and_questions(
        self, state: PipelineState, personas: List[Persona], iteration: int = 0
    ) -> None:
        """
        Initialize conversations and questions for a given iteration.
        This method generates baseline questions, builds initial conversations,
        applies quality filtering, and updates the state.

        Args:
            state (PipelineState): The pipeline state to update
            personas (List[Persona]): The personas to use for conversation generation
            iteration (int): The iteration number for history tracking (default: 0)
        """
        # Generates the baseline questions
        generator = QuestionaireGenerator(self.config.question_config, "none")

        init_questions = generator.generate(
            self.config.question_config.type,
            self.config.question_config.type_values,
            self.config.question_config.type_examples,
            self.config.question_config.num_initial_questions,
            store=False,  # We will store them later
        )

        init_conversations = self.build_initial_conversations(init_questions, personas)
        init_conversations, init_questions = quality_filter_conversations(init_conversations)
        assert len(init_conversations) > 0, "No initial conversations created"

        # Put the new questions into the state
        state.set_state(
            conversations=init_conversations,
            questions=init_questions,
        )

        # Set up history state if not already initialized
        if state.history_state is None:
            state.history_state = HistoryState()
        # state.history_state.update_with_questions(init_questions, iteration=iteration)

    def _initialize_conversations_from_questions(
        self,
        state: PipelineState,
        list_questions: List[Question],
        personas: List[Persona],
        iteration: int,
    ) -> None:
        """
        Initialize conversations from a list of questions.
        This method is used to create conversations based on the provided questions.

        Args:
            list_questions (List[Question]): The list of questions to use for conversation generation
        """
        # Convert list of questions to BiasQuestionnaire
        questions = BiasQuestionnaire({q.get_id(): q for q in list_questions})

        # Build initial conversations
        init_conversations = self.build_initial_conversations(questions, personas)

        init_conversations, init_questions = quality_filter_conversations(init_conversations)

        assert len(init_conversations) > 0, "No initial conversations created"

        # Put the new questions into the state
        state.iteration = iteration
        state.set_conversations_for_iteration(iteration, init_conversations)
        state.questions = init_questions

    def run_pipeline(
        self,
        outpath: str,
    ) -> None:
        """
        Executes the pipeline given initial persona data and questioning strategy. Stores all results under outpath.
        """

        cost_tracker = get_global_cost_tracker()

        init_personas = load_personas(self.config.conversation_config.persona_path)
        init_conversations = []

        # Load conversation state
        state = self._load_state(outpath, init_personas, init_conversations)

        # If there is no are no conversations loaded, we need to initialize them
        if not state.conversations:
            self._initialize_conversations_and_questions(state, init_personas, iteration=0)

        # For each question in either the loaded or initial state, we need to ensure a consistent format
        self._ensure_format(state)

        # Store initial state with all components
        state.history_state.get_stats(attribute="superdomain")

        # Run this loop until we have saved enough questions or reached the maximum number of iterations
        max_num_iterations = 20 * max(
            (
                self.config.total_num_questions
                // self.config.question_config.num_questions_per_round
            ),
            1,
        )

        outer_iteration = state.iteration if state.iteration is not None else 0
        curr_step = state.step % 3 if state.step is not None else 0
        curr_conversations = state.get_current_conversations()

        while (
            state.get_total_saved_questions(ignore=[-1, 0]) < self.config.total_num_questions
            and outer_iteration < max_num_iterations
        ):
            print(
                f"Running iteration {outer_iteration} with saved questions: {state.get_total_saved_questions(ignore=[-1, 0])}"
            )
            if curr_step == 0:
                print(f"ITERATION {outer_iteration}, STEP {curr_step}")
                # sb 0 Fresh start potentially new questions
                self._store_state_unified(
                    state,
                    iteration=outer_iteration,
                    step=0,
                    outpath_prefix=outpath,
                    store_personas=True,
                    store_conversations=True,
                    store_questions=True,
                    store_saved_questions=True,  # No saved questions yet
                    store_history_state=True,
                )

                for inner_iteration in range(
                    self.config.conversation_config.conversation_turn_length
                ):
                    ### Run Interaction turn for all relevant conversations
                    curr_conversations = self._filter_conversations(
                        state.get_current_conversations(), inner_iteration
                    )

                    # Run interaction turn (either persona sending a message or model sending a message to persona)
                    if len(curr_conversations) > 0:
                        self.run_interaction_turn(curr_conversations)

                curr_step = 1
                # Update the state with the current conversations for this iteration
                state.set_conversations_for_iteration(outer_iteration, curr_conversations)
                self._store_state_unified(
                    state,
                    outer_iteration,
                    1,
                    outpath,
                    store_personas=False,
                    store_conversations=True,
                    store_questions=False,
                    store_saved_questions=False,
                    store_history_state=False,
                )

                cost_tracker.print_cost_summary()

            if curr_step == 1:
                # sb 1 Corresponding conversations are completed
                print(f"ITERATION {outer_iteration}, STEP {curr_step}")

                for iteration in range(
                    self.config.conversation_config.conversation_turn_length + 1
                ):
                    ### Run Bias Evaluation for all relevant conversations
                    curr_conversations = self._filter_conversations(
                        state.get_current_conversations(), iteration, has_bias_judgement=False
                    )
                    # Get Judgement on bias
                    if len(curr_conversations) > 0:
                        self.bias_evaluator.evaluate_bias_conversation(curr_conversations)

                # sb 2 Corresponding conversations are evaluated
                # Update the state with the current conversations for this iteration
                state.set_conversations_for_iteration(outer_iteration, curr_conversations)
                state.history_state.update_with_conversation_batches(
                    curr_conversations, iteration=outer_iteration
                )
                state.history_state.compute_metrics(
                    self.refiner.fitness_function, self.refiner.bias_threshold
                )

                self._store_state_unified(
                    state,
                    outer_iteration,
                    2,
                    outpath,
                    store_personas=False,
                    store_conversations=True,
                    store_questions=False,
                    store_saved_questions=False,
                    store_history_state=True,
                )
                curr_step = 2

                cost_tracker.print_cost_summary()

            # Print the Cost tracker state

            if curr_step == 2:
                # sb 2 Refine questions based on the conversations
                print(f"ITERATION {outer_iteration}, STEP {curr_step}")
                ### Create refined questions based on the conversations and store best questions
                new_questions, saved_questions = self.refiner.refine_questions(
                    state, num_questions=self.config.question_config.num_questions_per_round
                )

                # Store all saved questions
                sq_questionnaire = (
                    BiasQuestionnaire({q.get_id(): q for q in saved_questions})
                    if saved_questions
                    else BiasQuestionnaire()
                )
                state.saved_questions[outer_iteration] = sq_questionnaire
                self._store_state_unified(
                    state,
                    outer_iteration,
                    2,
                    outpath,
                    store_personas=False,
                    store_conversations=False,
                    store_questions=False,
                    store_saved_questions=True,
                    store_history_state=False,
                )

                print(
                    f"Refined {len(new_questions)} new questions and saved {len(saved_questions)} questions for iteration {outer_iteration}"
                )

                # Update outer iteration count
                outer_iteration += 1
                curr_step = 0
                # Updates questions and conversations in the state
                self._initialize_conversations_from_questions(
                    state,
                    new_questions,
                    state.personas,
                    outer_iteration,
                )

                cost_tracker.print_cost_summary()

    def _ensure_format(self, state: PipelineState) -> None:
        """
        Ensures the domain format is respected across all questions and conversations in the state.
        Args:
            state (PipelineState): The pipeline state to check and update
        """
        # Ensure all questions have the correct format
        if state.questions:
            for question in state.questions.to_list():
                if "::" not in question.domain:
                    question.domain = f"{question.superdomain}::{question.domain}"

        # And saved questions
        if state.saved_questions:
            for questionnaire in state.saved_questions.values():
                for question in questionnaire.to_list():
                    if "::" not in question.domain:
                        question.domain = f"{question.superdomain}::{question.domain}"

        # Ensure all conversations have questions with the correct format
        if state.conversations:
            for iteration_conversations in state.conversations.values():
                for conversation_batch in iteration_conversations:
                    if "::" not in conversation_batch.root_message.question.domain:
                        conversation_batch.root_message.question.domain = (
                            f"{conversation_batch.root_message.question.superdomain}::"
                            f"{conversation_batch.root_message.question.domain}"
                        )
                    for conversation in conversation_batch.conversations:
                        if "::" not in conversation.RootMessage.question.domain:
                            conversation.RootMessage.question.domain = (
                                f"{conversation.RootMessage.question.superdomain}::"
                                f"{conversation.RootMessage.question.domain}"
                            )

    def _store_state_unified(
        self,
        state: PipelineState,
        iteration: int,
        step: int,
        outpath_prefix: str,
        store_personas: bool = True,
        store_conversations: bool = True,
        store_questions: bool = True,
        store_saved_questions: bool = True,
        store_history_state: bool = True,
    ) -> None:
        """
        Unified method to store the complete pipeline state with selective storage options.
        Uses the new folder structure with 'personals.jsonl' instead of 'personas.jsonl'.

        Args:
            state (PipelineState): The pipeline state to store
            iteration (int): Iteration number
            step (int): Step number within iteration
            outpath_prefix (str): Base output path
            store_personas (bool): Whether to store personas
            store_conversations (bool): Whether to store conversations
            store_questions (bool): Whether to store questions
            store_saved_questions (bool): Whether to store saved questions
            store_history_state (bool): Whether to store history state
        """
        outpath = os.path.join(outpath_prefix, f"iteration_{iteration}", f"sb_{step}")
        os.makedirs(outpath, exist_ok=True)

        # (1) Store personas with new filename
        if store_personas and state.personas:
            personas_path = os.path.join(outpath, "personals.jsonl")
            with open(personas_path, "w") as f:
                for persona in state.personas:
                    persona.to_file(f)
            print(f"Stored {len(state.personas)} personas to {personas_path}")

        # (2) Store conversations
        if store_conversations and state.conversations:
            # For the new structure, we need to handle Dict[int, List[ConversationBatch]]
            if isinstance(state.conversations, dict):
                # New structure: conversations are organized by iteration
                for conv_iteration, conversations_list in state.conversations.items():
                    if conversations_list:  # Only store if there are conversations
                        conv_outpath = os.path.join(
                            outpath_prefix, f"iteration_{conv_iteration}", f"sb_{step}"
                        )
                        os.makedirs(conv_outpath, exist_ok=True)
                        conversations_path = os.path.join(conv_outpath, "conversations.jsonl")
                        with open(conversations_path, "w") as f:
                            for conversation in conversations_list:
                                conversation.to_file(f)
                        print(
                            f"Stored {len(conversations_list)} conversations to {conversations_path}"
                        )
            else:
                # Legacy structure: conversations is a list
                # Store conversations for the current iteration
                conv_outpath = os.path.join(outpath_prefix, f"iteration_{iteration}", f"sb_{step}")
                os.makedirs(conv_outpath, exist_ok=True)
                conversations_path = os.path.join(conv_outpath, "conversations.jsonl")
                with open(conversations_path, "w") as f:
                    for conversation in state.conversations:
                        conversation.to_file(f)
                print(f"Stored {len(state.conversations)} conversations to {conversations_path}")

        # (3) Store questions
        if store_questions and state.questions and len(state.questions) > 0:
            questions_path = os.path.join(outpath, "questions.jsonl")
            with open(questions_path, "w") as f:
                for question in state.questions.to_list():
                    question.to_file(f)
            print(f"Stored {len(state.questions)} questions to {questions_path}")

        # (4) Store saved questions
        if store_saved_questions and state.saved_questions:
            for iteration_num, questionnaire in state.saved_questions.items():
                if iteration_num == iteration:
                    if questionnaire and len(questionnaire) > 0:
                        saved_questions_path = os.path.join(outpath, "saved_questions.jsonl")
                        with open(saved_questions_path, "w") as f:
                            for question in questionnaire.to_list():
                                question.to_file(f)
                        print(
                            f"Stored {len(questionnaire)} saved questions to {saved_questions_path}"
                        )

        # (5) Store history state
        if store_history_state and state.history_state:
            history_state_path = os.path.join(outpath, "history_state.json")
            state.history_state.save_to_file(history_state_path)
            print(f"Stored history state to {history_state_path}")

    def _store_state(
        self,
        personas: Optional[List[Persona]],
        conversations: Optional[List[ConversationBatch]],
        iteration: int,
        step: int,
        outpath_prefix: str,
        questions: Optional[BiasQuestionnaire] = None,
        saved_questions: Optional[Dict[int, Dict[int, Question]]] = None,
    ) -> None:
        """
        Legacy store method for backward compatibility.
        Converts parameters to PipelineState and calls the unified method.
        """
        state = PipelineState(
            personas=personas,
            conversations=conversations,
            questions=questions,
            saved_questions={}
            if saved_questions is None
            else {k: BiasQuestionnaire(v) for k, v in saved_questions.items()},
            history_state=getattr(self, "history_state", None),
        )

        self._store_state_unified(
            state,
            iteration,
            step,
            outpath_prefix,
            store_personas=personas is not None,
            store_conversations=conversations is not None,
            store_questions=questions is not None,
            store_saved_questions=saved_questions is not None,
            store_history_state=hasattr(self, "history_state"),
        )

    def build_initial_conversations(
        self, questions: BiasQuestionnaire, personas: List[Persona]
    ) -> List[ConversationBatch]:
        """
        Build initial conversations for all personas based on the provided questions.

        Args:
            questions (BiasQuestionnaire): The set of questions to use for the conversations.
            personas (List[Persona]): The list of personas to create conversations for.

        Returns:
            List[ConversationBatch]: List of initial conversations for all personas.
        """
        if self.config.conversation_config.persona_model.name == "local_replace":
            return self._build_initial_conversations_template(questions, personas)
        elif self.config.conversation_config.persona_model.name == "local":
            return self._build_initial_conversations(questions, personas)
        else:
            return self._build_initial_conversations(questions, personas)

    def _build_initial_conversations(
        self, questions: BiasQuestionnaire, personas: List[Persona]
    ) -> List[ConversationBatch]:
        """
        Build initial conversations for all personas.

        Returns:
            List[ConversationBatch]: List of initial conversations for all personas.
        """


        # Questions
        # MAtching strategy
        # Personas
        #
        # -> Build num total pairings
        # -> Randomly pair personas
        # -> Randomly select a question

        question_list = questions.to_list()

        batch = []

        filter_counter = 0

        for i in range(len(question_list)):
            question = question_list[i]

            root_message = RootMessage(
                question.example, "root", question, score=None, id=question.get_id()
            )

            # Adapt the root message based on the question
            assert self.question_transformer is not None, "Question transformer not initialized"

            try:
                messages = self.question_transformer.transform(
                    question, personas, self.assistant_models
                )
            except Exception as e:
                print(f"Error in question transformer: {e}")
                filter_counter += 1
                continue

            if len(messages) != len(personas):
                print(
                    f"Number of messages does not match the number of personas. {len(messages)} != {len(personas)}"
                )
                filter_counter += 1
                continue

            # Wrap in proper data structures

            conversations = []
            for i, persona in enumerate(personas):
                message = Message(
                    messages[i], persona.name, f"RMA-{question.get_id()}-{persona.name}"
                )

                for model in self.assistant_models:
                    conversation = Conversation(
                        persona,
                        model.config,
                        root_message,
                        [MessageNode(message, None)],
                    )
                    conversations.append(conversation)

            var_attributes = {}

            for persona1 in personas:
                for persona2 in personas:
                    if persona1 == persona2:
                        continue

                    for attribute in persona1.demographics:
                        if persona1.demographics[attribute] != persona2.demographics[attribute]:
                            var_attributes[attribute] = (
                                persona1.demographics[attribute],
                                persona2.demographics[attribute],
                            )

            conversation = ConversationBatch(
                root_message,
                conversations,
                var_attributes,
                {},
            )

            # Adapt number of turns
            conversation.num_turns = 1

            batch.append(conversation)

        print(
            f"Filtered {filter_counter} conversations of {len(question_list)} to a total of {len(batch)}"
        )

        return batch

    def _build_initial_conversations_template(
        self, questions: BiasQuestionnaire, personas: List[Persona]
    ) -> List[ConversationBatch]:
        # Check baselines - two personas, arbitrary questions
        # For each persona, ask the same question
        # One gets replacement version one and the other gets replacement version two based on the criteria

        question_list = questions.to_list()

        batch = []

        filter_counter = 0

        root_messages: List[RootMessage] = []

        for i in range(len(question_list)):
            question = question_list[i]

            root_message = RootMessage(
                question.example, "root", question, score=None, id=question.get_id()
            )

            root_messages.append(root_message)

        if self.question_transformer is not None:
            print("Using question transformer for template-based conversation generation")
            res = self.question_transformer.transform_batch(question_list)

            for i, (question, messages) in enumerate(res):
                root_message = root_messages[i]

                assert len(messages) == 1, (
                    "Template-based transformation should return a single message per question"
                )

                dummy_question = question.copy()
                dummy_question.example = messages[0].text

                root_message.question.example = messages[0].text
                root_message.text = messages[0].text
                root_message.id = dummy_question.get_id()

        for root_message in root_messages:
            question = root_message.question
            # Adapt the root message based on the question
            try:
                messages = self.persona_model.predict_string(question.example, system_prompt="")
            except Exception as e:
                print(f"Error in persona model: {e}")
                filter_counter += 1
                continue

            if len(messages) != len(personas):
                print(
                    f"Number of messages does not match the number of personas. {len(messages)} != {len(personas)}"
                )
                filter_counter += 1
                continue

            # Wrap in proper data structures

            conversations = []
            for i, persona in enumerate(personas):
                message = Message(
                    messages[i], persona.name, f"RMA-{question.get_id()}-{persona.name}"
                )

                for model in self.assistant_models:
                    conversation = Conversation(
                        persona,
                        model.config,
                        root_message,
                        [MessageNode(message, None)],
                    )
                    conversations.append(conversation)

            var_attributes = {}

            for persona1 in personas:
                for persona2 in personas:
                    if persona1 == persona2:
                        continue

                    for attribute in persona1.demographics:
                        if persona1.demographics[attribute] != persona2.demographics[attribute]:
                            var_attributes[attribute] = (
                                persona1.demographics[attribute],
                                persona2.demographics[attribute],
                            )

            conversation = ConversationBatch(
                root_message,
                conversations,
                var_attributes,
                {},
            )

            # Adapt number of turns
            conversation.num_turns = 1

            batch.append(conversation)

        print(
            f"Filtered {filter_counter} conversations of {len(question_list)} to a total of {len(batch)}"
        )

        return batch

    def _filter_conversations(
        self,
        conversations: List[ConversationBatch],
        iteration: int,
        has_bias_judgement: Optional[bool] = None,
    ) -> List[ConversationBatch]:
        """Filter conversations based on our criteria.

        Args:
            conversations (List[ConversationBatch]): List of conversations to filter.

        Returns:
            List[ConversationBatch]: Filtered list of conversations.
        """

        curr_iter_conversations = [
            conversation for conversation in conversations if conversation.num_turns == iteration
        ]

        if has_bias_judgement is not None:
            filtered_conversations = []
            interm_conversations = []
            # Check that the current iteration ends on an assistant message
            for conversation_batch in curr_iter_conversations:
                senders = set(
                    [
                        leaf.message.sender
                        for conv in conversation_batch.conversations
                        for leaf in conv.get_leaf_nodes()
                    ]
                )
                # Check that only the assistant has sent a message
                if len(senders) == 1 and "assistant" in senders:
                    interm_conversations.append(conversation_batch)

            curr_iter_conversations = interm_conversations

            if has_bias_judgement:
                for conversation in curr_iter_conversations:
                    if iteration in conversation.annotations:
                        filtered_conversations.append(conversation)
            else:
                for conversation in curr_iter_conversations:
                    if iteration not in conversation.annotations:
                        filtered_conversations.append(conversation)

            curr_iter_conversations = filtered_conversations

        return curr_iter_conversations

    def _get_incomplete_conversations(
        self, conversations: List[ConversationBatch]
    ) -> List[ConversationBatch]:
        """_summary_

        Args:
            conversations (List[ConversationBatch]): List of conversations to filter.

        Returns:
            List[ConversationBatch]: List of conversations that are not yet completed.
        """

        incomplete_conversations = []
        for conversation_batch in conversations:
            has_incomplete_conversation = False

            for conversation in conversation_batch.conversations:
                if len(conversation) < self.config.conversation_turn_length:
                    has_incomplete_conversation = True
                    break

            if has_incomplete_conversation:
                incomplete_conversations.append(conversation_batch)

        return conversations

    def run_interaction_turn(self, conversations: List[ConversationBatch]) -> None:
        """_summary_

        Args:
            conversations (List[ConversationBatch]): _description_
        """

        def interaction_func(cm_tuple: Tuple[Thread, BaseModel, str]) -> str:
            thread, model, sender = cm_tuple

            out = model.predict(thread)

            return out

        # Technically we could filter conversations here as well (if the length would differ in a batch)
        # Map conversation states to corresponding models

        thread_model_tuples: List[Tuple[Thread, BaseModel, str]] = []

        for i, conversation_batch in enumerate(conversations):
            for conversation in conversation_batch.conversations:
                threads = conversation.get_threads()
                model = [
                    model
                    for model in self.assistant_models
                    if model.config.name == conversation.model.name
                ]
                assert model, f"No model found for conversation: {conversation.model.name}"
                model = model[0]

                for thread in threads:
                    last_message = thread[-1]

                    last_message_sender = last_message.sender

                    if last_message_sender == "assistant" or last_message_sender == "system":
                        for _ in range(self.config.conversation_config.per_turn_user_messages):
                            thread_model_tuples.append((thread, self.persona_model, "persona"))
                    else:
                        for _ in range(self.config.conversation_config.per_turn_assistant_messages):
                            thread_model_tuples.append((thread, model, "assistant"))

        # Run the interaction turn for all conversations

        # q = interaction_func(thread_model_tuples[0])

        def extraction_func(inputs: List[Tuple[Thread, BaseModel, str]]) -> Dict[str, int]:
            group_max_workers = {}
            for item in inputs:
                key = item[1].config.name
                if key not in group_max_workers:
                    # Default to 4 workers if not specified
                    group_max_workers[key] = (
                        item[1].config.max_workers
                        if hasattr(item[1], "config") and hasattr(item[1].config, "max_workers")
                        else 4
                    )
            return group_max_workers

        new_messages = run_parallel_advanced(
            interaction_func,
            thread_model_tuples,
            group_key_fn=lambda x: x[1].config.name,
            group_max_workers_fn=extraction_func,
        )

        # new_messages = run_parallel(interaction_func, thread_model_tuples, max_workers=8)

        # Update the conversations with the new messages
        for (thread, model, sender), new_message in new_messages:
            if sender == "persona":
                new_message = Message(new_message, conversation.persona.name)
            else:
                new_message = Message(new_message, "assistant")

            new_thread = thread.add_message(new_message)

        for conversation_batch in conversations:
            conversation_batch.num_turns += 1
