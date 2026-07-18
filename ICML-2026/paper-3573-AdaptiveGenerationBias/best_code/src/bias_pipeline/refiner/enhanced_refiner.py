import json
from collections import deque, defaultdict
from src.configs import RefinerConfig
from src.models import get_model, run_parallel
import numpy as np
import random
import re
from src.bias_pipeline.data_types.conversation import ConversationBatch
from src.bias_pipeline.questionaires.questionaire import Question
from src.models import APIPrompt

from src.prompts.prompt_loader import get_prompt_loader
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.bias_pipeline.pipeline import PipelineState
    from src.bias_pipeline.history_state import HistoryState, IterationStats


class EnhancedRefiner:
    """
    Enhanced refiner that combines individual question refinement with domain-level analysis
    and strategic question generation to improve bias detection effectiveness.
    """

    def __init__(
        self,
        refiner_cfg: RefinerConfig,
        attribute: str,
        fitness_function=None,
        question_config=None,
    ) -> None:
        self.config = refiner_cfg
        self.attribute = attribute
        self.model = get_model(refiner_cfg.model)
        self.filter_model = get_model(refiner_cfg.filter_model)
        self.prompt_loader = get_prompt_loader()
        self.question_config = question_config  # Store question config for template formatting

        # Load schemas for structured generation
        with open("src/prompts/schemas/refiner/superdomain_generation_schema.json") as json_file:
            self.superdomain_schema = json.load(json_file)

        with open("src/prompts/schemas/refiner/domain_generation_schema.json") as json_file:
            self.domain_schema = json.load(json_file)

        with open("src/prompts/schemas/refiner/replace_question_schema.json") as json_file:
            self.replace_schema = json.load(json_file)

        with open("src/prompts/schemas/refiner/refine_question_schema.json") as json_file:
            self.refine_schema = json.load(json_file)

        with open("src/prompts/schemas/refiner/new_topic_schema.json") as json_file:
            self.new_topic_schema = json.load(json_file)

        with open("src/prompts/schemas/refiner/question_filter_schema.json") as json_file:
            self.question_filter_schema = json.load(json_file)

        self.filter_samples = ""

        if (
            self.config
            and hasattr(self.config, "filter_examples_path")
            and self.config.filter_examples_path
        ):
            with open(self.config.filter_examples_path) as file:
                self.filter_samples = file.read()

        self.good_question_samples = []

        if (
            self.config
            and hasattr(self.config, "good_question_samples_path")
            and self.config.good_question_samples_path
        ):
            with open(self.config.good_question_samples_path) as json_file:
                self.good_question_samples = json.load(json_file)

        # Initialize adaptive algorithm components
        self.fitness_function = fitness_function
        self._init_adaptive_algorithm()

    def _init_adaptive_algorithm(self) -> None:
        """Initialize data structures for the adaptive algorithm."""
        # Get algorithm parameters from config
        strategy_cfg = self.config.refinement_strategy

        # History tracking (domain, bias_score, fitness)
        self.hist = deque(maxlen=strategy_cfg.window_size)

        # Current counts for each unit (superdomain, domain, topic)
        self.current_counts = defaultdict(int)

        # Store algorithm parameters
        self.frac_existing = strategy_cfg.frac_existing
        self.super_max = strategy_cfg.super_max
        self.domain_max = strategy_cfg.domain_max
        self.bias_threshold = strategy_cfg.bias_threshold
        self.window_no_high_bias = strategy_cfg.window_no_high_bias
        self.window_positive_fitness = strategy_cfg.window_positive_fitness
        self.learning_rate_up = strategy_cfg.learning_rate_up
        self.learning_rate_dn = strategy_cfg.learning_rate_dn

        # Initialize fitness history tracking
        self.fitness_history = defaultdict(list)

    # ---------------------------------------------------------------------
    #  Public entry point
    # ---------------------------------------------------------------------
    def refine_questions(
        self,
        state: "PipelineState",
        num_questions: int = 10,
    ) -> Tuple[List[Question], List[Question]]:
        """
        Adaptive refinement pipeline.

        1.  Compute per-unit base quotas from the latest iteration.
        2.  Flag units for up- / down- scaling (strict windows logic).
        3.  Apply learning-rate scaling and normalise → `final_quotas`
        4.  Create exploration quotas for unseen units → `exploration_quotas`
        5.  Decide, per domain, how much of its quota is “refine” vs “replace”
            using `refine_replace_ratio` × domain-fitness weighting.
        6.  Generate / refine questions and return the combined list.
        """

        # -----------------------------------------------------------------
        # 0) House-keeping & config
        # -----------------------------------------------------------------
        self.num_questions_round = num_questions  # needed by helpers

        history_state = state.history_state
        assert history_state is not None, "HistoryState must be initialised."

        latest_iter = history_state.get_latest_iteration()

        current_conversations = state.get_current_conversations()
        question_fitness_tuples = [
            (q.root_message.question, q.compute_current_fitness(self.fitness_function))
            for q in current_conversations
        ]

        # Store questions with high enough fitness for later use
        saved_questions: List[Question] = [
            q for q, fit in question_fitness_tuples if fit >= self.bias_threshold
        ]

        all_saved_questions: List[Question] = []
        for index, vals in state.saved_questions.items():
            all_saved_questions.extend(vals.to_list())
        all_saved_questions.extend(saved_questions)

        # -----------------------------------------------------------------
        # 1) Base quotas (keep same share as last round, scaled)
        # -----------------------------------------------------------------
        base_quotas = self._compute_initial_quotas(latest_iter)

        # -----------------------------------------------------------------
        # 2) Up / Down qualification
        # -----------------------------------------------------------------
        upscale_units_superdomain, downscale_units_superdomain = self._identify_qualifying(
            history_state, level="superdomain"
        )
        upscale_units_domain, downscale_units_domain = self._identify_qualifying(
            history_state, level="domain"
        )

        # -----------------------------------------------------------------
        # 3) Apply scaling
        # -----------------------------------------------------------------
        for d in downscale_units_superdomain:
            if d in base_quotas["superdomain"]:
                base_quotas["superdomain"][d] *= 1.0 - self.learning_rate_dn
        for d in upscale_units_superdomain:
            if d in base_quotas["superdomain"]:
                base_quotas["superdomain"][d] *= 1.0 + self.learning_rate_up
        for d in downscale_units_domain:
            if d in base_quotas["domain"]:
                base_quotas["domain"][d] *= 1.0 - self.learning_rate_dn
        for d in upscale_units_domain:
            if d in base_quotas["domain"]:
                base_quotas["domain"][d] *= 1.0 + self.learning_rate_up

        # 3a) Apply default scaling with protection for new domains/superdomains
        base_quotas = self._apply_protected_scaling(base_quotas, history_state)

        # 3b) Normalise → integer quotas and exploration quotas
        quotas, to_explore = self._normalize_and_round_quotas(base_quotas)

        # -----------------------------------------------------------------
        # 4) Exploration pool (new super domains / domains / topics)
        # -----------------------------------------------------------------

        new_sds, new_dms = self._explore_new(state, to_explore)

        # Print overview over current quotas and exploration
        print("Current quotas:")
        print(quotas)
        print("To explore:")
        print(to_explore)

        # merge
        for elem in new_dms:
            sd_dom, quota = elem
            sd = sd_dom.split("::")[0]  # extract superdomain
            if sd not in quotas["superdomain"]:
                quotas["superdomain"][sd] = 0
            quotas["superdomain"][sd] += quota
            if sd_dom not in quotas["domain"]:
                quotas["domain"][sd_dom] = 0
            quotas["domain"][sd_dom] += quota

        # -----------------------------------------------------------------
        # 5) Decide “refine vs replace” for every domain
        # -----------------------------------------------------------------

        # New topic and question
        new_topic_percent = self.config.refinement_strategy.new_topic_percent
        # Refine question
        refine_question_percent = self.config.refinement_strategy.refine_question_percent
        # Keep topic but replace question
        replace_question_percent = self.config.refinement_strategy.replace_question_percent

        new_questions: List[Question] = []
        prompts: List[APIPrompt] = []

        for domain_key, quota in quotas["domain"].items():
            if quota <= 0:
                continue

            key_tuple = None
            if "::" in domain_key:
                superdomain_key, domain_key = domain_key.split("::")
                key_tuple = (superdomain_key, domain_key)
            assert key_tuple is not None

            # Compute the baseline values we need
            questions = [
                (q, fit)
                for q, fit in question_fitness_tuples
                if (q.superdomain, q.domain) == key_tuple
            ]
            questions.sort(key=lambda x: x[1], reverse=True)

            num_questions = len(questions)
            available_ids = set(range(num_questions))
            question_id_zip = list(zip(available_ids, questions))

            # Filter out saved_questions
            saved_questions_ids = [i for i, q in enumerate(questions) if q[0] in saved_questions]
            available_ids = available_ids - set(saved_questions_ids)

            topics = set(q.topic for q, _ in questions)

            num_refine_questions = min(int(round(quota * refine_question_percent)), num_questions)
            num_replace_question = min(int(round(quota * replace_question_percent)), num_questions)

            num_new = quota - num_refine_questions - num_replace_question

            # Refine prompts
            if num_refine_questions > 0:
                remaining_questions = [q for q in question_id_zip if q[0] in available_ids]

                if len(remaining_questions) < num_refine_questions:
                    num_refine_questions = len(remaining_questions)
                    num_replace_question = 0
                    num_new = quota - num_refine_questions

                refine_questions = remaining_questions[:num_refine_questions]
                available_ids = available_ids.difference(set(q[0] for q in refine_questions))
                questions_to_refine = [q[1][0] for q in refine_questions]

                # Get good question samples for refinement
                max_questions = min(2, len(self.good_question_samples))
                good_question_samples = []
                if max_questions > 0:
                    good_question_samples = random.sample(self.good_question_samples, max_questions)

                # Create prompts for refinement
                refine_prompts = self.prompt_loader.create_refine_prompts(
                    questions_to_refine,
                    self.attribute,
                    self.question_config,
                    good_question_samples=good_question_samples,
                )
                prompts.extend(refine_prompts)

            # Replace prompts
            if num_replace_question > 0:
                # Take remaining questions that need replacement
                remaining_questions = [q for q in question_id_zip if q[0] in available_ids]
                replace_questions = remaining_questions[:num_replace_question]
                available_ids = available_ids.difference(set(q[0] for q in replace_questions))
                questions_to_replace = [q[1][0] for q in replace_questions]

                # Get good question samples
                max_questions = min(2, len(self.good_question_samples))
                good_question_samples = []
                if max_questions > 0:
                    good_question_samples = random.sample(self.good_question_samples, max_questions)

                # Create prompts for replacement
                replace_prompts = self.prompt_loader.create_replace_prompts(
                    questions_to_replace,
                    self.attribute,
                    question_config=self.question_config,
                    good_question_samples=good_question_samples,
                )
                prompts.extend(replace_prompts)

            # Generate entirely new topics
            if num_new > 0:
                # Get existing topics in this domain
                existing_topics = {key_tuple: list(topics)}

                print(
                    f"Generating {num_new} new topics for domain {domain_key} and key {key_tuple}"
                )

                samples: List[Question] = []
                in_domain_questions = [
                    q for q in all_saved_questions if (q.superdomain, q.domain) == key_tuple
                ]
                out_domain_questions = [q for q in all_saved_questions if q.domain != domain_key]
                # Randomly sample 3 good questions in case we have enough
                max_questions = min(2, len(self.good_question_samples))
                good_question_samples = []
                if max_questions > 0:
                    good_question_samples = random.sample(self.good_question_samples, max_questions)

                if len(in_domain_questions) > 3:
                    samples = random.sample(in_domain_questions, 3)
                else:
                    samples = in_domain_questions
                    samples.extend(random.sample(out_domain_questions, 3 - len(samples)))

                # Create prompts for new topic generation
                new_topic_prompts = self.prompt_loader.create_new_topic_prompts(
                    [key_tuple],
                    self.attribute,
                    existing_topics,
                    samples,
                    good_question_samples,
                    question_config=self.question_config,
                )

                # We need to generate multiple questions for this domain
                for _ in range(num_new):
                    prompts.extend(new_topic_prompts)

        # Run all prompts in parallel creating the new questions
        if prompts:

            def func_generate_question(prompt_tuple):
                return self._run_generation_prompt(prompt_tuple)

            results = run_parallel(func_generate_question, prompts, self.model.config.max_workers)
            new_questions = [res[1] for res in results if res is not None]

            # Apply question filter to improve directness and consistency
            if new_questions:
                filtered_questions = self._filter_questions(new_questions, all_saved_questions)
                new_questions = filtered_questions

        else:
            new_questions = []

        return new_questions, saved_questions

    def _explore_new(
        self, state: "PipelineState", to_explore: Dict[str, int]
    ) -> Tuple[List[str], List[str]]:
        min_new_dom_per_sd = 2
        min_new_top_per_dom = 2
        min_new_top_sd = min_new_top_per_dom * min_new_dom_per_sd

        # Get existing superdomains and domains from state
        stats = state.history_state.get_latest_iteration()
        existing_superdomains = set(stats.superdomain_stats.keys())
        existing_domains = set(stats.domain_stats.keys())

        # Track generated items locally
        generated_superdomains = set()
        generated_domains = set()

        # Final results
        new_superdomains: List[tuple[str, int]] = []
        new_domains: List[tuple[str, int]] = []

        # Collect all generation tasks to run in parallel
        generation_tasks = []

        # Superdomain exploration tasks
        num_new_superdomains = max(to_explore.get("superdomain", 0) // min_new_top_sd, 1)
        available_sd_topics = to_explore.get("superdomain", 0)

        for _ in range(int(num_new_superdomains)):
            if available_sd_topics <= 0:
                break

            topics_for_sd = min(available_sd_topics, min_new_top_sd)
            available_sd_topics -= topics_for_sd

            # Create superdomain generation task
            focus_areas = getattr(state, "focus_areas", None) or ["general"]
            target_focus = focus_areas[0] if focus_areas else "general"

            generation_tasks.append(
                {
                    "type": "superdomain",
                    "state": state,
                    "target_focus": target_focus,
                    "topics_quota": topics_for_sd,
                }
            )

        # Domain exploration tasks for existing superdomains
        for sd, num_new_tops in to_explore.get("domain", {}).items():
            top_counter = num_new_tops

            while top_counter > 0:
                tops_for_dom = min(top_counter, min_new_top_per_dom)

                generation_tasks.append(
                    {
                        "type": "domain",
                        "state": state,
                        "target_superdomain": sd,
                        "topics_quota": tops_for_dom,
                    }
                )

                top_counter -= tops_for_dom

        # Generate superdomains and domains in parallel with retry for duplicates
        superdomain_results, domain_results = self._generate_units_parallel_with_retry(
            generation_tasks, existing_superdomains, existing_domains
        )

        # Process superdomain results
        for superdomain_name, topics_quota in superdomain_results:
            generated_superdomains.add(superdomain_name)
            new_superdomains.append((superdomain_name, topics_quota))

        # Process domain results
        for domain_name, topics_quota in domain_results:
            generated_domains.add(domain_name)
            new_domains.append((domain_name, topics_quota))

        # Generate domains under new superdomains
        if new_superdomains:
            domain_tasks_for_new_sds = []
            for superdomain_name, topics_quota in new_superdomains:
                topics_remaining = topics_quota

                while topics_remaining > 0:
                    topics_for_dom = min(topics_remaining, min_new_top_per_dom)

                    domain_tasks_for_new_sds.append(
                        {
                            "type": "domain",
                            "state": state,
                            "target_superdomain": superdomain_name,
                            "topics_quota": topics_for_dom,
                        }
                    )

                    topics_remaining -= topics_for_dom

            # Generate domains for new superdomains in parallel
            _, additional_domain_results = self._generate_units_parallel_with_retry(
                domain_tasks_for_new_sds,
                existing_superdomains | generated_superdomains,
                existing_domains | generated_domains,
            )

            # Add additional domain results
            for domain_name, topics_quota in additional_domain_results:
                new_domains.append((domain_name, topics_quota))

        return new_superdomains, new_domains

    def _apply_protected_scaling(
        self, base_quotas: Dict[str, Dict[str, float]], history_state: "HistoryState"
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply protected scaling that ensures domains and superdomains exist for at least
        two iterations before applying the fraction scaling. Domains and superdomains
        from iteration zero are exempt from this protection.

        Args:
            base_quotas: Dictionary containing quotas for superdomains and domains
            history_state: History state to check iteration appearances

        Returns:
            Updated base_quotas with protected scaling applied
        """
        current_iteration = max(history_state.get_available_iterations())

        # Apply scaling to superdomains with protection
        for unit, quota in base_quotas["superdomain"].items():
            if quota > 0:
                if self._should_apply_scaling(
                    unit, "superdomain", current_iteration, history_state
                ):
                    base_quotas["superdomain"][unit] *= self.frac_existing

        # Apply scaling to domains with protection
        for unit, quota in base_quotas["domain"].items():
            if quota > 0:
                if self._should_apply_scaling(unit, "domain", current_iteration, history_state):
                    base_quotas["domain"][unit] *= self.frac_existing

        return base_quotas

    def _should_apply_scaling(
        self, unit: str, level: str, current_iteration: int, history_state: "HistoryState"
    ) -> bool:
        """
        Determine if scaling should be applied to a unit based on protection rules.

        Protection rules:
        1. Units from iteration 0 are always scaled (no protection)
        2. Units must exist for at least 2 iterations before scaling is applied

        Args:
            unit: The unit name (domain or superdomain)
            level: Either "superdomain" or "domain"
            current_iteration: Current iteration number
            history_state: History state to check appearances

        Returns:
            True if scaling should be applied, False otherwise
        """
        # Get all iterations where this unit appeared
        unit_iterations = []

        for iteration in history_state.get_available_iterations():
            stats = history_state.iterations[iteration]
            level_stats = getattr(stats, f"{level}_stats")  # superdomain_stats or domain_stats

            if unit in level_stats:
                unit_iterations.append(iteration)

        # Sort iterations to get chronological order
        unit_iterations.sort()

        if not unit_iterations:
            # Unit doesn't exist, no scaling needed
            return False

        first_appearance = unit_iterations[0]

        # Rule 1: Units from iteration 0 are always scaled (no protection)
        if first_appearance == 0:
            return True

        # Rule 2: Unit must exist for at least 2 iterations before scaling
        return not len(unit_iterations) < 2

    def _identify_qualifying(
        self, history_state: "HistoryState", level: str
    ) -> Tuple[List[str], List[str]]:
        """
        Decide which units (superdomains or domains) should be up-scaled or
        down-scaled for the next round.

        Down-scale  ⟺  **very few high-bias successes** in the last
                      `window_no_high_bias` iterations
        Up-scale    ⟺  **strictly increasing average fitness** during the last
                      `window_positive_fitness` iterations

        Returns
        -------
        upscale, downscale : (List[str], List[str])
        """
        if level not in {"superdomain", "domain"}:
            raise ValueError("level must be 'superdomain' or 'domain'")

        win_low = self.window_no_high_bias
        win_high = self.window_positive_fitness

        # --- Helper to collect the last k iteration numbers ------------------
        def _last_k_iters(k: int) -> List[int]:
            return sorted(history_state.get_available_iterations(), reverse=True)[:k]

        # ---------------------------------------------------------------------
        # 1) DOWN-SCALE candidates: look at high-bias *success* counts
        # ---------------------------------------------------------------------
        downscale: set[str] = set()
        if len(history_state.get_available_iterations()) >= win_low:
            recent_iters = _last_k_iters(win_low)

            # successes / totals over the window
            success_counts: dict[str, int] = defaultdict(int)
            total_counts: dict[str, int] = defaultdict(int)

            for it in recent_iters:
                stats = history_state.iterations[it]
                lv_dict = getattr(stats, f"{level}_stats")  # superdomain_stats or domain_stats

                for unit_name, unit_stats in lv_dict.items():
                    total_counts[unit_name] += unit_stats.get("count", 0)

                    # count high-bias hits (score ≥ bias_threshold)
                    for score_list in unit_stats.get("scores", {}).values():
                        success_counts[unit_name] += sum(
                            1 for s in score_list if s >= self.bias_threshold
                        )

            for unit, tot in total_counts.items():
                succ = success_counts[unit]
                rate = succ / tot if tot else 0.0
                if tot and succ < 2 and rate < 0.05:
                    downscale.add(unit)

        # ---------------------------------------------------------------------
        # 2) UP-SCALE candidates: check monotonic ↑ fitness over last window
        # ---------------------------------------------------------------------
        upscale: set[str] = set()
        if len(history_state.get_available_iterations()) >= win_high:
            recent_iters = list(reversed(_last_k_iters(win_high)))  # chronological order

            fitness_per_unit: dict[str, list[float]] = defaultdict(list)
            for it in recent_iters:
                metrics = history_state.iterations[it].calculate_domain_metrics(
                    self.fitness_function, self.bias_threshold
                )
                lvl_metrics = metrics[f"{level}_metrics"]
                for unit_name, m in lvl_metrics.items():
                    fitness_per_unit[unit_name].append(
                        m["fitness_scores"]["avg"]  # uses avg fitness per iteration
                    )

            for unit, series in fitness_per_unit.items():
                if len(series) == win_high and all(
                    series[i] > series[i - 1] for i in range(1, len(series))
                ):
                    upscale.add(unit)
                elif len(series) < win_high:
                    # If we don't have enough data, we check if the unit has at the moment high bias (> 3 fitness)
                    if fitness_per_unit[unit][-1] >= self.bias_threshold:
                        upscale.add(unit)

        # make sure a unit is not on both lists
        downscale.difference_update(upscale)
        return list(upscale), list(downscale)

    def _compute_initial_quotas(
        self, iteration_state: "IterationStats"
    ) -> Dict[str, Dict[str, float]]:
        """Compute initial quotas based on current counts."""

        def compute_quota(elements: Dict[str, any]) -> Dict[str, float]:
            """Compute quota for a single unit based on current counts."""

            total_count = 0
            fractions = {}
            for key, val in elements.items():
                total_count += val["count"]
                fractions[key] = val["count"]

            assert total_count > 0, "Total count is zero, cannot compute quota."
            for key in fractions:
                fractions[key] /= total_count

            return fractions

        def compute_hierarchical_quota(
            elements: Dict[str, any], parent_key_extractor
        ) -> Dict[str, float]:
            """Compute quotas relative to their parent units."""

            # Group elements by their parent
            parent_groups = defaultdict(dict)
            for key, val in elements.items():
                parent_key = parent_key_extractor(key)
                parent_groups[parent_key][key] = val

            # Compute quotas within each parent group
            all_fractions = {}
            for parent_key, group_elements in parent_groups.items():
                group_fractions = compute_quota(group_elements)
                all_fractions.update(group_fractions)

            return all_fractions

        # Superdomain quotas remain relative to overall
        superdomain_quotas = compute_quota(iteration_state.superdomain_stats)

        # Domain quotas relative to their superdomain
        def extract_superdomain_from_domain(domain_key: str) -> str:
            """Extract superdomain from domain key (assumes format: 'superdomain::domain')"""
            return domain_key.split("::")[0] if "::" in domain_key else domain_key

        domain_quotas = compute_hierarchical_quota(
            iteration_state.domain_stats, extract_superdomain_from_domain
        )

        # Topic quotas relative to their domain
        def extract_domain_from_topic(topic_key: str) -> str:
            """Extract domain from topic key (assumes format: 'superdomain::domain::topic')"""
            parts = topic_key.split("::")
            if len(parts) >= 2:
                return "::".join(parts[:2])  # Return 'superdomain::domain'
            return topic_key

        topic_quotas = compute_hierarchical_quota(
            iteration_state.topic_stats, extract_domain_from_topic
        )

        res_dict = {
            "superdomain": superdomain_quotas,
            "domain": domain_quotas,
            "topic": topic_quotas,
        }

        return res_dict

    def _normalize_and_round_quotas(
        self, full_base_quotas: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, any]]:
        TOTAL_Q = self.num_questions_round

        def largest_remainder(fracs: Dict[str, float], total: int) -> Dict[str, int]:
            if total <= 0 or not fracs:
                return {k: 0 for k in fracs}
            scale = total / sum(fracs.values())
            raw = {k: v * scale for k, v in fracs.items()}

            ints = {k: int(v) for k, v in raw.items()}
            remainder = total - sum(ints.values())

            if remainder > 0:
                # add to biggest fractional parts
                ranked = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
                for k, _ in ranked[:remainder]:
                    ints[k] += 1
            return ints

        # ------------------------------------------------------------------ #
        # 1)  SUPER-DOMAIN ALLOCATION                                        #
        # ------------------------------------------------------------------ #
        sd_fracs = full_base_quotas.get("superdomain", {})
        sd_float_sum = sum(sd_fracs.values())  # ≤ 1.0  (could be 0)
        sd_target = round(sd_float_sum * TOTAL_Q)  # how many q’s they get
        sd_ints = largest_remainder(sd_fracs, sd_target)

        missing_super = TOTAL_Q - sum(sd_ints.values())  # ≥ 0

        # ------------------------------------------------------------------ #
        # 2)  DOMAIN  (inside each super-domain)                             #
        # ------------------------------------------------------------------ #
        dom_fracs_all = full_base_quotas.get("domain", {})
        dom_ints: Dict[str, int] = {}
        missing_dom: Dict[str, int] = {}

        # bucket domain fractions by their parent SD
        dom_by_sd: Dict[str, Dict[str, float]] = defaultdict(dict)
        for dom_key, frac in dom_fracs_all.items():
            sd = dom_key.split("::", 1)[0]
            dom_by_sd[sd][dom_key] = frac

        for sd, sd_quota in sd_ints.items():
            dom_fracs = dom_by_sd.get(sd, {})
            dom_float_sum = min(sum(dom_fracs.values()), 1.0)  # ≤ 1.0
            dom_target = round(dom_float_sum * sd_quota)  # ≤ sd_quota
            dom_ints.update(largest_remainder(dom_fracs, dom_target))
            missing_dom[sd] = sd_quota - dom_target  # ≥ 0

            if missing_dom[sd] < 0:
                raise ValueError(
                    f"Negative missing quota for superdomain '{sd}': {missing_dom[sd]}"
                )

        # ------------------------------------------------------------------ #
        # 3)  Compose outputs                                                #
        # ------------------------------------------------------------------ #
        alloc_ints = {
            "superdomain": {k: v for k, v in sd_ints.items() if v > 0},
            "domain": {k: v for k, v in dom_ints.items() if v > 0},
        }

        missing = {
            "superdomain": missing_super,
            "domain": missing_dom,
        }

        return alloc_ints, missing

    def _create_superdomain_prompt(
        self, state: "PipelineState", target_focus: str
    ) -> Optional[Tuple[str, str]]:
        """
        Create prompt for superdomain generation.

        Returns:
            Tuple of (system_prompt, query) or None if creation failed
        """
        try:
            # Separate high and low performing examples, filtering by superdomain
            stats = state.history_state.get_latest_iteration().superdomain_stats
            all_sds = list(stats.keys())

            # Get existing domains in this superdomain to avoid duplicates
            low_performing = []
            high_performing = []
            sorted_stats = sorted(
                stats.items(),
                key=lambda item: item[1]["fitness_scores"]["avg"],
                reverse=True,
            )
            high_performing = sorted_stats[: int(min(3, len(sorted_stats) // 2))]  # Top 20%
            low_performing = sorted_stats[-min(int(len(sorted_stats) // 2), 3) :]

            high_performing = [sd[0] for sd in high_performing]
            low_performing = [sd[0] for sd in low_performing]

            # Use templates to generate new superdomain with structured output
            system_prompt = self.prompt_loader.get_template(
                "refinement/generation/generate_superdomain_system.j2", attribute=self.attribute
            )

            query = self.prompt_loader.get_template(
                "refinement/generation/generate_superdomain_query.j2",
                attribute=self.attribute,
                target_focus=target_focus,
                high_performing_examples=high_performing,  # Limit examples
                low_performing_examples=low_performing,
                existing_superdomains=all_sds,
            )

            return (system_prompt, query)

        except Exception as e:
            print(f"Error creating superdomain prompt: {e}")
            return None

    def _create_domain_prompt(
        self, state: "PipelineState", target_superdomain: str, reasoning: str = ""
    ) -> Optional[Tuple[str, str]]:
        """
        Create prompt for domain generation.

        Returns:
            Tuple of (system_prompt, query) or None if creation failed
        """
        try:
            # Separate high and low performing examples, filtering by superdomain
            stats = state.history_state.get_latest_iteration().domain_stats
            all_doms = list(stats.keys())

            # Get existing domains in this superdomain to avoid duplicates
            existing_domains = []
            low_performing = []
            high_performing = []
            for domain_key in all_doms:
                if domain_key.startswith(f"{target_superdomain}::"):
                    domain_name = domain_key.split("::", 1)[1]
                    existing_domains.append(domain_name)

            sorted_stats = sorted(
                stats.items(),
                key=lambda item: item[1]["fitness_scores"]["avg"],
                reverse=True,
            )
            high_performing = sorted_stats[: int(min(3, len(sorted_stats) // 2))]
            low_performing = sorted_stats[-min(int(len(sorted_stats) // 2), 3) :]
            high_performing = [
                dom[0] for dom in high_performing if dom[0].startswith(target_superdomain)
            ]
            low_performing = [
                dom[0] for dom in low_performing if dom[0].startswith(target_superdomain)
            ]

            # Use templates to generate new domain with structured output
            system_prompt = self.prompt_loader.get_template(
                "refinement/generation/generate_domain_system.j2", attribute=self.attribute
            )

            query = self.prompt_loader.get_template(
                "refinement/generation/generate_domain_query.j2",
                attribute=self.attribute,
                target_superdomain=target_superdomain,
                high_performing_examples=high_performing[:3],  # Limit examples
                low_performing_examples=low_performing[:3],
                existing_domains=existing_domains,
                reasoning=reasoning,
            )

            return (system_prompt, query)

        except Exception as e:
            print(f"Error creating domain prompt: {e}")
            return None

    def _generate_units_parallel_with_retry(
        self, generation_tasks: List[Dict], existing_superdomains: set, existing_domains: set
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Generate superdomains and domains in parallel with retry mechanism for duplicates.

        Args:
            generation_tasks: List of generation task dictionaries
            existing_superdomains: Set of existing superdomain names
            existing_domains: Set of existing domain keys

        Returns:
            Tuple of (superdomain_results, domain_results) where each is a list of (name, quota) tuples
        """
        max_retries = 5
        superdomain_results = []
        domain_results = []

        # Track all generated items across retries
        all_generated_superdomains = set()
        all_generated_domains = set()

        # Separate tasks by type
        superdomain_tasks = [task for task in generation_tasks if task["type"] == "superdomain"]
        domain_tasks = [task for task in generation_tasks if task["type"] == "domain"]

        # Process superdomain tasks
        remaining_superdomain_tasks = superdomain_tasks.copy()
        for retry in range(max_retries):
            if not remaining_superdomain_tasks:
                break

            # Create prompts for remaining tasks
            prompts = []
            for task in remaining_superdomain_tasks:
                prompt_data = self._create_superdomain_prompt_with_duplicates(
                    task["state"],
                    task["target_focus"],
                    existing_superdomains | all_generated_superdomains,
                    list(all_generated_superdomains),
                )
                if prompt_data:
                    prompts.append((prompt_data[0], prompt_data[1], task, "superdomain"))

            if not prompts:
                break

            # Run prompts in parallel
            def func_generate_superdomain(prompt_tuple):
                return self._run_exploration_prompt(prompt_tuple)

            results = run_parallel(
                func_generate_superdomain, prompts, self.model.config.max_workers
            )

            # Process results and identify duplicates
            new_remaining_tasks = []
            for i, result in enumerate(results):
                task = remaining_superdomain_tasks[i]

                if result is None:
                    # Failed generation, retry
                    new_remaining_tasks.append(task)
                    continue

                result_type, unit_name, _ = result[-1]
                if result_type == "superdomain":
                    if (
                        unit_name not in existing_superdomains
                        and unit_name not in all_generated_superdomains
                    ):
                        # Success - unique superdomain
                        all_generated_superdomains.add(unit_name)
                        superdomain_results.append((unit_name, task["topics_quota"]))
                    else:
                        # Duplicate, retry
                        all_generated_superdomains.add(unit_name)  # Track duplicate
                        new_remaining_tasks.append(task)
                        print(
                            f"Superdomain '{unit_name}' is duplicate, retrying (attempt {retry + 1}/{max_retries})"
                        )

            remaining_superdomain_tasks = new_remaining_tasks

        # Process domain tasks
        remaining_domain_tasks = domain_tasks.copy()
        for retry in range(max_retries):
            if not remaining_domain_tasks:
                break

            # Create prompts for remaining tasks
            prompts = []
            for task in remaining_domain_tasks:
                prompt_data = self._create_domain_prompt_with_duplicates(
                    task["state"],
                    task["target_superdomain"],
                    existing_domains | all_generated_domains,
                    [
                        d.split("::")[-1]
                        for d in all_generated_domains
                        if d.startswith(f"{task['target_superdomain']}::")
                    ],
                )
                if prompt_data:
                    prompts.append((prompt_data[0], prompt_data[1], task, "domain"))

            if not prompts:
                break

            # Run prompts in parallel
            def func_generate_domain(prompt_tuple):
                return self._run_exploration_prompt(prompt_tuple)

            results = run_parallel(func_generate_domain, prompts, self.model.config.max_workers)

            # Process results and identify duplicates
            new_remaining_tasks = []
            for i, result in enumerate(results):
                task = remaining_domain_tasks[i]

                if result is None:
                    # Failed generation, retry
                    new_remaining_tasks.append(task)
                    continue

                result_type, unit_name, _ = result[-1]
                if result_type == "domain":
                    if unit_name not in existing_domains and unit_name not in all_generated_domains:
                        # Success - unique domain
                        all_generated_domains.add(unit_name)
                        domain_results.append((unit_name, task["topics_quota"]))
                    else:
                        # Duplicate, retry
                        all_generated_domains.add(unit_name)  # Track duplicate
                        new_remaining_tasks.append(task)
                        print(
                            f"Domain '{unit_name}' is duplicate, retrying (attempt {retry + 1}/{max_retries})"
                        )

            remaining_domain_tasks = new_remaining_tasks

        # Log any remaining failed tasks
        if remaining_superdomain_tasks:
            print(
                f"Failed to generate {len(remaining_superdomain_tasks)} unique superdomains after {max_retries} retries"
            )
        if remaining_domain_tasks:
            print(
                f"Failed to generate {len(remaining_domain_tasks)} unique domains after {max_retries} retries"
            )

        return superdomain_results, domain_results

    def _run_exploration_prompt(self, prompt_tuple) -> Optional[Tuple[str, str, Dict]]:
        """
        Run exploration prompt for superdomain or domain generation.

        Args:
            prompt_tuple: Tuple containing (system_prompt, query, task, prompt_type)

        Returns:
            Tuple of (result_type, unit_name, task) or None if generation failed
        """
        try:
            system_prompt, query, task, prompt_type = prompt_tuple

            if prompt_type == "superdomain":
                response = self.model.predict_string(
                    query, system_prompt=system_prompt, response_format=self.superdomain_schema
                )

                # Parse structured response
                try:
                    parsed = json.loads(response)
                    superdomain_analysis = parsed.get("superdomain_analysis", {})
                    new_superdomain = superdomain_analysis.get("superdomain", "").strip()

                    if new_superdomain:
                        return ("superdomain", new_superdomain, task)
                    else:
                        print(f"Generated superdomain is empty")
                        return None

                except json.JSONDecodeError:
                    print(f"Failed to parse superdomain generation response: {response}")
                    return None

            elif prompt_type == "domain":
                response = self.model.predict_string(
                    query, system_prompt=system_prompt, response_format=self.domain_schema
                )

                # Parse structured response
                try:
                    parsed = json.loads(response)
                    domain_analysis = parsed.get("domain_analysis", {})
                    new_domain = domain_analysis.get("domain", "").strip()

                    if "::" in new_domain:
                        new_domain = new_domain.split("::")[-1].strip()

                    # Create full domain key
                    target_superdomain = task["target_superdomain"]
                    domain_key = f"{target_superdomain}::{new_domain}"

                    if new_domain:
                        return ("domain", domain_key, task)
                    else:
                        print(f"Generated domain is empty")
                        return None

                except json.JSONDecodeError:
                    print(f"Failed to parse domain generation response: {response}")
                    return None

            else:
                print(f"Unknown exploration prompt type: {prompt_type}")
                return None

        except Exception as e:
            print(f"Error in _run_exploration_prompt: {e}")
            return None

    def _create_superdomain_prompt_with_duplicates(
        self,
        state: "PipelineState",
        target_focus: str,
        existing_superdomains: set,
        generated_duplicates: List[str],
    ) -> Optional[Tuple[str, str]]:
        """
        Create prompt for superdomain generation with information about duplicates.

        Returns:
            Tuple of (system_prompt, query) or None if creation failed
        """
        try:
            # Get base prompt
            prompt_data = self._create_superdomain_prompt(state, target_focus)
            if not prompt_data:
                return None

            system_prompt, query = prompt_data

            # Add information about duplicates to the query
            if generated_duplicates:
                duplicate_info = f"\n\nNOTE: The following superdomains have already been generated and are duplicates, please avoid generating them again: {', '.join(generated_duplicates)}"
                query += duplicate_info

            return (system_prompt, query)

        except Exception as e:
            print(f"Error creating superdomain prompt with duplicates: {e}")
            return None

    def _create_domain_prompt_with_duplicates(
        self,
        state: "PipelineState",
        target_superdomain: str,
        existing_domains: set,
        generated_duplicates: List[str],
    ) -> Optional[Tuple[str, str]]:
        """
        Create prompt for domain generation with information about duplicates.

        Returns:
            Tuple of (system_prompt, query) or None if creation failed
        """
        try:
            # Get base prompt
            prompt_data = self._create_domain_prompt(state, target_superdomain)
            if not prompt_data:
                return None

            system_prompt, query = prompt_data

            # Add information about duplicates to the query
            if generated_duplicates:
                duplicate_info = f"\n\nNOTE: The following domains have already been generated and are duplicates, please avoid generating them again: {', '.join(generated_duplicates)}"
                query += duplicate_info

            return (system_prompt, query)

        except Exception as e:
            print(f"Error creating domain prompt with duplicates: {e}")
            return None

    def _run_generation_prompt(self, prompt_tuple) -> Optional[Question]:
        """
        Run generation prompt for different types of question generation.

        Args:
            prompt_tuple: Tuple containing (system_prompt, query, context, prompt_type)
                         Context can be a Question object or domain_key string
                         prompt_type is one of: "refine", "replace", "new_topic"

        Returns:
            Generated Question object or None if generation failed
        """
        try:
            # Handle both old and new tuple formats for backward compatibility
            system_prompt, query, context, prompt_type = prompt_tuple

            # Route to appropriate handler based on prompt type
            if prompt_type in ["refine", "replace"]:
                return self._run_refine_replace_prompt(system_prompt, query, context, prompt_type)
            elif prompt_type == "new_topic":
                return self._run_new_topic_prompt(system_prompt, query, context)
            else:
                print(f"Unknown prompt type: {prompt_type}")
                return None

        except Exception as e:
            print(f"Error in _run_generation_prompt: {e}")
            return None

    def _run_refine_replace_prompt(
        self, system_prompt: str, query: str, original_question: Question, prompt_type: str = None
    ) -> Optional[Question]:
        """Run refine or replace prompt and return new question."""
        try:
            is_replace = prompt_type == "replace"
            is_refine = prompt_type == "refine"

            # Set appropriate schema
            if is_replace:
                schema = self.replace_schema
            elif is_refine:
                schema = self.refine_schema
            else:
                raise ValueError("Invalid prompt type for refinement/replacement")

            response = self.model.predict_string(
                query, system_prompt=system_prompt, response_format=schema
            )

            # Parse response based on type
            try:
                parsed = json.loads(response)

                if is_replace:
                    replacement_data = parsed.get("replacement_question", {})
                    question_text = replacement_data.get("question", "")
                elif is_refine:
                    refined_data = parsed.get("refined_question", {})
                    question_text = refined_data.get("question", "")
                else:
                    new_question_data = parsed.get("new_question", {})
                    question_text = new_question_data.get("question", "")

                if not question_text:
                    print(f"No question text found in response: {response}")
                    return None

                # Create new question with same domain/topic as original
                return Question(
                    original_question.superdomain,
                    original_question.domain,
                    original_question.topic,
                    question_text,
                )

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {response}, error: {e}")
                return None

        except Exception as e:
            print(f"Error in _run_refine_replace_prompt: {e}")
            return None

    def _run_new_topic_prompt(
        self, system_prompt: str, query: str, domain_key: tuple[str, str]
    ) -> Optional[Question]:
        """Run new topic generation prompt and return new question."""
        try:
            # Parse domain key
            superdomain, domain = domain_key

            # Set new topic schema

            response = self.model.predict_string(
                query, system_prompt=system_prompt, response_format=self.new_topic_schema
            )

            # Parse response
            try:
                parsed = json.loads(response)
                topic_data = parsed.get("new_topic_question", {})

                topic = topic_data.get("topic", "")
                question_text = topic_data.get("question", "")

                if not topic or not question_text:
                    print(f"Missing topic or question in response: {response}")
                    return None

                # Create new question with new topic
                return Question(superdomain, f"{superdomain}::{domain}", topic, question_text)

            except json.JSONDecodeError as e:
                print(f"Failed to parse new topic JSON response: {response}, error: {e}")
                return None

        except Exception as e:
            print(f"Error in _run_new_topic_prompt: {e}")
            return None

    def _filter_questions(
        self, questions: List[Question], additional_examples: List[Question] = None
    ) -> List[Question]:
        """
        Filter questions to improve directness and consistency using GPT-4.1-mini.

        Args:
            questions: List of questions to filter
            additional_examples: Optional list of additional example questions for reference

        Returns:
            List of filtered/reformatted questions
        """
        if not questions:
            return questions

        print(f"Filtering {len(questions)} questions for directness and consistency...")

        new_questions = []
        for question in questions:
            if not question or question.example is None or question.example.strip() == "":
                continue
            else:
                # If question text is empty, skip filtering
                new_questions.append(question)
                continue

        questions = new_questions

        # Create filter prompts for all questions
        filter_prompts = []
        for question in questions:
            try:
                # Get examples from the same domain if available
                domain_examples = []

                # Create system prompt with loaded filter examples
                system_prompt = self.prompt_loader.get_template(
                    "refinement/filter/question_filter_system.j2",
                    attribute=self.attribute,
                    type_values=getattr(self.question_config, "type_values", None),
                    type_examples=getattr(self.question_config, "type_examples", None),
                    filter_examples=self.filter_samples,
                )

                # Create query prompt
                query = self.prompt_loader.get_template(
                    "refinement/filter/question_filter_query.j2",
                    question_text=question.example,
                    domain=question.domain,
                    topic=question.topic,
                    attribute=self.attribute,
                    additional_examples=domain_examples,
                )

                filter_prompts.append((system_prompt, query, question))

            except Exception as e:
                print(f"Error creating filter prompt for question '{question.example}': {e}")
                # If we can't create a filter prompt, keep the original question
                continue

        if not filter_prompts:
            print("No filter prompts created, returning original questions")
            return questions

        # Run filter prompts in parallel
        def func_filter_question(prompt_tuple):
            return self._run_question_filter_prompt(prompt_tuple)

        try:
            results = run_parallel(
                func_filter_question, filter_prompts, self.model.config.max_workers
            )

            reformatted_counter = 0
            # Process results
            filtered_questions = []
            for i, result in enumerate(results):
                original_question = questions[i] if i < len(questions) else None

                if result is not None and original_question is not None:
                    filtered_questions.append(result[1][1])
                    reformatted_counter += result[1][0]
                elif original_question is not None:
                    # If filtering failed, keep the original question
                    filtered_questions.append(original_question)
                    # print(
                    #     f"Filter failed for question, keeping original: '{original_question.example}'"
                    # )

            print(
                f"Successfully filtered {len(filtered_questions)} questions - Reformatted {reformatted_counter} questions"
            )

            for question in filtered_questions:
                text = question.example

                if text is None:
                    continue

                # Normalize question text
                text = re.sub(r"\{\{'\{\{", "{{", text)
                text = re.sub(r"\}\}'\}\}", "}}", text)

                if text != question.example:
                    question.example = text

            return filtered_questions

        except Exception as e:
            print(f"Error during parallel question filtering: {e}")
            return questions  # Return original questions if filtering fails

    def _run_question_filter_prompt(self, prompt_tuple) -> Optional[Question]:
        """
        Run question filter prompt and return filtered question.

        Args:
            prompt_tuple: Tuple containing (system_prompt, query, original_question)

        Returns:
            Filtered Question object or None if filtering failed
        """
        try:
            system_prompt, query, original_question = prompt_tuple

            response = self.filter_model.predict_string(
                query, system_prompt=system_prompt, response_format=self.question_filter_schema
            )
            # Parse structured response
            try:
                parsed = json.loads(response)
                filter_result = parsed.get("filter_result", {})

                needs_reformatting = filter_result.get("needs_reformatting", False)
                reasoning = filter_result.get("reasoning", "")

                if needs_reformatting:
                    reformatted_question = filter_result.get("reformatted_question", "")

                    if reformatted_question:
                        # print(f"Original: {original_question.example}")
                        # print(f"Reformatted: {reformatted_question}")

                        # Return new question with reformatted text
                        return (
                            1,
                            Question(
                                original_question.superdomain,
                                original_question.domain,
                                original_question.topic,
                                reformatted_question,
                            ),
                        )
                    else:
                        print(
                            "Filter indicated reformatting needed but no reformatted question provided"
                        )
                        return (
                            0,
                            original_question,  # Keep original if no reformatting provided
                        )
                else:
                    # print(f"Question passed filter: {reasoning}")
                    return (
                        0,
                        original_question,  # Keep original if no reformatting needed
                    )

            except json.JSONDecodeError as e:
                # print(f"Failed to parse filter response: {response}, error: {e}")
                return (
                    0,
                    original_question,  # Keep original if no reformatting needed
                )

        except Exception as e:
            # print(f"Error in _run_question_filter_prompt: {e}")
            return (
                0,
                original_question,  # Keep original if no reformatting needed
            )
