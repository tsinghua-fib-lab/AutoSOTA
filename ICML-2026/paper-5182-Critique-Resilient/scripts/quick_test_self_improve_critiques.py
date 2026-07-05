import argparse
import json
import sys
from typing import Optional, Tuple

sys.path.append(".")  # noqa: E402

import dotenv
dotenv.load_dotenv()

from model_config import load_registry
from prompt_library import (
    build_critique_refine,
    build_critique_self_check,
    load_critique_guidance,
)
from self_improvement import self_improve_critiques


def resolve_model(
    config_path: str,
    model_name: Optional[str],
    temperature: Optional[float],
    reasoning: Optional[str],
) -> Tuple[str, Optional[float], Optional[str]]:
    registry = load_registry(config_path)
    spec = None

    if model_name:
        resolved = registry.resolve_model_name(model_name)
        spec = registry.models.get(resolved)
        model_name = resolved
    else:
        critique_models = registry.by_role("critique")
        if not critique_models:
            raise ValueError("No critique models found; pass --model explicitly.")
        spec = critique_models[0]
        model_name = spec.name

    if spec:
        temperature = temperature if temperature is not None else spec.temperature
        reasoning = reasoning if reasoning is not None else spec.reasoning

    if model_name.startswith("anthropic/") and temperature is not None and reasoning is not None:
        raise ValueError("Anthropic models cannot set both temperature and reasoning.")

    return model_name, temperature, reasoning


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for self_improve_critiques using a live model.")
    parser.add_argument("--config", default="configs/models.json")
    parser.add_argument("--model", help="Model name or slug (default: first critique model).")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning", type=str, default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    model, temperature, reasoning = resolve_model(
        args.config,
        args.model,
        args.temperature,
        args.reasoning,
    )

    question = "What is 2 + 2?"
    answer = "The answer is 5."
    initial_critique = "Looks fine to me."

    questions = [question]
    answer_texts = [answer]
    answer_lookup = {q: ans for q, ans in zip(questions, answer_texts)}
    guidance = load_critique_guidance()

    def eval_prompt(q: str, crit: str, idx: int) -> str:
        return build_critique_self_check(q, answer_texts[idx], crit, guidance)

    def refine_prompt(q: str, crit: str, fb: str) -> str:
        base_answer = answer_lookup.get(q, "")
        return build_critique_refine(q, base_answer, crit, fb)

    results = self_improve_critiques(
        model=model,
        questions=questions,
        initial_critiques=[initial_critique],
        build_eval_prompt=eval_prompt,
        build_refine_prompt=refine_prompt,
        temperature=temperature,
        reasoning=reasoning,
        max_rounds=args.max_rounds,
        disable_batch=True,
        raw_initial_critiques=[initial_critique],
    )

    result = results[0]
    summary = {
        "model": model,
        "status": result.status,
        "attempts": len(result.attempts),
        "final_critique": result.final_answer,
        "last_feedback": result.last_feedback,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
