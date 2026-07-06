import argparse
import os
import sys

from dotenv import load_dotenv

from model_api import query_llm_single


def _required_api_key(model: str) -> str:
    if model.startswith("openai/"):
        return "OPENAI_API_KEY"
    if "gemini" in model:
        return "GEMINI_API_KEY"
    if "anthropic" in model:
        return "ANTHROPIC_API_KEY"
    return "OPENROUTER_API_KEY"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a single query to any supported model via model_api."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. deepseek/deepseek-v3.2-speciale).",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="User message to send (defaults to stdin if omitted).",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt to send with the message.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["high", "medium", "none"],
        default="none",
        help="Reasoning effort level for OpenRouter or supported providers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omit to use model default).",
    )
    args = parser.parse_args()

    load_dotenv()
    required_key = _required_api_key(args.model)
    if not os.getenv(required_key):
        print(f"{required_key} not set in environment.", file=sys.stderr)
        return 1

    message = args.message or sys.stdin.read().strip()
    if not message:
        print("No message provided (arg or stdin).", file=sys.stderr)
        return 1

    reasoning = None if args.reasoning == "none" else args.reasoning
    response = query_llm_single(
        model=args.model,
        message=message,
        prompt=args.system,
        temperature=args.temperature,
        reasoning=reasoning,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
