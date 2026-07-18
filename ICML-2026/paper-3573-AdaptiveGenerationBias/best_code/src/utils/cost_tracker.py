import csv
import threading
from collections import defaultdict
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer
import tiktoken


class CostTracker:
    """Thread-safe cost tracking utility for API models."""

    def __init__(self, costs_csv_path: str = "llm_costs.csv"):
        self._lock = threading.Lock()
        self._costs_per_model: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._model_costs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Load cost data from CSV
        self._load_costs(costs_csv_path)

    def _load_costs(self, csv_path: str) -> None:
        """Load model costs from CSV file."""
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    provider = row["provider"].upper()
                    model = row["model"]
                    key = (provider, model)

                    self._model_costs[key] = {
                        "input_cost_per_1m": float(row["input_cost_per_1m"])
                        if row["input_cost_per_1m"]
                        else 0.0,
                        "output_cost_per_1m": float(row["output_cost_per_1m"])
                        if row["output_cost_per_1m"]
                        else 0.0,
                        "per_image": float(row["per_image"]) if row["per_image"] else 0.0,
                        "per_call": float(row["per_call"]) if row["per_call"] else 0.0,
                    }
        except FileNotFoundError:
            print(f"Warning: Cost file {csv_path} not found. Cost tracking will not work properly.")
        except Exception as e:
            print(f"Error loading costs: {e}")

    def _get_tokenizer(self, model_name: str, provider: str) -> Optional[AutoTokenizer]:
        """Get or create tokenizer for a model."""
        key = f"{provider}_{model_name}"

        if key not in self._tokenizers:
            encoding = tiktoken.encoding_for_model("gpt-4")

            # Create a wrapper that mimics AutoTokenizer interface
            class TiktokenWrapper:
                def __init__(self, encoding):
                    self.encoding = encoding

                def encode(self, text: str) -> list:
                    return self.encoding.encode(text)

                def __call__(self, text: str) -> dict:
                    tokens = self.encoding.encode(text)
                    return {"input_ids": tokens}

            self._tokenizers[key] = TiktokenWrapper(encoding)

        return self._tokenizers[key]

    def _count_tokens(self, text: str, model_name: str, provider: str) -> int:
        """Count tokens in text using appropriate tokenizer."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4

        try:
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
            else:
                tokens = tokenizer(text)
                return len(tokens["input_ids"])
        except Exception as e:
            print(f"Warning: Error counting tokens: {e}")
            # Fallback estimation
            return len(text) // 4

    def _get_model_costs(self, provider: str, model_name: str) -> Dict[str, float]:
        """Get cost information for a specific model."""
        key = (provider.upper(), model_name)

        # Try exact match first
        if key in self._model_costs:
            return self._model_costs[key]

        # Try partial matches for model names with prefixes
        for (p, m), costs in self._model_costs.items():
            if p == provider.upper() and (model_name.startswith(m) or m in model_name):
                return costs

        # Return default costs if not found
        # print(f"Warning: No cost data found for {provider}/{model_name}, using default costs")
        return {
            "input_cost_per_1m": 1.0,  # Default $1 per 1M input tokens
            "output_cost_per_1m": 2.0,  # Default $2 per 1M output tokens
            "per_image": 0.0,
            "per_call": 0.0,
        }

    def calculate_cost(
        self, input_text: str, output_text: str, model_name: str, provider: str
    ) -> float:
        """Calculate the cost for a single API call."""
        costs = self._get_model_costs(provider, model_name)

        # Count tokens
        input_tokens = self._count_tokens(input_text, model_name, provider)
        output_tokens = self._count_tokens(output_text, model_name, provider)

        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * costs["input_cost_per_1m"]
        output_cost = (output_tokens / 1_000_000) * costs["output_cost_per_1m"]
        per_call_cost = costs["per_call"]

        total_cost = input_cost + output_cost + per_call_cost

        return total_cost

    def add_cost(self, input_text: str, output_text: str, model_name: str, provider: str) -> float:
        """Add cost for a single API call and return the cost."""
        cost = self.calculate_cost(input_text, output_text, model_name, provider)

        with self._lock:
            model_key = f"{provider}_{model_name}"
            self._costs_per_model[model_key]["total_cost"] += cost
            self._costs_per_model[model_key]["call_count"] += 1

            # Also track input/output tokens for statistics
            input_tokens = self._count_tokens(input_text, model_name, provider)
            output_tokens = self._count_tokens(output_text, model_name, provider)
            self._costs_per_model[model_key]["input_tokens"] += input_tokens
            self._costs_per_model[model_key]["output_tokens"] += output_tokens

        return cost

    def get_costs(self) -> Dict[str, Dict[str, float]]:
        """Get current cost statistics for all models."""
        with self._lock:
            return dict(self._costs_per_model)

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        with self._lock:
            return sum(model_costs["total_cost"] for model_costs in self._costs_per_model.values())

    def reset_costs(self) -> None:
        """Reset all cost counters."""
        with self._lock:
            self._costs_per_model.clear()

    def print_cost_summary(self) -> None:
        """Print a summary of costs per model."""
        costs = self.get_costs()
        total_cost = self.get_total_cost()

        print("\n" + "=" * 60)
        print("API COST SUMMARY")
        print("=" * 60)

        if not costs:
            print("No API calls tracked yet.")
            return

        for model_key, stats in costs.items():
            print(f"\nModel: {model_key}")
            print(f"  Total Cost: ${stats['total_cost']:.6f}")
            print(f"  API Calls: {stats['call_count']}")
            print(f"  Input Tokens: {stats['input_tokens']:,}")
            print(f"  Output Tokens: {stats['output_tokens']:,}")
            if stats["call_count"] > 0:
                print(f"  Avg Cost/Call: ${stats['total_cost'] / stats['call_count']:.6f}")

        print(f"\nTOTAL COST: ${total_cost:.6f}")
        print("=" * 60)


# Global cost tracker instance
_global_cost_tracker = None


def get_global_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker
