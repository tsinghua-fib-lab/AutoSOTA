"""
Utility functions for cost tracking and reporting.
"""

from src.utils.cost_tracker import get_global_cost_tracker


def print_cost_summary():
    """Print a summary of API costs."""
    cost_tracker = get_global_cost_tracker()
    cost_tracker.print_cost_summary()


def get_total_cost() -> float:
    """Get the total cost across all models."""
    cost_tracker = get_global_cost_tracker()
    return cost_tracker.get_total_cost()


def get_costs_by_model() -> dict:
    """Get cost breakdown by model."""
    cost_tracker = get_global_cost_tracker()
    return cost_tracker.get_costs()


def reset_cost_tracking():
    """Reset all cost counters."""
    cost_tracker = get_global_cost_tracker()
    cost_tracker.reset_costs()
    print("Cost tracking has been reset.")


def save_cost_report(filepath: str):
    """Save cost report to a file."""
    import json
    from datetime import datetime

    cost_tracker = get_global_cost_tracker()
    costs = cost_tracker.get_costs()
    total_cost = cost_tracker.get_total_cost()

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_cost": total_cost,
        "costs_by_model": costs,
        "summary": {
            "total_api_calls": sum(model_costs["call_count"] for model_costs in costs.values()),
            "total_input_tokens": sum(
                model_costs["input_tokens"] for model_costs in costs.values()
            ),
            "total_output_tokens": sum(
                model_costs["output_tokens"] for model_costs in costs.values()
            ),
        },
    }

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Cost report saved to {filepath}")
