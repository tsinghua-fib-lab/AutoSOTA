# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Dict, Optional, Union

from ..analysis.evals.base import EvalResult
from .base import ComparisonData
from .comparison import ComparisonPlotter


def plot_evaluation_comparison(
    results: Dict[str, Union[EvalResult, str, Path]],
    output_dir: Union[str, Path],
    evaluator_type: str = "auto",
    **kwargs,
) -> None:
    """
    Convenience function to plot evaluation comparisons.

    Args:
        results: Dict mapping format names to EvalResult objects or file paths
        output_dir: Directory to save plots
        evaluator_type: Type of evaluator ("diversity", "ttct", "creativity_index", "length", "auto")
        **kwargs: Additional arguments passed to ComparisonPlotter
    """

    plotter = ComparisonPlotter(**kwargs)
    comparison_data = []

    for name, result in results.items():
        if isinstance(result, (str, Path)):
            # Load from file
            with open(result, "r") as f:
                result_dict = json.load(f)
                eval_result = EvalResult.from_dict(result_dict)
        else:
            eval_result = result

        comparison_data.append(ComparisonData(name=name, result=eval_result))

    plotter.create_comprehensive_comparison(comparison_data, output_dir, evaluator_type)


def plot_comparison_chart(
    results: Dict[str, Union[EvalResult, str, Path]],
    output_path: Union[str, Path],
    title: Optional[str] = None,
    show_error_bars: bool = True,
    error_bar_type: str = "ci",
    **kwargs,
) -> None:
    """Create a performance comparison chart with error bars."""
    plotter = ComparisonPlotter(**kwargs)
    print("Starting to plot comparison chart...")
    comparison_data = {}  # {exp_name: EvalResult}
    key_metric_names = []

    for metric, results in results.items():
        from ..analysis.evals import get_evaluator

        evaluator_class = get_evaluator(metric)
        if evaluator_class.key_plot_metrics:
            key_metric_names.extend(evaluator_class.key_plot_metrics)

        for exp_name, result in results.items():
            if isinstance(result, (str, Path)):
                # Load from file
                with open(result, "r") as f:
                    result_dict = json.load(f)
                    eval_result = EvalResult.from_dict(result_dict)
            else:
                eval_result = result
            if exp_name not in comparison_data:
                comparison_data[exp_name] = eval_result
            else:
                comparison_data[exp_name] = comparison_data[exp_name] + eval_result

    print(f"Number of experiments: {len(comparison_data)}")
    print("Plotting comparison chart...")
    plotter.create_performance_comparison_chart(
        comparison_data,
        key_metric_names,
        output_path,
        title=title,
        show_error_bars=show_error_bars,
        error_bar_type=error_bar_type,
    )
