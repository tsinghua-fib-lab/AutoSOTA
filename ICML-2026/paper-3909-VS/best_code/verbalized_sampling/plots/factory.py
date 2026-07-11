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

from typing import List, Optional

from .evaluator_specific import (
    EvaluatorSpecificPlotter,
    FactualityPlotter,
    GenericPlotter,
    ResponseCountPlotter,
)
from .plotters import (
    BasePlotter,
    BoxPlotter,
    HistogramPlotter,
    KDEPlotter,
    SNSHistogramPlotter,
    ViolinPlotter,
)


class PlotterFactory:
    """Factory for creating different plot types."""

    PLOTTERS = {
        "histogram": HistogramPlotter,
        "sns_histogram": SNSHistogramPlotter,
        "violin": ViolinPlotter,
        "kde": KDEPlotter,
        "box": BoxPlotter,
    }

    @classmethod
    def create_plotter(cls, plot_type: str, colors: List[str]) -> BasePlotter:
        if plot_type not in cls.PLOTTERS:
            raise ValueError(
                f"Unknown plot_type: {plot_type}. Available: {list(cls.PLOTTERS.keys())}"
            )
        return cls.PLOTTERS[plot_type](colors)

    @classmethod
    def get_available_plot_types(cls) -> List[str]:
        return list(cls.PLOTTERS.keys())


class EvaluatorPlotterFactory:
    """Factory for creating evaluator-specific plotters."""

    PLOTTERS = {"response_count": ResponseCountPlotter, "factuality": FactualityPlotter}

    @classmethod
    def create_plotter(cls, evaluator_type: str, **kwargs) -> Optional[EvaluatorSpecificPlotter]:
        if evaluator_type in cls.PLOTTERS:
            return cls.PLOTTERS[evaluator_type](**kwargs)
        return None

    @classmethod
    def create_generic_plotter(cls, comparison_plotter) -> GenericPlotter:
        return GenericPlotter(comparison_plotter)

    @classmethod
    def get_available_evaluator_types(cls) -> List[str]:
        return list(cls.PLOTTERS.keys())
