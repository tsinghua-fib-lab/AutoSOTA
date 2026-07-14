# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Utils package for ProEval
from .plotting import (
    WIDTH,
    COLOR_LIST,
    METHOD_COLORS,
    MARKER_LIST,
    HATCH_LIST,
    PREFERRED_METHOD_ORDER,
    set_style,
    set_size,
    save_fig,
    get_timestamp,
    get_method_color,
    sort_methods_by_preference,
)

# Re-export from experiment_plotting for backwards compatibility
from .experiment_plotting import (
    STRATEGY_NAMES,
    get_embedding,
    load_csv_data,
    load_v2_npz_data,
    load_v2_csv_data,
    plot_v2_from_csv,
    plot_v2_comparison,
    compute_failure_rates,
    plot_failure_rate,
    plot_embedding_scatter,
    plot_v2_embedding_scatter,
    plot_v2_topic_distribution,
)

__all__ = [
    # Core plotting utilities
    'WIDTH',
    'COLOR_LIST',
    'METHOD_COLORS',
    'MARKER_LIST',
    'HATCH_LIST',
    'PREFERRED_METHOD_ORDER',
    'set_style',
    'set_size',
    'save_fig',
    'get_timestamp',
    'get_method_color',
    'sort_methods_by_preference',
    # Experiment plotting utilities
    'STRATEGY_NAMES',
    'get_embedding',
    'load_csv_data',
    'load_v2_npz_data',
    'load_v2_csv_data',
    'plot_v2_from_csv',
    'plot_v2_comparison',
    'compute_failure_rates',
    'plot_failure_rate',
    'plot_embedding_scatter',
    'plot_v2_embedding_scatter',
    'plot_v2_topic_distribution',
]
