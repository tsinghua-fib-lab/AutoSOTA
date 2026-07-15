## dashboard/

### Overview

This module focuses on analyzing SAE features using **Interactive Visualization** tools. It integrates Logit Lens analysis and Feature Activation heatmaps to help users intuitively understand feature semantics and triggering patterns.

### Analysis Tools

#### 1. Logit Lens Explorer

This is an interactive dashboard built with `ipywidgets` for real-time analysis of model internal states or specific SAE features at the token level.

* **Logit Projection**: We utilize the Logit Lens technique to project the SAE decoder direction (Feature Direction) $\mathbf d_i \in \mathbb R^{d_{model}}$ into the vocabulary space:
    $$\text{logits} = W_U \cdot \text{LayerNorm}(\mathbf d_i)$$
    By observing the tokens with the highest probability (Top K) and lowest probability (Bottom K), we can directly infer the semantic inclination of the feature.
* **Sequence Heatmap**: The interface synchronously displays the activation intensity distribution of the selected feature across the entire text sequence. This helps in observing whether a feature is **locally triggered** (e.g., by a specific word) or **globally triggered** (e.g., by a specific context).
* **Interactivity**: Supports clicking on tokens to switch positions, manually specifying SAE feature indices, and dynamically toggling the application of the Final Block (`blocks[-1]`).

#### 2. Feature Activation Viewer

This component focuses on **dataset-level** feature analysis, aiming to answer "Under what circumstances is this feature activated?".

* **Global Search**: Retrieves all samples from the sparse activation data where the activation value of a specific feature $j$ exceeds a threshold $\theta$:
    $$S_j = \{ (x_i, t) \mid z_{i,t}^{(j)} > \theta \}$$
* **Contextual Visualization**: Renders high-activation segments as HTML heatmaps, highlighting the triggering tokens and their context windows. It supports automatic aggregation of consecutive activated tokens.
* **Interpretation Generation**: Provides the `get_markdown_text_with_activations` interface to export activated segments as formatted text with `<active>` tags, facilitating automatic feature interpretation via LLMs (Auto-Interpretability).

### Implementation

The core functionality is implemented by the following Python classes:

* **`LogitLensAnalyzer`**: The backend calculation engine. Responsible for handling Logit projections, Top-K statistics, and context data extraction.
* **`LogitLensGUI`**: The frontend interactive interface. Built on `ipywidgets`, it contains the token selector, heatmap rows, and multi-column data display areas.
* **`FeatureActivationViewer`**: The batch analysis tool. Responsible for cross-sample feature retrieval, statistical calculation (max, mean), and multi-sample visualization rendering.