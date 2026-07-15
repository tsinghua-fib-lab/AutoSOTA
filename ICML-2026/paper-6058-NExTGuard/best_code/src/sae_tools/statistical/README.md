## statistic/

### Overview

This module focuses on analyzing SAE features using **Statistical Metrics**. The core goal is to quantify the alignment between a specific SAE feature and a human-interpretable concept defined by a labeled dataset.

### Analysis Pipeline

#### 1. Data Definitions

We consider the following inputs:

* **SAE Activations**: A sparse vector $\mathbf z_{i,t}\in\mathbb R^{d_{sae}}$ for the $t$-th token of the $i$-th text entry.
* **Dataset Labels**: A label $y_i$ associated with the $i$-th text entry (e.g., a sentiment score or a category).

> **Note on Dimensions:**
> Most open-source datasets provide **entry-level** labels (one label per sentence/document), whereas SAE activations are **token-level**. Therefore, for a feature $j$, we have a sequence of activations $\{\mathbf z_{i,t}^{(j)}:t=1,...,T_i\}$ but only a single label $y_i$.

#### 2. Aggregation

To align the dimensions of the activations and the labels, we apply an **aggregation function** to the token sequence. The most common method is **Max Pooling**: $$\hat y_i^{(j)}=\max_{t=1}^{T_i}\{\mathbf z_{i,t}^{(j)}\}$$

This produces a single scalar score $\hat y_i^{(j)}$ representing the prediction made by feature $j$ in text entry .

* **Aggregation Functions**: `max`, `mean`, etc.

#### 3. Metric Calculation

We evaluate the feature by calculating statistical metrics between the aggregated feature scores  and the ground truth labels . High correlation (or F1 score) suggests the feature encodes the concept defined by the dataset.

* **Metrics**: Pearson Correlation, F1 Score, AUROC.
* **Visualization**: Histograms, scatter plots.