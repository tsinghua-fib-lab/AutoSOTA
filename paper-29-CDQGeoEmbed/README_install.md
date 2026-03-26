# Opposite-Score
Efficient embeddings and scoring mechanism for detecting contrasting or opposite relationships in text, based on a given context.

[opposite-score](https://github.com/cuishaobo/conditional-dichotomy-quantification) is designed to generate embeddings and compute the **opposite-score**, which quantifies the degree of contrast or opposition between two textual outputs within the same context. This package is particularly useful in scenarios like debates, legal reasoning, and causal analysis where contrasting perspectives need to be evaluated based on shared input.

## Features
- **Opposite-Score Calculation**: Computes a numerical score representing how opposite two texts are, conditioned on a shared context.
- **Opposite-Aware Embeddings**: Generates embeddings optimized for contrasting textual relationships.
- **Easy to Use**: Only a few lines of code to get sentence/token-level embeddings and calculate opposite scores.
- **Automatic Model Download**: The first initialization automatically downloads and installs the pre-trained Opposite-Score model.

## Installation
Install Opposite-Score via pip:
```bash
pip install opposite-score==0.1.0
```


# Usage Examples

## For Opposite Scores of two sentence conditioned on the context.

```python

from oppositescore.model.dichotomye import DichotomyE

# Example inputs
context = ["A company launches a revolutionary product."]
sentence1 = ["Competitors quickly release similar products, reducing the company's advantage."]
sentence2 = ["The company gains a significant advantage due to its unique product."]

# Initialize the model
opposite_scorer = DichotomyE.from_pretrained('google-bert/bert-base-uncased', pooling_strategy='cls').cuda()

# Calculate opposite-score (using cosine similarity as an example)
opposite_score = opposite_scorer.calculate_opposite_score(ctx=context, sent1=sentence1, sent2=sentence2)
print('Opposite Score:', opposite_score)
# Output: Opposite Score: 1.5123178
```

# 📧 Contact

Please feel free to contact us via email: firstname.lastname@epfl.ch

# Reference Papers


# &copy; License
This project is licensed under the MIT License. :blush: