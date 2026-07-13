# Robustness Evaluation

This repository contains the code for generating perturbation datasets and running the robustness ("Noise Perturbations") evaluation experiments.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd prototype_attention

python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Generation

To generate the perturbation dataset, open and run the notebook inside:

```text
robustness/perturbation_dataset/
```

Run all cells in the dataset generation notebook.

This will create the perturbation dataset used by the robustness evaluation.

---

## Running the Experiment

After setup and dataset generation, open:

```text
robustness/blackbox_evaluation.ipynb
```

Run all cells in the notebook.

The notebook:
- loads the model checkpoints,
- runs the robustness benchmark,
- computes the evaluation metrics,
- and writes the final results.

---

## Output

The generated evaluation results are written to:

```text
robustness/results_full/
```