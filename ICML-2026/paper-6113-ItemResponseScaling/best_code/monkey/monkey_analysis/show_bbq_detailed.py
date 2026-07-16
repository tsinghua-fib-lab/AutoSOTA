"""
Show detailed BBQ questions with answer choices and model performance.
"""

import torch
import numpy as np
from huggingface_hub import snapshot_download
import pickle
import pandas as pd
from scipy.stats import spearmanr

# Load test-time response data
print("Loading test-time response data...")
FILE_NAME = "irsl_testtime_resmat1"
cache_dir_resmat = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")
testtime_resmat = torch.load(f"{cache_dir_resmat}/resmat.pt", map_location="cpu", weights_only=False)
data_tensor = testtime_resmat["data_tensor"].numpy()
models = testtime_resmat["models"]
questions = testtime_resmat["questions"]
scenarios = testtime_resmat["scenarios"]
helm_zs = np.array(testtime_resmat["zs"])

# Load calibrated z-scores
import os
calibrated_path = os.path.join(os.path.dirname(__file__), f"{FILE_NAME}_withz.pt")
testtime_calibrated = torch.load(calibrated_path, weights_only=False)
testtime_zs = testtime_calibrated["zs"]
if hasattr(testtime_zs, 'numpy'):
    testtime_zs = testtime_zs.numpy()

# Load BBQ question details from monkey_query_pre
print("Loading BBQ question details...")
cache_dir_bbq = snapshot_download(repo_id="stair-lab/monkey_query_pre", repo_type="dataset")
bbq_file = f"{cache_dir_bbq}/meta_llama-3-8b_bbq_pre_query.pkl"

with open(bbq_file, 'rb') as f:
    bbq_df = pickle.load(f)

# Create mapping from question text to details
bbq_details = {}
for _, row in bbq_df.iterrows():
    question_text = row['instance.input.text']
    bbq_details[question_text] = {
        'prompt': row['request.prompt'],
        'solution': row['solution'],
        'prompt_index': row['prompt_index']
    }

# Find BBQ questions with highest z-score discrepancy
bbq_mask = np.array([s == "bbq" for s in scenarios])
bbq_indices = np.where(bbq_mask)[0]
bbq_z_diffs = np.abs(testtime_zs[bbq_mask] - helm_zs[bbq_mask])

# Get top 5 discrepancy questions
top_bbq_idxs = bbq_indices[np.argsort(bbq_z_diffs)[-5:]][::-1]

print("\n" + "="*80)
print("BBQ QUESTIONS WITH HIGHEST Z-SCORE DISCREPANCIES")
print("="*80)

for rank, q_idx in enumerate(top_bbq_idxs, 1):
    print(f"\n{'='*80}")
    print(f"QUESTION #{rank}")
    print(f"{'='*80}")

    question_text = questions[q_idx]
    helm_z = helm_zs[q_idx]
    testtime_z = testtime_zs[q_idx]
    z_diff = abs(helm_z - testtime_z)

    # Get detailed question info
    if question_text in bbq_details:
        details = bbq_details[question_text]

        # Parse the prompt to extract the actual question and choices
        prompt = details['prompt']
        correct_answer = details['solution']

        # Find the last question in the prompt (the one being asked)
        lines = prompt.split('\n')

        # Find passage and question
        passage_start = None
        question_start = None
        answer_choices_start = None

        for i, line in enumerate(lines):
            if line.startswith('Passage:'):
                passage_start = i
            elif line.startswith('Question:'):
                question_start = i
            elif line.startswith('A.') or line.startswith('A)'):
                answer_choices_start = i
                break

        # Extract the last passage and question
        if passage_start is not None:
            passage_lines = []
            for i in range(passage_start, len(lines)):
                if lines[i].startswith('Question:'):
                    break
                if lines[i].startswith('Passage:'):
                    passage_lines.append(lines[i][9:])  # Remove 'Passage: '
                elif passage_lines:  # continuation of passage
                    passage_lines.append(lines[i])

            passage = '\n'.join(passage_lines).strip()

        if question_start is not None:
            question_lines = []
            for i in range(question_start, len(lines)):
                if lines[i].startswith('A.') or lines[i].startswith('A)'):
                    break
                if lines[i].startswith('Question:'):
                    question_lines.append(lines[i][10:])  # Remove 'Question: '
                elif question_lines:
                    question_lines.append(lines[i])

            question = '\n'.join(question_lines).strip()

        # Extract answer choices
        choices = []
        if answer_choices_start is not None:
            for i in range(answer_choices_start, len(lines)):
                line = lines[i].strip()
                if line and (line[0] in ['A', 'B', 'C', 'D', 'E'] and (len(line) > 1 and line[1] in ['.', ')'])):
                    choices.append(line)
                elif line.startswith('Answer:'):
                    break

        print(f"\n{'-'*80}")
        print("PASSAGE:")
        print(f"{'-'*80}")
        print(passage if passage_start is not None else question_text)

        print(f"\n{'-'*80}")
        print("QUESTION:")
        print(f"{'-'*80}")
        print(question if question_start is not None else question_text)

        if choices:
            print(f"\n{'-'*80}")
            print("ANSWER CHOICES:")
            print(f"{'-'*80}")
            for choice in choices:
                marker = " ← CORRECT ANSWER" if choice[0] == correct_answer else ""
                print(f"  {choice}{marker}")

        print(f"\n{'-'*80}")
        print("DIFFICULTY ASSESSMENT:")
        print(f"{'-'*80}")
        print(f"HELM z-score:      {helm_z:7.3f} (higher z = easier)")
        print(f"Test-time z-score: {testtime_z:7.3f} (higher z = easier)")
        print(f"Δz:                {z_diff:7.3f}")

        if helm_z < 0 and testtime_z > 0:
            print("⚠️  HELM: Hard (z < 0) → Test-time: Easy (z > 0) - MASSIVE SHIFT!")
        elif helm_z > 0 and testtime_z < 0:
            print("⚠️  HELM: Easy (z > 0) → Test-time: Hard (z < 0) - MASSIVE SHIFT!")
        else:
            direction = "both easy" if helm_z > 0 else "both hard"
            print(f"→  Both evaluations agree on direction ({direction}), but magnitudes differ")

    else:
        print(f"\nQuestion text: {question_text}")
        print("(Detailed question info not found in BBQ dataset)")

    # Get model performance
    responses = data_tensor[:, q_idx, :]  # (8 models, 10000 samples)

    print(f"\n{'-'*80}")
    print("MODEL PERFORMANCE:")
    print(f"{'-'*80}")

    for model_idx, model_name in enumerate(models):
        pass_at_1 = np.mean(responses[model_idx, :])
        std = np.std(responses[model_idx, :])

        # Create visual bar
        bar_length = int(pass_at_1 * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)

        print(f"\n{model_name}:")
        print(f"  Accuracy: {pass_at_1:6.1%}  {bar}")
        print(f"  Std dev:  {std:.4f}")

    # Aggregate stats
    model_means = np.mean(responses, axis=1)
    model_stds = np.std(responses, axis=1)

    print(f"\n{'-'*80}")
    print("AGGREGATE STATISTICS:")
    print(f"{'-'*80}")
    print(f"Mean accuracy across models: {np.mean(model_means):.1%}")
    print(f"Std dev across models:       {np.std(model_means):.4f}")
    print(f"Avg intra-model variance:    {np.mean(model_stds**2):.4f}")
    print(f"Inter-model variance:        {np.var(model_means):.4f}")

    avg_std = np.mean(model_stds)
    if avg_std < 0.1:
        print(f"\n✓ Very consistent responses (std={avg_std:.3f})")
    elif avg_std < 0.3:
        print(f"\n→ Moderate consistency (std={avg_std:.3f})")
    else:
        print(f"\n✗ High variability in responses (std={avg_std:.3f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
