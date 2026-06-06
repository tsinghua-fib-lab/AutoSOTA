"""
GSM8K evaluation using Qwen2.5-Math-1.5B-Instruct with vLLM.
Uses standard zero-shot chain-of-thought evaluation.
"""
import json
import re
import os
import sys

# Add /pkgs to Python path
sys.path.insert(0, '/pkgs')

from vllm import LLM, SamplingParams
from transformers import set_seed

set_seed(42)

# Configuration
MODEL_PATH = "/data/Qwen2.5-Math-1.5B-Instruct"
DATA_PATH = "/repo/datasets/gsm8k/test.jsonl"
OUTPUT_DIR = "/repo/output/gsm8k"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eval_results.json")

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def extract_answer(text):
    """Extract the final answer from model output."""
    # Try to find \boxed{answer} — primary signal for Qwen math models
    boxed_matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Try #### prefix (GSM8K format)
    hash_match = re.search(r'####\s*([\-]?[\d,\.\s/]+)', text)
    if hash_match:
        return hash_match.group(1).strip().replace(',', '').replace(' ', '')

    # Try "answer is X" pattern (extended to handle more formats)
    answer_match = re.search(
        r'(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\-]?[\d,\.\s/]+)',
        text
    )
    if answer_match:
        return answer_match.group(1).strip().replace(',', '')

    # Try "= X" at end of sentence
    eq_match = re.search(r'=\s*([\-]?[\d,\.]+)\s*(?:\.|$|\n)', text)
    if eq_match:
        return eq_match.group(1).strip().replace(',', '')

    # Try "total/result is X" patterns
    total_match = re.search(
        r'(?i)(?:total|result|sum|cost|price|amount|number|value)\s+(?:is|are|=|:)\s*([\-]?[\d,\.]+)',
        text
    )
    if total_match:
        return total_match.group(1).strip().replace(',', '')

    # Last resort: find the last standalone number in the text
    numbers = re.findall(r'(?<!\w)(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+)(?!\w)', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def normalize(value):
    """Normalize a numeric string for comparison."""
    if value is None:
        return None
    # Remove commas, whitespace
    value = str(value).strip().replace(',', '').replace('$', '').replace('%', '')
    # Remove trailing .0
    try:
        f = float(value)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return value.lower()


def check_correctness(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize(predicted)
    gt_norm = normalize(ground_truth)

    if pred_norm is None or gt_norm is None:
        return False

    # Exact string match after normalization
    if pred_norm == gt_norm:
        return True

    # Numeric comparison with tolerance
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        if abs(pred_float - gt_float) < 1e-3:
            return True
    except (ValueError, TypeError):
        pass

    return False


def main():
    # Load test data
    print(f"Loading data from {DATA_PATH}...")
    test_data = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            test_data.append(item)

    # The eval script expects 1319 examples
    assert len(test_data) == 1319, f"Expected 1319 test examples, got {len(test_data)}"
    print(f"Loaded {len(test_data)} test examples")

    # Initialize vLLM
    print(f"Loading model from {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=False,
    )

    # Self-consistency: N samples with temperature, then majority vote
    N_SAMPLES = 5
    TEMPERATURE = 0.6

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=0.95,
        max_tokens=2048,
        n=N_SAMPLES,  # Generate N samples per prompt
    )

    # Build prompts
    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_data:
        question = item.get('question', item.get('problem', ''))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Run inference
    print(f"Running self-consistency inference ({N_SAMPLES} samples/question, T={TEMPERATURE}) on {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate with majority voting
    from collections import Counter
    correct = 0
    results = []
    for i, (item, output) in enumerate(zip(test_data, outputs)):
        # Get ground truth
        gt_answer = item.get('answer', '')
        gt_match = re.search(r'####\s*([\d,\.\-]+)', gt_answer)
        if gt_match:
            gt_final = gt_match.group(1).strip().replace(',', '')
        else:
            gt_final = gt_answer.strip()

        # Extract answers from all N samples
        candidate_answers = []
        for sample_output in output.outputs:
            pred = extract_answer(sample_output.text)
            norm_pred = normalize(pred) if pred else None
            candidate_answers.append(norm_pred)

        # Majority vote: pick most common non-None answer
        valid_answers = [a for a in candidate_answers if a is not None]
        if valid_answers:
            majority_answer = Counter(valid_answers).most_common(1)[0][0]
        else:
            majority_answer = None

        is_correct = check_correctness(majority_answer, gt_final)
        if is_correct:
            correct += 1

        results.append({
            'question': item.get('question', item.get('problem', '')),
            'ground_truth': gt_final,
            'predicted': majority_answer,
            'all_candidates': candidate_answers,
            'correct': is_correct,
        })

    accuracy = round(correct / len(test_data) * 100, 2)

    print(f"\n=== GSM8K Evaluation Results ===")
    print(f"Correct: {correct}/{len(test_data)}")
    print(f"Accuracy: {accuracy}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eval_results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_data),
        'results': results,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
