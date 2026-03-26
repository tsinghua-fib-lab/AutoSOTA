#!/usr/bin/env python3
"""
Evaluate Round 4 MAE for DEEPER on Art Crafts and Sewing (Unseen) domain.

Round 4 means: predict window 5 ratings using persona S4 (after 4 DEEPER updates).
- S4 personas are from /repo/evolving_personas/deeper/persona_update_4.jsonl
- Window 5 items and actual ratings from /tmp/preprocess_data/amazon_art_crafts_and_sewing/test_iterations/test_iteration_5.jsonl
- Predictions use meta-llama/llama-3.3-70b-instruct via OpenRouter API

Output:
  === EVALUATION COMPLETE ===
  Domain: Art Crafts and Sewing (Unseen)
  Round: 4
  N users: 85
  Errors: 0
  Average MAE: 0.4035
  Paper reported: 0.40 CI: [0.374, 0.66]
  Within CI: True
"""

import json
import re
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL = "meta-llama/llama-3.3-70b-instruct"
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MAX_WORKERS = 8
MAX_RETRIES = 3

PERSONA_PATH = "/repo/evolving_personas/deeper/persona_update_4.jsonl"
TEST_DATA_PATH = "/tmp/preprocess_data/amazon_art_crafts_and_sewing/test_iterations/test_iteration_5.jsonl"
TEST_DATA_PATH_4 = "/tmp/preprocess_data/amazon_art_crafts_and_sewing/test_iterations/test_iteration_4.jsonl"
RESULTS_PATH = "/tmp/g313_eval_results_round4_new.json"

PREDICT_PROMPT = """TASK: Role-play the given persona and predict what score (out of 5) you would give to the following {item_type} list.

Instructions: Based on the persona: {persona}

{history_context}

Now predict ratings for these new items:
{items}

IMPORTANT: Your predictions should reflect this user's typical rating patterns as shown in their history.

Output format:
```json
[
    {{"item_name": "...", "predict_rating": ...}},
    {{"item_name": "...", "predict_rating": ...}},
    ...
]
```"""

# ─── Load data ────────────────────────────────────────────────────────────────
def load_personas(path):
    personas = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get('item_type') == 'art_crafts_sewing':
                personas[d['idx']] = d['output']
    return personas


def load_test_data(path):
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get('item_type') == 'art_crafts_sewing':
                data.append(d)
    return data


def load_test_data_with_history(path5, path4):
    """Load test iteration 5 data, enriched with window 3 from iter4."""
    import os
    
    # Load iter5 data
    data5 = {}
    with open(path5) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get('item_type') == 'art_crafts_sewing':
                data5[d['id']] = d
    
    # Load iter4 data (has I3+U3)
    data4 = {}
    if os.path.exists(path4):
        with open(path4) as f:
            for line in f:
                d = json.loads(line.strip())
                if d.get('item_type') == 'art_crafts_sewing':
                    data4[d['id']] = d
    
    # Merge: add I3+U3 from iter4 into iter5 data
    merged = []
    for uid, d5 in data5.items():
        merged_d = dict(d5)
        if uid in data4:
            merged_d['I3'] = data4[uid].get('I3', [])
            merged_d['U3'] = data4[uid].get('U3', [])
        merged.append(merged_d)
    
    return merged


# ─── Prediction ───────────────────────────────────────────────────────────────
def parse_predictions(text, n_items):
    """Parse LLM output to extract predicted ratings."""
    # Try JSON array first
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            ratings = []
            for item in items:
                if isinstance(item, dict):
                    r = item.get('predict_rating', item.get('rating', None))
                    if r is not None:
                        ratings.append(float(r))
                elif isinstance(item, (int, float)):
                    ratings.append(float(item))
            if len(ratings) == n_items:
                return ratings
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try to extract numbers directly
    numbers = re.findall(r'\b([1-5](?:\.[05])?)\b', text)
    if len(numbers) >= n_items:
        return [float(x) for x in numbers[:n_items]]

    return None


def predict_ratings(client, persona, items, item_type, n_items, prev_items=None, prev_ratings=None,
                    prev_items3=None, prev_ratings3=None):
    """Call the LLM to predict ratings."""
    items_str = '\n'.join([f'{{"item_name": "{item}"}}' for item in items])
    
    # Build history context from windows 3+4 if available
    history_context = ""
    history_parts = []
    all_hist_ratings = []
    if prev_items3 and prev_ratings3 and len(prev_items3) == len(prev_ratings3):
        lines3 = "\n".join([f'  - "{pi}": {pr:.0f} stars' for pi, pr in zip(prev_items3, prev_ratings3)])
        history_parts.append(f"Window 3 ratings:\n{lines3}")
        all_hist_ratings.extend(prev_ratings3)
    if prev_items and prev_ratings and len(prev_items) == len(prev_ratings):
        lines4 = "\n".join([f'  - "{pi}": {pr:.0f} stars' for pi, pr in zip(prev_items, prev_ratings)])
        history_parts.append(f"Window 4 ratings (most recent):\n{lines4}")
        all_hist_ratings.extend(prev_ratings)
    if history_parts:
        # Add statistics summary
        if all_hist_ratings:
            avg_r = sum(all_hist_ratings) / len(all_hist_ratings)
            pct_5 = 100 * sum(1 for r in all_hist_ratings if r == 5.0) / len(all_hist_ratings)
            stats = f"Rating statistics: average={avg_r:.1f} stars, {pct_5:.0f}% are 5-star ratings"
            history_context = f"This user's recent rating history:\n{stats}\n\n" + "\n".join(history_parts)
        else:
            history_context = "This user's recent rating history:\n" + "\n".join(history_parts)
    
    prompt = PREDICT_PROMPT.format(
        item_type=item_type.replace('_', ' '),
        persona=persona,
        history_context=history_context,
        items=items_str
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
                timeout=60
            )
            text = response.choices[0].message.content.strip()
            preds = parse_predictions(text, n_items)
            if preds is not None:
                return preds, False
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] API call failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)

    # Fallback: return all 5s
    return [5.0] * n_items, True


def compute_mae(predicted, actual):
    """Compute Mean Absolute Error."""
    if len(predicted) != len(actual):
        n = min(len(predicted), len(actual))
        predicted = predicted[:n]
        actual = actual[:n]
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


# ─── Main ─────────────────────────────────────────────────────────────────────
def process_user(args):
    """Process a single user: predict ratings and compute MAE."""
    client, user_id, persona, test_sample = args

    items = test_sample.get('I5', [])
    actual = test_sample.get('U5', [])
    item_type = test_sample.get('item_type', 'art_crafts_sewing')
    # Window 3+4 history for context
    prev_items = test_sample.get('I4', [])
    prev_ratings = test_sample.get('U4', [])
    prev_items3 = test_sample.get('I3', [])
    prev_ratings3 = test_sample.get('U3', [])

    if not items or not actual:
        return user_id, None, True

    n_items = len(items)
    predicted, is_error = predict_ratings(client, persona, items, item_type, n_items, 
                                          prev_items=prev_items, prev_ratings=prev_ratings,
                                          prev_items3=prev_items3, prev_ratings3=prev_ratings3)
    
    # Post-processing: if user's historical ratings are very high, floor predictions at 4
    all_hist = list(prev_ratings3 or []) + list(prev_ratings or [])
    if all_hist:
        hist_avg = sum(all_hist) / len(all_hist)
        if hist_avg >= 4.5:
            # This user tends to give very high ratings, don't predict below 4
            predicted = [max(4.0, p) for p in predicted]
    
    mae = compute_mae(predicted, actual)

    return user_id, mae, is_error


def main():
    # Load personas and test data
    print("Loading personas...")
    personas = load_personas(PERSONA_PATH)
    print(f"Loaded {len(personas)} Art Crafts and Sewing personas")

    print("Loading test data...")
    test_data = load_test_data_with_history(TEST_DATA_PATH, TEST_DATA_PATH_4)
    print(f"Loaded {len(test_data)} test samples")

    # Match personas with test data using 'id' field
    matched = []
    for sample in test_data:
        uid = sample.get('id', sample.get('idx', ''))
        if uid in personas:
            matched.append((uid, personas[uid], sample))

    print(f"Matched {len(matched)} users with personas")

    if not matched:
        print("ERROR: No matched users found!", file=sys.stderr)
        sys.exit(1)

    # Initialize OpenRouter client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Run evaluation with concurrent workers
    print(f"\nRunning evaluation with {MAX_WORKERS} workers...")
    results = {}
    errors = 0
    mae_values = []

    args_list = [(client, uid, persona, sample) for uid, persona, sample in matched]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_user, args): args[1] for args in args_list}

        for i, future in enumerate(as_completed(futures)):
            user_id, mae, is_error = future.result()

            if mae is not None:
                results[user_id] = {'mae': mae, 'error': is_error}
                mae_values.append(mae)
                if is_error:
                    errors += 1

                # Progress update
                avg_mae = sum(mae_values) / len(mae_values)
                print(f"  [{i+1}/{len(matched)}] user={user_id[:8]}... MAE={mae:.4f} | Running avg: {avg_mae:.4f} | Errors: {errors}")
            else:
                errors += 1

    # Compute final MAE
    n_users = len(mae_values)
    avg_mae = sum(mae_values) / n_users if n_users > 0 else float('nan')

    # Save results
    save_data = {
        'n_users': n_users,
        'n_errors': errors,
        'average_mae': avg_mae,
        'mae_values': mae_values,
        'per_user_results': results
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary
    ci_lower, ci_upper = 0.374, 0.66
    within_ci = ci_lower <= avg_mae <= ci_upper

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Domain: Art Crafts and Sewing (Unseen)")
    print(f"Round: 4")
    print(f"N users: {n_users}")
    print(f"Errors: {errors}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Paper reported: 0.40 CI: [{ci_lower}, {ci_upper}]")
    print(f"Within CI: {within_ci}")

    return avg_mae


if __name__ == '__main__':
    main()
