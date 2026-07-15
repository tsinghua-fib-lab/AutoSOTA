"""
LLM causal coefficient estimation with LLM-determined causal ordering.

At the start of each conversation run, the LLM is asked to produce a causal
ordering of the variables. Coefficients are then queried pair-by-pair
according to that ordering.

Dataset: 7 variables (Population Density, Literacy, Income, Sanitation,
Smoking, Happiness, Life Expectancy)
"""

import pandas as pd
import numpy as np
import os
import time
import re
import boto3
from botocore.config import Config


# =============================================================================
# Dataset configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

DATA_FILES = {
    'population_density': 'population_density.csv',
    'literacy_rate': 'literacy_rate.csv',
    'daily_income': 'daily_income.csv',
    'sanitation_access': 'sanitation_access.csv',
    'smoking': 'smoking.csv',
    'happiness_score': 'happiness_score.csv',
    'life_expectancy': 'life_expectancy.csv',
}

VARIABLES = list(DATA_FILES.keys())

VARIABLE_DESCRIPTIONS = {
    'population_density': 'Average number of people per square kilometer of land in the given country',
    'literacy_rate': 'Adult literacy rate is the percentage of people ages 15 and above who can, with understanding, read and write a short, simple statement on their everyday life',
    'daily_income': 'Mean daily household per capita income or consumption expenditure in constant international dollars',
    'sanitation_access': 'Percentage of people using at least basic sanitation services (improved sanitation facilities not shared with other households)',
    'smoking': 'Percentage of both men and women over age 15 that smoke',
    'happiness_score': 'National average response to happiness survey, score ranging from 0 (worst) to 100 (best)',
    'life_expectancy': 'Average life expectancy at birth in years',
}


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load the dataset as a long DataFrame with one row per (country, year)."""
    dataframes = {}
    all_years = set()

    for var_name, filename in DATA_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)
        dataframes[var_name] = df
        year_cols = [col for col in df.columns if col not in ['geo', 'name']]
        all_years.update(year_cols)

    all_years = sorted([int(y) for y in all_years if 1950 <= int(y) <= 2025])

    all_data = []
    for year in all_years:
        year_str = str(year)
        year_data = {}
        for var_name, df in dataframes.items():
            df_indexed = df.set_index('geo')
            if year_str in df_indexed.columns:
                year_data[var_name] = df_indexed[year_str]
        if year_data:
            year_df = pd.DataFrame(year_data)
            year_df['year'] = year
            all_data.append(year_df)

    combined_df = pd.concat(all_data).reset_index()
    return combined_df


def compute_correlation_matrix():
    """Compute the correlation matrix across all variables."""
    df = load_data()
    return df[list(DATA_FILES.keys())].corr()


# =============================================================================
# Response parsing helpers
# =============================================================================

def extract_content_text(response):
    """Extract text content from a Bedrock converse response."""
    output = response.get('output', {})
    message = output.get('message', {})
    content_list = message.get('content', [])

    text_parts = []
    for item in content_list:
        if isinstance(item, dict):
            if 'text' in item:
                text_parts.append(item['text'])
            elif 'reasoningContent' in item:
                reasoning = item['reasoningContent']
                if isinstance(reasoning, dict) and 'reasoningText' in reasoning:
                    rt = reasoning['reasoningText']
                    if isinstance(rt, dict) and 'text' in rt:
                        text_parts.append(rt['text'])
        elif isinstance(item, str):
            text_parts.append(item)

    return '\n'.join(text_parts) if text_parts else None


def parse_coefficient(content):
    """Parse the causal coefficient from response text."""
    if content is None:
        return None
    match = re.search(r'CAUSAL_COEFFICIENT:\s*([-+]?\d*\.?\d+)', content)
    if match:
        return float(match.group(1))
    return None


def parse_ordering(content):
    """Parse the causal ordering from response text.

    Expected format: <ordering>var1, var2, var3, ...</ordering>
    Returns a list of variable names, or None if parsing fails.
    """
    if content is None:
        return None
    match = re.search(r'<ordering>\s*(.+?)\s*</ordering>', content, re.DOTALL)
    if not match:
        return None
    names = [s.strip() for s in match.group(1).split(',')]
    if set(names) != set(VARIABLES) or len(names) != len(VARIABLES):
        return None
    return names


# =============================================================================
# Model configuration
# =============================================================================

def get_model_config(model_id):
    """Get the inference configuration for a Bedrock model."""
    inference_config = {
        "maxTokens": 2048,
        "temperature": 0.6,
        "topP": 0.7,
    }

    # Anthropic Claude models do not allow topP
    if "anthropic" in model_id.lower():
        inference_config = {
            "maxTokens": 2048,
            "temperature": 0.6,
        }

    if "." in model_id:
        model_name = model_id.split(".")[-1].split(":")[0]
    else:
        model_name = model_id

    return {
        'inference_config': inference_config,
        'model_name': model_name,
    }


# =============================================================================
# Main experiment
# =============================================================================

def estimate_all_coefficients_conversation(bedrock_client, model_id, num_runs=5,
                                           model_config=None, max_total_runs=10,
                                           on_run_complete=None):
    """Estimate causal coefficients using an LLM-determined ordering per run.

    Runs are discarded if the ordering or any coefficient fails to parse.
    Retries until num_runs valid runs are obtained, or max_total_runs attempts
    have been made.

    If ``on_run_complete`` is provided, it is called as
    ``on_run_complete(run_index, matrix)`` immediately after each valid run
    (run_index is 0-based within this call).
    """
    corr_matrix = compute_correlation_matrix()

    n_vars = len(VARIABLES)
    print(f"Variables ({n_vars}): {', '.join(VARIABLES)}")
    print(f"Running {num_runs} complete conversation runs")
    print()

    system_prompt = (
        "You are a causality expert, tasked to estimate standardized TOTAL "
        "causal effects between country development indicators.\n"
        "Return your answer in HTML format:\n\n"
        "<answer>CAUSAL_COEFFICIENT: <number></answer>\n\n"
        "For example: <answer>CAUSAL_COEFFICIENT: 0.35</answer> or "
        "<answer>CAUSAL_COEFFICIENT: -0.62</answer>\n"
        "The causal coefficient quantifies the expected change of the effect "
        "variable in standard deviations, given an intervention that changes "
        "the cause variable by 1 standard deviation. It includes the effect "
        "of all direct causal pathways from the cause to the effect variable. "
        "Do not assume away confounding; use realistic domain knowledge. "
        "No other text."
    )
    system = [{"text": system_prompt}]

    if model_config is None:
        model_config = get_model_config(model_id)

    inference_config = model_config['inference_config']

    max_retries = 10
    base_delay = 2

    def make_api_call(msgs):
        return bedrock_client.converse(
            modelId=model_id,
            messages=msgs,
            system=system,
            inferenceConfig=inference_config,
        )

    per_run_matrices = []
    all_conversation_logs = []

    attempt_num = 0
    while len(per_run_matrices) < num_runs and attempt_num < max_total_runs:
        attempt_num += 1
        valid_count = len(per_run_matrices)
        print(f"=== Attempt {attempt_num}/{max_total_runs} "
              f"(valid runs: {valid_count}/{num_runs}) ===")

        # Randomize variable presentation order per attempt
        rng = np.random.RandomState(
            seed=hash((attempt_num, model_id)) & 0xFFFFFFFF
        )
        shuffled_vars = list(VARIABLES)
        rng.shuffle(shuffled_vars)
        print(f"  Presentation order: {', '.join(shuffled_vars)}")

        corr_shuffled = corr_matrix.loc[shuffled_vars, shuffled_vars]
        corr_str = corr_shuffled.to_string()
        var_desc_block = '\n'.join(
            f"- {var}: {VARIABLE_DESCRIPTIONS[var]}" for var in shuffled_vars
        )

        ordering_prompt = f"""I have observational data on {n_vars} country-level development indicators.

Correlation matrix:
{corr_str}

Variable descriptions:
{var_desc_block}

Before we begin estimating causal coefficients, please determine a plausible \
causal ordering of these {n_vars} variables (from root causes to downstream \
effects). Return the ordering as a comma-separated list of variable names \
inside <ordering> tags. Use the exact variable names shown above.

For example: <ordering>var_a, var_b, var_c, ...</ordering>"""

        messages = [
            {"role": "user", "content": [{"text": ordering_prompt}]}
        ]

        conversation_log = {'attempt': attempt_num, 'ordering': None, 'pairs': {}}

        # ---- Step 1: get causal ordering ----
        ordering = None
        max_ordering_attempts = 5
        for oa in range(max_ordering_attempts):
            resp_text = None
            for attempt in range(max_retries):
                try:
                    response = make_api_call(messages)
                    resp_text = extract_content_text(response)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Error: {e}, retrying in {delay}s...")
                        time.sleep(delay)

            if resp_text:
                messages.append({"role": "assistant", "content": [{"text": resp_text}]})
                ordering = parse_ordering(resp_text)
                if ordering is not None:
                    break
                elif oa < max_ordering_attempts - 1:
                    retry_msg = (
                        "Please provide the causal ordering as a comma-separated "
                        "list of the exact variable names inside <ordering> tags. "
                        f"The variable names are: {', '.join(VARIABLES)}"
                    )
                    messages.append({"role": "user", "content": [{"text": retry_msg}]})
                    print("  ordering retry...", end=" ")

        if ordering is None:
            print("  [X] Failed to parse ordering, discarding run")
            continue

        conversation_log['ordering'] = ordering
        print(f"  Ordering: {' -> '.join(ordering)}")

        # ---- Step 2: transition to coefficient questions ----
        transition_msg = (
            "I will now ask you to estimate the total linear "
            "causal coefficient for several pairs of variables, following "
            "the ordering you provided. For each question, please provide "
            "your answer in the format: "
            "<answer>CAUSAL_COEFFICIENT: <number></answer>"
        )
        messages.append({"role": "user", "content": [{"text": transition_msg}]})

        for attempt in range(max_retries):
            try:
                response = make_api_call(messages)
                ack_content = extract_content_text(response)
                if ack_content:
                    messages.append({"role": "assistant", "content": [{"text": ack_content}]})
                    break
            except Exception:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)

        # ---- Step 3: ask coefficients pair by pair ----
        pairs = []
        for i, cause in enumerate(ordering):
            for effect in ordering[i + 1:]:
                pairs.append((cause, effect))

        total_pairs = len(pairs)
        pair_coefficients = {}

        for idx, (cause, effect) in enumerate(pairs, 1):
            print(f"  [{idx}/{total_pairs}] {cause} -> {effect}...", end=" ")

            question = (
                f'Estimate the total linear causal coefficient for the causal '
                f'effect of "{cause}" on "{effect}".'
            )
            messages.append({"role": "user", "content": [{"text": question}]})

            max_parse_attempts = 5
            coefficient = None
            response_text = None

            for parse_attempt in range(max_parse_attempts):
                for attempt in range(max_retries):
                    try:
                        response = make_api_call(messages)
                        response_text = extract_content_text(response)
                        break
                    except Exception:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print("throttled...", end=" ")
                            time.sleep(delay)
                        else:
                            response_text = None

                if response_text:
                    messages.append({"role": "assistant", "content": [{"text": response_text}]})
                    coefficient = parse_coefficient(response_text)
                    if coefficient is not None:
                        break
                    elif parse_attempt < max_parse_attempts - 1:
                        retry_msg = (
                            "Please provide your answer in the correct format: "
                            "<answer>CAUSAL_COEFFICIENT: <number></answer>"
                        )
                        messages.append({"role": "user", "content": [{"text": retry_msg}]})
                        print("retry...", end=" ")

            if coefficient is not None:
                pair_coefficients[(cause, effect)] = coefficient
                print(f"[OK] {coefficient}")
            else:
                print("[X] Failed to parse, discarding run")
                break

            conversation_log['pairs'][(cause, effect)] = {
                'coefficient': coefficient,
                'response': response_text,
            }
        else:
            # All coefficients parsed: build matrix in LLM-determined ordering
            run_matrix = pd.DataFrame(
                np.zeros((n_vars, n_vars)),
                index=ordering,
                columns=ordering,
            )
            for var in ordering:
                run_matrix.loc[var, var] = 1.0
            for (cause, effect), coef in pair_coefficients.items():
                run_matrix.loc[effect, cause] = coef

            per_run_matrices.append(run_matrix)
            all_conversation_logs.append(conversation_log)
            if on_run_complete is not None:
                try:
                    on_run_complete(len(per_run_matrices) - 1, run_matrix)
                except Exception as save_err:
                    print(f"  [!] on_run_complete callback failed: {save_err}")
            print()
            continue

        # If we broke out of the pair loop, the run is discarded
        print()

    print(f"Completed {len(per_run_matrices)} valid runs "
          f"out of {attempt_num} attempts")

    return per_run_matrices, all_conversation_logs


def run_single_model(bedrock_client, model_id, model_config, output_dir,
                     num_runs=5, start_run=1):
    """Run the experiment for a single model and persist the results."""
    model_name = model_config['model_name']

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Runs: {start_run} to {start_run + num_runs - 1}")
    print("=" * 70)

    responses_dir = os.path.join(output_dir, 'LLM_responses_linear')
    os.makedirs(responses_dir, exist_ok=True)

    # Save correlation matrix up-front so partial results are always usable
    corr_matrix = compute_correlation_matrix()
    corr_matrix.to_csv(os.path.join(responses_dir, 'correlation_matrix.csv'))

    saved_run_nums = []

    def save_run(run_idx, matrix):
        run_num = start_run + run_idx
        out_path = os.path.join(
            responses_dir,
            f'causal_coefficients_{model_name}_run{run_num}.csv',
        )
        matrix.to_csv(out_path)
        saved_run_nums.append(run_num)
        print(f"  [saved] causal_coefficients_{model_name}_run{run_num}.csv")

    try:
        estimate_all_coefficients_conversation(
            bedrock_client, model_id, num_runs=num_runs,
            model_config=model_config, on_run_complete=save_run,
        )

        print(f"\nSaved to LLM_responses_linear/:")
        for run_num in saved_run_nums:
            print(f"  - causal_coefficients_{model_name}_run{run_num}.csv")
        print(f"  - correlation_matrix.csv")
    except Exception as e:
        print(f"ERROR for {model_name}: {str(e)}")
        if saved_run_nums:
            print(f"  Partial results saved for runs: {saved_run_nums}")


if __name__ == "__main__":
    import sys

    AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
    NUM_RUNS = 10
    START_RUN = 1

    MODELS_TO_RUN = [
        "openai.gpt-oss-120b-1:0",
        "openai.gpt-oss-20b-1:0",
        "mistral.magistral-small-2509",
        "mistral.mistral-large-3-675b-instruct",
        "qwen.qwen3-next-80b-a3b",
        "qwen.qwen3-235b-a22b-2507-v1:0",
        "google.gemma-3-4b-it",
        "google.gemma-3-27b-it",
        "moonshot.kimi-k2-thinking",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-opus-4-6-v1",
    ]

    args = sys.argv[1:]

    if '--start-run' in args:
        idx = args.index('--start-run')
        if idx + 1 < len(args):
            START_RUN = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --start-run requires a value")
            sys.exit(1)

    if len(args) > 0 and args[0].lower() == "list":
        print("Available models:")
        for i, mid in enumerate(MODELS_TO_RUN):
            print(f"  {i+1}. {mid}")
        sys.exit(0)

    if len(args) > 0:
        arg = args[0].lower()
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(MODELS_TO_RUN):
                MODELS_TO_RUN = [MODELS_TO_RUN[idx]]
            else:
                print(f"Invalid model index. Use 1-{len(MODELS_TO_RUN)}")
                sys.exit(1)
        elif arg == "all":
            pass
        else:
            matching = [mid for mid in MODELS_TO_RUN if arg in mid.lower()]
            if matching:
                MODELS_TO_RUN = matching
            else:
                print(f"No models matching '{arg}'. Use 'list' to see available models.")
                sys.exit(1)

    bedrock_config = Config(
        region_name=AWS_REGION,
        retries={'max_attempts': 3, 'mode': 'adaptive'},
    )
    bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)

    output_dir = os.path.dirname(__file__)

    print("=" * 70)
    print("LLM CAUSAL COEFFICIENT ESTIMATION (LLM-DETERMINED ORDERING)")
    print(f"Region: {AWS_REGION}")
    print(f"Variables: {len(VARIABLES)}")
    print(f"Runs: {START_RUN} to {START_RUN + NUM_RUNS - 1} ({NUM_RUNS} runs)")
    print(f"Models to run: {len(MODELS_TO_RUN)}")
    print("=" * 70)

    for i, model_id in enumerate(MODELS_TO_RUN):
        print("\n" + "#" * 70)
        print(f"# MODEL {i+1}/{len(MODELS_TO_RUN)}")
        print("#" * 70)

        model_config = get_model_config(model_id)

        run_single_model(
            bedrock_client=bedrock_client,
            model_id=model_id,
            model_config=model_config,
            output_dir=output_dir,
            num_runs=NUM_RUNS,
            start_run=START_RUN,
        )

    print("\n" + "=" * 70)
    print("ALL MODELS COMPLETE")
    print("=" * 70)
