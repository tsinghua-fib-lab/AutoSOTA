"""
LLM Experiments for Graphical Causal Structure Discovery.

This script asks LLMs to identify:
1. Total causal effects (and their direction) between pairs of variables
2. Confounding between pairs of variables

The results are parsed into a "statement graph" with:
- Directed edges for causal effects (cause -> effect)
- Bidirected edges for confounding (X <-> Y)

Uses the conversation-style approach from experiments_llm_linear.py.
"""

import pandas as pd
import os
import time
import re
import boto3
from botocore.config import Config


# =============================================================================
# Data Configuration
# =============================================================================

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'LLM_responses_graphical')

DATASET_FILES = {
    'population_density': 'population_density.csv',
    'literacy_rate': 'literacy_rate.csv',
    'daily_income': 'daily_income.csv',
    'sanitation_access': 'sanitation_access.csv',
    'smoking': 'smoking.csv',
    'happiness_score': 'happiness_score.csv',
    'life_expectancy': 'life_expectancy.csv'
}

VARIABLE_NAMES = list(DATASET_FILES.keys())

VARIABLE_DESCRIPTIONS = {
    'population_density': 'Average number of people per square kilometer of land in the given country',
    'literacy_rate': 'Adult literacy rate is the percentage of people ages 15 and above who can, with understanding, read and write a short, simple statement on their everyday life',
    'daily_income': 'Mean daily household per capita income or consumption expenditure in constant international dollars',
    'sanitation_access': 'Percentage of people using at least basic sanitation services (improved sanitation facilities not shared with other households)',
    'smoking': 'Percentage of both men and women over age 15 that smoke',
    'happiness_score': 'National average response to happiness survey, score ranging from 0 (worst) to 100 (best)',
    'life_expectancy': 'Average life expectancy at birth in years'
}


def load_correlation_matrix():
    """Load the correlation matrix."""
    filepath = os.path.join(DATASET_DIR, 'correlation_matrix.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0)
    
    # Fallback: compute from data
    print("Computing correlation matrix from data...")
    dataframes = {}
    all_years = set()
    
    for var_name, filename in DATASET_FILES.items():
        filepath = os.path.join(DATASET_DIR, filename)
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
    
    combined_df = pd.concat(all_data)
    var_columns = list(DATASET_FILES.keys())
    corr_matrix = combined_df[var_columns].corr()
    
    # Save for future use
    corr_matrix.to_csv(os.path.join(DATASET_DIR, 'correlation_matrix.csv'))
    
    return corr_matrix


def extract_content_text(response):
    """Extract the actual text content from Bedrock response."""
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


def parse_causal_effect(content):
    """
    Parse the causal effect response.
    
    Expected format:
    <causal_effect>
        <exists>YES|NO</exists>
        <direction>A_TO_B|B_TO_A|NONE</direction>
    </causal_effect>
    
    Returns:
        dict with 'exists', 'direction' or None if parsing fails
    """
    if content is None:
        return None
    
    # Try to find the causal_effect block
    effect_match = re.search(r'<causal_effect>(.*?)</causal_effect>', content, re.DOTALL | re.IGNORECASE)
    if not effect_match:
        return None
    
    block = effect_match.group(1)
    
    exists_match = re.search(r'<exists>\s*(YES|NO)\s*</exists>', block, re.IGNORECASE)
    direction_match = re.search(r'<direction>\s*(A_TO_B|B_TO_A|NONE)\s*</direction>', block, re.IGNORECASE)
    
    return {
        'exists': exists_match.group(1).upper() if exists_match else None,
        'direction': direction_match.group(1).upper() if direction_match else None
    }


def parse_confounding(content):
    """
    Parse the confounding response.
    
    Expected format:
    <confounding>
        <exists>YES|NO</exists>
    </confounding>
    
    Returns:
        dict with 'exists' or None if parsing fails
    """
    if content is None:
        return None
    
    conf_match = re.search(r'<confounding>(.*?)</confounding>', content, re.DOTALL | re.IGNORECASE)
    if not conf_match:
        return None
    
    block = conf_match.group(1)
    
    exists_match = re.search(r'<exists>\s*(YES|NO)\s*</exists>', block, re.IGNORECASE)
    
    return {
        'exists': exists_match.group(1).upper() if exists_match else None
    }


def get_model_config(model_id):
    """Get the inference configuration for a specific model."""
    inference_config = {
        "maxTokens": 2048,
        "temperature": 0.6,
        "topP": 0.7
    }
    additional_model_fields = None
    
    if "." in model_id:
        model_name = model_id.split(".")[-1].split(":")[0]
    else:
        model_name = model_id
    
    # Special handling for Anthropic Claude models (no topP)
    if "anthropic" in model_id.lower():
        inference_config = {
            "maxTokens": 2048,
            "temperature": 0.6
        }
    
    return {
        'inference_config': inference_config,
        'additional_model_fields': additional_model_fields,
        'model_name': model_name
    }


def build_statement_graph(pair_results):
    """
    Build a statement graph from the pair results.
    
    Returns:
        dict with:
            'directed_edges': list of (cause, effect) tuples
            'bidirected_edges': list of (var1, var2) tuples
            'adjacency_matrix': DataFrame for directed edges
            'confounding_matrix': DataFrame for bidirected edges
    """
    n = len(VARIABLE_NAMES)
    
    # Directed edges (causal effects)
    directed_adj = pd.DataFrame(
        0, index=VARIABLE_NAMES, columns=VARIABLE_NAMES, dtype=int
    )
    directed_edges = []
    
    # Bidirected edges (confounding)
    confounding_adj = pd.DataFrame(
        0, index=VARIABLE_NAMES, columns=VARIABLE_NAMES, dtype=int
    )
    bidirected_edges = []
    
    for (var_a, var_b), result in pair_results.items():
        causal = result.get('causal_effect')
        confounding = result.get('confounding')
        
        # Process causal effect
        if causal and causal.get('exists') == 'YES':
            direction = causal.get('direction')
            
            if direction == 'A_TO_B':
                directed_adj.loc[var_b, var_a] = 1  # cause -> effect (lower triangular)
                directed_edges.append((var_a, var_b))
            elif direction == 'B_TO_A':
                directed_adj.loc[var_a, var_b] = 1
                directed_edges.append((var_b, var_a))
        
        # Process confounding
        if confounding and confounding.get('exists') == 'YES':
            confounding_adj.loc[var_a, var_b] = 1
            confounding_adj.loc[var_b, var_a] = 1
            bidirected_edges.append((var_a, var_b))
    
    return {
        'directed_edges': directed_edges,
        'bidirected_edges': bidirected_edges,
        'adjacency_matrix': directed_adj,
        'confounding_matrix': confounding_adj
    }


def run_graphical_experiment_conversation(
    bedrock_client, 
    model_id, 
    num_runs=5, 
    model_config=None,
    seed=42,
    on_run_complete=None,
    start_run=1
):
    """
    Run the graphical structure discovery experiment using conversation approach.
    
    For each pair of variables, asks:
    1. Is there a causal effect? If so, in which direction?
    2. Is there strong confounding between the two variables?
    
    Parameters:
        bedrock_client: The boto3 Bedrock runtime client
        model_id: The Bedrock model ID to use
        num_runs: Number of complete conversation runs (default: 5)
        model_config: Optional dict with 'inference_config' and 'additional_model_fields'
        seed: Random seed for reproducibility
        on_run_complete: Optional callback ``on_run_complete(statement_graph)``
            called immediately after each run finishes, allowing the caller to
            persist results as they are produced.
        start_run: Run number for the first run (default: 1). Subsequent runs
            are numbered consecutively. Useful for appending to an existing
            set of saved runs.
    
    Returns:
        list: Statement graphs for each run
        dict: Aggregated results across runs
    """
    import random
    random.seed(seed)
    
    # Load correlation matrix
    corr_matrix = load_correlation_matrix()
    corr_str = corr_matrix.to_string()
    
    # Generate all unordered pairs
    pairs = []
    for i, var_a in enumerate(VARIABLE_NAMES):
        for var_b in VARIABLE_NAMES[i+1:]:
            pairs.append((var_a, var_b))
    
    total_pairs = len(pairs)
    n_vars = len(VARIABLE_NAMES)
    
    print(f"[Graphical Structure Discovery - Conversation-based]")
    print(f"Analyzing {total_pairs} pairs ({n_vars} variables)")
    print(f"Running {num_runs} complete conversation runs")
    print()
    
    # System prompt
    system_prompt = """You are a causality expert analyzing relationships between country-level development indicators.

For each pair of variables, you will assess:
1. Whether there is a total causal effect between them, and in which direction
2. Whether there is confounding (correlation not explained by the causal effect between the pair)

Guidelines:
- Answer YES for causal effect if you expect that an intervention on the cause variable would significantly change the effect variable
- Be conservative about confounding - only answer YES for confounding if MOST of the correlation cannot be explained by the causal effect
- Use realistic domain knowledge about socioeconomic factors

Return your answer in the following HTML format:

<causal_effect>
    <exists>YES or NO</exists>
    <direction>A_TO_B or B_TO_A or NONE</direction>
</causal_effect>
<confounding>
    <exists>YES or NO</exists>
</confounding>

Where A_TO_B means the first variable causes the second, and B_TO_A means the second causes the first."""

    system = [{"text": system_prompt}]
    
    # Use provided config or default
    if model_config is None:
        model_config = get_model_config(model_id)
    
    inference_config = model_config['inference_config']
    additional_model_fields = model_config.get('additional_model_fields')
    
    # Initial context message
    context_prompt = f"""I have observational data on {n_vars} country-level development indicators.

Correlation matrix:
{corr_str}

Variable descriptions:
{chr(10).join(f"- {var}: {desc}" for var, desc in VARIABLE_DESCRIPTIONS.items())}

I will now ask you about each pair of variables. For each pair:
1. Assess if there is a causal effect 
2. Assess if there is strong confounding"""

    # Collect results across all runs
    all_run_results = []
    all_statement_graphs = []
    
    for run in range(start_run, start_run + num_runs):
        print(f"=== Run {run} (target: {start_run}..{start_run + num_runs - 1}) ===")
        
        # Start fresh conversation with context
        messages = [
            {"role": "user", "content": [{"text": context_prompt}]}
        ]
        
        run_results = {}
        
        # Helper to make API call
        def make_api_call(msgs):
            kwargs = {
                'modelId': model_id,
                'messages': msgs,
                'system': system,
                'inferenceConfig': inference_config
            }
            if additional_model_fields:
                kwargs['additionalModelRequestFields'] = additional_model_fields
            return bedrock_client.converse(**kwargs)
        
        # Get initial acknowledgment
        max_retries = 10
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = make_api_call(messages)
                ack_content = extract_content_text(response)
                if ack_content:
                    messages.append({"role": "assistant", "content": [{"text": ack_content}]})
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Error: {str(e)}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  Failed to get initial acknowledgment after {max_retries} attempts")
                    break
        
        # Ask about each pair
        for idx, (var_a, var_b) in enumerate(pairs, 1):
            corr_val = corr_matrix.loc[var_a, var_b]
            
            # Apply random permutation to the pair presentation
            if random.random() < 0.5:
                presented_a, presented_b = var_a, var_b
                swapped = False
            else:
                presented_a, presented_b = var_b, var_a
                swapped = True
            
            print(f"  [{idx}/{total_pairs}] {var_a} <-> {var_b} (r={corr_val:.3f})...", end=" ")
            
            question = f"""Consider the pair: "{presented_a}" (A) and "{presented_b}" (B).

1. Is there a causal effect between them? If yes, in which direction?
2. Is there confounding (significant correlation not explained by the causal effect between these two)?

Provide your assessment in the required format."""

            messages.append({"role": "user", "content": [{"text": question}]})
            
            # Try to get a valid answer
            max_parse_attempts = 5
            causal_effect = None
            confounding = None
            response_text = None
            
            for parse_attempt in range(max_parse_attempts):
                for attempt in range(max_retries):
                    try:
                        response = make_api_call(messages)
                        response_text = extract_content_text(response)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            time.sleep(delay)
                        else:
                            response_text = None
                
                if response_text:
                    messages.append({"role": "assistant", "content": [{"text": response_text}]})
                    causal_effect = parse_causal_effect(response_text)
                    confounding = parse_confounding(response_text)
                    
                    if causal_effect is not None and confounding is not None:
                        break
                    elif parse_attempt < max_parse_attempts - 1:
                        retry_msg = """Please provide your answer in the correct format:
<causal_effect>
    <exists>YES or NO</exists>
    <direction>A_TO_B or B_TO_A or NONE</direction>
</causal_effect>
<confounding>
    <exists>YES or NO</exists>
</confounding>"""
                        messages.append({"role": "user", "content": [{"text": retry_msg}]})
                        print("retry...", end=" ")
            
            # Store results - translate direction back if swapped
            stored_causal_effect = causal_effect
            if causal_effect and swapped and causal_effect.get('direction') in ['A_TO_B', 'B_TO_A']:
                # Swap the direction back to match original (var_a, var_b) ordering
                original_direction = causal_effect['direction']
                if original_direction == 'A_TO_B':
                    stored_causal_effect = {**causal_effect, 'direction': 'B_TO_A'}
                elif original_direction == 'B_TO_A':
                    stored_causal_effect = {**causal_effect, 'direction': 'A_TO_B'}
            
            run_results[(var_a, var_b)] = {
                'causal_effect': stored_causal_effect,
                'confounding': confounding,
                'response': response_text,
                'correlation': corr_val,
                'presented_order': (presented_a, presented_b)
            }
            
            # Print summary
            if stored_causal_effect and confounding:
                ce_str = f"CE:{stored_causal_effect.get('exists', '?')}"
                if stored_causal_effect.get('exists') == 'YES':
                    ce_str += f"({stored_causal_effect.get('direction', '?')})"
                conf_str = f"Conf:{confounding.get('exists', '?')}"
                print(f"[OK] {ce_str}, {conf_str}")
            else:
                print("[X] Parse failed")
        
        # Build statement graph for this run
        statement_graph = build_statement_graph(run_results)
        statement_graph['run'] = run
        statement_graph['pair_results'] = {
            f"{k[0]} <-> {k[1]}": {
                'causal_effect': v['causal_effect'],
                'confounding': v['confounding'],
                'correlation': v['correlation']
            }
            for k, v in run_results.items()
        }
        
        all_run_results.append(run_results)
        all_statement_graphs.append(statement_graph)
        
        # Print run summary
        n_directed = len(statement_graph['directed_edges'])
        n_bidirected = len(statement_graph['bidirected_edges'])
        print(f"  Run {run} summary: {n_directed} directed edges, {n_bidirected} bidirected edges")

        if on_run_complete is not None:
            try:
                on_run_complete(statement_graph)
            except Exception as save_err:
                print(f"  [!] on_run_complete callback failed: {save_err}")
        print()
    
    return all_statement_graphs, all_run_results


def save_results(model_name, statement_graphs, output_dir=None):
    """Save the statement graphs and results to JSON files."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save adjacency matrices for each run as CSV files
    for graph in statement_graphs:
        save_run_graph(model_name, graph, output_dir=output_dir)
    
    print(f"Saved {len(statement_graphs)} adjacency matrix pairs to: {output_dir}")
    
    return output_dir


def save_run_graph(model_name, graph, output_dir=None):
    """Save adjacency matrices for a single run's statement graph."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    run = graph['run']
    directed_path = os.path.join(output_dir, f'directed_adj_{model_name}_run{run}.csv')
    graph['adjacency_matrix'].to_csv(directed_path)

    bidirected_path = os.path.join(output_dir, f'bidirected_adj_{model_name}_run{run}.csv')
    graph['confounding_matrix'].to_csv(bidirected_path)

    print(f"  [saved] directed_adj_{model_name}_run{run}.csv, "
          f"bidirected_adj_{model_name}_run{run}.csv")


def run_experiment_for_model(model_id, num_runs=5, region='us-east-2', start_run=1):
    """Run the full experiment for a single model."""
    # Initialize Bedrock client
    config = Config(
        read_timeout=600,
        connect_timeout=60,
        retries={'max_attempts': 10, 'mode': 'adaptive'}
    )
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=config
    )
    
    model_config = get_model_config(model_id)
    model_name = model_config['model_name']
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    # Run experiment, saving each run as soon as it completes
    def _save_callback(graph):
        save_run_graph(model_name, graph)

    statement_graphs, all_results = run_graphical_experiment_conversation(
        bedrock_client, 
        model_id, 
        num_runs=num_runs,
        model_config=model_config,
        on_run_complete=_save_callback,
        start_run=start_run,
    )

    print(f"Completed {len(statement_graphs)} runs for {model_name}; "
          f"results saved incrementally to: {OUTPUT_DIR}")
    
    return statement_graphs, all_results


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    import sys
    
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
    NUM_RUNS = 10
    START_RUN = 1
    
    # Define all models to run
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
    
    # Parse command line arguments
    # Usage:
    #   python experiments_llm_graphical.py                    # Run all models
    #   python experiments_llm_graphical.py 1                  # Run model 1
    #   python experiments_llm_graphical.py claude             # Run matching models
    #   python experiments_llm_graphical.py list               # List available models
    #   python experiments_llm_graphical.py --start-run 6      # Number runs starting at 6
    
    args = sys.argv[1:]

    if '--start-run' in args:
        idx = args.index('--start-run')
        if idx + 1 < len(args):
            START_RUN = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --start-run requires a value")
            sys.exit(1)
    
    # Check for help commands first
    if len(args) > 0:
        if args[0].lower() == "list":
            print("Available models:")
            for i, mid in enumerate(MODELS_TO_RUN):
                print(f"  {i+1}. {mid}")
            sys.exit(0)
    
    # Parse model selection
    if len(args) > 0:
        arg = args[0].lower()
        if arg.isdigit():
            # Run specific model by index (1-based)
            idx = int(arg) - 1
            if 0 <= idx < len(MODELS_TO_RUN):
                MODELS_TO_RUN = [MODELS_TO_RUN[idx]]
            else:
                print(f"Invalid model index. Use 1-{len(MODELS_TO_RUN)}")
                sys.exit(1)
        elif arg == "all":
            pass  # Run all models
        else:
            # Try to match model name
            matching = [mid for mid in MODELS_TO_RUN if arg in mid.lower()]
            if matching:
                MODELS_TO_RUN = matching
            else:
                print(f"No models matching '{arg}'. Use 'list' to see available models.")
                sys.exit(1)
    
    print("=" * 70)
    print("LLM GRAPHICAL STRUCTURE DISCOVERY - BATCH EXPERIMENT")
    print(f"Region: {AWS_REGION}")
    print(f"Dataset: data ({len(VARIABLE_NAMES)} variables)")
    print(f"Runs per model: {NUM_RUNS} (numbered {START_RUN}..{START_RUN + NUM_RUNS - 1})")
    print(f"Models to run: {len(MODELS_TO_RUN)}")
    print("=" * 70)
    
    for i, model_id in enumerate(MODELS_TO_RUN):
        print("\n" + "#" * 70)
        print(f"# MODEL {i+1}/{len(MODELS_TO_RUN)}")
        print("#" * 70)
        
        run_experiment_for_model(model_id, num_runs=NUM_RUNS, region=AWS_REGION, start_run=START_RUN)
    
    print("\n" + "=" * 70)
    print("ALL MODELS COMPLETE")
    print("=" * 70)
