#!/usr/bin/env python3
"""
Cloud-based Error Schema Generator for GPT and Gemini models.
Generates error schemata from error logs using cloud APIs.
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import google.generativeai as genai

def read_error_logs(directory_path: str):
    """
    Read and parse JSON files from the specified directory.
    """
    error_logs = []
    try:
        # Get JSON files, excluding file_mapping.json
        json_files = [f for f in os.listdir(directory_path)
                     if f.endswith('.json') and f != 'file_mapping.json']
        json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Verify it has the expected structure
                if isinstance(data, dict) and 'history' in data:
                    error_logs.append(data)
                else:
                    print(f"  ⚠️  Skipping {json_file}: unexpected format")
        return error_logs
    except Exception as e:
        print(f"Error reading error logs: {e}")
        raise

def create_prompt(error_log: dict) -> str:
    """
    Create a prompt for schema generation based on the error log.
    """
    # Extract relevant information
    chat_history = error_log.get("history", [])
    question = error_log.get("question", "")
    ground_truth = error_log.get("ground_truth", error_log.get("groundtruth", ""))
    mistake_agent = error_log.get("mistake_agent", "")
    mistake_step = error_log.get("mistake_step", "")
    mistake_reason = error_log.get("mistake_reason", "")

    # Determine agent key based on dataset type
    # Check if it's handcrafted (uses "role") or algorithm-generated (uses "name")
    agent_key = "role"
    if chat_history and "name" in chat_history[0]:
        agent_key = "name"

    # Format chat history with step numbers
    chat_content = "\n".join([
        f"Step {idx}: {entry.get(agent_key, 'Unknown')}: {entry.get('content', '')}"
        for idx, entry in enumerate(chat_history)
    ])

    # Create focused prompt for error identification
    prompt_text = f"""Given an error analysis from a multi-agent conversation, create a error schema to help identify similar errors in the future.

Context:
Question: {question}
Ground Truth: {ground_truth}
Error Agent: {mistake_agent}
Error Step: {mistake_step}
Error Reason: {mistake_reason}

Conversation History:
{chat_content}

Based on this error case, please create a error schema that will help IDENTIFY similar errors in future conversations. Focus primarily on recognition patterns rather than mitigation strategies. The schema should include:

1. Error Signatures:
   - What distinctive patterns or signals indicate this type of error is occurring?
   - What are the telltale signs in the agent's behavior or responses?

2. Error Context Analysis:
   - What contextual conditions typically surround this type of error?
   - What sequence of interactions tends to precede this error?

3. Detection Heuristics:
   - What specific questions can be asked to determine if this error is present?
   - What analytical framework can help identify this error pattern?
   - What key phrases or conversation patterns serve as reliable indicators?

Please format your response as a structured schema that focuses specifically on ERROR IDENTIFICATION, not on how to improve agent behavior.

Provide a concise, actionable schema in the following format:

Agent Name: {mistake_agent}
Step Number: {mistake_step}
Reason for Mistake: [Your analysis of why this specific error occurred and how to identify similar patterns]
"""

    return prompt_text

def generate_schema_gpt(error_log: dict, client: OpenAI, model: str = "gpt-4o-mini", max_tokens: int = 1024) -> str:
    """
    Generate a error schema using GPT models.
    """
    prompt = create_prompt(error_log)

    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations and creating schemata for error detection."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Handle GPT-5 models differently (they use max_completion_tokens)
        if 'gpt-5' in model:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # temperature=0.7,
                max_completion_tokens=max_tokens
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # temperature=0.7,
                max_tokens=max_tokens
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating schema with GPT: {e}")
        return f"Error: Failed to generate schema - {str(e)}"

def generate_schema_gemini(error_log: dict, model, max_tokens: int = 1024) -> str:
    """
    Generate a error schema using Gemini models.
    """
    prompt = create_prompt(error_log)

    # Convert to Gemini format
    full_prompt = f"""System: You are a helpful assistant skilled in analyzing conversations and creating schemata for error detection.

User: {prompt}"""

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating schema with Gemini: {e}")
        return f"Error: Failed to generate schema - {str(e)}"

def generate_schemata_parallel(error_logs: list, client, model_type: str, model_name: str,
                              max_tokens: int = 1024, max_workers: int = 5) -> list:
    """
    Generate schemata for multiple error logs in parallel using cloud APIs.
    """
    schemata = []

    print(f"Generating schemata using {model_type} model: {model_name}")
    print(f"Processing {len(error_logs)} error logs with {max_workers} concurrent workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        if model_type == "gpt":
            futures = {
                executor.submit(generate_schema_gpt, log, client, model_name, max_tokens): i
                for i, log in enumerate(error_logs)
            }
        else:  # gemini
            futures = {
                executor.submit(generate_schema_gemini, log, client, max_tokens): i
                for i, log in enumerate(error_logs)
            }

        # Process completed tasks
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Schemata"):
            idx = futures[future]
            try:
                schema = future.result(timeout=60)
                schemata.append((idx, schema))
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                print(f"  ✗ Failed to generate schema for log {idx + 1}: {e}")
                schemata.append((idx, f"Error: Failed to generate schema - {str(e)}"))

    # Sort schemata by original index
    schemata.sort(key=lambda x: x[0])
    return [t[1] for t in schemata]

def process_single_dataset(dataset_name, dataset_path, client, model_type, model_name,
                         max_tokens, max_workers, output_base_dir):
    """
    Process a single dataset and generate error schemata.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Check if dataset_path contains 'individual_trajectories' or has numbered JSON files
        if 'individual_trajectories' in dataset_path or \
           (os.path.exists(dataset_path) and
            any(f.endswith('.json') and f[0].isdigit()
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f)))):
            # Direct path to JSON files
            trajectories_path = dataset_path
            print(f"Reading error logs directly from: {trajectories_path}")
        else:
            # Try to find individual_trajectories subdirectory
            trajectories_path = os.path.join(dataset_path, "individual_trajectories")
            print(f"Reading error logs from: {trajectories_path}")

            if not os.path.exists(trajectories_path):
                print(f"  ✗ No individual_trajectories directory found for {dataset_name}")
                return

        error_logs = read_error_logs(trajectories_path)

        if not error_logs:
            print(f"  ✗ No error logs found for {dataset_name}")
            return

        print(f"  ✓ Found {len(error_logs)} error logs")

        # Generate schemata
        print(f"  Generating schemata...")
        schemata = generate_schemata_parallel(
            error_logs, client, model_type, model_name, max_tokens, max_workers
        )

        # Create output directory for this dataset
        output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Write results
        output_file = os.path.join(output_dir, "error_schemata.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (error_log, schema) in enumerate(zip(error_logs, schemata), 1):
                f.write(f"=== Schema for Error Log {i} ===\n")
                f.write("Generated Schema:\n")
                f.write(schema)
                f.write("\n\n" + "="*50 + "\n\n")

        print(f"  ✓ Schemata written to: {output_file}")

        # Also write a summary
        summary_file = os.path.join(output_dir, "generation_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Schema Generation Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Total error logs: {len(error_logs)}\n")
            f.write(f"Schemata generated: {len(schemata)}\n")
            f.write(f"Output file: {output_file}\n")

        print(f"  ✓ Summary written to: {summary_file}")

    except Exception as e:
        print(f"  ✗ Error processing dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description='Generate error identification schemata from error logs using cloud models (GPT/Gemini).'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-nano',
                'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-flash-8b'],
        help='Model to use for schema generation'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['arc', 'hotpot', 'musique', 'wikimqa', 'math500', 'mmlu_pro', 'gaia'],
        help='List of datasets to process (default: 7 CORRECT-Error datasets)'
    )

    parser.add_argument(
        '--dataset_paths',
        nargs='+',
        help='Specific paths to datasets (must match --datasets length)'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='data/correct_error',
        help='Base directory containing dataset results'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/schemata',
        help='Output directory for generated schemata'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help='Maximum tokens for schema generation. For reasoning models '
             '(gpt-5, gpt-5-nano) this is the combined reasoning + output '
             'budget — keep this above ~2048 or content may come back empty.'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        default=5,
        help='Maximum concurrent API calls'
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default=os.getenv('OPENAI_API_KEY', ''),
        help='API key for GPT models (uses OPENAI_API_KEY env var if not provided)'
    )

    parser.add_argument(
        '--gemini_api_key',
        type=str,
        default=os.getenv('GEMINI_API_KEY', ''),
        help='API key for Gemini models (uses GEMINI_API_KEY env var if not provided)'
    )

    args = parser.parse_args()

    # Initialize client based on model type
    client = None
    model_type = None

    if args.model.startswith('gpt'):
        model_type = 'gpt'
        if not args.api_key:
            print("Error: OpenAI API key required for GPT models")
            print("Provide via --api_key or set OPENAI_API_KEY environment variable")
            return

        try:
            client = OpenAI(api_key=args.api_key)
            print(f"✓ Initialized OpenAI client for model: {args.model}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return

    elif args.model.startswith('gemini'):
        model_type = 'gemini'
        if not args.gemini_api_key:
            print("Error: Gemini API key required for Gemini models")
            print("Provide via --gemini_api_key or set GEMINI_API_KEY environment variable")
            return

        try:
            genai.configure(api_key=args.gemini_api_key)
            client = genai.GenerativeModel(args.model)
            print(f"✓ Initialized Gemini client for model: {args.model}")
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            return

    # Output directory is taken verbatim; per-dataset subdirs are added below.
    # (We intentionally do NOT nest by model — the inference scripts look up
    # schemata at output_dir/{dataset}/error_schemata.txt without a model layer.)
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Schema Generation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_base_dir}")
    print(f"Max concurrent workers: {args.max_workers}")
    print(f"Max tokens: {args.max_tokens}")

    # Process each dataset
    for i, dataset_name in enumerate(args.datasets):
        # Determine dataset path
        if args.dataset_paths and i < len(args.dataset_paths):
            dataset_path = args.dataset_paths[i]
        else:
            dataset_path = os.path.join(args.results_dir, dataset_name)

        process_single_dataset(
            dataset_name,
            dataset_path,
            client,
            model_type,
            args.model,
            args.max_tokens,
            args.max_workers,
            output_base_dir
        )

    print(f"\n{'='*60}")
    print(f"Schema generation complete!")
    print(f"Results saved in: {output_base_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()