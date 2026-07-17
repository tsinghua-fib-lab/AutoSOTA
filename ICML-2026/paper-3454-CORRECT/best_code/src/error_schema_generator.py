import os
import json
import argparse
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def setup_vllm_model(model_name: str, tensor_parallel_size: int = 8):
    """
    Initialize VLLM model with the provided model name.
    """
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            # max_model_len=65536,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 10,
                "original_max_position_embeddings": 32768,  # Model's original context length
            },
        )
        return llm
    except Exception as e:
        print(f"Error initializing VLLM model: {e}")
        raise

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

def setup_tokenizer(model_name: str):
    """
    Initialize tokenizer for the provided model name.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'  # Set left padding for decoder-only model
        return tokenizer
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        raise

def create_prompt(error_log: dict, tokenizer=None) -> str:
    """
    Create a prompt for schema generation based on the error log.
    """
    # Extract relevant information
    chat_history = error_log.get("history", [])
    question = error_log.get("question", "")
    ground_truth = error_log.get("groundtruth", "")
    mistake_agent = error_log.get("mistake_agent", "")
    mistake_step = error_log.get("mistake_step", "")
    mistake_reason = error_log.get("mistake_reason", "")

    # Format chat history
    chat_content = "\n".join([
        f"{entry.get('role', 'Unknown')}: {entry.get('content', '')}"
        for entry in chat_history
    ])

    # Original prompt (commented out for easy recovery)
    """
    prompt_text = f'''Given an error analysis from a multi-agent conversation, create a error schema to help identify similar errors in the future.

Context:
Question: {question}
Ground Truth: {ground_truth}
Error Agent: {mistake_agent}
Error Step: {mistake_step}
Error Reason: {mistake_reason}

Conversation History:
{chat_content}

Based on this error case, please create a error schema that will help identify similar errors in future conversations. The schema should include:

1. Error Pattern Recognition:
   - What are the key indicators that signal this type of error might occur?
   - What patterns in agent behavior or responses should raise concerns?

2. Critical Points Analysis:
   - What are the critical decision points where this error could have been prevented?
   - What alternative approaches could have been taken?

3. Future Detection Guidelines:
   - What specific aspects should be monitored to catch similar errors early?
   - What verification steps should be implemented?

Please format your response as a structured schema that can be applied to future cases.
'''
    """

    # New prompt focused on error identification
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
"""
    
    # If tokenizer is provided, apply chat schema
    if tokenizer:
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations and creating schemata for error detection."},
            {"role": "user", "content": prompt_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    return prompt_text

def generate_error_schemata(error_logs: list, llm: LLM, tokenizer=None, batch_size: int = 4) -> list:
    """
    Generate error schemata for multiple error logs in parallel using batched processing.
    """
    # Create all prompts first
    print("Creating prompts for all error logs...")
    prompts = [create_prompt(error_log, tokenizer) for error_log in tqdm(error_logs, desc="Creating Prompts")]
    
    # Generate schemata using VLLM in batches
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    all_schemata = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"Processing {len(prompts)} prompts in {num_batches} batches...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Schemata", total=num_batches):
        batch_prompts = prompts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(prompts))))
        
        print(f"\nBatch {i//batch_size + 1}/{num_batches}: Processing logs {batch_indices[0]+1}-{batch_indices[-1]+1}")
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Extract generated text from outputs
        batch_schemata = []
        for j, output in enumerate(outputs):
            log_num = batch_indices[j] + 1
            if output.outputs:
                print(f"  ✓ Schema generated for log {log_num}")
                batch_schemata.append(output.outputs[0].text)
            else:
                print(f"  ✗ Failed to generate schema for log {log_num}")
                batch_schemata.append("Error: Failed to generate schema")
        
        all_schemata.extend(batch_schemata)
        
    return all_schemata

def process_single_dataset(dataset_name, dataset_path, llm, tokenizer, batch_size, output_base_dir):
    """
    Process a single dataset and generate error schemata.
    """
    try:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Check if dataset_path contains 'individual_trajectories' in its path or has numbered JSON files
        if 'individual_trajectories' in dataset_path or (os.path.exists(dataset_path) and any(f.endswith('.json') and f[0].isdigit() for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)))):
            # Direct path to JSON files (either contains 'individual_trajectories' or has numbered JSON files like 1.json, 2.json)
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
        schemata = generate_error_schemata(error_logs, llm, tokenizer, batch_size=batch_size)
        
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
        
        print(f"  ✓ Schemata saved to: {output_file}")
        
    except Exception as e:
        print(f"  ✗ Error processing {dataset_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate error schemata from error logs using VLLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for schema generation"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/schemata",
        help="Base directory to save the generated schemata"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/correct_error",
        help="Path to the results directory containing all datasets"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel schema generation"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["arc", "hotpot", "musique", "wikimqa", "math500", "mmlu_pro", "gaia"],
        help="List of datasets to process (default: 7 CORRECT-Error datasets)"
    )
    parser.add_argument(
        "--single_dir",
        type=str,
        default=None,
        help="Process a single directory containing JSON files directly (e.g., /path/to/individual_trajectories)"
    )

    args = parser.parse_args()

    try:
        # Initialize tokenizer and VLLM model
        print(f"Initializing tokenizer for model: {args.model_name}")
        tokenizer = setup_tokenizer(args.model_name)
        
        print(f"Initializing VLLM model: {args.model_name}")
        llm = setup_vllm_model(args.model_name, args.tensor_parallel_size)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Check if processing a single directory
        if args.single_dir:
            # Process single directory mode
            print(f"\nProcessing single directory: {args.single_dir}")
            dataset_name = os.path.basename(os.path.dirname(args.single_dir)) if "individual_trajectories" in args.single_dir else os.path.basename(args.single_dir)
            process_single_dataset(dataset_name, args.single_dir, llm, tokenizer, args.batch_size, args.output_dir)
        else:
            # Process multiple datasets
            print(f"\nProcessing {len(args.datasets)} datasets...")
            for dataset_name in args.datasets:
                dataset_path = os.path.join(args.results_dir, dataset_name)
                if os.path.exists(dataset_path):
                    process_single_dataset(dataset_name, dataset_path, llm, tokenizer, args.batch_size, args.output_dir)
                else:
                    print(f"\n✗ Dataset directory not found: {dataset_path}")
        
        print(f"\n✓ All datasets processed. Results saved in: {args.output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
