import multiprocessing
import random
import os
import argparse
import contextlib
import sys
import datetime
import json
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from Lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search
)

from Lib.local_model import (
    analyze_all_at_once_local,
    analyze_step_by_step_local,
    analyze_binary_search_local,
    analyze_all_at_once_vllm,
    analyze_all_at_once_vllm_with_schemata,
)


KNOWN_GPT_MODELS = {"gpt-4o", "gpt4", "gpt4o-mini"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-7b-1m", "qwen-72b"}
LOCAL_QWQ_ALIASES = {"qwq-32b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES | LOCAL_QWQ_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-7b-1m": "Qwen/Qwen2.5-7B-Instruct-1M",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwq-32b": "Qwen/QwQ-32B",
}

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")

def load_error_schemata_for_dataset(dataset_name, schemata_dir="error_schemata_results"):
    """
    Load error schemata for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'assistant', 'gaia', etc.)
        schemata_dir: Base directory containing schema files
    
    Returns:
        dict: A dictionary mapping error log numbers to their error schemata
    """
    schemata_file = os.path.join(schemata_dir, dataset_name, "error_schemata.txt")
    
    if not os.path.exists(schemata_file):
        print(f"Warning: Schema file not found for {dataset_name}: {schemata_file}")
        return {}
    
    schemata = {}
    current_log_num = None
    current_schema = ""
    
    print(f"Loading schemata from {schemata_file}")
    
    try:
        with open(schemata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Start of a new schema section
                if line.startswith("=== Schema for Error Log "):
                    # Save previous schema if there was one
                    if current_log_num is not None and current_schema:
                        schemata[current_log_num] = current_schema.strip()
                    
                    # Extract new log number
                    try:
                        current_log_num = int(line.strip().split("===")[1].strip().replace("Schema for Error Log ", ""))
                        current_schema = ""
                    except:
                        print(f"Warning: Could not parse log number from: {line}")
                    
                    i += 1
                    
                    # Collect all lines until the separator
                    while i < len(lines) and not lines[i].startswith("=" * 50):
                        # Skip "Generated Schema:" line if present
                        if not lines[i].strip() == "Generated Schema:":
                            current_schema += lines[i]
                        i += 1
                else:
                    i += 1
            
            # Save the last schema
            if current_log_num is not None and current_schema:
                schemata[current_log_num] = current_schema.strip()
                
        print(f"Successfully loaded {len(schemata)} error schemata for {dataset_name}")
        return schemata
        
    except Exception as e:
        print(f"Error loading error schemata for {dataset_name}: {e}")
        return {}

def load_trajectory_similarities_for_dataset(dataset_name, similarities_dir="trajectory_similarities"):
    """
    Load trajectory similarities for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        similarities_dir: Directory containing similarity files
    
    Returns:
        dict: A dictionary mapping indices to lists of similar trajectory indices
    """
    similarities_file = os.path.join(similarities_dir, f"{dataset_name}_trajectory_similarities.json")
    
    if not os.path.exists(similarities_file):
        print(f"Warning: Similarity file not found for {dataset_name}: {similarities_file}")
        return {}
    
    try:
        with open(similarities_file, 'r', encoding='utf-8') as f:
            similarities = json.load(f)
        
        # Convert string keys to integers for easier lookup
        similarities_int = {}
        for key, value in similarities.items():
            similarities_int[int(key)] = value
            
        print(f"Successfully loaded trajectory similarities for {dataset_name}: {len(similarities_int)} entries")
        return similarities_int
    except Exception as e:
        print(f"Error loading trajectory similarities for {dataset_name}: {e}")
        return {}

def get_json_number(json_file):
    """Extract the number from a JSON filename."""
    return int(''.join(filter(str.isdigit, json_file)) or 0)

class DatasetSimilaritySchemaAnalyzer:
    """
    Class to modify prompts with error schemata based on trajectory similarities.
    Handles multiple datasets and their corresponding schemata/similarities.
    """
    def __init__(self, dataset_name, schemata_dir="error_schemata_results", 
                 similarities_dir="trajectory_similarities", use_random_fallback=False):
        self.dataset_name = dataset_name
        self.schemata = load_error_schemata_for_dataset(dataset_name, schemata_dir)
        self.similarities = load_trajectory_similarities_for_dataset(dataset_name, similarities_dir)
        self.use_random_fallback = use_random_fallback
        
        # Convert schemata dict to list for random fallback
        self.schema_list = list(self.schemata.values()) if self.schemata else []
        self.schema_keys = list(self.schemata.keys()) if self.schemata else []
        
        print(f"\nDatasetSimilaritySchemaAnalyzer initialized for {dataset_name}")
        print(f"  Schemata loaded: {len(self.schemata)}")
        print(f"  Similarities loaded: {len(self.similarities)}")
        
    def get_similarity_based_schema(self, file_num, num_schemata=1):
        """
        Get schema(s) based on trajectory similarity.
        Uses the top N similar trajectory's schemata.
        """
        schema_keys = []
        schema_contents = []
        
        # Check if we have similarity data for this file number
        if file_num in self.similarities:
            similar_indices = self.similarities[file_num]
            
            if similar_indices:
                # Try to get schemata from the most similar trajectories
                checked_count = 0
                for similar_idx in similar_indices:
                    if similar_idx in self.schemata:
                        schema_keys.append(similar_idx)
                        schema_contents.append(self.schemata[similar_idx])
                        
                        if len(schema_contents) >= num_schemata:
                            break
                    
                    checked_count += 1
                    # Stop checking after looking at many indices to avoid long search
                    if checked_count > num_schemata * 5:
                        break
        
        # Fallback to random if enabled and we don't have enough schemata
        if self.use_random_fallback and len(schema_contents) < num_schemata and self.schema_list:
            num_random_needed = num_schemata - len(schema_contents)
            
            # Get random schemata (avoiding duplicates)
            available_indices = [i for i in range(len(self.schema_list)) 
                               if self.schema_keys[i] not in schema_keys]
            
            for _ in range(min(num_random_needed, len(available_indices))):
                if available_indices:
                    idx = random.choice(available_indices)
                    available_indices.remove(idx)
                    schema_keys.append(self.schema_keys[idx])
                    schema_contents.append(self.schema_list[idx])
        
        return schema_keys, schema_contents
        
    # def modify_prompt(self, prompt, json_file, num_schemata=1):
    #     """
    #     Modify a prompt to include error schema(s) based on trajectory similarity.
    #     """
    #     file_num = get_json_number(json_file)
    #     schema_keys, schema_contents = self.get_similarity_based_schema(file_num, num_schemata)
        
    #     if schema_contents:
    #         # Combine multiple schemata if more than one
    #         if len(schema_contents) == 1:
    #             combined_schema = schema_contents[0]
    #         else:
    #             # Format multiple schemata
    #             schema_parts = []
    #             for i, (key, content) in enumerate(zip(schema_keys, schema_contents)):
    #                 schema_parts.append(f"Schema {i+1} (from log #{key}):\n{content}")
    #             combined_schema = "\n\n".join(schema_parts)
            
    #         return (
    #             f"{prompt}\n\n"
    #             f"Here {'are error schemata' if len(schema_contents) > 1 else 'is a error schema'} to help guide your analysis:\n\n"
    #             f"{combined_schema}"
    #         )
    #     else:
    #         return prompt
    def modify_prompt(self, prompt, json_file, num_schemata=1):
        """
        Natural integration version: Step counting guidance is naturally embedded with schema usage.
        Schemata and step analysis flow together as one cohesive analytical approach.

        Args:
            prompt: The original prompt
            json_file: The JSON filename
            num_schemata: Number of schemata to include (default: 1)
        """
        file_num = get_json_number(json_file)
        schema_keys, schema_contents = self.get_similarity_based_schema(file_num, num_schemata)

        if schema_contents:
            # Log which schemata were selected
            print(f"\n[SIMILARITY-BASED] File {json_file} (#{file_num}) -> Using {len(schema_contents)} schema(s) from logs: {schema_keys}")

            schema_section_parts = []

            if len(schema_contents) == 1:
                schema_content = schema_contents[0]
                schema_key = schema_keys[0]

                schema_section_parts.append(
                    f"\n\n==== ERROR SCHEMA FOR GUIDANCE ====\n"
                    f"Here is how a similar error was identified in Case #{schema_key}:\n\n"
                    f"{schema_content}\n"
                )

                schema_section_parts.append(
                    "HOW TO USE THIS REFERENCE EXAMPLE:\n"
                    "This schema demonstrates one type of error pattern for reference. To apply it to your analysis:\n\n"
                    "1. Study the ERROR PATTERN shown: What type of mistake does this example identify?\n"
                    "2. Use this as reference to analyze YOUR conversation:\n"
                    "   • Read through your conversation systematically (Step 0, Step 1, Step 2...)\n"
                    "   • At each step, ask: 'Is there an error here, and does it match this pattern or a different one?'\n"
                    "   • The error in your case may follow the same pattern or be completely different\n"
                    "3. Remember this is just a reference example:\n"
                    "   • Your error may occur at any step number\n"
                    "   • Your error may be a different type entirely\n"
                    "   • Use this schema to help you recognize what errors look like, not to assume your error matches\n"
                )
                print(f"[DEBUG] Single schema with naturally integrated step analysis")

            else:
                schema_section_parts.append(
                    f"\n\n==== ERROR SCHEMATA FOR GUIDANCE ====\n"
                    f"Here are {len(schema_contents)} examples of how similar errors were identified:\n"
                )

                for i, (key, content) in enumerate(zip(schema_keys, schema_contents), 1):
                    schema_section_parts.append(
                        f"\n--- Example {i} (Case #{key}) ---\n"
                        f"{content}\n"
                        f"--- End Example {i} ---\n"
                    )

                schema_section_parts.append(
                    "HOW TO USE THESE REFERENCE EXAMPLES:\n"
                    f"These {len(schema_contents)} examples show different error patterns for reference. For your analysis:\n\n"
                    "1. Study the various error patterns demonstrated above\n"
                    "2. Read through your conversation step by step (Step 0, Step 1, Step 2...)\n"
                    "3. At each step, check for errors - they may match one of these patterns or be different types\n"
                    "4. When you identify an error, determine if it follows a similar pattern or is a new type\n\n"
                    "Important: These are reference examples only. Your conversation may contain:\n"
                    "• The same type of error as shown in the examples\n"
                    "• A completely different type of error not shown here\n"
                    "• An error at any step number, regardless of the examples\n"
                )
                print(f"[DEBUG] Multiple schemata with naturally integrated step analysis")

            schema_section = '\n'.join(schema_section_parts)

            modified_prompt = (
                f"{prompt}"
                f"{schema_section}\n\n"
                "Now analyze your conversation using these reference examples as guidance:\n"
                "1. Examine your conversation step by step (starting from Step 0)\n"
                "2. Look for errors at each step - they may match the example patterns or be different types\n"
                "3. Identify where an error occurs and what type it is\n\n"
                "Format your response as:\n"
                "Agent Name: [agent who made the error]\n"
                "Step Number: [step where error occurred, counting from Step 0]\n"
                "Reason for Mistake: [explain the error - may match example patterns or be a different type]\n"
            )

            return modified_prompt

        else:
            print(f"\n[WARNING] No similar schema found for file {json_file} (#{file_num})")

            return (
                f"{prompt}\n\n"
                "Since no reference schema is available, analyze the conversation step by step:\n"
                "1. Read the entire conversation first to understand the flow\n"
                "2. Go through each step (Step 0, Step 1, Step 2, etc.) and evaluate:\n"
                "   • Is the information accurate?\n"
                "   • Is the reasoning sound?\n"
                "   • Does it advance toward the correct answer?\n"
                "3. Identify where the first error occurs\n\n"
                "Format your response as:\n"
                "Agent Name: [your prediction]\n"
                "Step Number: [step where error occurred, counting from Step 0]\n"
                "Reason for Mistake: [your explanation]\n"
            )

def main():
    parser = argparse.ArgumentParser(description="Run multi-agent failure attribution with similarity-based schemata")
    
    # Basic arguments
    parser.add_argument("--method", type=str, choices=["all_at_once", "step_by_step", "binary_search"], 
                       default="all_at_once", help="Analysis method to use")
    parser.add_argument("--model", type=str, default="qwen-7b", choices=ALL_MODELS, 
                       help="Model to use for analysis")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["arc", "hotpot", "musique", "wikimqa", "math500", "mmlu_pro", "gaia", "gaia_level1"],
                       help="Dataset to process")
    
    # Directory arguments
    parser.add_argument("--results_dir", type=str, default="data/correct_error",
                       help="Base directory containing all datasets")
    parser.add_argument("--schemata_dir", type=str, default="data/schemata",
                       help="Directory containing error schemata")
    parser.add_argument("--similarities_dir", type=str, default="data/similarities",
                       help="Directory containing similarity mappings")
    
    # Schema arguments
    parser.add_argument("--num_schemata", type=int, default=1,
                       help="Number of similar schemata to use per file")
    parser.add_argument("--use_random_fallback", action="store_true",
                       help="Use random schemata as fallback when similar ones not found")
    
    # Model arguments
    parser.add_argument("--use_vllm", action="store_true", 
                       help="Use vllm for local model inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="Tensor parallel size for vllm")
    parser.add_argument("--is_handcrafted", type=str_to_bool, default=True,
                       help="Whether to prefer the trajectory 'role' field for agent labels. CORRECT-Error uses role.")
    
    # Output argument
    parser.add_argument("--output_file", type=str, help="Output file path")

    # Data-parallel sharding: split the trajectory list across N workers
    parser.add_argument("--shard_id", type=int, default=0,
                       help="0-indexed shard id (when --num_shards>1)")
    parser.add_argument("--num_shards", type=int, default=1,
                       help="Total number of shards for data-parallel sharding")

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="vLLM sampling temperature (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="vLLM sampling top_p")

    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Set up paths
    input_dir = os.path.join(args.results_dir, args.dataset, "individual_trajectories")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Baseline mode: num_schemata=0 skips schema retrieval entirely (LLM-as-a-Judge).
    baseline_mode = (args.num_schemata == 0)

    if baseline_mode:
        schema_analyzer = None
        print("Baseline mode: skipping schemata retrieval (LLM-as-a-Judge)")
    else:
        schema_analyzer = DatasetSimilaritySchemaAnalyzer(
            dataset_name=args.dataset,
            schemata_dir=args.schemata_dir,
            similarities_dir=args.similarities_dir,
            use_random_fallback=args.use_random_fallback
        )
        if not schema_analyzer.schemata:
            print(f"Error: No schemata loaded for {args.dataset}", file=sys.stderr)
            sys.exit(1)
        if not schema_analyzer.similarities:
            print(f"Error: No similarities loaded for {args.dataset}", file=sys.stderr)
            sys.exit(1)
    
    # Set up output file
    if args.output_file:
        output_file = args.output_file
    else:
        model_name = args.model.replace('/', '_')
        output_file = f"outputs/{args.method}_similarity_{args.num_schemata}schemata_{model_name}_{args.dataset}.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process based on method
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print("--- Starting Analysis with Similarity-Based Schemata ---")
        print(f"Timestamp: {datetime.datetime.now()}")
        print(f"Method: {args.method}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of Schemata per File: {args.num_schemata}")
        if schema_analyzer is not None:
            print(f"Schemata Loaded: {len(schema_analyzer.schemata)}")
            print(f"Similarities Loaded: {len(schema_analyzer.similarities)}")
        else:
            print("Mode: baseline (LLM-as-a-Judge, no schemata)")
        print("-" * 50)
        
        try:
            if args.method == "all_at_once" and args.use_vllm and args.model in LOCAL_MODEL_ALIASES:
                # Use VLLM with schemata
                model_path = LOCAL_MODEL_MAP[args.model]
                
                # Prepare schemata for VLLM
                similarity_schemata = {}
                json_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f != 'file_mapping.json']
                json_files = sorted(json_files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

                # Data-parallel sharding: stride-based slice keeps load balanced
                # across workers when trajectory lengths follow a long-tail.
                if args.num_shards > 1:
                    json_files = json_files[args.shard_id::args.num_shards]
                    print(f"Shard {args.shard_id}/{args.num_shards}: {len(json_files)} trajectories")

                if baseline_mode:
                    analyze_all_at_once_vllm(
                        model_path=model_path,
                        directory_path=input_dir,
                        is_handcrafted=args.is_handcrafted,
                        tensor_parallel_size=args.tensor_parallel_size,
                        file_list=json_files,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                else:
                    for json_file in json_files:
                        file_num = get_json_number(json_file)
                        schema_keys, schema_contents = schema_analyzer.get_similarity_based_schema(file_num, args.num_schemata)
                        if schema_contents:
                            similarity_schemata[file_num] = schema_contents
                            print(f"File {file_num}: Using {len(schema_contents)} schema(s) from indices: {schema_keys}")

                    analyze_all_at_once_vllm_with_schemata(
                        model_path=model_path,
                        directory_path=input_dir,
                        is_handcrafted=args.is_handcrafted,
                        schemata=similarity_schemata,
                        tensor_parallel_size=args.tensor_parallel_size,
                        file_list=json_files,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
            else:
                print(f"Note: Only all_at_once method with VLLM is currently supported for similarity-based schemata")
                print(f"Requested method: {args.method}, use_vllm: {args.use_vllm}")
            
            print("\n" + "-" * 50)
            print("--- Analysis Complete ---")
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            sys.stdout = original_stdout
    
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()
