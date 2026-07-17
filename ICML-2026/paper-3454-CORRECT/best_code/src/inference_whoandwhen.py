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
from openai import AzureOpenAI
try:
    from google import genai as google_genai
    from google.oauth2 import service_account
    GEMINI_AVAILABLE = True
except ImportError:
    google_genai = None
    service_account = None
    GEMINI_AVAILABLE = False

from Lib.utils import (
    all_at_once as gpt_all_at_once,
    all_at_once_with_schemata as gpt_all_at_once_with_schemata,
    all_at_once_with_schemata_gemini,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search,
    make_openai_client,
)

from Lib.cloud_paper import (
    all_at_once_baseline_parallel as cloud_paper_baseline,
    analyze_all_at_once_cloud_with_schemata_parallel as cloud_paper_correct,
)

from Lib.local_model import (
    analyze_all_at_once_local,
    analyze_step_by_step_local,
    analyze_binary_search_local,
    analyze_all_at_once_vllm_with_schemata
)


KNOWN_GPT_MODELS = {
    "gpt-4o", "gpt4", "gpt4o-mini", "gpt-4o-mini",
    "gpt-5", "gpt-5-2025-08-07",
    "gpt-5-nano", "gpt-5-nano-2025-08-07",
    "gpt-5-mini", "gpt-5-mini-2025-08-07",
}
KNOWN_GEMINI_MODELS = {
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash-8b",
    "gemini-3-pro-preview", "gemini-3.1-pro-preview",
}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-7b-1m", "qwen-72b", "qwen-8b","qwen-4b","qwen-30b","qwen-80b"}
LOCAL_QWQ_ALIASES = {"qwq-32b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES | LOCAL_QWQ_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | KNOWN_GEMINI_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "/models/Qwen/Qwen2.5-7B-Instruct",      # Original model
    "qwen-7b-1m": "Qwen/Qwen2.5-7B-Instruct-1M", # 1M variant
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwq-32b": "Qwen/QwQ-32B",
    "qwen-8b": "Qwen/Qwen3-8B-Instruct",
    "qwen-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen-30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen-80b": "Qwen/Qwen3-Next-80B-A3B-Instruct"
}

def load_error_schemata(schemata_file):
    """
    Load error schemata from the generated file.
    Only includes the schema content, not metadata like question or error details.
    
    Returns:
        dict: A dictionary mapping error log numbers to their error schemata
    """
    schemata = {}
    current_log_num = None
    current_schema = None
    line_count = 0
    debug_info = []
    
    print(f"Starting to load schemata from {schemata_file}")
    
    try:
        with open(schemata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines from schema file")
            
            i = 0
            while i < len(lines):
                line = lines[i]
                line_count += 1
                
                # Start of a new schema section. Accept the legacy
                # "Template" marker so the cleaned repo can reproduce the
                # original Who&When cache/results without conversion.
                is_schema_marker = line.startswith("=== Schema for Error Log ")
                is_template_marker = line.startswith("=== Template for Error Log ")
                if is_schema_marker or is_template_marker:
                    # Save previous schema if there was one
                    if current_log_num is not None and current_schema:
                        debug_info.append(f"Saving schema for log {current_log_num}, length: {len(current_schema)}")
                        # Only save if not already in schemata (prevents duplicates)
                        if current_log_num not in schemata:
                            schemata[current_log_num] = current_schema.strip()
                    
                    # Extract new log number
                    marker_text = line.strip().split("===")[1].strip()
                    marker_text = marker_text.replace("Schema for Error Log ", "")
                    marker_text = marker_text.replace("Template for Error Log ", "")
                    current_log_num = int(marker_text)
                    debug_info.append(f"Found log marker at line {i}: {line.strip()}")
                    debug_info.append(f"New log number: {current_log_num}")
                    
                    # Start collecting schema content from the next line
                    current_schema = ""
                    i += 1  # Move to next line after marker
                    
                    # Collect all lines until the separator
                    while i < len(lines) and not lines[i].startswith("=" * 50):
                        # Skip generated-content header lines if present.
                        if lines[i].strip() not in {"Generated Schema:", "Generated Template:"}:
                            current_schema += lines[i]
                        i += 1
                    
                    # When we reach the separator, save this schema
                    if i < len(lines) and lines[i].startswith("=" * 50):
                        debug_info.append(f"Found end marker at line {i}")
                        debug_info.append(f"Collected schema for log {current_log_num}, length: {len(current_schema)}")
                        # Only save if not already in schemata (prevents duplicates)
                        if current_log_num not in schemata:
                            schemata[current_log_num] = current_schema.strip()
                else:
                    i += 1  # Move to next line if not a schema marker
            # Save the last schema if we ended without a separator
            if current_log_num is not None and current_schema and current_log_num not in schemata:
                debug_info.append(f"Saving final schema for log {current_log_num}, length: {len(current_schema)}")
                schemata[current_log_num] = current_schema.strip()
                
        print(f"Successfully loaded {len(schemata)} error schemata")
        
        if len(schemata) == 0:
            print("Debug information for schema loading:")
            for info in debug_info:
                print(f"  {info}")
                
            print("\nFirst few lines of schema file:")
            for i, line in enumerate(lines[:10]):
                print(f"  Line {i}: {line.strip()}")
            
        return schemata
    except Exception as e:
        print(f"Error loading error schemata: {e}")
        print("Debug information for schema loading:")
        for info in debug_info:
            print(f"  {info}")
        return {}

def load_trajectory_similarities(similarities_file):
    """
    Load trajectory similarities from JSON file.
    
    Returns:
        dict: A dictionary mapping indices to lists of similar trajectory indices
    """
    try:
        with open(similarities_file, 'r', encoding='utf-8') as f:
            similarities = json.load(f)
        
        # Convert string keys to integers for easier lookup
        similarities_int = {}
        for key, value in similarities.items():
            similarities_int[int(key)] = value
            
        print(f"Successfully loaded trajectory similarities for {len(similarities_int)} indices")
        return similarities_int
    except Exception as e:
        print(f"Error loading trajectory similarities: {e}")
        return {}

def get_json_number(json_file):
    """
    Extract the number from a JSON filename.
    """
    return int(''.join(filter(str.isdigit, json_file)) or 0)

class SimilarityBasedSchemaAnalyzer:
    """
    Class to modify prompts with error schemata based on trajectory similarities
    """
    def __init__(self, schemata, similarities, use_random_fallback=False):
        self.schemata = schemata
        self.similarities = similarities
        self.use_random_fallback = use_random_fallback
        
        # Convert schemata dict to list for random fallback
        self.schema_list = list(schemata.values()) if schemata else []
        self.schema_keys = list(schemata.keys()) if schemata else []
        
        print(f"SimilarityBasedSchemaAnalyzer initialized with {len(self.schema_list)} schemata")
        print(f"Trajectory similarities loaded for {len(self.similarities)} indices")
        
        # Print detailed mapping information
        print("\n=== SIMILARITY-BASED SCHEMA MAPPING DEBUG INFO ===")
        print(f"Available schema indices: {sorted(self.schemata.keys())}")
        print(f"\nFirst 10 similarity mappings:")
        for idx, (key, similar_list) in enumerate(sorted(self.similarities.items())[:10]):
            if similar_list:
                top_1 = similar_list[0]
                has_schema = top_1 in self.schemata
                print(f"  Index {key}: Top-1 similar = {top_1} (schema {'EXISTS' if has_schema else 'MISSING'})")
            else:
                print(f"  Index {key}: No similar indices")
        print("=== END DEBUG INFO ===\n")
        
    def get_similarity_based_schema(self, file_num, num_schemata=1):
        """
        Get schema(s) based on trajectory similarity.
        Uses the top N similar trajectory's schemata.
        
        Args:
            file_num: The file number to get schemata for
            num_schemata: Number of schemata to retrieve (default: 1)
        
        Returns tuple of (schema_keys, schema_contents) for debugging
        """
        # Debug print
        print(f"\n[DEBUG] Looking up {num_schemata} schema(s) for file #{file_num}")
        
        schema_keys = []
        schema_contents = []
        
        # Check if we have similarity data for this file number
        if file_num in self.similarities:
            similar_indices = self.similarities[file_num]
            print(f"[DEBUG] Similar indices for #{file_num}: {similar_indices[:min(10, len(similar_indices))]}...")  # Show first 10
            
            if similar_indices:
                # Get the top N similar indices
                for i, similar_idx in enumerate(similar_indices[:num_schemata]):
                    print(f"[DEBUG] Top-{i+1} similar index: {similar_idx}")
                    
                    # Check if we have a schema for this similar index
                    if similar_idx in self.schemata:
                        print(f"[DEBUG] Schema found for index {similar_idx}")
                        schema_keys.append(similar_idx)
                        schema_contents.append(self.schemata[similar_idx])
                    else:
                        print(f"[WARNING] No schema found for similar index {similar_idx}")
                        
                if len(schema_contents) < num_schemata:
                    print(f"[WARNING] Only found {len(schema_contents)} schemata out of {num_schemata} requested")
        else:
            print(f"[WARNING] No similarity data found for file #{file_num}")
            print(f"[DEBUG] Available similarity indices: {sorted(self.similarities.keys())[:10]}...")
        
        # Fallback to random if enabled and we don't have enough schemata
        if self.use_random_fallback and len(schema_contents) < num_schemata and self.schema_list:
            num_random_needed = num_schemata - len(schema_contents)
            print(f"[DEBUG] Using random fallback for {num_random_needed} schemata")
            
            # Get random schemata (avoiding duplicates)
            available_indices = [i for i in range(len(self.schema_list)) 
                               if self.schema_keys[i] not in schema_keys]
            
            for _ in range(min(num_random_needed, len(available_indices))):
                idx = random.choice(available_indices)
                available_indices.remove(idx)
                schema_keys.append(self.schema_keys[idx])
                schema_contents.append(self.schema_list[idx])
                print(f"[DEBUG] Added random schema from index: {self.schema_keys[idx]}")
        
        if not schema_contents:
            print(f"[DEBUG] No schemata selected for file #{file_num}")
            return [], []
            
        return schema_keys, schema_contents
        
    def modify_prompt(self, prompt, json_file, num_schemata=1):
        """
        Modify a prompt to include error schema(s) based on trajectory similarity.

        This mirrors the paper-aligned prompt framing in the original
        `inference_with_selected_templates_cloud.py`: retrieved examples are
        references for recognizing error patterns, not copied step labels.
        
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

                # Prompt strings use "error schema" wording (paper's final
                # terminology) while keeping the Unicode "•" bullets that
                # match the original paper code's bullet character.
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
                    "When applying these schemata:\n"
                    "• Look for common error patterns across these examples\n"
                    "• Each example shows different step numbers - focus on the ERROR TYPE, not the step position\n"
                    "• Systematically check each step in your conversation (starting from Step 0)\n"
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

            return (
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
        else:
            print(f"\n[WARNING] No similar schema found for file {json_file} (#{file_num})")
            return (
                f"{prompt}\n\n"
                "Since no reference schema is available, analyze the conversation step by step:\n"
                "1. Read the entire conversation first to understand the flow\n"
                "2. Go through each step (Step 0, Step 1, Step 2, etc.) and evaluate:\n"
                "   - Is the information accurate?\n"
                "   - Is the reasoning sound?\n"
                "   - Does it advance toward the correct answer?\n"
                "3. Identify where the first error occurs\n\n"
                "Format your response as:\n"
                "Agent Name: [your prediction]\n"
                "Step Number: [step where error occurred, counting from Step 0]\n"
                "Reason for Mistake: [your explanation]\n"
            )

def modify_analyze_all_at_once_local(schema_analyzer, original_func):
    """
    Create a modified version of analyze_all_at_once_local that incorporates similarity-based error schemata.
    """
    def modified_func(model_obj, directory_path, is_handcrafted, model_family, **kwargs):
        # Check if using vllm
        use_vllm = kwargs.get('use_vllm', False)
        model_path = kwargs.get('model_path', None)
        tensor_parallel_size = kwargs.get('tensor_parallel_size', 8)
        num_schemata = kwargs.get('num_schemata', 1)  # Get number of schemata from kwargs
        
        # Use vllm with schemata if specified and supported for this model family
        if use_vllm and model_path and model_family in ['qwen', 'qwq']:
            print(f"Using vllm with similarity-based schemata for {model_family} model")
            # Create a schemata dict with similarity-based mapping
            similarity_schemata = {}
            json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
            for json_file in json_files:
                file_num = get_json_number(json_file)
                schema_keys, schema_contents = schema_analyzer.get_similarity_based_schema(file_num, num_schemata)
                if schema_contents:
                    # Pass schemata as list for vllm
                    similarity_schemata[file_num] = schema_contents
                    
            return analyze_all_at_once_vllm_with_schemata(
                model_path=model_path,
                directory_path=directory_path,
                is_handcrafted=is_handcrafted,
                schemata=similarity_schemata,
                tensor_parallel_size=tensor_parallel_size
            )
        
        # Otherwise use the standard HuggingFace implementation with schemata
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        index_agent = "role" if is_handcrafted else "name"
        
        print(f"\n--- Starting Similarity-Based Schema-Enhanced Local All-at-Once Analysis ({model_family}) ---")
        print(f"Number of JSON files to process: {len(json_files)}")
        print(f"Number of schemata loaded: {len(schema_analyzer.schemata)}")
        print(f"Number of schemata per file: {num_schemata}")
        print(f"Schema numbers available: {sorted(schema_analyzer.schemata.keys())}")

        # Create a record of which schema was used for each file
        schema_usage = {}

        for json_file in tqdm(json_files, desc=f"All-at-Once with Similarity-Based Schemata ({model_family})"):
            file_path = os.path.join(directory_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            chat_history = data.get("history", [])
            problem = data.get("question", "")
            ground_truth = data.get("ground_truth", "")

            if not chat_history:
                continue

            chat_content = "\n".join([
            f"Step {idx}\n\n\n{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
            for idx, entry in enumerate(chat_history)
            ])

            # Original prompt
            original_prompt = (
                "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
                f"The problem is:  {problem} \n"
                f"The Answer for the problem is: {ground_truth}\n"
                "Identify which agent made an error, at which step, and explain the reason for the error. "
                "Here's the conversation:\n\n" + chat_content +
                "\n\nBased on this conversation, please predict the following:\n"
                "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
                "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
                '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
                "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
                "3. The reason for your prediction."
                "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
            )
            
            # Modify prompt to include similarity-based error schemata
            prompt = schema_analyzer.modify_prompt(original_prompt, json_file, num_schemata)
            file_num = get_json_number(json_file)
            
            # Track schema usage
            schema_keys, _ = schema_analyzer.get_similarity_based_schema(file_num, num_schemata)
            if schema_keys:
                schema_usage[json_file] = schema_keys
            
            # Create system prompt and messages
            system_prompt = "You are a helpful assistant skilled in analyzing conversations."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            # Use the original function's _run_local_generation
            from Lib.local_model import _run_local_generation
            assistant_response = _run_local_generation(model_obj, messages, model_family)

            print(f"Prediction for {json_file}:")
            if assistant_response:
                print(assistant_response)
            else:
                print("Failed to get prediction from local model.")
            print("\n" + "="*50 + "\n")
        
        # Print summary of schema usage
        print("\n=== SIMILARITY-BASED SCHEMA USAGE SUMMARY ===")
        print(f"Total files processed: {len(json_files)}")
        print(f"Files with schemata applied: {len(schema_usage)}")
        print(f"\nDetailed mapping:")
        for json_file, schema_keys in sorted(schema_usage.items()):
            file_num = get_json_number(json_file)
            print(f"  File {json_file} (#{file_num}) -> Schemata from logs: {schema_keys}")
        
        # Print statistics
        print(f"\n=== SCHEMA USAGE STATISTICS ===")
        schema_counts = {}
        for schema_keys_list in schema_usage.values():
            for schema_key in schema_keys_list:
                schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1
        
        print(f"Unique schemata used: {len(schema_counts)}")
        print(f"Most frequently used schemata:")
        for schema_key, count in sorted(schema_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Schema #{schema_key}: used {count} times")
        print("=================================\n")
            
    return modified_func

def modify_analyze_step_by_step_local(schema_analyzer, original_func):
    """
    Create a modified version of analyze_step_by_step_local that incorporates similarity-based error schemata.
    """
    def modified_func(model_obj, directory_path, is_handcrafted, model_family, **kwargs):
        num_schemata = kwargs.get('num_schemata', 1)  # Get number of schemata from kwargs
        
        print(f"\n--- Starting Similarity-Based Schema-Enhanced Local Step-by-Step Analysis ({model_family}) ---")
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        index_agent = "role" if is_handcrafted else "name"
        
        print(f"Number of JSON files to process: {len(json_files)}")
        print(f"Number of schemata per file: {num_schemata}")

        schema_usage = {}

        for json_file in tqdm(json_files, desc=f"Step-by-Step with Similarity-Based Schemata ({model_family})"):
            file_path = os.path.join(directory_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            chat_history = data.get("history", [])
            problem = data.get("question", "")
            ground_truth = data.get("ground_truth", "")

            if not chat_history:
                continue
                
            # Get similarity-based error schemata for this file
            file_num = get_json_number(json_file)
            schema_keys, schema_contents = schema_analyzer.get_similarity_based_schema(file_num, num_schemata)
            
            error_schema = ""
            if schema_contents:
                schema_usage[json_file] = schema_keys
                
                # Format multiple schemata if more than one
                if len(schema_contents) == 1:
                    error_schema = (
                        f"\n\nHere's a error schema to help guide your analysis:\n\n"
                        f"{schema_contents[0]}"
                    )
                else:
                    schema_parts = []
                    for i, (key, content) in enumerate(zip(schema_keys, schema_contents)):
                        schema_parts.append(f"Schema {i+1} (from log #{key}):\n{content}")
                    combined_schema = "\n\n".join(schema_parts)
                    error_schema = (
                        f"\n\nHere are error schemata to help guide your analysis:\n\n"
                        f"{combined_schema}"
                    )

            current_conversation_history = ""
            error_found = False
            for idx, entry in enumerate(chat_history):
                agent_name = entry.get(index_agent, 'Unknown Agent')
                content = entry.get('content', '')
                current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n"
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                    f"{error_schema}"
                )

                system_prompt = "You are a helpful assistant skilled in analyzing conversations."

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                from Lib.local_model import _run_local_generation
                answer = _run_local_generation(model_obj, messages, model_family)

                if not answer:
                    print("Failed to get evaluation for this step from local model. Stopping analysis for this file.")
                    error_found = True
                    break

                if answer.lower().strip().startswith("1. yes"):
                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {agent_name}")
                    print(f"Step Number: {idx}")
                    try:
                        reason = answer.split('Reason:', 1)[-1].strip()
                    except:
                        reason = "[Could not extract reason]"
                    print(f"Reason provided by LLM: {reason}")
                    error_found = True
                    break
                elif answer.lower().strip().startswith("1. no"):
                    pass
                else:
                    print(f"Warning: Unexpected response format from local LLM for step {idx} in {json_file}. Response: {answer[:100]}...")

            if not error_found:
                print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

            print("\n" + "="*50 + "\n")
        
        # Print summary of schema usage
        print("\n=== SIMILARITY-BASED SCHEMA USAGE SUMMARY ===")
        print(f"Total files processed: {len(json_files)}")
        print(f"Files with schemata applied: {len(schema_usage)}")
        print(f"\nDetailed mapping:")
        for json_file, schema_keys in sorted(schema_usage.items()):
            file_num = get_json_number(json_file)
            print(f"  File {json_file} (#{file_num}) -> Schemata from logs: {schema_keys}")
            
    return modified_func

def modify_analyze_binary_search_local(schema_analyzer, original_func):
    """
    Create a modified version of analyze_binary_search_local that incorporates similarity-based error schemata.
    """
    from Lib.local_model import _construct_binary_search_prompt_local, _find_error_in_segment_local
    
    # Get num_schemata from outer scope
    num_schemata_global = 1
    
    # Track schema usage across all files
    schema_usage = {}
    
    def modified_find_error_in_segment(model_obj, chat_history, problem, answer, start, end, json_file, is_handcrafted, model_family):
        if start > end:
            from Lib.local_model import _report_binary_search_error_local
            print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
            _report_binary_search_error_local(chat_history, end if end >= 0 else 0, json_file, is_handcrafted)
            return
        if start == end:
            from Lib.local_model import _report_binary_search_error_local
            _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
            return

        index_agent = "role" if is_handcrafted else "name"

        segment_history = chat_history[start : end + 1]
        if not segment_history:
            from Lib.local_model import _report_binary_search_error_local
            print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Reporting start index.")
            _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
            return

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
            for entry in segment_history
        ])

        mid = start + (end - start) // 2

        range_description = f"from step {start} to step {end}"
        upper_half_desc = f"from step {start} to step {mid}"
        lower_half_desc = f"from step {mid + 1} to step {end}"

        original_prompt = _construct_binary_search_prompt_local(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)
        
        # Add similarity-based error schemata
        file_num = get_json_number(json_file)
        prompt = original_prompt
        
        # Get similarity-based schemata - use same schemata for entire binary search of one file
        if json_file not in schema_usage:
            schema_keys, schema_contents = schema_analyzer.get_similarity_based_schema(file_num, num_schemata_global)
            if schema_contents:
                # Format and store schemata
                if len(schema_contents) == 1:
                    combined_schema = schema_contents[0]
                else:
                    schema_parts = []
                    for i, (key, content) in enumerate(zip(schema_keys, schema_contents)):
                        schema_parts.append(f"Schema {i+1} (from log #{key}):\n{content}")
                    combined_schema = "\n\n".join(schema_parts)
                
                schema_usage[json_file] = (schema_keys, combined_schema)
                print(f"\n[SIMILARITY] File {json_file} (#{file_num}) is using schemata from logs: {schema_keys}")
        
        if json_file in schema_usage:
            schema_keys, combined_schema = schema_usage[json_file]
            prompt = (
                f"{original_prompt}\n\n"
                f"Here {'are error schemata' if len(schema_keys) > 1 else 'is a error schema'} to help guide your analysis:\n\n"
                f"{combined_schema}"
            )

        system_prompt = "You are a helpful assistant skilled in analyzing conversations."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        from Lib.local_model import _run_local_generation
        result = _run_local_generation(model_obj, messages, model_family)

        if not result:
            print(f"Model call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
            return

        result_lower = result.lower().strip()

        if "upper half" in result_lower:
            modified_find_error_in_segment(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)
        elif "lower half" in result_lower:
            new_start = min(mid + 1, end)
            modified_find_error_in_segment(model_obj, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, model_family)
        else:
            print(f"Warning: Ambiguous response '{result}' from local LLM for segment {start}-{end}. Defaulting to upper half.")
            modified_find_error_in_segment(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)
    
    def modified_binary_search(model_obj, directory_path, is_handcrafted, model_family, **kwargs):
        nonlocal num_schemata_global
        num_schemata_global = kwargs.get('num_schemata', 1)
        
        print(f"\n--- Starting Similarity-Based Schema-Enhanced Local Binary Search Analysis ({model_family}) ---")
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        json_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        
        print(f"Number of JSON files to process: {len(json_files)}")
        print(f"Number of schemata per file: {num_schemata_global}")

        for json_file in tqdm(json_files, desc=f"Binary Search with Similarity-Based Schemata ({model_family})"):
            file_path = os.path.join(directory_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            chat_history = data.get("history", [])
            problem = data.get("question", "")
            answer = data.get("ground_truth", "")

            if not chat_history:
                continue

            modified_find_error_in_segment(
                model_obj=model_obj,
                chat_history=chat_history,
                problem=problem,
                answer=answer,
                start=0,
                end=len(chat_history) - 1,
                json_file=json_file,
                is_handcrafted=is_handcrafted,
                model_family=model_family
            )
        
        # Print summary of schema usage
        print("\n=== SIMILARITY-BASED SCHEMA USAGE SUMMARY ===")
        print(f"Total files processed: {len(json_files)}")
        print(f"Files with schemata applied: {len(schema_usage)}")
        print(f"\nDetailed mapping:")
        for json_file, (schema_keys, _) in sorted(schema_usage.items()):
            file_num = get_json_number(json_file)
            print(f"  File {json_file} (#{file_num}) -> Schemata from logs: {schema_keys}")
            
    return modified_binary_search

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent chat history using specific models with error schemata.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default="data/whoandwhen/Algorithm-Generated",
        help="Path to the Who&When subset directory (Algorithm-Generated or Hand-Crafted)."
    )
    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], # If you want to test Hand-Crafted, set is_handcrafted to be True.
        help="Specify 'True' or 'False'. Default: 'False'."
    )
    parser.add_argument(
        "--schemata_file",
        type=str,
        default="data/schemata_whoandwhen/Algorithm-Generated/error_schemata.txt",
        help="Path to the file containing error schemata generated by "
             "error_schema_generator_cloud.py. Shipped schemata are GPT-5 "
             "generated (paper §A.3) at data/schemata_whoandwhen/<subset>/error_schemata.txt."
    )
    parser.add_argument(
        "--schema_selection",
        type=str,
        default="similarity",
        choices=["similarity"],
        help="Schema selection method. This release supports 'similarity' (the paper method)."
    )
    parser.add_argument(
        "--similarities_file",
        type=str,
        default="data/similarities_whoandwhen/trajectory_similarities.json",
        help="Path to the JSON file containing trajectory similarities (required for similarity-based selection)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenAI or Azure OpenAI API key. If unset, reads OPENAI_API_KEY or AZURE_OPENAI_API_KEY from the env."
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default=None,
        help="Azure OpenAI endpoint URL. If unset, falls back to standard OpenAI (or to AZURE_OPENAI_ENDPOINT env)."
    )
    parser.add_argument(
        "--api_version", type=str, default="2024-08-01-preview",
        help="Azure OpenAI API Version. Used only for GPT models."
    )
    parser.add_argument(
        "--gemini_api_key", type=str, default=None,
        help="(deprecated) Legacy AI Studio API key. The Vertex AI path uses "
             "a service-account JSON instead — see --google_credentials."
    )
    parser.add_argument(
        "--google_credentials", type=str, default=None,
        help="Path to a Google Cloud service-account JSON for Vertex AI. "
             "Falls back to the GOOGLE_APPLICATION_CREDENTIALS env var."
    )
    parser.add_argument(
        "--google_project", type=str, default=None,
        help="Google Cloud project for Vertex AI. "
             "Falls back to the GOOGLE_CLOUD_PROJECT env var."
    )
    parser.add_argument(
        "--google_location", type=str, default=None,
        help="Vertex AI location (default: GOOGLE_CLOUD_LOCATION env or 'us-central1')."
    )
    parser.add_argument(
        "--cloud_paper_path", action="store_true",
        help="Route cloud (gpt/gemini) inference through Lib/cloud_paper.py — "
             "byte-identical to paper Sep 2025 utils_cloud_parallel.py "
             "(baseline) and cloud_model_parallel.py (CORRECT)."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to write the prediction output into. Overrides the "
             "built-in default. The filename is derived from method/model/schema-count."
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Full path of the output file. Overrides both --output_dir and the "
             "derived filename. Use this when a runner script controls output locations."
    )
    parser.add_argument(
        "--batch_size", type=int, default=30,
        help="Batch size for the cloud_paper_path. Paper used 30 for baseline, 40-60 for CORRECT."
    )
    parser.add_argument(
        "--max_workers", type=int, default=15,
        help="Max concurrent workers for cloud_paper_path. Paper used 15 for baseline, 20-30 for CORRECT."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response. Used only for GPT models."
    )
    parser.add_argument(
        "--use_parallel", action="store_true",
        help="Enable parallel processing for local model inference (currently only supported for Qwen models)"
    )
    parser.add_argument(
        "--use_vllm", action="store_true",
        help="Use vllm for parallel inference (recommended for better throughput)"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=8,
        help="Number of GPUs to use for tensor parallelism in vllm"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu",
        help="Device for local model inference (e.g., 'cuda', 'cuda:0', 'cpu'). Default: 'cuda' if available, else 'cpu'."
    )
    parser.add_argument(
        "--random_seed", type=int, default=None,
        help="Random seed for reproducible random schema selection (only used when schema_selection='random')"
    )
    parser.add_argument(
        "--use_random_fallback", action="store_true",
        help="Use random schema selection as fallback when similarity-based selection fails"
    )
    parser.add_argument(
        "--num_schemata", type=int, default=1,
        help="Number of schemata to use per file (for similarity-based selection)"
    )

    args = parser.parse_args()

    # Set random seed if provided (used for similarity-fallback sampling)
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")

    client_or_model_obj = None
    model_type = None # gpt, llama, qwen
    model_family = None 
    model_id_or_deployment = args.model

    if args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
        model_family = 'gpt'
        print(f"Selected GPT model: {args.model}")

        try:
            from Lib.utils import make_openai_client
            client_or_model_obj = make_openai_client(
                api_key=args.api_key,
                azure_endpoint=args.azure_endpoint,
                api_version=args.api_version,
            )
            kind = "AzureOpenAI" if args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") else "OpenAI"
            print(f"Successfully initialized {kind} client.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            sys.exit(1)

    elif args.model in KNOWN_GEMINI_MODELS:
        model_type = 'gemini'
        model_family = 'gemini'
        print(f"Selected Gemini model: {args.model}")

        if not GEMINI_AVAILABLE:
            print("Error: google-genai is not installed (pip install google-genai).")
            sys.exit(1)

        # Vertex AI via service account JSON. Path comes from
        # --google_credentials or the GOOGLE_APPLICATION_CREDENTIALS env var.
        cred_path = args.google_credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path or not os.path.exists(cred_path):
            print("Error: Vertex AI service-account JSON not found. "
                  "Pass --google_credentials or set GOOGLE_APPLICATION_CREDENTIALS "
                  "to the path of your service-account key.")
            sys.exit(1)
        project = args.google_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            print("Error: Google Cloud project not set. "
                  "Pass --google_project or set GOOGLE_CLOUD_PROJECT.")
            sys.exit(1)
        location = args.google_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        try:
            creds = service_account.Credentials.from_service_account_file(
                cred_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client_or_model_obj = google_genai.Client(
                vertexai=True, project=project, location=location, credentials=creds,
            )
            print(f"Successfully initialized Vertex AI Gemini client (project={project}, location={location}).")
        except Exception as e:
            print(f"Error initializing Vertex AI Gemini client: {e}")
            sys.exit(1)

    elif args.model in LOCAL_MODEL_ALIASES:
        model_type = 'local'
        model_id_or_deployment = LOCAL_MODEL_MAP[args.model]

        if args.model in LOCAL_LLAMA_ALIASES:
            model_family = 'llama'
            print(f"Selected local Llama model: {args.model} ({model_id_or_deployment}) on device {args.device}")
            if not pipeline:
                 print("Error: transformers library not found or pipeline could not be imported.")
                 sys.exit(1)
            try:
                 print(f"Initializing Llama pipeline for {model_id_or_deployment}...")
                 client_or_model_obj = pipeline(
                     "text-generation",
                     model=model_id_or_deployment,
                     model_kwargs={"torch_dtype": torch.bfloat16},
                     device=args.device,
                 )
                 print(f"Successfully initialized Llama pipeline on {args.device}.")
            except Exception as e:
                print(f"Error initializing Llama pipeline for {model_id_or_deployment}: {e}")
                sys.exit(1)

        elif args.model in LOCAL_QWEN_ALIASES:
            model_family = 'qwen'
            print(f"Selected local Qwen model: {args.model} ({model_id_or_deployment})")
            
            if args.use_vllm:
                # When using vllm, we don't need to load the model here
                client_or_model_obj = None
                print(f"Using vllm for inference with tensor parallel size {args.tensor_parallel_size}")
            else:
                # Only load with HuggingFace if not using vllm
                if not AutoModelForCausalLM or not AutoTokenizer:
                    print("Error: transformers library not found or specific classes could not be imported.")
                    sys.exit(1)
                try:
                    print(f"Initializing Qwen model and tokenizer for {model_id_or_deployment}...")
                    qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_id_or_deployment,
                        torch_dtype="auto",
                        device_map=args.device # Use device_map for potentially large models
                    )
                    qwen_tokenizer = AutoTokenizer.from_pretrained(model_id_or_deployment)
                    qwen_tokenizer.padding_side = 'left'  # Set left padding for decoder-only model
                    client_or_model_obj = (qwen_model, qwen_tokenizer) # Store as tuple
                    print(f"Successfully initialized Qwen model and tokenizer on {args.device}.")
                except Exception as e:
                    print(f"Error initializing Qwen model/tokenizer for {model_id_or_deployment}: {e}")
                    print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
                    sys.exit(1)
        elif args.model in LOCAL_QWQ_ALIASES:
            model_family = 'qwq'
            print(f"Selected local QwQ model: {args.model} ({model_id_or_deployment})")
            
            if args.use_vllm:
                # When using vllm, we don't need to load the model here
                client_or_model_obj = None
                print(f"Using vllm for inference with tensor parallel size {args.tensor_parallel_size}")
            else:
                # Only load with HuggingFace if not using vllm
                if not AutoModelForCausalLM or not AutoTokenizer:
                    print("Error: transformers library not found or specific classes could not be imported.")
                    sys.exit(1)
                try:
                    print(f"Initializing QwQ model and tokenizer for {model_id_or_deployment}...")
                    qwq_model = AutoModelForCausalLM.from_pretrained(
                        model_id_or_deployment,
                        torch_dtype="auto",
                        device_map=args.device # Use device_map for potentially large models
                    )
                    qwq_tokenizer = AutoTokenizer.from_pretrained(model_id_or_deployment)
                    qwq_tokenizer.padding_side = 'left'  # Set left padding for decoder-only model
                    client_or_model_obj = (qwq_model, qwq_tokenizer) # Store as tuple
                    print(f"Successfully initialized QwQ model and tokenizer on {args.device}.")
                except Exception as e:
                    print(f"Error initializing QwQ model/tokenizer for {model_id_or_deployment}: {e}")
                    print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
                    sys.exit(1)
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)

    # Adjust directory path if needed based on environment
    if args.is_handcrafted == "True":
        if args.directory_path == "../Who&When/Algorithm-Generated":
            args.directory_path = "../Who&When/Hand-Crafted"
            print(f"Using Hand-Crafted directory: {args.directory_path}")
            
    # Resolve the directory path 
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.directory_path))
    print(f"Resolved directory path: {directory_path}")
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        # Try falling back to a relative path from current working directory
        directory_path = args.directory_path
        print(f"Falling back to: {directory_path}")
        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} does not exist")
            sys.exit(1)
    
    # Count JSON files in directory
    json_count = len([f for f in os.listdir(directory_path) if f.endswith('.json')])
    print(f"Found {json_count} JSON files in {directory_path}")
    
    # Load error schemata
    print(f"Loading error schemata from {args.schemata_file}")
    schemata = load_error_schemata(args.schemata_file)

    # Create appropriate schema analyzer based on selection method
    if args.schema_selection == "similarity":
        # Load trajectory similarities
        print(f"Loading trajectory similarities from {args.similarities_file}")
        similarities = load_trajectory_similarities(args.similarities_file)
        if not similarities:
            print("Error: Failed to load trajectory similarities. Cannot proceed with similarity-based selection.")
            sys.exit(1)
        
        # Print mapping validation info
        print("\n=== SCHEMA MAPPING VALIDATION ===")
        print(f"Total schemata available: {len(schemata)}")
        print(f"Total similarity mappings: {len(similarities)}")
        
        # Check coverage
        missing_schemata = []
        for idx, similar_list in similarities.items():
            if similar_list and similar_list[0] not in schemata:
                missing_schemata.append((idx, similar_list[0]))
        
        if missing_schemata:
            print(f"\nWARNING: {len(missing_schemata)} files will have missing schemata:")
            for idx, top_similar in missing_schemata[:10]:  # Show first 10
                print(f"  File #{idx} needs schema #{top_similar} (not available)")
            if len(missing_schemata) > 10:
                print(f"  ... and {len(missing_schemata) - 10} more")
        else:
            print("\nAll similarity mappings have corresponding schemata available!")
        
        print("===================================\n")
        
        schema_analyzer = SimilarityBasedSchemaAnalyzer(schemata, similarities, args.use_random_fallback)

    output_dir = args.output_dir or "outputs_cloud_schemata_no_step_label_prompt_qwen72b_schema"
    handcrafted_suffix = "_handcrafted" if args.is_handcrafted == "True" else "_alg_generated"

    # Output filename suffix (similarity-based selection)
    schema_suffix = f"_similarity_based_{args.num_schemata}schemata"

    output_filename = f"{args.method}_with{schema_suffix}_schemata_{args.model.replace('/','_')}{handcrafted_suffix}.txt"

    # --output_file fully overrides path; otherwise place derived filename in output_dir.
    if args.output_file:
        output_filepath = args.output_file
        parent = os.path.dirname(output_filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, output_filename)

    print(f"Analysis method: {args.method}")
    print(f"Model Alias: {args.model} (Family: {model_family})")
    print(f"Output will be saved to: {output_filepath}")
    print(f"Schema selection method: {args.schema_selection}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis with {args.schema_selection.upper()} Error Schemata: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Family: {model_family}")
            print(f"Model Used: {model_id_or_deployment}")
            print(f"Input Directory: {args.directory_path}")
            print(f"Is Handcrafted: {args.is_handcrafted}")
            print(f"Schemata File: {args.schemata_file}")
            print(f"Number of Schemata Loaded: {len(schemata)}")
            print(f"Schema Selection Method: {args.schema_selection}")
            print(f"Number of Schemata per File: {args.num_schemata}")
            if args.schema_selection == "similarity":
                print(f"Similarities File: {args.similarities_file}")
                print(f"Number of Similarity Mappings: {len(similarities) if 'similarities' in locals() else 0}")
                print(f"Use Random Fallback: {args.use_random_fallback}")
            elif args.schema_selection == "random":
                print(f"Random Seed: {args.random_seed if args.random_seed is not None else 'Timestamp-based'}")
            print("-" * 20)

            if model_type == 'gpt':
                if args.method == "all_at_once":
                    if args.cloud_paper_path:
                        # Paper byte-identical cloud path (utils_cloud_parallel.py
                        # baseline / cloud_model_parallel.py CORRECT).
                        # Legacy cloud/template drivers passed the raw
                        # CLI string into helper functions. That means
                        # "False" is truthy, so Algorithm-Generated prompts
                        # use `role` (user/assistant), matching old raw files.
                        is_hc_legacy = args.is_handcrafted
                        if args.num_schemata == 0:
                            cloud_paper_baseline(
                                client=client_or_model_obj,
                                directory_path=args.directory_path,
                                is_handcrafted=is_hc_legacy,
                                model=args.model,
                                max_tokens=args.max_tokens,
                                model_type='gpt',
                                batch_size=args.batch_size,
                                max_workers=args.max_workers,
                            )
                        else:
                            cloud_paper_correct(
                                client=client_or_model_obj,
                                directory_path=args.directory_path,
                                is_handcrafted=is_hc_legacy,
                                model=args.model,
                                max_tokens=args.max_tokens,
                                model_type='gpt',
                                schema_analyzer=schema_analyzer,
                                num_schemata=args.num_schemata,
                                batch_size=args.batch_size,
                                max_workers=args.max_workers,
                                wording='template',  # paper byte-identical for gpt
                            )
                    else:
                        # Schema-guided GPT inference: use SimilarityBasedSchemaAnalyzer's
                        # `modify_prompt` to inject k retrieved schemata into each prompt.
                        # When num_schemata == 0, modify_prompt returns the prompt unchanged
                        # (baseline / LLM-as-a-Judge).
                        def _gpt_prompt_modifier(prompt, json_file):
                            return schema_analyzer.modify_prompt(prompt, json_file, args.num_schemata)

                        gpt_all_at_once_with_schemata(
                            client=client_or_model_obj,
                            directory_path=args.directory_path,
                            is_handcrafted=args.is_handcrafted,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            prompt_modifier=_gpt_prompt_modifier,
                            max_workers=int(os.environ.get("OPENAI_MAX_WORKERS", "16")),
                        )
                elif args.method == "step_by_step":
                    gpt_step_by_step(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "binary_search":
                    gpt_binary_search(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
            elif model_type == 'gemini':
                if args.method == "all_at_once":
                    if args.cloud_paper_path:
                        # Preserve legacy string truthiness for paper/raw
                        # output compatibility; see GPT branch above.
                        is_hc_legacy = args.is_handcrafted
                        if args.num_schemata == 0:
                            cloud_paper_baseline(
                                client=client_or_model_obj,
                                directory_path=args.directory_path,
                                is_handcrafted=is_hc_legacy,
                                model=args.model,
                                max_tokens=args.max_tokens,
                                model_type='gemini',
                                batch_size=args.batch_size,
                                max_workers=args.max_workers,
                            )
                        else:
                            cloud_paper_correct(
                                client=client_or_model_obj,
                                directory_path=args.directory_path,
                                is_handcrafted=is_hc_legacy,
                                model=args.model,
                                max_tokens=args.max_tokens,
                                model_type='gemini',
                                schema_analyzer=schema_analyzer,
                                num_schemata=args.num_schemata,
                                batch_size=args.batch_size,
                                max_workers=args.max_workers,
                                wording='template',  # paper byte-identical (Gemini-pro mishandles "schema" wording)
                            )
                    else:
                        def _gemini_prompt_modifier(prompt, json_file):
                            return schema_analyzer.modify_prompt(prompt, json_file, args.num_schemata)

                        all_at_once_with_schemata_gemini(
                            client=client_or_model_obj,
                            directory_path=args.directory_path,
                            is_handcrafted=args.is_handcrafted,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            prompt_modifier=_gemini_prompt_modifier if args.num_schemata > 0 else None,
                            max_workers=int(os.environ.get("GEMINI_MAX_WORKERS", os.environ.get("OPENAI_MAX_WORKERS", "16"))),
                        )
                else:
                    print(f"Gemini support is implemented only for all_at_once in this reproduction path.")
            elif model_type == 'local':
                if args.method == "all_at_once":
                    # Use the modified function that incorporates schemata
                    modified_all_at_once = modify_analyze_all_at_once_local(schema_analyzer, analyze_all_at_once_local)
                    modified_all_at_once(
                        model_obj=client_or_model_obj,
                        directory_path=directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family,
                        use_parallel=args.use_parallel,
                        use_vllm=args.use_vllm,
                        model_path=model_id_or_deployment,
                        tensor_parallel_size=args.tensor_parallel_size,
                        num_schemata=args.num_schemata
                    )
                elif args.method == "step_by_step":
                    # Use the modified function that incorporates schemata
                    modified_step_by_step = modify_analyze_step_by_step_local(schema_analyzer, analyze_step_by_step_local)
                    modified_step_by_step(
                        model_obj=client_or_model_obj,
                        directory_path=directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family,
                        num_schemata=args.num_schemata
                    )
                elif args.method == "binary_search":
                    # Use the modified function that incorporates schemata
                    modified_binary_search = modify_analyze_binary_search_local(schema_analyzer, analyze_binary_search_local)
                    modified_binary_search(
                        model_obj=client_or_model_obj,
                        directory_path=directory_path,
                        is_handcrafted=args.is_handcrafted,
                        model_family=model_family,
                        num_schemata=args.num_schemata
                    )
            else:
                print(f"Internal Error: Unknown model_type '{model_type}' during function call.")

            print("-" * 20)
            print(f"--- Analysis Complete ---")

        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
