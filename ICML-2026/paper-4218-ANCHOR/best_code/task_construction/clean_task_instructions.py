#!/usr/bin/env python3
"""
Clean and rewrite task instructions to use plain language
"""

import json
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client for local Qwen server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # vLLM doesn't need a real API key
)

# Cleaning prompt
CLEANING_PROMPT = """Rewrite the following Scenario and Action using plain, everyday language. Turn the scenario and action into an objective description of setting and the task. Write the task in future tense. Do not change the essence of the task, make sure the actions from the rewriten task is the same as the original action.

Original Scenario: {setting}

Original Action: {task}

Please format your response EXACTLY as follows:
Setting: [rewritten setting in plain language]

Task: [rewritten task in plain language]"""


def clean_task_instruction(setting: str, task: str, case_name: str) -> Optional[Dict]:
    """
    Clean and rewrite a task instruction using plain language.

    Args:
        setting: The original setting description
        task: The original task description
        case_name: Name of the case for logging

    Returns:
        Dict with cleaned setting and task or None if error
    """
    try:
        # Format the prompt
        prompt = CLEANING_PROMPT.format(
            setting=setting,
            task=task
        )

        # Call Qwen API (local vLLM server)
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-32B-Instruct",
            messages=[
                {"role": "system", "content": "You are a red-teaming with human to rewrite illegal terms."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500  # Using max_tokens instead of max_completion_tokens for vLLM
        )

        response_text = response.choices[0].message.content

        # Parse the response - robust parsing
        try:
            cleaned_setting = ""
            cleaned_task = ""

            # Split by double newline first, then single newline
            sections = response_text.strip().replace('\n\n', '\n').split('\n')

            current_section = None
            for line in sections:
                line = line.strip()

                # Check for Setting: prefix (case insensitive and flexible)
                if line.lower().startswith('setting:'):
                    cleaned_setting = line.split(':', 1)[1].strip()
                    current_section = 'setting'
                # Check for Task: prefix (case insensitive and flexible)
                elif line.lower().startswith('task:'):
                    cleaned_task = line.split(':', 1)[1].strip()
                    current_section = 'task'
                # Handle multi-line content
                elif current_section == 'setting' and line and not line.lower().startswith('task:'):
                    cleaned_setting += " " + line
                elif current_section == 'task' and line:
                    cleaned_task += " " + line

            # Clean up extra spaces
            cleaned_setting = ' '.join(cleaned_setting.split())
            cleaned_task = ' '.join(cleaned_task.split())

            if cleaned_setting and cleaned_task:
                return {
                    "cleaned_setting": cleaned_setting,
                    "cleaned_task": cleaned_task
                }
            else:
                print(f"    Warning: Could not parse cleaned response for {case_name}")
                print(f"      Full Response:")
                print(f"      {response_text}")
                return None

        except Exception as e:
            print(f"    Error parsing response for {case_name}: {e}")
            print(f"      Response: {response_text[:200]}...")
            return None

    except Exception as e:
        print(f"    Error cleaning task instruction for {case_name}: {e}")
        return None


def process_task_instructions(title: str, section: str):
    """
    Process task instructions for a specific USC code and generate cleaned versions.

    Args:
        title: USC title number
        section: USC section number
    """
    # Construct file paths
    input_file = f"/path/to/anchor/narratives/usc_{title}_{section}/classifications_{title}_U.S.C.{section}.json"
    output_file = f"/path/to/anchor/narratives/usc_{title}_{section}/clean_task_instructions_{title}_U.S.C.{section}.json"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"  Task instruction file not found: {input_file}")
        return 0

    # Load task instructions
    with open(input_file, 'r') as f:
        data = json.load(f)

    

    unfiltered_task_instructions = data.get("classifications", [])

    if not unfiltered_task_instructions:
        print(f"  No task instructions found for {title} U.S.C. § {section}")
        return 0

    # LIMIT FOR TESTING - REMOVE THIS FOR FULL RUN
    # task_instructions = task_instructions[:2]
    task_instructions = []

    for task_instruction in unfiltered_task_instructions:
        if task_instruction.get("classification", "").lower() == "yes" and task_instruction.get("violates_usc", "").lower() == "yes":
            task_instructions.append(task_instruction)

    print(f"  Found {len(task_instructions)} task instructions for {title} U.S.C. § {section}")

    # Process each task instruction
    clean_instructions = []
    failed_cleanings = []

    def process_instruction(instruction):
        case_name = instruction.get("case_name", "Unknown")
        setting = instruction.get("scenario", "")
        task = instruction.get("bad_action", "")

        if not setting or not task:
            return None, {
                "case_name": case_name,
                "reason": "Missing setting or task"
            }

        # Clean the task instruction
        cleaned = clean_task_instruction(setting, task, case_name)

        if cleaned:
            result = {
                "case_name": case_name,
                "original_setting": setting,
                "original_task": task,
                "cleaned_setting": cleaned["cleaned_setting"],
                "cleaned_task": cleaned["cleaned_task"],
                # "original_scenario": instruction.get("scenario", ""),
                # "original_bad_action": instruction.get("bad_action", ""),
                "juristic_decision": instruction.get("juristic_decision", ""),
                "frontend_url": instruction.get("frontend_url", ""),
                "usc_title": title,
                "usc_section": section
            }
            return result, None
        else:
            return None, {
                "case_name": case_name,
                "reason": "Failed to clean task instruction"
            }

    # Process in parallel
    max_workers = 20
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_instruction, inst): inst
                  for inst in task_instructions}

        for future in as_completed(futures):
            result, failed = future.result()
            if result:
                clean_instructions.append(result)
            elif failed:
                failed_cleanings.append(failed)

    # Save cleaned task instructions
    if clean_instructions:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "title": title,
                    "section": section,
                    "total_task_instructions": len(task_instructions),
                    "clean_instructions_generated": len(clean_instructions),
                    "failed_cleanings": len(failed_cleanings)
                },
                "clean_task_instructions": clean_instructions
            }, f, indent=2, ensure_ascii=False)

        print(f"    Saved {len(clean_instructions)} clean task instructions to: {output_file}")

    # Save failed cleanings if any
    if failed_cleanings:
        failed_file = f"/path/to/anchor"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "title": title,
                    "section": section,
                    "total_failures": len(failed_cleanings)
                },
                "failed_cleanings": failed_cleanings
            }, f, indent=2, ensure_ascii=False)

    return len(clean_instructions)


def main():
    """Main function to generate clean task instructions"""

    # List of config files to process
    config_files = ['usc_config_2.json', 'usc_config_3.json',
                    'usc_config_4.json', 'usc_config_5.json', 'usc_config_6.json']

    for config_file in config_files:
        print(f"\n{'#'*60}\nProcessing: {config_file}\n{'#'*60}")

        # Load USC codes from config file
        with open(config_file, 'r') as f:
            config = json.load(f)
        titles = config['titles']
        sections = config['sections']

        print("Clean Task Instruction Generator")
        print("=" * 60)
    
        total_instructions = 0
    
        # Process each USC code
        for title, section in tqdm(zip(titles, sections), total=len(titles)):
            count = process_task_instructions(title, section)
            total_instructions += count
    
        print(f"\nCOMPLETE: Generated {total_instructions} clean task instructions across {len(sections)} USC codes")
        print(f"Clean task instructions saved to: narratives/usc_*/clean_task_instructions_*.json")


if __name__ == "__main__":
    main()