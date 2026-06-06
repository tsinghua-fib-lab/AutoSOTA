import json
import os
import argparse
import threading
import time
from typing import List, Dict
from langchain.chat_models import ChatOpenAI  # Please ensure you have installed langchain
from langchain.schema import HumanMessage
from datetime import datetime
from tqdm import tqdm  # For displaying progress
from concurrent.futures import ThreadPoolExecutor, as_completed

# API Configuration (Global Read-Only)
api_config = {
        'deepseek': {'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
                     'url': 'https://api.deepseek.com/v1/', 'model': 'deepseek-chat'},
        'gpt-4o': {'api_key': os.getenv('Gpt_API_KEY', ''),
                   'url': 'https://api.openai.com/v1/', 'model': 'gpt-4o'},
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process user preference data using large language models, with only one LLM call.")
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help=f"The model to be used. Options: {', '.join(api_config.keys())}"
    )
    parser.add_argument(
        '--processes',  # Parameter name is processes, but internally represents the number of threads
        type=int,
        default=40,
        help="Number of parallel threads to use."
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='  ',
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help="Base directory for output files."
    )
    return parser.parse_args()


def initialize_llm(selected_model: str) -> ChatOpenAI:
    """
    Initialize a ChatOpenAI instance based on the selected model.
    """
    if selected_model not in api_config:
        raise ValueError(f"The selected model '{selected_model}' is not in the API configuration.")
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_config[selected_model]['api_key'],
        openai_api_base=api_config[selected_model]['url'],
        model_name=api_config[selected_model]['model']
    )
    return llm


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read a JSONL file and return a list of JSON objects, skipping lines with formatting errors and printing errors.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decoding error on line {i}: {e}")
    return data


def write_jsonl(file_path: str, data: List[Dict]):
    """
    Write a list of JSON objects to a JSONL file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def validate_user_data(user: Dict) -> bool:
    """
    Validate whether user data contains necessary fields and the format is correct.
    """
    if 'user_id' not in user:
        print("Missing user_id, skipping this user.")
        return False
    if 'reviews_attraction' not in user or not isinstance(user['reviews_attraction'], list):
        print(f"Missing or incorrect format for reviews_attraction of user {user.get('user_id', 'unknown')}, skipping.")
        return False
    return True


def aggregate_preferences(reviews: List[Dict]) -> Dict[str, List[str]]:
    """
    Aggregate likes and dislikes from all reviews of a single user, while filtering out invalid data.
    """
    likes = []
    dislikes = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        review_likes = review.get('likes', [])
        review_dislikes = review.get('dislikes', [])

        # Filter out items with content "None" (case-insensitive)
        likes.extend([like for like in review_likes if isinstance(like, str) and like.lower() != 'none'])
        dislikes.extend([dislike for dislike in review_dislikes if isinstance(dislike, str) and dislike.lower() != 'none'])

    # If aggregated result is empty, set to ["None"]
    if not likes:
        likes = ["None"]
    if not dislikes:
        dislikes = ["None"]

    return {"likes": likes, "dislikes": dislikes}


def construct_profile_prompt(preferences: Dict[str, List[str]]) -> str:
    """
    Construct a single prompt text for generating the final LLM response (profile).
    Uses likes and dislikes aggregated from the original file (aggregate_preferences).
    """
    likes = preferences.get('likes', [])
    dislikes = preferences.get('dislikes', [])

    likes_text = "; ".join(likes) if likes else "None."
    dislikes_text = "; ".join(dislikes) if dislikes else "None."

    # You can modify the prompt template as needed
    prompt_profile = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"User Preferences:\n"
        f"Likes: {likes_text}\n"
        f"Dislikes: {dislikes_text}\n\n"
        "Based on these preferences, please generate a concise user profile.\n"
        "Guidelines:\n"
        "1. Focus on high-level characteristics.\n"
        "2. Avoid directly repeating the preference lists; infer underlying traits.\n"
        "3. Output in this format:\n"
        "Profile: [user profile]\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "```"
    )
    return prompt_profile


def parse_profile_response(response: str) -> str:
    """
    Parse the user profile (Profile) returned by the LLM.
    """
    profile = "Profile could not be determined."
    try:
        if "Profile:" in response:
            # Take the content after the first 'Profile:'
            profile_split = response.split("Profile:")
            if len(profile_split) > 1:
                profile_content = profile_split[1]
                # Avoid including content after <|eot_id|>
                profile_content = profile_content.split("<|eot_id|>")[0]
                profile = profile_content.strip()
    except Exception as e:
        print(f"Error parsing profile response: {e}")
        profile = "Profile could not be determined."
    return profile


def call_llm(llm, messages, retries=3, delay=1):
    """
    Add retry mechanism for LLM calls to prevent empty responses due to network or temporary errors.
    """
    for attempt in range(1, retries + 1):
        try:
            response = llm(messages)
            content = response.content.strip() if response and response.content else ""
            if content:
                return response
            else:
                print(f"LLM returned empty response on attempt {attempt}.")
        except Exception as e:
            print(f"LLM call failed on attempt {attempt}: {e}")
        time.sleep(delay)
    return None


def process_user(user: Dict, selected_model: str) -> Dict:
    """
    Process a single user: aggregate preferences -> construct single prompt -> call LLM -> parse LLM response.
    In the final output, "preference" uses the aggregated likes/dislikes,
    "profile" uses the parsed result from the LLM response.
    """
    try:
        if not validate_user_data(user):
            return None

        thread_name = threading.current_thread().name
        user_id = user.get('user_id', 'unknown')
        reviews = user.get('reviews_attraction', [])

        # 1) Aggregate likes and dislikes from raw reviews
        aggregated_prefs = aggregate_preferences(reviews)

        # 2) Construct a single prompt using aggregated likes & dislikes
        prompt_profile = construct_profile_prompt(aggregated_prefs)

        # 3) Initialize LLM and make the call
        llm = initialize_llm(selected_model)
        response_prefs = call_llm(llm, [HumanMessage(content=prompt_profile)])
        if response_prefs is None or not response_prefs.content.strip():
            print(f"[{thread_name}] LLM returned empty for user {user_id}, setting to default.")
            final_profile = "Profile could not be determined."
        else:
            # 4) Parse the returned profile
            final_profile = parse_profile_response(response_prefs.content.strip())

        # 5) Return the final structure
        output_entry = {
            "user_id": user_id,
            "preference": aggregated_prefs,  # Use aggregated likes/dislikes
            "profile": final_profile,        # Result from a single LLM call
            "reviews_attraction": reviews
        }
        return output_entry

    except Exception as e:
        print(f"[{threading.current_thread().name}] Error processing user {user.get('user_id', 'unknown')}: {e}")
        return None


def process_users_parallel(users: List[Dict], selected_model: str, num_threads: int) -> List[Dict]:
    """
    Process user data in parallel using multiple threads and display progress using tqdm.
    """
    output_data = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_user = {executor.submit(process_user, user, selected_model): user for user in users}
        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Processing Users"):
            try:
                result = future.result()
                if result is not None:
                    output_data.append(result)
            except Exception as e:
                user = future_to_user.get(future, {})
                print(f"Exception occurred processing user {user.get('user_id', 'unknown')}: {e}")
    return output_data


def main():
    args = parse_arguments()
    selected_model = args.model
    num_threads = args.processes  # Parameter name is processes, but actually represents the number of threads
    input_file = args.input_file
    base_output_dir = args.output_dir

    if selected_model not in api_config:
        raise ValueError(f"The selected model '{selected_model}' is not in the API configuration. Available models: {', '.join(api_config.keys())}")

    # Create a subdirectory named with the model name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = f"{selected_model}_{timestamp}"
    output_path = os.path.join(base_output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'user_preference.jsonl')
    sim_output_file = os.path.join(output_path, f'sim_profile_{selected_model}.jsonl')  # Simplified output file

    # Read input data
    users = read_jsonl(input_file)
    print(f"Total users to process: {len(users)}")
    print(f"Using model '{selected_model}' and initiating {num_threads} threads.")
    print(f"Output files will be saved to '{output_file}' and '{sim_output_file}'")

    # Process user data in parallel using multiple threads
    output_data = process_users_parallel(users, selected_model, num_threads)

    # Write complete output data
    write_jsonl(output_file, output_data)
    print(f"Processing completed. Full output written to {output_file}")

    # Write simplified output data (only including user id, aggregated preferences, and LLM-returned profile)
    sim_output_data = [
        {
            "user_id": user["user_id"],
            "preference": user["preference"],
            "profile": user["profile"]
        }
        for user in output_data
    ]
    write_jsonl(sim_output_file, sim_output_data)
    print(f"Simplified output written to {sim_output_file}")


if __name__ == "__main__":
    main()