import json
import ast
import argparse
from openai import OpenAI
from collections import defaultdict
import time
from tqdm import tqdm
import sys
import os
import traceback
from datetime import datetime
import concurrent.futures

# ----------------- Global API Configuration -----------------
API_CONFIG = {
    'deepseek': {
        'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
        'url': '<url id="cvljkg5f4394kb3thhp0" type="url" status="failed" title="" wc="0">https://api.deepseek.com/v1/</url> ',
        'model': 'deepseek-chat'
    },
    'gpt-4o': {
        'api_key': os.getenv('Gpt_API_KEY', ''),
        'url': '<url id="cvljkg5f4394kb3thhpg" type="url" status="failed" title="" wc="0">https://api.openai.com/v1/</url> ',
        'model': 'gpt-4o'
    },
}

# ----------------- Hyperparameters Configuration -----------------
HYPERPARAMS = {
    "max_retries": 3,
    "batch_size": 10,
    "max_workers": 20,        # Default number of parallel threads, can be overridden by command-line arguments
    "enable_api_logging": True  # Controls whether to log API input and output
}


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Tag standardization process.")
    parser.add_argument("--input_file", type=str, default="",
                        help="Input JSONL file containing user profiles.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save the output files.")
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        help="Which model to use from the API_CONFIG (e.g., deepseek, etc.).")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Number of threads for concurrent processing.")
    args = parser.parse_args()
    return args


def analyze_possible_error_reason(exception):
    """
    Analyze the possible reasons for an exception and return an explanatory string
    """
    reasons = []
    if isinstance(exception, FileNotFoundError):
        reasons.append("Input file path error or file not found")
    elif isinstance(exception, json.JSONDecodeError):
        reasons.append("JSON parsing error, possibly due to incorrect LLM response format or data anomaly")
    elif isinstance(exception, ConnectionError):
        reasons.append("Network connection error, possibly due to network issues or API request problems")
    elif isinstance(exception, TimeoutError):
        reasons.append("Request timeout, possibly due to network latency or slow API response")
    elif isinstance(exception, ValueError):
        reasons.append("Value error, possibly due to unexpected data format")
    else:
        reasons.append("Unknown error, cause not determined")
    return '; '.join(reasons)


class ProgressTracker:
    """
    Manage and display progress information during processing, while saving error logs.
    Provides progress bars, status updates, and error logging functionality.
    """

    def __init__(self):
        self.current_stage = ""
        self.error_logs = []  # Save all error logs

    def update_stage(self, stage_message):
        """Display the status of a new stage with a timestamp"""
        self.current_stage = stage_message
        timestamp = time.strftime('%H:%M:%S')
        print(f"\n[{timestamp}] {stage_message}")
        sys.stdout.flush()  # Ensure immediate output

    def progress_bar(self, iterable, desc):
        """Create a progress bar for an iterable object"""
        return tqdm(iterable, desc=desc, leave=True)

    def log_error(self, error_message, exception=None, api_request=None, api_response=None):
        """
        Log errors, display timestamps, error messages, and optional exception details, API input and output, and save error logs.

        - api_request: API request parameters when calling the large model API
        - api_response: API response when calling the large model API (if available)
        """
        timestamp = time.strftime('%H:%M:%S')
        print(f"\n[{timestamp}] ERROR: {error_message}")
        error_entry = {
            "timestamp": timestamp,
            "error_message": error_message,
            "exception": None,
            "traceback": None,
            "analysis": None,
            "api_request": api_request,
            "api_response": api_response
        }

        if exception:
            exception_str = str(exception)
            tb_str = traceback.format_exc()
            analysis = analyze_possible_error_reason(exception)
            print(f"Exception details: {exception_str}")
            print("Traceback:")
            traceback.print_exc()
            print(f"Possible error reasons: {analysis}")

            error_entry["exception"] = exception_str
            error_entry["traceback"] = tb_str
            error_entry["analysis"] = analysis

        sys.stdout.flush()
        self.error_logs.append(error_entry)

    def save_error_logs(self, file_path):
        """Save all error logs to the specified file"""
        try:
            ensure_directory_exists(file_path)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.error_logs, f, ensure_ascii=False, indent=2)
            print(f"\nError logs saved to {file_path}")
        except Exception as e:
            print("Failed to save error logs", e)


def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the specified file path exists, create it if it does not
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False


# ----------------- Functions to Parse LLM Response -----------------
def parse_json_response(response_content):
    """
    Try to parse the response content into a JSON object:
      1. First use json.loads() to parse;
      2. If it fails, try using ast.literal_eval() for fault-tolerant parsing.
    If both methods fail, raise an exception.
    """
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        # If there is a JSONDecodeError due to single quotes or other issues
        try:
            return ast.literal_eval(response_content)
        except Exception as e:
            # If both methods fail, raise an exception
            raise e


def remove_json_markers(response_content):
    """
    Remove the ```json wrapping symbols that may be included in the LLM response content
    """
    if response_content.startswith('```json'):
        response_content = response_content[len('```json'):]
    if response_content.endswith('```'):
        response_content = response_content[:-len('```')]
    return response_content.strip()


def create_llm_client(api_name="deepseek"):
    """
    Create and return a configured API client that can call different APIs

    Parameters:
        api_name: The name of the API to use, defaults to "deepseek"
    Returns:
        A configured client instance (with the default_model attribute saved)
    """
    if api_name not in API_CONFIG:
        raise ValueError(f"Unknown API name: {api_name}")
    config = API_CONFIG[api_name]
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['url']
    )
    # Save the default model to the client object
    client.default_model = config['model']
    return client


def load_profiles(file_path, tracker):
    """
    Load user profiles from a JSONL file and display progress.
    If a line's JSON parsing fails, try to extract the user_id and log the error.
    """
    tracker.update_stage(f"Loading user profiles from JSONL file: {file_path}")

    profiles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with tracker.progress_bar(lines, desc="Reading profiles") as pbar:
            for line in pbar:
                line_content = line.strip()
                if not line_content:
                    continue
                try:
                    data = json.loads(line_content)
                    profiles.append(data)
                except Exception as e:
                    # If parsing fails, try to extract user_id
                    user_id = "unknown"
                    try:
                        partial_data = json.loads(line_content)
                        user_id = partial_data.get('user_id', 'unknown')
                    except:
                        pass
                    tracker.log_error(
                        f"Error parsing profile. user_id={user_id}, line={line_content}",
                        exception=e
                    )
    except Exception as e:
        tracker.log_error(f"Error loading profiles from {file_path}", e)
        raise

    tracker.update_stage(f"Successfully loaded {len(profiles)} profiles")
    return profiles


def filter_irrelevant_tags(tags, tags_to_user_ids, client, tracker,
                           max_retries=3, batch_size=100, max_workers=50,
                           enable_api_logging=True):
    """
    Use LLM to filter tags that are not relevant to tourist attraction features,
    improving efficiency through parallel batch processing, while displaying batch progress.

    New parameters:
    - tags_to_user_ids: {tag: set(user_ids)}, used to record related user IDs in case of errors
    - When an error occurs, all related user_ids in the batch are recorded in the log.
    """
    tracker.update_stage(f"Filtering irrelevant tags from {len(tags)} tags in parallel batches of {batch_size}...")

    relevant_tags = []
    irrelevant_tags = []

    tag_batches = [tags[i:i + batch_size] for i in range(0, len(tags), batch_size)]

    def process_batch(batch_idx, batch):
        nonlocal relevant_tags, irrelevant_tags
        # Collect all user_ids associated with this batch
        user_ids_in_batch = set()
        for t in batch:
            user_ids_in_batch.update(tags_to_user_ids.get(t, []))

        prompt = f"""
        Given these user preference tags: {', '.join(batch)}

        Please identify and filter out tags that are NOT relevant to describing features of a tourist attraction. Follow these updated rules:

        1. **Keep tags that describe specific features, characteristics, or experiences of a tourist attraction.**
        2. **Remove tags that are irrelevant to tourist attractions.**
        3. **Return a JSON object with two keys:**
           - "relevant_tags": A list of tags that are relevant to tourist attractions.
           - "irrelevant_tags": A list of tags that are irrelevant and should be removed.
        """

        local_api_request = {
            "model": client.default_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that filters out irrelevant tags. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        for attempt in range(max_retries):
            response_content = None
            try:
                response = client.chat.completions.create(**local_api_request)
                response_content = response.choices[0].message.content.strip()
                response_content = remove_json_markers(response_content)

                filter_result = parse_json_response(response_content)
                if not isinstance(filter_result, dict):
                    raise ValueError("Filter result is not a dictionary")
                if "relevant_tags" not in filter_result or "irrelevant_tags" not in filter_result:
                    raise ValueError("Missing 'relevant_tags' or 'irrelevant_tags' keys")

                relevant_tags.extend(filter_result["relevant_tags"])
                irrelevant_tags.extend(filter_result["irrelevant_tags"])

                tracker.update_stage(f"Successfully processed batch {batch_idx + 1}")
                return

            except Exception as e:
                # If an error occurs, record the user_ids in this batch
                error_msg = (
                    f"Error in batch {batch_idx + 1} (attempt {attempt + 1}/{max_retries}), "
                    f"user_ids_in_batch={list(user_ids_in_batch)}"
                )
                tracker.log_error(
                    error_msg,
                    exception=e,
                    api_request=local_api_request if enable_api_logging else None,
                    api_response=response_content if enable_api_logging else None
                )
                if attempt == max_retries - 1:
                    tracker.update_stage(f"All retry attempts failed for batch {batch_idx + 1}. Skipping batch.")
                    # Custom handling logic can be added here; in this example, we just add the original tags to relevant_tags
                    relevant_tags.extend(batch)
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch_idx, batch): batch_idx
            for batch_idx, batch in enumerate(tag_batches)
        }
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            pass

    tracker.update_stage(
        f"Completed filtering. Found {len(relevant_tags)} relevant tags and {len(irrelevant_tags)} irrelevant tags."
    )
    return relevant_tags, irrelevant_tags


def standardize_tags(tags, client, tracker, max_retries=3, enable_api_logging=True):
    """
    Use LLM to standardize similar tags into consistent categories,
    including progress display and retry mechanism.
    """
    tracker.update_stage(f"Beginning tag standardization for {len(tags)} unique tags...")

    prompt = f"""
    Given these user preference tags: {', '.join(tags)}

    Please standardize similar tags into consistent categories. Return a JSON dictionary mapping original tags to standardized tags.
    """

    local_api_request = {
        "model": client.default_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that standardizes similar tags into consistent categories. "
                    "Always return valid JSON in the format: {'original_tag': 'standardized_tag', ...}"
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    for attempt in range(max_retries):
        response_content = None
        try:
            tracker.update_stage(
                f"Sending request to LLM for tag standardization (attempt {attempt + 1}/{max_retries})..."
            )
            response = client.chat.completions.create(**local_api_request)

            response_content = response.choices[0].message.content.strip()
            response_content = remove_json_markers(response_content)

            standardization_map = parse_json_response(response_content)
            if not isinstance(standardization_map, dict):
                raise ValueError("Standardization map is not a dictionary")

            for key, value in standardization_map.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("Both keys and values in the map must be strings")

            tracker.update_stage("Successfully received and parsed LLM response")
            return standardization_map

        except Exception as e:
            tracker.log_error(
                f"Error in standardize_tags (attempt {attempt + 1}/{max_retries})",
                exception=e,
                api_request=local_api_request if enable_api_logging else None,
                api_response=response_content if enable_api_logging else None
            )
            if attempt == max_retries - 1:
                tracker.update_stage("All retry attempts failed. Returning unchanged tags.")
                return {tag: tag for tag in tags}
            time.sleep(2)

    return {tag: tag for tag in tags}


def select_top5_tags_for_profile(profile, relevant_tags, client, tracker, max_retries=3, enable_api_logging=True):
    """
    Use LLM to select the top 5 tags from a user's filtered likes and dislikes.

    If there are fewer than 5 tags in a category, return all of them.
    The returned JSON format example:
      {"likes_top5": ["tag1", "tag2", ...], "dislikes_top5": ["tagA", "tagB", ...]}
    """
    if 'preference' not in profile:
        return {"likes_top5": [], "dislikes_top5": []}
    likes = profile['preference'].get('likes', [])
    dislikes = profile['preference'].get('dislikes', [])
    filtered_likes = [tag for tag in likes if tag in relevant_tags]
    filtered_dislikes = [tag for tag in dislikes if tag in relevant_tags]

    # If both are empty, return directly
    if not filtered_likes and not filtered_dislikes:
        return {"likes_top5": filtered_likes, "dislikes_top5": filtered_dislikes}

    prompt = f"""
    Given the following user preference tags after filtering irrelevant ones:
    Likes: {', '.join(filtered_likes)}
    Dislikes: {', '.join(filtered_dislikes)}

    Please select the top 5 tags for likes and top 5 tags for dislikes that best represent the user's most important preferences regarding tourist attractions.
    If there are fewer than 5 tags in a category, return all of them.
    Return a JSON object with two keys: "likes_top5" and "dislikes_top5".
    """

    local_api_request = {
        "model": client.default_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that selects the top 5 tags from user preferences."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response_content = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**local_api_request)
            response_content = response.choices[0].message.content.strip()
            response_content = remove_json_markers(response_content)
            result = parse_json_response(response_content)
            if "likes_top5" not in result or "dislikes_top5" not in result:
                raise ValueError("Missing keys in top5 selection result")
            if not isinstance(result["likes_top5"], list) or not isinstance(result["dislikes_top5"], list):
                raise ValueError("Top5 selections are not lists")
            return result
        except Exception as e:
            tracker.log_error(
                f"Error selecting top5 tags for profile user_id={profile.get('user_id', 'unknown')} (attempt {attempt + 1}/{max_retries})",
                exception=e,
                api_request=local_api_request if enable_api_logging else None,
                api_response=response_content if enable_api_logging else None
            )
            if attempt == max_retries - 1:
                # If all retries fail, fall back to truncating the first 5
                return {"likes_top5": filtered_likes[:5], "dislikes_top5": filtered_dislikes[:5]}
            time.sleep(2)


def process_user_preferences(profiles, model_name, tracker):
    """
    Process all user preferences, standardize tags, and display corresponding progress.
    Main workflow:
      1. Collect all user tags.
      2. Call LLM to filter irrelevant tags.
      3. For each user, call LLM to select the top5 tags from their filtered likes and dislikes, and save them to the profile.
      4. Calculate the union of all users' top5 tags and call LLM to standardize these tags.
      5. When updating the profile, use the top5 tags for mapping.
    """
    tracker.update_stage("Collecting unique tags from all profiles...")

    all_tags = set()
    # To record all user_ids associated with each tag
    tags_to_user_ids = defaultdict(set)

    # Collect all user IDs and their associated tags
    for profile in tracker.progress_bar(profiles, desc="Extracting tags"):
        user_id = profile.get('user_id', 'unknown')
        if 'preference' in profile:
            if 'likes' in profile['preference']:
                for tag in profile['preference']['likes']:
                    all_tags.add(tag)
                    tags_to_user_ids[tag].add(user_id)
            if 'dislikes' in profile['preference']:
                for tag in profile['preference']['dislikes']:
                    all_tags.add(tag)
                    tags_to_user_ids[tag].add(user_id)

    tracker.update_stage(f"Found {len(all_tags)} unique tags")

    # Create LLM client using the specified model name
    client = create_llm_client(api_name=model_name)

    # Step 1: Filter irrelevant tags
    relevant_tags, irrelevant_tags = filter_irrelevant_tags(
        list(all_tags),
        tags_to_user_ids,  # Pass the mapping table to record user_ids in case of errors
        client,
        tracker,
        max_retries=HYPERPARAMS["max_retries"],
        batch_size=HYPERPARAMS["batch_size"],
        max_workers=HYPERPARAMS["max_workers"],
        enable_api_logging=HYPERPARAMS["enable_api_logging"]
    )

    # New step: Call LLM to select top5 tags from each user's filtered likes and dislikes
    tracker.update_stage("Selecting top 5 tags for each profile using LLM...")
    for profile in tracker.progress_bar(profiles, desc="Selecting top5 tags"):
        if 'preference' in profile:
            likes = profile['preference'].get('likes', [])
            dislikes = profile['preference'].get('dislikes', [])
            filtered_likes = [tag for tag in likes if tag in relevant_tags]
            filtered_dislikes = [tag for tag in dislikes if tag in relevant_tags]
            if not filtered_likes and not filtered_dislikes:
                profile['preference']['likes_top5'] = []
                profile['preference']['dislikes_top5'] = []
            else:
                result = select_top5_tags_for_profile(profile, relevant_tags, client, tracker,
                                                       max_retries=HYPERPARAMS["max_retries"],
                                                       enable_api_logging=HYPERPARAMS["enable_api_logging"])
                profile['preference']['likes_top5'] = result.get("likes_top5", [])
                profile['preference']['dislikes_top5'] = result.get("dislikes_top5", [])

    # Calculate the union of all users' top5 tags for subsequent standardization
    union_top5_tags = set()
    for profile in profiles:
        if 'preference' in profile:
            union_top5_tags.update(profile['preference'].get('likes_top5', []))
            union_top5_tags.update(profile['preference'].get('dislikes_top5', []))
    tracker.update_stage(f"Union of top5 tags count: {len(union_top5_tags)}")

    # Step 2: Standardize the union of top5 tags
    standardization_map = standardize_tags(
        list(union_top5_tags),
        client,
        tracker,
        max_retries=HYPERPARAMS["max_retries"],
        enable_api_logging=HYPERPARAMS["enable_api_logging"]
    )

    # Step 3: Update profiles, using top5 tags for mapping, and mark removed irrelevant tags
    standardized_profiles = []
    for profile in tracker.progress_bar(profiles, desc="Updating profiles with standardized top5 tags"):
        standardized_profile = profile.copy()
        if 'preference' in profile:
            likes = profile['preference'].get('likes', [])
            dislikes = profile['preference'].get('dislikes', [])
            standardized_profile['preference']['del_likes'] = [tag for tag in likes if tag in irrelevant_tags]
            standardized_profile['preference']['del_dislikes'] = [tag for tag in dislikes if tag in irrelevant_tags]
            # Map using top5 tags
            standardized_profile['preference']['standardized_likes'] = [
                standardization_map.get(tag, tag)
                for tag in profile['preference'].get('likes_top5', [])
            ]
            standardized_profile['preference']['standardized_dislikes'] = [
                standardization_map.get(tag, tag)
                for tag in profile['preference'].get('dislikes_top5', [])
            ]
            # Remove duplicates for the new standardized tags
            standardized_profile['preference']['new_likes'] = list(
                set(standardized_profile['preference']['standardized_likes'])
            )
            standardized_profile['preference']['new_dislikes'] = list(
                set(standardized_profile['preference']['standardized_dislikes'])
            )
        standardized_profiles.append(standardized_profile)

    return {
        'standardization_map': standardization_map,
        'standardized_profiles': standardized_profiles,
        'irrelevant_tags': irrelevant_tags
    }


def analyze_tag_reduction(original_profiles, standardized_results, tracker):
    """Analyze the reduction in unique tags before and after standardization and display progress"""
    tracker.update_stage("Analyzing tag reduction results...")

    original_tags = set()
    standardized_tags = set()

    for profile in tracker.progress_bar(original_profiles, desc="Analyzing original tags"):
        if 'preference' in profile:
            if 'likes' in profile['preference']:
                original_tags.update(profile['preference']['likes'])
            if 'dislikes' in profile['preference']:
                original_tags.update(profile['preference']['dislikes'])

    for profile in tracker.progress_bar(standardized_results['standardized_profiles'],
                                        desc="Analyzing standardized tags"):
        if 'preference' in profile:
            if 'standardized_likes' in profile['preference']:
                standardized_tags.update(profile['preference']['standardized_likes'])
            if 'standardized_dislikes' in profile['preference']:
                standardized_tags.update(profile['preference']['standardized_dislikes'])

    if len(original_tags) == 0:
        return {
            'original_tag_count': 0,
            'standardized_tag_count': len(standardized_tags),
            'reduction_percentage': 0
        }

    return {
        'original_tag_count': len(original_tags),
        'standardized_tag_count': len(standardized_tags),
        'reduction_percentage': ((len(original_tags) - len(standardized_tags)) / len(original_tags)) * 100
    }


def save_results(standardized_results, analysis, output_path, tracker):
    """
    Save standardized results and analysis data to files
    """
    tracker.update_stage("Saving results to files...")

    try:
        ensure_directory_exists(output_path)

        profiles_path = os.path.join(output_path, 'profiles.jsonl')
        ensure_directory_exists(profiles_path)
        with open(profiles_path, 'w', encoding='utf-8') as f:
            for profile in tracker.progress_bar(standardized_results['standardized_profiles'], desc="Saving profiles"):
                json.dump(profile, f, ensure_ascii=False)
                f.write('\n')

        mapping_path = os.path.join(output_path, 'mapping.json')
        ensure_directory_exists(mapping_path)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(standardized_results['standardization_map'], f, ensure_ascii=False, indent=2)

        analysis_path = os.path.join(output_path, 'analysis.json')
        ensure_directory_exists(analysis_path)
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        tracker.update_stage("Results saved successfully")

    except Exception as e:
        tracker.log_error("Error saving results", e)
        raise


def main():
    # 1. Parse command-line arguments
    args = parse_arguments()

    # Dynamically set hyperparameters like the number of parallel threads based on command-line arguments
    HYPERPARAMS["max_workers"] = args.max_workers

    tracker = ProgressTracker()
    tracker.update_stage("Starting tag standardization process...")

    try:
        # 2. Load user profiles (JSONL format)
        profiles = load_profiles(args.input_file, tracker)

        # 3. Process user preferences, filter & standardize tags (new step to select top5 tags)
        standardized_results = process_user_preferences(profiles, args.model_name, tracker)

        # 4. Analyze the reduction in tags before and after standardization
        analysis = analyze_tag_reduction(profiles, standardized_results, tracker)

        # 5. Generate output directory (based on command-line arguments)
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = os.path.join(output_dir, f'standardization_{timestamp}')

        # 6. Save results
        save_results(standardized_results, analysis, output_subdir, tracker)

        # If there are error logs, save them to a file
        error_log_path = os.path.join(output_subdir, 'error_log.json')
        if tracker.error_logs:
            tracker.save_error_logs(error_log_path)

        # 7. Output results
        tracker.update_stage("Process completed. Final results:")
        print(f"\nTag Standardization Results:")
        print(f"Original unique tags: {analysis['original_tag_count']}")
        print(f"Standardized unique tags: {analysis['standardized_tag_count']}")
        print(f"Reduction percentage: {analysis['reduction_percentage']:.2f}%")

        print("\nExample standardizations:")
        # Display the first 5 example mappings
        count = 0
        for original, standardized in standardized_results['standardization_map'].items():
            if original != standardized:
                print(f"'{original}' -> '{standardized}'")
                count += 1
                if count >= 5:
                    break

    except Exception as e:
        tracker.log_error("An error occurred in the main process", e)
        sys.exit(1)


if __name__ == "__main__":
    main()