import json
import os
import csv
import argparse
import datetime
import concurrent.futures
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from tqdm import tqdm


# -----------------------------
# 1. Read target user list
# -----------------------------
def read_target_user_ids(csv_file_path: str) -> List[str]:
    target_user_ids = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, desc="Reading target user IDs", unit="row"):
            target_user_ids.append(row['user_id'])
    return target_user_ids


# -----------------------------
# 2. Read user reviews
# -----------------------------
def read_user_reviews(file_path: str, target_user_ids: List[str]) -> List[dict]:
    """
    Read reviews from a JSON Lines file and filter based on target_user_ids.
    The returned list format is:
    [
      {
         "user_id": xxx,
         "reviews_attraction": [
             {"text": "...", "attraction_id": "...", ...},
             ...
         ]
      },
      ...
    ]
    """
    user_reviews = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Reading user reviews", unit="line"):
            data = json.loads(line.strip())
            if data['user_id'] in target_user_ids:
                user_reviews.append(data)
    return user_reviews


# -----------------------------
# 3. Call LLM to generate likes/dislikes
# -----------------------------
def generate_likes_dislikes(review_text: str, api_config: dict, model_name: str) -> str:
    """
    Call ChatOpenAI to generate [Like]/[Dislike] for attractions based on review_text.
    Return an empty string if an exception occurs.
    """
    try:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_config[model_name]['api_key'],
            openai_api_base=api_config[model_name]['url'],
            model_name=api_config[model_name]['model']
        )

        input_text = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "Given a review written by a user, list the preferences the user liked and disliked about the attraction under [Like] and [Dislike] in bullet points, respectively. "
            "If there is nothing to mention about like/dislike, simply write 'None' under the corresponding tag. "
            "DO NOT write any content that is not revealed in the review. Please do not repeat the expressions in the original text, but use one or more words to describe the characteristics of the attractions that the user is interested in.\n"
            "Analyze user reviews of attractions through these lenses:\n"
            "Attraction type, Cultural Value, Facilities & Services, Activities\n"

            "List preferences under [Like]/[Dislike] using these strict criteria:\n"
            "1. Focus on: Attraction Type, Cultural Value, Facilities & Services, On-site Activities\n"
            "2. EXCLUDE: Transportation, weather, personal scheduling, or off-site locations\n"
            "3. Require direct textual evidence in the review\n"
            "4. Express characteristics as concise descriptors (1-3 words)\n\n"

            "For EACH bullet point, validate:\n"
            "- Directly concerns the attraction's core features/services\n"
            "- Not affected by external/temporary factors\n"
            "- Not about adjacent locations/activities outside attraction boundaries\n\n"

            "If no valid aspects exist for a section, output 'None'.\n"

            "Now, analyze the following review and extract meaningful likes and dislikes:\n"
            "### Output Format:\n"
            "[Like]\n"
            "- Encapsulate the preferences the user liked in bullet points.\n"
            "If no relevant likes found: None\n\n"
            "[Dislike]\n"
            "- Encapsulate the preferences the user disliked in bullet points.\n"
            "If no relevant dislikes found: None\n\n"
            f"Review: {review_text}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        response = llm.predict(input_text)
        return response
    except Exception as e:
        print(f"[ERROR] Error during LLM call: {e}")
        return ""


# -----------------------------
# 4. Parse LLM response
# -----------------------------
def parse_likes_dislikes(llm_response: str) -> Tuple[List[str], List[str]]:
    """
    Extract [Like] and [Dislike] content from the LLM response text.
    Return (likes, dislikes) two lists.
    """
    likes = []
    dislikes = []

    like_section = False
    dislike_section = False

    for line in llm_response.split('\n'):
        # Mark entering [Like] section
        if line.strip().startswith('[Like]'):
            like_section = True
            dislike_section = False
            continue

        # Mark entering [Dislike] section
        if line.strip().startswith('[Dislike]'):
            like_section = False
            dislike_section = True
            continue

        # Collect specific items
        if like_section and line.strip().startswith('-'):
            likes.append(line.strip('- ').strip())
        elif dislike_section and line.strip().startswith('-'):
            dislikes.append(line.strip('- ').strip())

    return likes, dislikes


# -----------------------------
# 5. Process individual user reviews
# -----------------------------
def process_user_reviews(user_reviews: dict, api_config: dict, max_reviews_per_user: int, model_name: str) -> dict:
    """
    Process the input user_reviews (all reviews of a single user).
    1) Prioritize longer reviews
    2) Extract likes/dislikes from each review
    3) Fill the results back into user_reviews
    """
    user_id = user_reviews['user_id']
    reviews_attraction = user_reviews['reviews_attraction']

    # Sort by review length, prioritize longer reviews
    reviews_attraction.sort(key=lambda x: len(x['text']), reverse=True)

    # Process each review (process up to max_reviews_per_user)
    # Use tqdm to display the processing progress of reviews for this user (if the data is large, many progress bars may be output)
    for review in tqdm(reviews_attraction[:max_reviews_per_user],
                       desc=f"User {user_id}", unit="review", leave=False):
        llm_response = generate_likes_dislikes(review['text'], api_config, model_name)
        likes, dislikes = parse_likes_dislikes(llm_response)
        review['likes'] = likes
        review['dislikes'] = dislikes

    return user_reviews


# -----------------------------
# 6. Save results to JSON Lines
# -----------------------------
def save_to_jsonl(all_user_preferences: list, output_file: str) -> None:
    """
    Write the processed user preference information to a JSON Lines file, one JSON per user per line.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for user_preference in tqdm(all_user_preferences, desc="Saving results", unit="user"):
            json.dump(user_preference, file, ensure_ascii=False)
            file.write('\n')
    print(f"[INFO] Results have been saved to: {output_file}")


# -----------------------------
# 7. Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process user reviews and generate likes/dislikes using multi-threading.")
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='The model to use for generating likes/dislikes.')
    parser.add_argument('--threads', type=int, default=25,
                        help='Number of threads for concurrent processing.')
    parser.add_argument('--user_ids_csv', type=str, default='input/user_id_counts_1200.csv',
                        help='CSV file containing target user_ids.')
    parser.add_argument('--reviews_jsonl', type=str, default='input/user_id_review_merged_attraction.json',
                        help='JSON Lines file containing user reviews.')
    parser.add_argument('--max_reviews_per_user', type=int, default=25,
                        help='Max number of reviews to process per user.')
    parser.add_argument('--output_folder', type=str, default='',
                        help='Output folder to save result .jsonl.')
    args = parser.parse_args()

    # 1) Read target user user_id
    target_user_ids = read_target_user_ids(args.user_ids_csv)

    # 2) Read user reviews, only keep target users
    user_reviews_list = read_user_reviews(args.reviews_jsonl, target_user_ids)

    # 3) Configure API parameters for each model (supplement/modify as needed)
    api_config = {
        'deepseek': {'api_key': os.getenv('DEEPSEEK_API_KEY', ''),
                     'url': 'https://api.deepseek.com/v1/', 'model': 'deepseek-chat'},
        'gpt-4o': {'api_key': os.getenv('Gpt_API_KEY', ''),
                   'url': 'https://api.openai.com/v1/', 'model': 'gpt-4o'},
    }

    if args.model not in api_config:
        raise ValueError(f"Model '{args.model}' is not configured. Available models: {list(api_config.keys())}")

    # 4) Set output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_folder, f"{args.model}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"all_users_likes_dislikes_{args.model}_{timestamp}.jsonl")
    if os.path.exists(output_file):
        os.remove(output_file)

    # 5) Use thread pool for concurrent processing
    max_threads = args.threads
    print(f"[INFO] Using up to {max_threads} threads...")

    all_results = []
    # Construct tasks first, then process concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for user_reviews in user_reviews_list:
            futures.append(
                executor.submit(
                    process_user_reviews,
                    user_reviews,
                    api_config,
                    args.max_reviews_per_user,
                    args.model
                )
            )

        # Use as_completed to get execution results in real-time and display progress in tqdm
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing users",
                           unit="user"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                # Avoid the entire execution crashing due to an exception in a thread; can record or handle it
                print(f"[ERROR] A thread encountered an error: {e}")

    # 6) Save results
    save_to_jsonl(all_results, output_file)


if __name__ == "__main__":
    main()