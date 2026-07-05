import os
import re
import logging
import click
import numpy as np
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from semantic_isotropy.prompts.segscore import SYSTEM_PROMPT, create_prompt
from semantic_isotropy.llm.api import chat_api as query_api
from semantic_isotropy.llm.utils import estimate_tokens, TokenRateLimiter
from semantic_isotropy.datasets.utils import get_entity_page_idx
from semantic_isotropy.datasets.loaders import load_triviaqa, load_factscore, load_data, load_booksummaries
from semantic_isotropy.pipeline.utils import save_config, write_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def parse_response(response_text, logprobs):
    # Regular expression to extract statements and their classes
    statement_pattern = re.compile(r"<statement>(.*?)</statement>\s*<class>(1|0)</class>")

    # Find all matches in the response
    matches = statement_pattern.findall(response_text)

    # Verify that all <statement> tags have corresponding <class> tags
    if len(re.findall(r"<statement>(.*?)</statement>", response_text)) != len(matches):
        raise ValueError("Mismatch between the number of <statement> and <class> tags in the response.")

    # Initialize the result list
    results = []

    # Iterate over matches and pair them with corresponding logprobs
    token_index = 0
    for statement, cls in matches:
        # Find the logprob of the class (True or False)
        logprob_raw = None
        top_probs = None
        prob_norm = None
        total_prob = None
        while token_index < len(logprobs['tokens']):
            token = logprobs['tokens'][token_index]

            # Check if the token matches the class and is preceded by "<class>" and followed by "</class>"
            if token.lower() == cls.lower():
                #print(f"Found class!: {token} at index {token_index}")
                # Backtrack to check if "<class>" is present before the class token
                backtrack_index = max(0, token_index - 7)  # Limit backtracking to at most 7 tokens
                found_class_prefix = ''.join(logprobs['tokens'][backtrack_index:token_index]).lower().endswith("<class>")
                #print(f"Found class prefix <class>: {found_class_prefix}")
                # Check for "</class>" after the class token
                #print(f"Token (Token Index): {logprobs['tokens'][token_index]}({token_index})")
                suffix_index = token_index + 1
                class_suffix = ''.join(logprobs['tokens'][suffix_index:suffix_index + 7])
                #print(f"Suffix found (from_idx: to_idx): {class_suffix}({suffix_index}:{suffix_index+7})")
                found_class_suffix = class_suffix.lower().startswith("</class>")
                #print(found_class_suffix)
                if found_class_prefix and found_class_suffix:
                    top_logprobs_raw = logprobs['top_logprobs'][token_index]
                    prob_dict = {t.token: np.exp(t.logprob) for t in top_logprobs_raw}
                    prob_0 = prob_dict.get('0', -1)
                    prob_1 = prob_dict.get('1', -1)
                    if np.isclose(prob_0, 1) and prob_1 == -1:
                        logger.debug(f"Could not find matching logprob for: {statement}: True => {top_logprobs_raw}. However, opposite case is found to be close to 100%. Setting to 0.0")
                        prob_1 = 0.0
                    elif np.isclose(prob_1, 1) and prob_0 == -1:
                        logger.debug(f"Could not find matching logprob for: {statement}: False => {top_logprobs_raw}. However, opposite case is found to be close to 100%. Setting to 0.0")
                        prob_0 = 0.0
                    elif np.isclose(prob_0, -1) and np.isclose(prob_1, -1):
                        logger.warning(f"Could not find matching logprob for: {statement}: {cls} => {top_logprobs_raw}")
                        prob_0, prob_1 = -1, -1
                    total_prob = prob_0 + prob_1 if prob_0 != -1 and prob_1 != -1 else 1
                    top_probs = {'0': prob_0, '1': prob_1}
                    logprob_raw = logprobs['logprobs'][token_index]
                    prob_norm = np.exp(logprob_raw)/(total_prob)
                    token_index += 1
                    break

            token_index += 1

        # Append the statement tuple to the results
        results.append((statement.strip(), cls, logprob_raw, prob_norm, top_probs))

    return results

def coalesce_results(existing_results: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted_results = [None] * len(results)
    # Update with new results
    for i, result in enumerate(results):
        # Convert numpy types to native Python types
        result_entry = {
            'index': int(result['index']),  # Convert np.int64 to int
            'idx_cat': result['idx_cat'],
            'entity': result['entity'],
            'entity_page_idx': result['entity_page_idx'],
            'responses': [{
                'response': response['response'],
                'logprobs': response['logprobs'],
                'statements': response['statements']
            } for response in result['responses']]
        }
        formatted_results[i] = result_entry
    existing_results.extend(formatted_results)
    return existing_results

def process_row(idx, row, entity, reference_doc, key, model, rate_limiter, dryrun):
    """Process a single row using threads"""
    logger.debug(f"Processing row {idx} for {key} at {pd.Timestamp.now()}")

    if not row['response'] or pd.isna(row['response']):
        logger.warning(f"Skipping empty response for {key}")
        return None

    if dryrun:
        logger.debug(f"DRY RUN: Would process response for {key}")
        parsed_statements = [("This is a mock statement 1", '1', 0.1, 0.1, {'0': 0.1, '1': 0.1}),
                              ("This is a mock statement 2", '0', 0.1, 0.1, {'0': 0.1, '1': 0.1})]
    else:
        prompt = create_prompt(entity=entity, reference_doc=reference_doc, response=row['response'])

        estimated_tokens = estimate_tokens(prompt + SYSTEM_PROMPT) + estimate_tokens(row['response']) + estimate_tokens("<statements>" + "<statement></statement><class>1</class>"*40 + "</statements>")
        rate_limiter.add_tokens(estimated_tokens)
        result = query_api(prompt, api='openai', system=SYSTEM_PROMPT,
                             logprobs=True, top_logprobs=2, model=model)
        logger.debug(f"{key}: {idx} Completed API call.")
        logprobs = {"tokens": [x.token for x in result['logprobs']],
                 "logprobs": [x.logprob for x in result['logprobs']],
                 "top_logprobs": [(x.top_logprobs[0], x.top_logprobs[1]) for x in result['logprobs']]}

        # Parse the response
        try:
            parsed_statements = parse_response(result['response'], logprobs)
        except Exception as e:
            if 'Mismatch' in str(e):
                logger.error(f"Error parsing response for {key}: {e}. Retrying with temperature=0.0.")
                rate_limiter.add_tokens(estimated_tokens)
                result = query_api(prompt, api='openai', temperature=0.0, system=SYSTEM_PROMPT,
                                    logprobs=True, top_logprobs=2, model=model)
                logprobs = {"tokens": [x.token for x in result['logprobs']],
                        "logprobs": [x.logprob for x in result['logprobs']],
                        "top_logprobs": [(x.top_logprobs[0], x.top_logprobs[1]) for x in result['logprobs']]}
                parsed_statements = parse_response(result['response'], logprobs)

    if not parsed_statements:
        logger.warning(f"No statements found for {key}")
        return (True, {
            'response': row['response'],
            'logprobs': row['logprobs'],
            'statements': [],
            'reason': 'no_statements'
        })

    return (False, {
        'response': row['response'],
        'logprobs': row['logprobs'],
        'statements': [{'text': statement, 'class': 'True' if cls == '1' else 'False',
                        'logprob_raw': logprob_raw,
                        'prob_norm': prob_norm,
                        'top_probs': top_probs}
                        for statement, cls, logprob_raw, prob_norm, top_probs in parsed_statements]
    })

def process_group(group, entity, reference_doc, key, model, rate_limiter, dryrun, max_workers):
    """Process a group of rows with concurrent API calls using ThreadPoolExecutor"""
    results = []

    # Create a partial function with the common arguments
    process_row_partial = partial(process_row, entity=entity, reference_doc=reference_doc, key=key, model=model, rate_limiter=rate_limiter, dryrun=dryrun)

    # Use ThreadPoolExecutor to process rows concurrently

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_row_partial, idx, row): (idx, row)
            for idx, row in enumerate(group['responses'])
        }

        # Collect results as they complete
        for future in as_completed(future_to_row):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                idx, row = future_to_row[future]
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.append((True, {
                    'response': row['response'],
                    'logprobs': row['logprobs'],
                    'statements': [],
                    'reason': str(e)
                }))

    return results

@click.command()
@click.option('--input-path', required=True, help='Path to the input CSV file of generated open ended responses')
@click.option('--output-path', required=True, help='Output path for the evaluation results')
@click.option('--group-batch-size', default=20, help='Number of entries to process before writing')
@click.option('--subset', default=None, type=int, help='Subset of data to process (testing purposes)')
@click.option('--model', default='gpt-4.1-mini', help='Model to use for segmentation')
@click.option('--dataset', default='triviaqa', help='Dataset: TriviaQA (triviaqa), BookSummaries (bs), or FactScore-Bio (factscore)', type=click.Choice(['triviaqa', 'factscore', 'bs']))
@click.option('--overwrite', is_flag=True, help='Overwrite output file if it exists')
@click.option('--dryrun', is_flag=True, help='Simulate pipeline execution without making API calls')
@click.option('--seed', default=None, type=int, help='Random seed for reproducible sampling')
@click.option('--token-limit', default=150000, type=int, help='Max tokens per minute for model')
@click.option('--max-workers', default=10, help='Maximum number of concurrent API calls')
@click.option('--restart-from-checkpoint', is_flag=True, help='Restart from checkpoint. Resume from the last checkpoint file in the output file.')
@click.pass_context  # Add this decorator to get Click's context
def main(ctx: click.Context, input_path: str, output_path: str, group_batch_size: int, subset: int,
         model: str, dataset: str, overwrite: bool, dryrun: bool, seed: int, max_workers: int, token_limit: int, restart_from_checkpoint: bool):
    """Segment and evaluate responses against TriviaQA reference documents"""

    output_dir = os.path.dirname(output_path)
    save_config(output_dir, ctx, dryrun)

    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")

    if dryrun:
        logger.info("DRY RUN MODE - No API calls or file writes will be performed")

    assert not (restart_from_checkpoint and overwrite), "Cannot restart from checkpoint and overwrite at the same time. Please use either --overwrite or --restart-from-checkpoint, not both."

    # Check if output file exists at program start
    output_fp = Path(output_path)
    if output_fp.exists() and output_fp.stat().st_size > 0:
        if overwrite:
            if not dryrun:
                logger.info(f"Removing existing output file: {output_path}")
                output_fp.unlink()
            else:
                logger.info(f"DRY RUN: Would remove existing output file: {output_path}")
        elif restart_from_checkpoint:
            logger.info(f"Restarting from checkpoint: {output_path}")
            existing_results = load_data(output_path)
            existing_results_map = { f"{r['index']}-{r['idx_cat']}": r for r in existing_results }
        else:
            raise ValueError(f"Output file already exists and is non-empty: {output_path}. Use --overwrite to replace.")

    # Load data
    data = load_data(input_path)

    if not restart_from_checkpoint and subset and subset < len(data):
        logger.info(f"Sampling {subset} examples from {len(data)} total entries")
        data = random.sample(data, subset)
    elif restart_from_checkpoint and subset:
        raise ValueError("Cannot use --restart-from-checkpoint and --subset at the same time. Please use either --overwrite or --restart-from-checkpoint, not both.")

    if restart_from_checkpoint:
        keys_to_run = set()
        for d in data:
            key = f"{d['index']}-{d['idx_cat']}"
            if key not in existing_results_map or len(existing_results_map[key]['responses']) < len(d['responses']):
                keys_to_run.add(key)

        if len(keys_to_run) == 0:
            logger.info("No new keys to run. All entries have been processed and line up with between the checkpoint file and the input file.")
            return

        data = [d for d in data if f"{d['index']}-{d['idx_cat']}" in keys_to_run]
        for k in keys_to_run:
            if k in existing_results_map:
                del existing_results_map[k]

        logger.info(f"Subsetted data to {len(keys_to_run)} bad keys in checkpoint file.")

        if not dryrun:
            logger.info(f"Removing checkpoint file: {output_path}")
            output_fp.unlink()
            write_results(output_path, existing_results_map.values(), coalesce_results)
        else:
            logger.info(f"DRY RUN: Would remove checkpoint file: {output_path}")

    results_buffer = []
    response_result_count = 0
    failed_entries = []

    if dataset == 'triviaqa':
        triviaqa = load_triviaqa()
    elif dataset == 'factscore':
        factscore = load_factscore()
    elif dataset == 'bs':
        bs = load_booksummaries()
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Starting segmentation and scoring.")
    rate_limiter = TokenRateLimiter(token_limit)

    try:
        for group in tqdm(data, desc="Scoring"):
            index = group['index']
            idx_cat = group['idx_cat']
            entity = group['entity']
            key = f"{index}-{idx_cat}"

            # Get reference document from TriviaQA
            if dataset == 'triviaqa':
                dataset_split = triviaqa['train'] if idx_cat == 'train' else triviaqa['validation']
                trivia_entry = dataset_split[index]
                entity_page_idx = get_entity_page_idx(entity, trivia_entry)
                if entity_page_idx == -1:
                    logger.warning(f"Skipping group {key} {entity}: No valid pages found for entity \"{entity}\" in trivia entry {trivia_entry['question_id']}")
                    continue
                reference_doc = trivia_entry['entity_pages']['wiki_context'][entity_page_idx]
            elif dataset == 'factscore':
                if entity == "Paul ONeill (racing driver)":
                    reference_doc = factscore["Paul O'Neill (racing driver)"] # Correct edge case.
                else:
                    reference_doc = factscore[entity]
                entity_page_idx = 0
            elif dataset == 'bs':
                reference_doc = bs[entity]
                entity_page_idx = 0

            try:
                group_results = process_group(group, entity, reference_doc, key, model, rate_limiter, dryrun, max_workers)

                # Separate successful and failed results
                failed_results = [result for is_failed, result in group_results if is_failed]
                success_results = [result for is_failed, result in group_results if not is_failed]

                if failed_results:
                    failing_entry = {
                        'index': index,
                        'idx_cat': idx_cat,
                        'entity': entity,
                        'entity_page_idx': entity_page_idx,
                        'responses': [{'response': result['response'], 'logprobs': result['logprobs'], 'statements': result['statements'], 'reason': result['reason']} for result in failed_results],
                    }
                    failed_entries.append(failing_entry)

                if success_results:
                    success_entry = {
                        'index': index,
                        'idx_cat': idx_cat,
                        'entity': entity,
                        'entity_page_idx': entity_page_idx,
                        'responses': [{'response': result['response'], 'logprobs': result['logprobs'], 'statements': result['statements']} for result in success_results],
                    }
                    results_buffer.append(success_entry)
                    response_result_count += len(success_results)

                    # Write results when buffer reaches batch size
                if len(results_buffer) >= group_batch_size:
                    if not dryrun:
                        write_results(output_path, results_buffer, coalesce_results)
                        logger.info(f"Wrote batch of {len(results_buffer)} results containing {response_result_count} responses.")
                    else:
                        logger.info(f"DRY RUN: Would write batch of {len(results_buffer)} results containing {response_result_count} statements.")
                    results_buffer = []
                    response_result_count = 0

            except (ConnectionError, TimeoutError) as e:
                # Handle API-specific errors that shouldn't crash the program
                logger.error(f"API error processing group {key}: {str(e)}")
                failing_entry = {
                    'index': index,
                    'idx_cat': idx_cat,
                    'entity': entity,
                    'entity_page_idx': entity_page_idx,
                    'responses': [{'response': row['response'], 'logprobs': row['logprobs'], 'statements': [], 'reason': str(e)} for row in group['responses']],
                }
                failed_entries.append(failing_entry)
                continue
            except Exception as e:
                # Log unexpected errors but allow the process to continue
                logger.error(f"Unexpected error processing group {key}: {str(e)}")
                failing_entry = {
                    'index': index,
                    'idx_cat': idx_cat,
                    'entity': entity,
                    'entity_page_idx': entity_page_idx,
                    'responses': [{'response': row['response'], 'logprobs': row['logprobs'], 'statements': [], 'reason': str(e)} for row in group['responses']],
                }
                failed_entries.append(failing_entry)
                continue

        # Write any remaining results
        if results_buffer:
            if not dryrun:
                write_results(output_path, results_buffer, coalesce_results)
                logger.info(f"Wrote final batch of {len(results_buffer)} results containing {response_result_count} responses.")
            else:
                logger.info(f"DRY RUN: Would write final batch of {len(results_buffer)} results containing {response_result_count} responses.")

        # Write failed entries report
        if failed_entries:
            failed_path = output_path + '.failed.json'
            if not dryrun:
                with open(failed_path, 'w') as f:
                    json.dump(failed_entries, f, indent=2)
                logger.warning(f"Wrote {len(failed_entries)} failed entries to {failed_path}")
            else:
                logger.info(f"DRY RUN: Would write {len(failed_entries)} failed entries to {failed_path}")

        logger.info("Segmentation and scoring complete!")
        if not dryrun:
            file_path = Path(f"{output_path}.done")
            file_path.touch(mode=0o777, exist_ok=True)

    except (FileNotFoundError, PermissionError) as e:
        # Handle file-related errors
        logger.error(f"File system error: {str(e)}")
        raise
    except KeyboardInterrupt:
        # Handle clean shutdown
        logger.info("Process interrupted by user")
        raise
    except Exception as e:
        # Handle truly unexpected errors
        logger.error(f"Fatal error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
