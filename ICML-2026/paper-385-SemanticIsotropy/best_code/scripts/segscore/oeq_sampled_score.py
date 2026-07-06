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
from typing import List, Dict, Any, Tuple

from scipy.stats import mode

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
logging.getLogger("httpx").setLevel(logging.WARNING)


SYSTEM_PROMPT = """
You are an NLP segmentation and evaluation engine designed to analyze text-based scenarios.
Your primary task is to classify segments of a given text into as either 'True' or 'False' based solely on the provided reference document.
Adhere strictly to the instructions and formatting requirements in the user prompt, ensuring accuracy and consistency.
"""

def create_scoring_prompt(entity: str, reference_doc: str, statements: List[str]) -> str:
    """Create evaluation prompt from reference document and response"""
    base_context = """
You are an NLP segmentation and evaluation engine. Examine the scenario below. You are given:
1. The name of an entity/person/place/thing etc. in <entity> tags.
2. A reference document regarding the entity in <reference_doc> tags.
3. A list of <statements> about the entity that must be classified.

**Factual Classification Task:**
For each `<statement>`, classify it as 'True' or 'False' based solely on the information in the `<reference_doc>` and the context of the preceeding `<statement>`s. Follow these guidelines:
- If a statement is factually accurate and supported by the `<reference_doc>`, classify it as 'True'.
- If a statement is inaccurate, unverifiable, or not supported by the `<reference_doc>`, classify it as 'False'.
- If a statement is partially true, but contains incorrect or unsupported information, classify it as 'False'.
- Do not rely on any external knowledge or context beyond the `<reference_doc>`.

**Error Handling:**
- If the `<statement>` contains unparseable text, incomplete sentences, or conflicting information that cannot be resolved using the `<reference_doc>`, include the flagged statement as is and classify it as 'False'.

Examples:
##### EXAMPLE 1 ######
Entity:
<entity>
London, UK
</entity>

Reference Document:
<reference_doc>
London, England's capital, boasts a rich history spanning millennia. Founded by the Romans as Londinium around 47 AD, it became a major port and trading center. After the Roman withdrawal, Anglo-Saxons established Lundenwic, which later fell to Viking raids. The Norman Conquest in 1066 led to the construction of the Tower of London, a symbol of royal power. London thrived during the medieval period, becoming a major center for trade, finance, and culture. It weathered plagues, fires, and civil wars, emerging as a global metropolis and the heart of the British Empire. Today, London remains a vibrant hub, blending its historical legacy with modern dynamism, home to over 9 million people.
</reference_doc>

Statements:
<statements>
1. <statement>London, the capital city of England and the United Kingdom</statement>
2. <statement>is a vibrant metropolis steeped in history and brimming with modern energy</statement>
3. <statement>With a population of over 9 million people</statement>
</statements>

Classifications:
<classifications>
1. True
2. True
3. True
</classifications>

########################
##### EXAMPLE 2 ######
Entity:
<entity>
Obsidian
</entity>

Reference Document:
<reference_doc>
Obsidian is a naturally occurring volcanic glass formed from rapidly cooling lava. Its glassy texture and conchoidal fracture result from minimal crystal growth during the cooling process. Typically jet-black, obsidian can also appear red, brown, or even iridescent due to the presence of mineral inclusions.

Prized for its sharpness and beauty since ancient times, obsidian was used for tools, weapons, and ornaments. Its glassy nature made it ideal for crafting arrowheads, knives, and mirrors. Today, obsidian remains popular in jewelry and decorative objects.

Found in volcanic regions worldwide, obsidian provides valuable insights into volcanic activity and Earth's geological processes. Obsidian relics have been found at ancient sites in Syria, Israel and Mexico.
</reference_doc>

Statements:
<statements>
1. <statement>Obsidian is a naturally occurring volcanic glass formed when lava cools rapidly, preventing the formation of crystalline structures</statement>
2. <statement>Its amorphous, non-crystalline structure gives it a smooth, homogeneous texture, making it distinct from most igneous rocks</statement>
3. <statement>Obsidian is remarkably brittle yet strong, with a Mohs hardness of about 1-2</statement>
4. <statement>Its unique fracture pattern, known as conchoidal fracturing, allows it to be shaped into extremely sharp edges, sharper than even modern steel surgical scalpels</statement>
5. <statement>This quality made obsidian a vital material for crafting tools and weapons in ancient cultures</statement>
6. <statement>and continues to find use in precision cutting applications in modern surgery</statement>
7. <statement>Obsidian tools have been found at historical sites such as Tell Brak, Gilat and</statement>
8. <statement>beaches in Seychelles</statement>
</statements>

Classifications:
<classifications>
1. True
2. True
3. True
4. False
5. True
6. True
7. True
8. False
</classifications>

########################
Entity:
<entity>
"""
    formatted_statements = "\n".join([f"{i+1}. <statement>{s}</statement>" for i, s in enumerate(statements)])
    dynamic_context = f"""{entity}
</entity>

Reference Document:
<reference_doc>
{reference_doc}
</reference_doc>

Statements:
<statements>
{formatted_statements}
</statements>
"""
    prompt = f"{base_context}{dynamic_context}"
    return prompt

def parse_classification_labels(res: str) -> List[bool]:
    matches = re.findall(r'\d+\.\s*(True|False)', res)
    return [match.lower() == 'true' for match in matches]

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
                'statements': response['statements'],
                'sampled_labels': response['sampled_labels']
            } for response in result['responses']]
        }
        formatted_results[i] = result_entry
    existing_results.extend(formatted_results)
    return existing_results

def process_statements(args: Tuple[str, str, str, TokenRateLimiter, int, int]) -> Tuple[int, str, Dict]:
    """Process a single statement through the API"""
    model, prompt, system_prompt, rate_limiter, samples, statement_count = args

    # Estimate tokens for this request (multiply by samples to account for all samples)
    estimated_tokens = (estimate_tokens(prompt + system_prompt) + statement_count + estimate_tokens("<classifications></classifications>"))*samples
    rate_limiter.add_tokens(estimated_tokens)

    responses = []
    for i in range(samples):
        try:
            ml = model.lower()
            if ml == "openai":
                response = query_api(prompt, api=ml, system=system_prompt, \
                                     logprobs=False, model="gpt-4.1-mini")
            elif ml == "deepseek":
                response = query_api(prompt, api=ml, system=system_prompt, logprobs=False, model='deepseek-chat')
            elif ml == "claude":
                response = query_api(prompt, api=ml, system=system_prompt, logprobs=False, model="claude-sonnet-4-20250514")
            elif ml == "gemini":
                response = query_api(prompt, api=ml, system=system_prompt, logprobs=False, model="gemini-2.5-flash")
            responses.append(response['response'])
        except Exception as e:
            logger.error(f"Error processing sample {i} of response: {e}")
            responses.append("")
    return responses

@click.command()
@click.option('--input-path', required=True, help='Path to the input CSV file of generated open ended responses')
@click.option('--output-path', required=True, help='Output path for the evaluation results')
@click.option('--group-batch-size', default=20, help='Number of entries to process before writing')
@click.option('--subset', default=None, type=int, help='Subset of data to process (testing purposes)')
@click.option('--samples', default=3, type=int, help='Number of samples to generate for each response')
@click.option('--models', default='gpt-4o-mini',
              help='Comma-separated list of models to use for scoring (e.g., gpt-4o-mini,gpt-3.5-turbo)',
              callback=lambda ctx, param, value: value.split(',') if value else [])
@click.option('--token-limits', default=None,
              help='Comma-separated list of token limits for each model (e.g., 190000,190000). Defaults to 200000.',
              callback=lambda ctx, param, value: [int(x) for x in value.split(',')] if value else None)
@click.option('--dataset', default='triviaqa', help='Dataset: TriviaQA (triviaqa), BookSummaries (bs) or FactScore-Bio (factscore)', type=click.Choice(['triviaqa', 'factscore', 'bs']))
@click.option('--overwrite', is_flag=True, help='Overwrite output file if it exists')
@click.option('--dryrun', is_flag=True, help='Simulate pipeline execution without making API calls')
@click.option('--seed', default=None, type=int, help='Random seed for reproducible sampling')
@click.option('--max-workers', default=10, help='Maximum number of concurrent API calls')
@click.option('--restart-from-checkpoint', is_flag=True, help='Restart from checkpoint. Resume from the last checkpoint file in the output file.')
@click.pass_context  # Add this decorator to get Click's context
def main(ctx: click.Context, input_path: str, output_path: str, group_batch_size: int, subset: int, samples: int,
         models: List[str], token_limits: List[int], dataset: str, overwrite: bool, dryrun: bool, seed: int, max_workers: int, restart_from_checkpoint: bool):
    """Evaluate and score individual statements against reference documents to estimate true labels via sampling"""

    output_dir = os.path.dirname(output_path)
    save_config(output_dir, ctx, dryrun)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Set random seed to {seed}")

    if dryrun:
        logger.info("DRY RUN MODE - No API calls or file writes will be performed")

    if not token_limits:
        token_limits = [200000] * len(models)

    assert len(models) == len(token_limits), "Number of models and token limits must match."
    assert not (restart_from_checkpoint and overwrite), "Cannot restart from checkpoint and overwrite at the same time. Please use either --overwrite or --restart-from-checkpoint, not both."

    logger.info("Dry run mode - configuration:" if dryrun else "Configuration:")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Token limits: {', '.join(map(str, token_limits))}")
    logger.info(f"Samples: {samples}")
    logger.info(f"Group batch size: {group_batch_size}")
    logger.info(f"Subset size: {subset}")

    # Check if output file exists at program start
    output_fp = Path(output_path)
    existing_results_map = {}
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
            logger.info("No new keys to run. All entries have been processed and line up between the checkpoint file and the input file.")
            return

        data = [d for d in data if f"{d['index']}-{d['idx_cat']}" in keys_to_run]
        for k in keys_to_run:
            if k in existing_results_map:
                del existing_results_map[k]

        logger.info(f"Subsetted data to {len(keys_to_run)} keys to process from checkpoint file.")

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

    logger.info("Starting sampled scoring.")
    rate_limiters = {model: TokenRateLimiter(limit) for model, limit in zip(models, token_limits)}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for group in tqdm(data, desc="Scoring"):
                index = group['index']
                idx_cat = group['idx_cat']
                entity = group['entity']
                key = f"{index}-{idx_cat}"

                # Get reference document from relevant dataset
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
                        reference_doc = factscore["Paul O'Neill (racing driver)"]
                    else:
                        reference_doc = factscore[entity]
                    entity_page_idx = 0
                elif dataset == 'bs':
                    reference_doc = bs[entity]
                    entity_page_idx = 0

                try:
                    result = {
                        'index': index,
                        'idx_cat': idx_cat,
                        'entity': entity,
                        'entity_page_idx': entity_page_idx,
                        'responses': []
                    }
                    response_map = {i: response for i, response in enumerate(group['responses'])}
                    task_tuples = []
                    for model in models:
                        for i, response in enumerate(group['responses']):
                            task_tuples.append((i, model, (model, create_scoring_prompt(entity=entity,
                                                    reference_doc=reference_doc,
                                                    statements=[s['text'] for s in response['statements']]),
                                SYSTEM_PROMPT, rate_limiters.get(model), samples, len(response['statements']))))

                    scored_responses = []
                    if dryrun:
                        # Mock responses in dryrun mode
                        scored_responses = [(task[0], task[1], [[True] * task[-1][-1]]*samples) for task in task_tuples]
                    else:
                        # Submit all tasks to the thread pool
                        future_to_stmt = {
                            executor.submit(process_statements, task[-1]): task
                            for task in task_tuples
                        }

                        # Collect results as they complete
                        for future in as_completed(future_to_stmt):
                            task = future_to_stmt[future]
                            try:
                                fut_result = future.result()
                                scored_responses.append((task[0], task[1], [parse_classification_labels(res) for res in fut_result]))
                            except Exception as e:
                                logger.error(f"Task failed for entity \"{entity}\", response index {task[0]} and model {task[1]}. Error: {e}")
                                continue

                    has_failures = False
                    for scored_response in scored_responses:
                        idx, model, sampled_labels = scored_response
                        response = response_map[idx]
                        if 'sampled_labels' not in response:
                            response['sampled_labels'] = {model: sampled_labels}
                        else:
                            response['sampled_labels'][model] = sampled_labels

                        if not len(set([len(x) for x in sampled_labels])) == 1:
                            logger.warning(f"Number of sampled labels varies: ({[str(len(x)) for x in sampled_labels]}). Will continue if majority of sampled labels are correct.")
                            response_lengths = mode([len(x) for x in sampled_labels])
                            if response_lengths.mode != len(response['statements']):
                                logger.warning(f"Majority of sampled labels are incorrect for entity {entity}, response {idx}, model {model}. Will flag this entry.")
                                has_failures = True
                                continue
                            correct_samples = [sampled_labels[i] for i in range(len(sampled_labels)) if len(sampled_labels[i]) == response_lengths.mode]
                            if not len(correct_samples):
                                logger.warning(f"All sampled labels are incorrect for entity {entity}, response {idx}, model {model}. Will flag this entry.")
                                has_failures = True
                                continue
                            response['sampled_labels'][model] = correct_samples
                        response_map[idx] = response

                    result['responses'] = [response_map[i] for i in range(len(group['responses']))]

                    if has_failures:
                        failing_entry = {
                            'index': index,
                            'idx_cat': idx_cat,
                            'entity': entity,
                            'entity_page_idx': entity_page_idx,
                            'responses': result['responses'],
                            'reason': 'sampled_labels_mismatch'
                        }
                        failed_entries.append(failing_entry)
                    else:
                        results_buffer.append(result)
                        response_result_count += len(result['responses'])

                    # Write results when buffer reaches batch size
                    if len(results_buffer) >= group_batch_size:
                        if not dryrun:
                            write_results(output_path, results_buffer, coalesce_results)
                            logger.info(f"Wrote batch of {len(results_buffer)} results containing {response_result_count} responses.")
                        else:
                            logger.info(f"DRY RUN: Would write batch of {len(results_buffer)} results containing {response_result_count} responses.")
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
                        'responses': [{'response': row['response'], 'logprobs': row['logprobs'], 'statements': row['statements'], 'sampled_labels': {}} for row in group['responses']],
                        'reason': str(e)
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
                        'responses': [{'response': row['response'], 'logprobs': row['logprobs'], 'statements': row['statements'], 'sampled_labels': {}} for row in group['responses']],
                        'reason': str(e)
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

        logger.info("Sampled scoring complete!")
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
