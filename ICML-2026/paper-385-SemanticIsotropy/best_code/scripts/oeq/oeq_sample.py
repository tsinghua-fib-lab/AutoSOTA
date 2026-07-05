import click
import os
import logging
from typing import List, Dict, Any
import json
from pathlib import Path
from tqdm import tqdm

from semantic_isotropy.datasets.loaders import load_csv
from semantic_isotropy.datasets.utils import strip_and_return
from semantic_isotropy.pipeline.utils import init_logger
from semantic_isotropy.llm.query import LLM, SamplingParams
from semantic_isotropy.prompts.oeq import create_prompts
from semantic_isotropy.pipeline.utils import save_config, write_results
from semantic_isotropy.llm.utils import detect_api_model


logger = init_logger(__name__, 'INFO')
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def coalesce_results(existing_results: List[Dict[str, Any]], result_map: Dict[str, Any]) -> Dict[str, Any]:
    all_results_map = {f'{k["index"]}-{k["idx_cat"]}': k for k in existing_results} if len(existing_results) > 0 else {}

    # Update with new results
    for key, results in result_map.items():
        # Convert numpy types to native Python types
        if key in all_results_map:
            k = all_results_map[key]
            responses = k['responses']
            responses.extend([{
                    'response': result['response'],
                    'logprobs': result['logprobs']
                } for result in results])
            k['responses'] = responses
            all_results_map[key] = k
        else:
            result_entry = {
                'index': int(results[0]['index']),  # Convert np.int64 to int
                'idx_cat': results[0]['idx_cat'],
                'entity': results[0]['entity'],
                'responses': [{
                    'response': result['response'],
                    'logprobs': result['logprobs']
                } for result in results]
            }
            all_results_map[key] = result_entry
    return list(all_results_map.values())

def process_batch_outputs(batch_metadata_to_process, batch_prompts_to_process, result_map, llm, sampling_params, dry_run):
    """Process batch outputs from LLM and prepare results for writing.

    Args:
        outputs: List of LLM generation outputs
        batch_metadata_to_process: List of metadata for each prompt
        batch_prompts_to_process: List of input prompts
        result_map: Dictionary to store processed results

    Returns:
        Updated result_map with new results
    """
    if dry_run:
        result_map = {}
        for metadata in batch_metadata_to_process:
            key = f'{metadata["index"]}-{metadata["idx_cat"]}'
            if key not in result_map:
                result_map[key] = []
            result_map[key].append({
                'index': metadata['index'],
                'idx_cat': metadata['idx_cat'],
                'entity': metadata['entity'],
                'open_ended_question': metadata['question'],
                'response': 'Dry run mode - no LLM calls will be made',
                'logprobs': [0.4, 0.4, 0.4, 0.4, 0.4]
            })
        return result_map, [], []

    outputs = llm.generate(batch_prompts_to_process, sampling_params)

    failed_prompts = []
    failed_metadata = []

    for idx, (output, prompt, metadata) in enumerate(zip(outputs, batch_prompts_to_process, batch_metadata_to_process)):
        try:
            res = output.outputs[0].text
            res_split = res.split("<response>")
            res_tail_split = res_split[1].split("</response>")
            response, left_trim, _ = strip_and_return(res_tail_split[0])
            len_prefix = len(llm.get_tokenizer().encode(res_split[0] + "<response>" + left_trim))
            logprobs = output.outputs[0].logprobs[len_prefix - 1:len_prefix + len(llm.get_tokenizer().encode(response)) - 1]
            assert not len(logprobs) or len(logprobs) == len(llm.get_tokenizer().encode(response)), f"Logprobs length does not match response length for prompt {idx} {metadata['entity']}."
            assert len(response) > 100, f"Response length is too short for prompt {idx} {metadata['entity']}."
        except Exception as e:
            logger.error(f"Error processing prompt {idx}: {e}")
            failed_prompts.append(prompt)
            failed_metadata.append(metadata)
            continue

        key = f'{metadata["index"]}-{metadata["idx_cat"]}'
        if key not in result_map:
            result_map[key] = []

        if llm.api_model:
            logprobs_values = logprobs if sampling_params.top_logprobs == 1 else [round(x.logprob, 8) for x in logprobs]
        else:
            logprobs_values = [round(sorted(x.values(), key=lambda x: x.logprob, reverse=True)[0].logprob, 8) for x in logprobs]

        result_map[key].append({
            'index': metadata['index'],
            'idx_cat': metadata['idx_cat'],
            'entity': metadata['entity'],
            'open_ended_question': metadata['question'],
            'response': response,
            'logprobs': logprobs_values
        })

    return result_map, failed_prompts, failed_metadata

@click.command()
@click.option('--model', required=True, help='Name or path of the model to use. Either a HuggingFace model name or an an API based model (e.g. "openai/gpt-4o-mini") etc.')
@click.option('--input-path', required=True, help='Path to the input JSON file from open_ended_trivia.py')
@click.option('--output-path', required=True, help='Path to save the output results')
@click.option('--n', default=5, help='Number of responses to generate per question')
@click.option('--word-count', default=500, help='Number of words to generate per response')
@click.option('--batch-size', default=8, help='Batch size for vLLM inference. -1 when using API based models.')
@click.option('--group-batch-size', default=20, help='Batch size for group writing')
@click.option('--subset', default=None, type=int, help='Number of questions to sample (for testing)')
@click.option('--dtype', default='BFloat16', help='Data type for model precision (e.g., half, BFloat16)')
@click.option('--temperature', default=0.7, help='Temperature for sampling')
@click.option('--tensor_parallel_size', default=1, help='Multi-GPU Parallelism to use (Machine GPU count)')
@click.option('--logprobs', is_flag=True, default=False, help='Whether to return logprobs for generated tokens')
@click.option('--top-logprobs', default=3, type=int, help='Number of top logprobs to return per token (defaults to 3)')
@click.option('--dryrun', is_flag=True, help='Run in dry run mode without making actual LLM calls')
@click.option('--restart-from-checkpoint', is_flag=True, help='Restart from checkpoint. Resume from the last checkpoint file in the output file.')
@click.pass_context
def main(ctx: click.Context, model: str, input_path: str, output_path: str, n: int, word_count: int, batch_size: int,
         group_batch_size: int, subset: int, dtype: str, temperature: float, tensor_parallel_size: int, logprobs: bool, top_logprobs: int, dryrun: bool,
         restart_from_checkpoint: bool):
    """Generate responses for open-ended questions using vLLM"""

    if '-batch' in model and batch_size != -1:
        raise ValueError("Batch size must be -1 when using API based models in batch mode!")

    if batch_size == -1 and '-batch' in model:
        logger.info("Using API based model in batch mode - batch size will be set to a very large int")
        batch_size = int(1e9)

    top_logprobs = top_logprobs if logprobs else 1

    output_dir = os.path.dirname(output_path)
    save_config(output_dir, ctx, False)

    if dryrun:
        logger.info("Running in dry run mode - no LLM calls will be made")

    if restart_from_checkpoint:
        if os.path.exists(output_path):
            logger.info(f"Restarting from checkpoint: {output_path}")
            with open(output_path, 'r') as f:
                existing_results = json.load(f)
            existing_results_map = {f'{k["index"]}-{k["idx_cat"]}': k for k in existing_results} if len(existing_results) > 0 else {}
            logger.info(f"Loaded {len(existing_results_map)} results from checkpoint. Resuming from the last checkpoint.")
        else:
            logger.info(f"No checkpoint found at {output_path}")
            restart_from_checkpoint = False

    # Load data
    df = load_csv(input_path).sort_values(by="index")

    if df.empty:
        raise ValueError("Input data is empty")
    logger.info(f"Loaded {len(df)} examples")

    logger.info(f"Processing {len(df)} examples with:")
    logger.info(f"- Model: {model}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Group batch size: {group_batch_size}")
    logger.info(f"- Responses per question: {n}")
    logger.info(f"- Output path: {output_path}")
    logger.info(f"- Logprobs: {logprobs}")
    logger.info(f"- Top logprobs: {top_logprobs}")
    logger.info(f"- Restarting from checkpoint: {restart_from_checkpoint}")

    if dryrun:
        output_path = output_path.replace('.json', '.dryrun.json')
        logger.info(f"Dry run mode - output path: {output_path}")

    # Create unique key and group data
    if not restart_from_checkpoint and subset and subset < len(df):
        logger.info(f"Sampling {subset} examples from {len(df)} total examples")
        df = df.sample(subset).sort_values(by="index")

    df['_n'] = n
    if restart_from_checkpoint:
        df['index_locator'] = df.apply(lambda x: f'{x["index"]}-{x["idx_cat"]}', axis=1)
        df.set_index('index_locator', inplace=True)
        keys_to_run = set()
        for index, idx_cat, num_n in df[["index", "idx_cat", "_n"]].values:
            key = f'{index}-{idx_cat}'
            if key in existing_results_map:
                res = existing_results_map[key]
                missing = 0

                if len(res['responses']) < num_n:
                    missing = num_n - len(res['responses'])

                if missing > 0:
                    df.loc[key, '_n'] = missing
                    keys_to_run.add(key)
            else:
                keys_to_run.add(key)

        df = df.loc[list(keys_to_run)]
        logger.info(f"Subsetted data to {len(keys_to_run)} bad keys in checkpoint file. {len(df)} entries left to run.")
        if len(df) == 0:
            logger.info("No new keys to run. All entries have been processed and line up with between the checkpoint file and the input file.")
            return
        df.reset_index(drop=True, inplace=True)

    # Initialize vLLM
    logger.info(f"Initializing model: {model}")

    if not detect_api_model(model):
        logger.info(f"Initializing using vLLM inference with dtype: {dtype}")

    if not dryrun:
        llm = LLM(model=model, trust_remote_code=True, dtype=dtype, tensor_parallel_size=tensor_parallel_size, max_model_len=4096)
    else:
        logger.info("Dry run mode - vLLM will not be initialized")
        llm = None

    sampling_args = {}
    sampling_args['max_tokens'] = 2000
    sampling_args['logprobs'] = logprobs
    sampling_args['top_logprobs'] = top_logprobs

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        **sampling_args
    )

    # Process questions in batches
    batch_prompts = []
    batch_metadata = []
    result_map = {}

    for index, idx_cat, question, num_n in tqdm(df[["index", "idx_cat", "open_ended_question", "_n"]].values, total=df.shape[0], desc="Processing entities / topics"):

        # Process batch when it reaches batch_size
        while len(batch_prompts) >= batch_size:
            # Truncate to batch_size
            batch_prompts_to_process = batch_prompts[:batch_size]
            batch_metadata_to_process = batch_metadata[:batch_size]

            batch_prompts = batch_prompts[batch_size:]
            batch_metadata = batch_metadata[batch_size:]

            logger.info(f"Processing batch of {len(batch_prompts_to_process)} prompts")

            # Process batch outputs
            result_map, failed_prompts, failed_metadata = process_batch_outputs(
                batch_metadata_to_process,
                batch_prompts_to_process,
                result_map,
                llm,
                sampling_params,
                dryrun
            )

            batch_prompts.extend(failed_prompts)
            batch_metadata.extend(failed_metadata)

        # Write batch results
        if len(result_map) >= group_batch_size and len(result_map) > 0:
            logger.info(f"Writing batch of {len(result_map)} results")
            write_results(output_path, result_map, coalesce_results)
            result_map = {}

        # Create k copies of each prompt
        prompts = [create_prompts([question], word_count=word_count)[0]] * num_n
        batch_prompts.extend(prompts)
        batch_metadata.extend([{'index': index, 'idx_cat': idx_cat, 'question': question,
                                'entity': question[question.index("'")+1:].replace("'", "").rstrip('.').strip()}] * num_n)

    iterations = 0
    while len(batch_prompts) > 0 and iterations <= 3:
        # Truncate to batch_size
        batch_prompts_to_process = batch_prompts[:batch_size]
        batch_metadata_to_process = batch_metadata[:batch_size]

        batch_prompts = batch_prompts[batch_size:]
        batch_metadata = batch_metadata[batch_size:]

        logger.info(f"Processing batch of {len(batch_prompts_to_process)} prompts")

        # Process batch outputs
        result_map, failed_prompts, failed_metadata = process_batch_outputs(
            batch_metadata_to_process,
            batch_prompts_to_process,
            result_map,
            llm,
            sampling_params,
            dryrun
        )

        batch_prompts.extend(failed_prompts)
        batch_metadata.extend(failed_metadata)
        iterations += 1

    if len(failed_prompts) > 0:
        logger.warning(f"Failed to process {len(failed_prompts)} prompts. Saving failed metadata...")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(os.path.join(os.path.dirname(output_path), os.path.basename(output_path).replace('.json', '_failed_metadata.json')), 'w') as f:
            json.dump(failed_metadata, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    if len(result_map) > 0:
        write_results(output_path, result_map, coalesce_results)
    if not dryrun:
        file_path = Path(f"{output_path}.done")
        file_path.touch(mode=0o777, exist_ok=True)

    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
