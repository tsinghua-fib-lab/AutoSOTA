import os
import click
import random
import torch
import gc
import logging

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from typing import Dict, Any
from pathlib import Path

from semantic_isotropy.datasets.loaders import load_data
from semantic_isotropy.pipeline.utils import save_config, write_results, init_logger
from semantic_isotropy.metrics.luq import entailment_score_func, luq
from semantic_isotropy.metrics.isotropy import embedding_density
from semantic_isotropy.metrics.entailment import entailment_metrics
from semantic_isotropy.metrics.graph_uncertainty import graph_uncertainty
from semantic_isotropy.llm.utils import estimate_tokens, TokenRateLimiter


logger = init_logger(__name__, 'INFO')
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

def truncate_response(response: dict, response_count: int) -> dict:
    token_cutoff = response_count * 4/3
    item = response.copy()
    statements = response['statements']

    token_cutoff_ctr = token_cutoff
    statement_cutoff = 0
    while (token_cutoff_ctr > 0) and (statement_cutoff < len(statements)):
        token_cutoff_ctr -= estimate_tokens(statements[statement_cutoff]['text'])
        statement_cutoff += 1

    truncated_response = " ".join([s['text'] for s in statements[:statement_cutoff]])
    item['response'] = truncated_response + ('' if (truncated_response.endswith('.') or truncated_response.endswith('!')) else '.')
    item['statements'] = statements[:statement_cutoff]
    return item

def coalesce_results(existing_results: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    if not len(existing_results):
        existing_results = {}
    existing_results.update(results)
    return existing_results

def get_available_gpus():
    """Get the number of available GPUs in the system."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

@click.command()
@click.option('--input-path', required=True, help='Path to the input json file of generated open ended responses')
@click.option('--output-path', required=True, help='Output path for the evaluation results')
@click.option('--metric', required=True, default='isotropy', help='Metric to calculate')
@click.option('--embedding-model', required=True, default='Alibaba-NLP/gte-qwen2.5-7b-instruct', help='Embedding model to use')
@click.option('--response-count', default=1000, type=int, help='Number of word to truncate responses to. If equal to response_max, no truncation is done.')
@click.option('--response-max', default=1000, type=int, help='Maximum length of responses in corpus.')
@click.option('--subset', default=None, type=int, help='Subset of group of data to process')
@click.option('--group-batch-size', default=100, type=int, help='Number of entries to process before writing')
@click.option('--device', default='auto', type=str, help='Device to use (auto, cpu, cuda, mps, or specific like cuda:0)')
@click.option('--pooling', default='mean', type=str, help='Pooling method to use (MES only)')
@click.option('--dryrun', is_flag=True, help='Run without making API calls or saving results')
@click.option('--overwrite', is_flag=True, help='Overwrite existing output file')
@click.option('--dtype', default=None, help='Dtype to use')
@click.option('--restart-from-checkpoint', is_flag=True, help='Restart from checkpoint. Resume from the last checkpoint file in the output file.')
@click.pass_context  # Add this decorator to get Click's context
def main(ctx: click.Context, input_path: str, output_path: str, metric: str, embedding_model: str, response_count: int,
         response_max: int, subset: int, group_batch_size: int, device: str, pooling: str,
         dryrun: bool, overwrite: bool, dtype: str, restart_from_checkpoint: bool):
    """Calculate metric for generated open ended responses"""

    output_dir = os.path.dirname(output_path)
    save_config(output_dir, ctx, dryrun)

    # Determine device setup
    if device == 'auto':
        num_gpus = get_available_gpus()
        if num_gpus > 0:
            device = 'cuda'
            logger.info(f"Auto-detected {num_gpus} GPUs available")
        else:
            device = 'cpu'
            logger.info("No GPUs detected, using CPU")

    # Check if we should use model parallelism (multiple GPUs)
    use_model_parallelism = device == 'cuda' and get_available_gpus() > 1

    logger.info("Dry run mode - configuration:" if dryrun else "Configuration:")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Metric: {metric}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"Response count: {response_count}")
    logger.info(f"Response max: {response_max}")
    logger.info(f"Subset size: {subset}")
    if metric == 'isotropy':
        logger.info(f"Pooling method: {pooling}")
    logger.info(f"Group batch size: {group_batch_size}")
    logger.info(f"Dtype: {dtype}")
    logger.info(f"Device: {device}")
    logger.info(f"Restart from checkpoint: {restart_from_checkpoint}")
    if use_model_parallelism:
        logger.info(f"Using model parallelism across {get_available_gpus()} GPUs")

    assert not (restart_from_checkpoint and overwrite), "Cannot restart from checkpoint and overwrite at the same time. Please use either --overwrite or --restart-from-checkpoint, not both."

    # Check if output file exists and handle overwrite
    existing_keys = set()
    if restart_from_checkpoint and os.path.exists(output_path):
        logger.info(f"Restarting from checkpoint: {output_path}")
        if os.path.exists(f"{output_path}.done"):
            logger.warning(f"Checkpoint file {output_path}.done exists. Pipeline already completed. Please use --overwrite without --restart-from-checkpoint to restart from scratch or remove the checkpoint file if you suspect results are incomplete.")
            return
        existing_results = load_data(output_path)
        existing_keys = set([key.split(':')[-1].strip() for key in existing_results.keys()])
        logger.info(f"Loaded {len(existing_keys)} existing keys from checkpoint file.")
    elif os.path.exists(output_path):
        if overwrite:
            logger.warning(f"Removing existing file: {output_path}")
            os.remove(output_path)
            if os.path.exists(f"{output_path}.done"):
                os.remove(f"{output_path}.done")
        else:
            if os.path.exists(f"{output_path}.done"):
                logger.info(f"Pipeline complete file {output_path}.done exists. Exiting.")
            else:
                logger.error(f"Output file {output_path} already exists. Use --overwrite to replace.")
            return

    # Load data
    data = load_data(input_path)
    logger.info(f"Loaded {len(data)} data entries from input file.")

    if restart_from_checkpoint and len(existing_keys) > 0:
        data = [d for d in data if f"{d['index']}-{d['idx_cat']}" not in existing_keys]
        logger.info(f"Filtered data down to {len(data)} entries from input file.")

    tdevice = torch.device(device)

    scoring_ctx = {}
    model_args = {'torch_dtype': torch.float16} if dtype == 'half' else {}

    if metric == 'luq':
        scoring_ctx['entailment_func'] = entailment_score_func(embedding_model, device=device, model_args=model_args)
    elif metric == 'isotropy':
        if ('gemini' not in embedding_model) and ('openai' not in embedding_model) and ('cohere' not in embedding_model):
            # 1. Load config with hidden states on
            config = AutoConfig.from_pretrained(embedding_model, output_hidden_states=True, trust_remote_code=True if 'Phi' not in embedding_model else False)
            tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True if 'Phi' not in embedding_model else False)
            tokenizer.add_special_tokens({'pad_token': tokenizer.special_tokens_map.get('pad_token', tokenizer.eos_token)})

            # Load model with appropriate parallelism settings
            if use_model_parallelism:
                # Enable model parallelism across available GPUs
                model = AutoModel.from_pretrained(
                    embedding_model,
                    trust_remote_code=True if 'Phi' not in embedding_model else False,
                    device_map="auto",  # Automatically distribute across available devices
                    config=config,
                    **{k:v for k,v in model_args.items() if k != 'dtype'}
                )
            else:
                # Standard single device loading
                model = AutoModel.from_pretrained(
                    embedding_model,
                    trust_remote_code=True if 'Phi' not in embedding_model else False,
                    config=config,
                    **{k:v for k,v in model_args.items() if k != 'dtype'}
                )
                model.to(tdevice)

            scoring_ctx['tokenizer'] = tokenizer
            scoring_ctx['embedding_model'] = model
        else:
            rate_limiter_gemini = (TokenRateLimiter(2500000), 'tokens')
            rate_limiter_openapi = (TokenRateLimiter(1500000), 'tokens')
            rate_limiter_cohere = (TokenRateLimiter(200), 'requests')
            scoring_ctx['tokenizer'] = None
            scoring_ctx['embedding_model'] = None
            scoring_ctx['rate_limiter'] = rate_limiter_gemini if 'gemini' in embedding_model else rate_limiter_openapi if 'openai' in embedding_model else rate_limiter_cohere
            if 'gemini' in embedding_model:
                scoring_ctx['task_type'] = 'SEMANTIC_SIMILARITY'

    elif metric == 'entailment':
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)  #microsoft/deberta-v2-xlarge-mnli
        model = AutoModelForSequenceClassification.from_pretrained(embedding_model, **model_args).to(device)
        scoring_ctx['tokenizer'] = tokenizer
        scoring_ctx['model'] = model
    elif metric == 'graph_uncertainty':
        scoring_ctx['factscore_data'] = load_data(f"{input_path.replace('.json', '')}_factscore_output_factscore_output.json")['decisions']

    results = {}
    for idx, item in enumerate(tqdm(data, desc=f"Scoring using metric: {metric}")):
        index = item['index']
        idx_cat = item['idx_cat']
        entity = item['entity']
        key = f"{entity}: {index}-{idx_cat}"
        if key in existing_keys:
            continue

        responses = item['responses']

        subset_idx = list(range(len(responses)))
        if subset and subset < len(responses):
            subset_idx = random.sample(subset_idx, subset)
            responses = [responses[i] for i in subset_idx]

        if not dryrun:
            if response_count < response_max:
                responses_truncated = [truncate_response(response, response_count) for response in responses]
                responses = responses_truncated

            if metric == 'luq':
                confidence, uncertainty = luq(responses, scoring_ctx['entailment_func'])
                results[key] = {
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'subset_idx': subset_idx
                }
            elif metric == 'isotropy':
                add_args = {"device": device} if not use_model_parallelism else {"use_multi_gpu": True}
                if 'task_type' in scoring_ctx and 'gemini' in embedding_model:
                    add_args['task_type'] = scoring_ctx['task_type']
                if 'api_key' in scoring_ctx and ('openai' in embedding_model or 'cohere' in embedding_model or 'gemini' in embedding_model):
                    add_args['api_key'] = scoring_ctx['api_key']
                if 'rate_limiter' in scoring_ctx:
                    add_args['rate_limiter'] = scoring_ctx['rate_limiter']
                si, pooled_state, eigenscore_vec = embedding_density(
                    responses,
                    scoring_ctx['embedding_model'],
                    scoring_ctx["tokenizer"],
                    entity,
                    pooling_method=pooling,
                    model_name=embedding_model,
                    **add_args,
                )
                results[key] = {
                    'semantic_isotropy': si,
                    'pooled_state': pooled_state,
                    'eigenscore_vec': eigenscore_vec,
                    'subset_idx': subset_idx
                }

                # Clean up memory
                if device == "mps":
                    torch.mps.empty_cache()
                elif device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
            elif metric == 'entailment':
                entailment_matrix, metric_dict = entailment_metrics(responses, model, tokenizer, device)
                results[key] = {
                    'entailment_matrix': entailment_matrix,
                    'metric_dict': metric_dict,
                    'subset_idx': subset_idx
                }
            elif metric == 'graph_uncertainty':
                rate_limiter_openapi = (TokenRateLimiter(1500000), 'tokens')
                claim_metrics_dict, bipartite_matching, claims_list, _ = graph_uncertainty(
                    responses, scoring_ctx['factscore_data'], idx, subset_idx, len(item['responses']), rate_limiter_openapi, scoring_ctx['api_key'], max_workers=20)
                results[key] = {
                    'claim_level_metrics_dict': claim_metrics_dict,
                    'subset_idx': subset_idx,
                    'claims_list': claims_list,
                    'bipartite_matching': bipartite_matching,
                }
            else:
                raise ValueError(f"Metric {metric} not supported")

            if len(results) % group_batch_size == 0:
                logger.info(f"Wrote batch of {len(results)} results to {output_path}")
                write_results(output_path, results, coalesce_results)
                results = {}

    if results:
        logger.info(f"Wrote final batch of {len(results)} results to {output_path}")
        write_results(output_path, results, coalesce_results)

    if not dryrun:
        file_path = Path(f"{output_path}.done")
        file_path.touch(mode=0o777, exist_ok=True)


if __name__ == "__main__":
    main()
