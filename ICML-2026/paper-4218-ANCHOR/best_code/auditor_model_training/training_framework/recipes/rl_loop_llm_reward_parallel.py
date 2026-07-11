import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
import requests
import re

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-loop-new-judge"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 256
    reward_workers: int = 16  # Number of parallel workers for reward calculation


def get_reward(generated_response: str, reference_answer: str, user_prompt: str) -> float:
    """Use local LLM on port 8001 to evaluate the response quality compared to reference."""

    prompt = f"""Compare and rate the quality of a generated response against a reference answer.

User's question: {user_prompt}

Reference answer (from SFT dataset): {reference_answer}

Generated response: {generated_response}

Rate the generated response on a scale from 0 to 10, where:
- 0: All sentences are different compared to the reference answer.
- 5: Half of the sentences are exactly the same meaning as the reference answer, whereas the other half is not exactly the same meaning as the reference answer.
- 10: All sentences from the generated response are exactly the same meaning as the reference anwer.

Just respond with a single number from 0 to 10."""

    try:
        # Call local model on port 8001
        request_data = {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 10,
            "stream": False
        }

        response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()

        # Extract the rating from response
        result = response.json()
        rating_text = result['choices'][0]['message']['content'].strip()

        # Try to extract a number from the response
        numbers = re.findall(r'\d+(?:\.\d+)?', rating_text)
        if numbers:
            rating = float(numbers[0])
            # Normalize to 0-1 range
            return min(max(rating / 10.0, 0.0), 1.0)
        else:
            # Default to 0.5 if can't parse
            return 0.5

    except Exception as e:
        logger.warning(f"Error getting reward from LLM: {e}")
        return 0.5  # Return neutral score on error


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load STAIR-SFT dataset
    logger.info("Loading STAIR-SFT dataset...")
    dataset = datasets.load_dataset("thu-ml/STAIR-SFT")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    n_train_batches = len(train_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    # Create a thread pool for parallel reward calculation
    reward_executor = ThreadPoolExecutor(max_workers=config.reward_workers)

    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Set up sampling parameters

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []
        batch_user_prompts: list[str] = []

        # Process each row in the STAIR-SFT dataset
        for row in batch_rows:
            user_prompt = row["prompt"]
            batch_user_prompts.append(user_prompt)

            # Create conversation format with just the user prompt
            messages = [
                Message(role="user", content=user_prompt)
            ]
            model_input = renderer.build_generation_prompt(messages)
            prompt_tokens = model_input.to_ints()

            # Generate multiple responses
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        for sample_futures, prompt_tokens, row, user_prompt in zip(
            batch_futures, batch_prompts, batch_rows, batch_user_prompts
        ):
            reference_answer = row["answer"]
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            group_responses: list[str] = []

            # First, collect all sampling results and prepare responses
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                generated_response = parsed_message["content"]
                group_responses.append(generated_response)

            # Submit all reward calculations in parallel
            reward_futures = []
            for generated_response in group_responses:
                reward_future = reward_executor.submit(
                    get_reward,
                    generated_response,
                    reference_answer,
                    user_prompt
                )
                reward_futures.append(reward_future)

            # Collect all rewards
            group_rewards = [future.result() for future in reward_futures]

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip this prompt because all responses got the same reward
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        ml_logger.log_metrics(metrics, step=batch_idx)

    # Clean up executor
    reward_executor.shutdown(wait=True)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)