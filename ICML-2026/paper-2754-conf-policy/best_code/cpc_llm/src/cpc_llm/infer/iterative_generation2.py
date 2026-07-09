from __future__ import annotations

import datasets
import hydra
import logging
import os
import pandas as pd
import pprint

from ..test_functions.finetune_utils import (
    formatting_texts_func_edit_pairs,
    parse_particle_and_score,
    parse_particle_and_score_permissive,
    truncate_after_right_bracket,
    remove_integer_padding_from_list,
)
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from ..core.model_client import ModelClient
from ..core.model_loading import init_model_client_with_retry
from ..data_contracts import NUM_PARTICLES_GENERATED, PARTICLE, SCORE
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig
from tqdm import tqdm


def run_iterative_generation_inmemory(
    input_ds: datasets.Dataset,
    model_client: ModelClient,
    test_fn: Ehrlich | RoughMtFuji,
    gen_config: GenerationConfig,
    cfg: DictConfig,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Generate samples over max_iterations rounds, accumulating results.

    This is the looping wrapper used by the subprocess entry point. For
    in-memory callers that want single-batch control, use
    ``generate_single_batch`` directly.

    Args:
        input_ds: datasets.Dataset with seed sequences for generation.
        model_client: Pre-initialized ModelClient instance.
        test_fn: Ehrlich or RoughMtFuji test function for scoring.
        gen_config: HuggingFace generation config (instantiated).
        cfg: Hydra config (needs batch_size, max_iterations, subsample_seeds,
             permissive_parsing, higher_score_particle_field,
             lower_score_particle_field).
        logger: Logger instance.

    Returns:
        DataFrame with columns: particle, score, num_particles_generated.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    all_dfs = []
    logger.info(f"cfg.max_iterations : {cfg.max_iterations}")
    for _iter in tqdm(
        range(1, cfg.max_iterations + 1), desc="Generation iterations..."
    ):
        batch_df = generate_single_batch(
            input_ds, model_client, test_fn, gen_config, cfg, logger
        )
        if len(batch_df) > 0:
            all_dfs.append(batch_df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame(columns=[PARTICLE, SCORE, NUM_PARTICLES_GENERATED])


def generate_single_batch(
    input_ds: datasets.Dataset,
    model_client: ModelClient,
    test_fn: Ehrlich | RoughMtFuji,
    gen_config: GenerationConfig,
    cfg: DictConfig,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Generate one batch of samples from a model.

    Formats seed inputs, runs one forward generation pass, truncates outputs
    at the closing bracket, and parses valid particles with scores.

    Args:
        input_ds: datasets.Dataset with seed sequences for generation.
        model_client: Pre-initialized ModelClient instance.
        test_fn: Ehrlich or RoughMtFuji test function for scoring.
        gen_config: HuggingFace generation config (instantiated).
        cfg: Hydra config (needs batch_size, subsample_seeds,
             permissive_parsing, higher_score_particle_field,
             lower_score_particle_field).
        logger: Logger instance.

    Returns:
        DataFrame with columns: particle, score, num_particles_generated.
        May be empty if no outputs parsed successfully.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ## Format text inputs from seeds
    input_texts = formatting_texts_func_edit_pairs(
        input_ds,
        include_target=False,
        higher_score_particle_field=cfg.higher_score_particle_field,
        lower_score_particle_field=cfg.lower_score_particle_field,
    )
    ## Remove integer padding (default pad is -1) from input texts
    for j, text in enumerate(input_texts):
        input_texts[j] = remove_integer_padding_from_list(text)
        if (len(input_texts[j])) < len(text):
            logger.info(f"Removed integer padding from input text {text}, new input is {input_texts[j]}")

    # input_texts = [remove_integer_padding_from_list(text) for text in input_texts]

    logger.info(
        f"Generating texts with cfg.subsample_seeds={cfg.subsample_seeds}, "
        f"len(input_texts)={len(input_texts)}, "
        f"len(set(input_texts))={len(set(input_texts))}"
    )

    logger.info(f"cfg.permissive_parsing : {cfg.permissive_parsing}")

    ## Generate and decode
    output_strs = model_client.generate_texts_batched(
        input_texts,
        batch_size=cfg.batch_size,
        generation_config=gen_config,
        return_likelihoods=False,
        subsample_seeds=cfg.subsample_seeds,
        logger=logger,
    )

    ## Truncate after closing bracket and parse
    outputs = []
    num_particles_generated = 0
    for output_str in output_strs:
        truncated = truncate_after_right_bracket(output_str)

        if cfg.permissive_parsing:
            # logger.info(f"Permissive parsing enabled. Truncated: {truncated}")
            result = parse_particle_and_score_permissive(truncated, test_fn, cfg, logger)
        else:
            # logger.info(f"Permissive parsing disabled. Truncated: {truncated}")
            result = parse_particle_and_score(truncated, test_fn)

        num_particles_generated += 1
        if result is None:
            continue

        outputs.append(
            {
                PARTICLE: result[0],
                SCORE: result[1],
                NUM_PARTICLES_GENERATED: num_particles_generated,
            }
        )

    return pd.DataFrame(outputs)


def run_iterative_generation(
    cfg: DictConfig, logger: logging.Logger | None = None
) -> None:
    test_fn_params = pd.read_json(cfg.test_fn_fp, orient="records", lines=True).to_dict(
        "records"
    )[0]
    if cfg.test_fn_type == "ehrlich":
        test_fn = Ehrlich(
            num_states=test_fn_params["num_states"],
            dim=test_fn_params["dim"],
            num_motifs=test_fn_params["num_motifs"],
            motif_length=test_fn_params["motif_length"],
            quantization=test_fn_params["quantization"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    else:
        test_fn = RoughMtFuji(
            dim=test_fn_params["dim"],
            additive_term=test_fn_params["additive_term"],
            random_term_std=test_fn_params["random_term_std"],
            noise_std=test_fn_params["noise_std"],
            negate=test_fn_params["negate"],
            random_seed=test_fn_params["random_seed"],
        )
    gen_config = hydra.utils.instantiate(cfg.generation_config)

    model_client = init_model_client_with_retry(
        cfg.model_name_or_path, gen_config.max_new_tokens, logger
    )
    df = pd.read_json(cfg.data_path, orient="records", lines=True)
    if cfg.sanity_check:
        logger.info(
            "Running in sanity check mode... reducing data down to 10 examples."
        )
        df = df.sample(n=min(10, len(df)))

    if cfg.higher_score_field in df.columns and cfg.lower_score_field in df.columns:
        # Before sampling, save all the particles as tuples in a set so that we can check whether
        # generations are regurgitations from the training data
        set(
            [tuple(ex[cfg.higher_score_particle_field]) for _, ex in df.iterrows()]
        ).union(
            set([tuple(ex[cfg.lower_score_particle_field]) for _, ex in df.iterrows()])
        )
        # Now dedupe the lower_score_particles and sample the lowest scoring examples from the data
        # to use as seeds for generation
        # df = df.drop_duplicates(subset=[cfg.lower_score_particle_field])
        logger.info(f"sample_size : {cfg.sample_size}")
        if cfg.sampling_method == "best_scoring": # and not cfg.first_iter:
            ## Only use best_scoring if is not first iteration
            logger.info("sampling_method : best_scoring")
            df = df.sort_values(by=[cfg.lower_score_field], ascending=True)[
                : cfg.sample_size
            ]
        elif cfg.sampling_method == "uniform": # or cfg.first_iter:
            ## Always use "uniform" for first iteration, for a safe initial policy
            logger.info("sampling_method : uniform")
            df = df.sample(n=min(len(df), cfg.sample_size), random_state=cfg.seed)
        elif cfg.sampling_method == "combination":
            half_sample_size = int(cfg.sample_size / 2)
            df = pd.concat(
                [
                    df.sort_values(by=[cfg.lower_score_field], ascending=True)[
                        :half_sample_size
                    ],
                    df.sample(n=min(len(df), half_sample_size), random_state=cfg.seed),
                ]
            )
        else:
            raise ValueError(f"Unknown sampling method '{cfg.sampling_method}.'")

        # Start by using the lower score particle from each pair as the seed
        ## ds: Dataset of *pairs*, by the best (lowest) sample_size num scoring particles
        ds = datasets.Dataset.from_pandas(df)

        ## input_ds: Dataset of *sequences*, by the best (lowest) sample_size num scoring particles
        input_ds = datasets.Dataset.from_list(
            [
                {
                    cfg.higher_score_particle_field: ex[cfg.lower_score_particle_field],
                    SCORE: ex[cfg.lower_score_field],
                }
                for ex in ds
            ]
        )

        ## If selected seeds here, then write them to disk (needed for computing seed-marginalized likelihoods used in policy control)
        input_df = input_ds.to_pandas()
        input_df.to_json(
            os.path.join(cfg.output_dir, f"seeds_{cfg.output_filename}"),
            orient="records",
            lines=True,
        )

    else:
        ## Else, assume that seeds are pre-selected, don't need to select
        set([tuple(ex[cfg.higher_score_particle_field]) for _, ex in df.iterrows()])

        # Start by using the lower score particle from each pair as the seed
        ## ds: Dataset of *pairs*, by the best (lowest) sample_size num scoring particles
        input_ds = datasets.Dataset.from_pandas(df)

    all_outputs = run_iterative_generation_inmemory(
        input_ds, model_client, test_fn, gen_config, cfg, logger
    )
    all_outputs.to_json(
        os.path.join(cfg.output_dir, cfg.output_filename), orient="records", lines=True
    )


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config")


@hydra.main(config_path=_CONFIG_PATH, config_name="iterative_generation")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    run_iterative_generation(cfg, logger)


if __name__ == "__main__":
    main()
