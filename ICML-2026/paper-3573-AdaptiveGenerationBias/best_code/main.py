import argparse
import sys
import os
import json

from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    check_credentials,
    get_out_file,
)
from src.configs import Task
from src.bias_pipeline import BiasDetectionPipeline
from src.bias_pipeline.model_evaluator import ModelEvaluationPipeline
from src.personas import generate_personas, migrate_personas_from_old_format
from src.bias_pipeline.questionaires.generator import QuestionaireGenerator
from src.ablations.generate_neutral import generate_neutral_questions
from src.ablations.transfer.generate_transfer import generate_transfer_questions
from src.ablations.baseline.generate_baseline import generate_baseline_questions
from src.ablations.transformations.transform_questions import transform_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/thread/thread.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)

    check_credentials(cfg)

    f, path = get_out_file(cfg)

    outpath = cfg.get_out_path()
    try:
        print(cfg)
        # Store the config file
        if cfg.task != Task.MODELEVAL:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            with open(outpath + "config.json", "w") as f:
                json.dump(cfg.model_dump(), f)

        if cfg.task == Task.GENPROFILES:
            if cfg.task_config.format_old:
                migrate_personas_from_old_format(cfg)
            else:
                generate_personas(cfg, f)
        elif cfg.task == Task.GENQUESTIONS:
            generator = QuestionaireGenerator(cfg.task_config, outpath)
            generator.generate(
                cfg.task_config.type, cfg.task_config.type_values, cfg.task_config.type_examples
            )
        elif cfg.task == Task.RUN:
            pipeline = BiasDetectionPipeline(cfg.task_config)
            pipeline.run_pipeline(outpath)
        elif cfg.task == Task.EVAL:
            pass
        elif cfg.task == Task.MODELEVAL:
            pipeline = ModelEvaluationPipeline(cfg.task_config)
            pipeline.run_evaluation()
            # pipeline.run_evaluation_convs()
        elif cfg.task == Task.NEUTRAL_GENERATE:
            generate_neutral_questions(cfg)
        elif cfg.task == Task.BASELINE_GENERATE:
            generate_baseline_questions(cfg)
        elif cfg.task == Task.BIAS_TRANSFER:
            generate_transfer_questions(cfg)
        elif cfg.task == Task.QUESTION_TRANSFORM:
            transform_questions(cfg)
        else:
            raise NotImplementedError(f"Task {cfg.task} not implemented")

    except ValueError as e:
        sys.stderr.write(f"Error: {e}")
    finally:
        if cfg.store:
            f.close()
