import sys
sys.path.insert(0, "/repo")
import hydra
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
from hydra.utils import instantiate
import torch
import logging
import numpy as np
from logdiff.evaluate.query_generator import ComplexQueryGenerator
from logdiff.evaluate.evaluation_utils import get_null_token, load_models, run_task_evaluation, save_results
from logdiff.score.pipelines import CondDDIMPipeline
from logdiff.score.sampling_compositional import LogicModelWrapper, And, Or_CI, Or_ME, Not
from logdiff.score.sampling_cmnist import Digit, Color
from logdiff.cs_metric import ConformityScorer
from logdiff.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda")

@hydra.main(config_path="/repo/configs", version_base="1.2", config_name="cmnist_inference")
def main(cfg):
    set_seed(cfg.seed)
    output_dir = "/autosota_cache/paper-4940-eval/minimal"
    
    EVAL_TOTAL_SAMPLES = 10000
    BATCH_SIZE = 100
    GUIDANCE = cfg.get("guidance", None)
    NUM_STEPS = 50
    
    logger.info("Loading models...")
    scheduler = instantiate(cfg.noise_scheduler)
    model, judge_classifier, composition_classifier = load_models(cfg, device)

    expr_wrapper = LogicModelWrapper(model, composition_classifier, False)
    pipe = CondDDIMPipeline(net=expr_wrapper, scheduler=scheduler)
    null_token = get_null_token(cfg, BATCH_SIZE, device)

    dataset_config = {
        "LOGIC_GROUP_NAMES": ["Digit", "Color"],
        "ATTRIBUTE_CLASSES": {"Digit": Digit, "Color": Color},
        "ATTRIBUTE_OPTIONS": {"Digit": 10, "Color": 10},
    }
    dataset_config["CLASSES_ATTRIBUTES"] = {v: k for k, v in dataset_config["ATTRIBUTE_CLASSES"].items()}

    cs = ConformityScorer(judge_classifier, dataset_config["LOGIC_GROUP_NAMES"], dataset_config["CLASSES_ATTRIBUTES"])
    complex_query_generator = ComplexQueryGenerator(dataset_config["LOGIC_GROUP_NAMES"], dataset_config["ATTRIBUTE_CLASSES"], dataset_config["ATTRIBUTE_OPTIONS"])
    
    # ONLY run Complex: 2 expressions (N=2)
    tasks = [
        ("Complex: 2 expressions", lambda: complex_query_generator.gen_complex_query(expressions=2)),
    ]
    
    results = {}
    for task_name, query_generator in tasks:
        acc = run_task_evaluation(task_name, query_generator, logger, pipe, cs, 
                                  EVAL_TOTAL_SAMPLES, BATCH_SIZE, GUIDANCE, NUM_STEPS, 
                                  null_token, dataset_config["ATTRIBUTE_OPTIONS"], 
                                  output_dir, eval_baselines=False)
        results[task_name] = acc
        logger.info(f"RESULT: {task_name} = {acc}")
    
    save_results(results, EVAL_TOTAL_SAMPLES, output_dir)

if __name__ == "__main__":
    main()
