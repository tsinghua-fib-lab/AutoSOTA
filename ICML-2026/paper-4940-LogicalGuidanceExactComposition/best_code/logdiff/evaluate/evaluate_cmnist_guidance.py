import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from cs_metric import ConformityScorer
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging
import numpy as np
from logdiff.score.pipelines import CondDDIMPipeline
from logdiff.score.sampling_compositional import LogicModelWrapper, And, Or_CI, Or_ME, Not
from logdiff.score.sampling_cmnist import Digit, Color
from logdiff.evaluate.query_generator import ComplexQueryGenerator
import torch
from logdiff.utils import set_seed
from logdiff.evaluate.evaluation_utils import get_null_token, load_models, run_task_evaluation, save_results

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@hydra.main(config_path='../../configs', version_base='1.2', config_name='cmnist_inference')
def main(cfg):
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    
    EVAL_TOTAL_SAMPLES = 1000
    BATCH_SIZE = cfg.get("batch_size", 100)
    GUIDANCE = cfg.get("guidance", None)
    USE_NEG_GUIDANCE = cfg.get("use_neg_guidance", False)
    NUM_STEPS = 50
    output_dir = f"{output_dir}_{cfg.dataset.train_dataset._target_}_{GUIDANCE['atom']}_{cfg.output_suffix}"
    
    ########## Load model #############
    logger.info("Loading Diffusion Model and Classifiers")
    scheduler = instantiate(cfg.noise_scheduler)
    model, judge_classifier, composition_classifier = load_models(cfg, device)

    ########## Setup Pipeline #############
    expr_wrapper = LogicModelWrapper(model, composition_classifier, USE_NEG_GUIDANCE)
    pipe = CondDDIMPipeline(net=expr_wrapper, scheduler=scheduler)
    null_token = get_null_token(cfg, BATCH_SIZE, device)

    results = {}

    # Task Definitions
    dataset_config = {
        "LOGIC_GROUP_NAMES": ["Digit", "Color"],
        "ATTRIBUTE_CLASSES": {
            "Digit": Digit,
            "Color": Color,
        },
        "ATTRIBUTE_OPTIONS": {
            "Digit": 10,
            "Color": 10,    
        },
    }

    dataset_config["CLASSES_ATTRIBUTES"] = {v: k for k, v in dataset_config["ATTRIBUTE_CLASSES"].items()}

    cs = ConformityScorer(judge_classifier, dataset_config["LOGIC_GROUP_NAMES"], dataset_config["CLASSES_ATTRIBUTES"])
    
    tasks = [
        (
            "AND (Digit, Color)", 
            lambda: And(Color(np.random.randint(10)), Digit(np.random.randint(10)))
        ),
        (
            "NOT (Color)",        
            lambda: Not(Color(np.random.randint(10))) if np.random.rand() < 0.5 else Not(Digit(np.random.randint(10)))

        ),
        (
            "OR (ME) (Same Attribute)",        
            lambda: Or_ME(Digit(np.random.randint(10)), Digit(np.random.randint(10))) 
                    if np.random.rand() < 0.5 
                    else Or_ME(Color(np.random.randint(10)), Color(np.random.randint(10)))
        ),
        (
            "OR (CI)",        
            lambda: Or_CI(Digit(np.random.randint(10)), Color(np.random.randint(10)))
        ), 
    ]

    # Execution Loop
    for task_name, query_generator in tasks:
        acc = run_task_evaluation(task_name, query_generator, logger, pipe, cs, 
                                  EVAL_TOTAL_SAMPLES, BATCH_SIZE, GUIDANCE, NUM_STEPS, 
                                  null_token, dataset_config["ATTRIBUTE_OPTIONS"], output_dir)
        results[task_name] = acc
        save_results(results, GUIDANCE['atom'], output_dir)

    save_results(results, GUIDANCE['atom'], output_dir)
if __name__ == "__main__":
    main()
