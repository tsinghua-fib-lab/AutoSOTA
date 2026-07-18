import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from cs_metric import ConformityScorer
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging
import numpy as np
from score.pipelines import CondDDIMPipeline
from logdiff.score.sampling_compositional import LogicModelWrapper, And, Or_CI, Or_ME, Not
from logdiff.score.sampling_3dshapes import Shape, FloorHue, WallHue, ObjectHue, Scale, Orientation
from logdiff.evaluate.query_generator import ComplexQueryGenerator
import torch
from utils import set_seed
from logdiff.evaluate.evaluation_utils import get_null_token, load_models, run_task_evaluation, save_results


if torch.cuda.is_available():
    device = torch.device('cuda')

@hydra.main(config_path='../../configs', version_base='1.2', config_name='shapes3d_inference')
def main(cfg):
    ########## Hyperparameters and settings ##########
    set_seed(cfg.seed) 
    output_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    
    # Settings for evaluation
    EVAL_TOTAL_SAMPLES = 1000
    BATCH_SIZE = cfg.get("batch_size", 100)
    GUIDANCE = cfg.get("guidance", None)
    USE_NEG_GUIDANCE = cfg.get("use_neg_guidance", False)
    NUM_STEPS = 50
    output_dir = f"{output_dir}_{cfg.dataset.train_dataset._target_}_{GUIDANCE["atom"]}_{cfg.output_suffix}"
    
    ########## Load model #############
    logger.info("Loading Diffusion Model and Classifiers")
    scheduler = instantiate(cfg.noise_scheduler)
    model, judge_classifier, composition_classifier = load_models(cfg, device)

    ########## Setup Pipeline #############
    expr_wrapper = LogicModelWrapper(model, composition_classifier, USE_NEG_GUIDANCE)
    pipe = CondDDIMPipeline(net=expr_wrapper, scheduler=scheduler)
    null_token = get_null_token(cfg, BATCH_SIZE, device)

    dataset_config = {
        "LOGIC_GROUP_NAMES": ["FloorHue", "WallHue", "ObjectHue", "Scale", "Shape", "Orientation"],
        
        "ATTRIBUTE_CLASSES": {
            "FloorHue": FloorHue, "WallHue": WallHue, "ObjectHue": ObjectHue,
            "Scale": Scale, "Shape": Shape, "Orientation": Orientation,
        },
        
        "ATTRIBUTE_OPTIONS": {
            "FloorHue": 10, "WallHue": 10, "ObjectHue": 10,
            "Scale": 8, "Shape": 4, "Orientation": 15,
        },
    }
    dataset_config["CLASSES_ATTRIBUTES"] = {v: k for k, v in dataset_config["ATTRIBUTE_CLASSES"].items()}
    
    cs = ConformityScorer(judge_classifier, dataset_config["LOGIC_GROUP_NAMES"], dataset_config["CLASSES_ATTRIBUTES"])
    results = {}

    def get_random_shapes3d_atom(AtomClass, config):
        """Returns a random Atom instance with a valid value."""
        attr_name = AtomClass.__name__
        max_val = config["ATTRIBUTE_OPTIONS"].get(attr_name, 1) # Default to 1 if missing
        return AtomClass(np.random.randint(max_val))


    complex_query_generator = ComplexQueryGenerator(dataset_config["LOGIC_GROUP_NAMES"],
                                                    dataset_config["ATTRIBUTE_CLASSES"],
                                                    dataset_config["ATTRIBUTE_OPTIONS"])
    
    tasks = [
        (
            "AND (Random Attribute, Random Attribute)", 
            lambda: (
                lambda names: And(
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][names[0]], dataset_config),
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][names[1]], dataset_config)
                )
            )(np.random.choice(dataset_config["LOGIC_GROUP_NAMES"], size=2, replace=False))
        ),
        (
            "NOT (Random Attribute)",        
            lambda: (
                lambda name: Not(get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][name], dataset_config))
            )(np.random.choice(dataset_config["LOGIC_GROUP_NAMES"]))
        ),
        (
            "OR (ME) (Same Attribute - Random)",        
            lambda: (
                lambda name: Or_ME(
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][name], dataset_config), 
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][name], dataset_config)
                ) 
            )(np.random.choice(dataset_config["LOGIC_GROUP_NAMES"], size=1)[0])
        ),
        (
            "OR (CI) (Different Attributes - Random)",   
            lambda: (
                lambda names: Or_CI(
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][names[0]], dataset_config),
                    get_random_shapes3d_atom(dataset_config["ATTRIBUTE_CLASSES"][names[1]], dataset_config)
                )
            )(np.random.choice(dataset_config["LOGIC_GROUP_NAMES"], size=2, replace=False))
        ), 
    ]

    # Execution Loop
    for task_name, query_generator in tasks:
        acc = run_task_evaluation(task_name, query_generator, logger, pipe, cs, 
                                  EVAL_TOTAL_SAMPLES, BATCH_SIZE, GUIDANCE, NUM_STEPS, 
                                  null_token, dataset_config["ATTRIBUTE_OPTIONS"], 
                                  output_dir)
        results[task_name] = acc
        save_results(results, GUIDANCE["atom"], output_dir)

    save_results(results, GUIDANCE["atom"], output_dir)
if __name__ == "__main__":
    main()
