import multiprocessing
import json
from evaluation.LiveCodeBench.testing_util import run_test
from loguru import logger
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not multiprocessing.get_start_method(allow_none=True):
    multiprocessing.set_start_method('spawn')
    
def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)
    
def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
            
    return result[0], metadata_list[0]


def evaluate_generation(generations, sample, debug: bool = False, timeout:int=6):
    res = []
    metadata = []
    
    for o_idx, o in enumerate(generations):
        curr_res = [-2]
        try:
            # print(sample, o)
            curr_res, curr_metadata = check_correctness(sample, o, timeout, debug)
            if debug:
                logger.info(f"sample generation {o_idx} passed {curr_res}")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    logger.info(f"Results were not True for all test cases {curr_res=}\n")
        except Exception as e:
            if debug:
                logger.warning(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            curr_metadata = {
                "error": str(e),
                "error_code": -5,
                "error_message": "TestRunnerError"
            }
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(res):
            logger.info("Sample\n")
            logger.info(sample)
            logger.info("\n")
            logger.info("Result\n")
            logger.info(res[i])
            logger.info("*" * 30 + "\n\n")
    return res, metadata
    