import argparse
import json
from common_utils.logging_utils import get_logger
from modules.citeeval_runner import CiteEvalRunner
from data.data_loader import load_response_output

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--response_output_file", type=str, required=True, help="Response output file path (input for citation evaluation). Should have field `id`, `question` and `output`")
    parser.add_argument("--eval_output_dir", type=str, required=True, help="Directory to save evaluation output files.")
    parser.add_argument("--modules", type=str, default="sc,ce,cv,cr_light,cc", required=False, help="Metrics for source attribution evaluation.")
    parser.add_argument("--version", type=str, default="", required=False, help="Prompt template version for CiteEval-Auto evaluation.")
    parser.add_argument("--model_name", type=str, default="bedrock:anthropic.claude-v2:1", required=False, help="LLM backbone for CiteEval-Auto.")
    parser.add_argument("--citing_sentences_only", action="store_true", help="Exclude answer sentences w/o citations in attribution evaluation.")
    parser.add_argument('--skip_data_processing', default=False, action='store_true', help="For iterative improvement, the response file is produced after applying citation edits, and can be used for citation rating directly.")
    parser.add_argument("--n_threads", type=int, default=4, help="Number of parallel threads to run LLM calls.")
    parser.add_argument("--ca_output_file", type=str, required=False, default="", help="Use this specified sc output file, rather the auto-inferred file")

    args = parser.parse_args()

    data = load_response_output(
        file_path=args.response_output_file, 
        skip_data_processing=args.skip_data_processing
    )

    modules = args.modules.split(',')
    runner_params = {
        'data': data,
        "version": args.version,
        "citing_sentences_only": args.citing_sentences_only,
        "n_threads": args.n_threads,
        "model_name": args.model_name
    }
    citeeval_runner = CiteEvalRunner(**runner_params)

    # build output file path for each module
    module2output_fp = {}
    dataset_prefix = args.response_output_file.split("/")[-1]
    allowed_module_names = ["ca", "ce", "cr_itercoe", "cr_editdist"]

    for module_name in allowed_module_names:
        output_fn = f"{dataset_prefix}.{args.version}.{module_name}.{args.model_name}.out"
        output_fp =  f"{args.eval_output_dir}/{output_fn}"
        module2output_fp[module_name] = output_fp

    # run the pipeline
    for module_name in modules:
        logger.info(f"Running module: {module_name}")
        assert module_name in allowed_module_names, f"Invalid module name: {allowed_module_names}. Must be one of {allowed_module_names}"
        eval_params = {
            "module_name": module_name
        }

        if module_name.startswith("cr"):  # cr requires outputs from sc, ce
            eval_params = {
                **eval_params,
                "ca_output_file": module2output_fp["ca"] if not args.ca_output_file else args.ca_output_file,
                "ce_output_file": module2output_fp["ce"],
            }
        
        cite_results, citeeval_outputs = citeeval_runner.run(**eval_params)
        output_fp = module2output_fp[module_name]
        result_fp = f"{output_fp[:-len('.out')]}.res"
        
        json.dump(citeeval_outputs, open(output_fp, 'w'), indent=2)
        json.dump(cite_results, open(result_fp, "a"), indent=2)
        
        logger.info(f"Output written to: {output_fp}")
        logger.info(f"Metrics written to: {result_fp}")


if __name__ == "__main__":
    main()