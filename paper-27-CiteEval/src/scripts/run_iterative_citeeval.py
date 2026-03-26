import io
import json
import subprocess
import os.path
from collections import defaultdict
import argparse
from data.data_loader import load_response_output


def generated_edited_response_output_file(
        response_output_file, 
        ce_output_file,
        new_response_output_file,
        dump=True
    ):
    """Generate a new version of responses and citations by applying edits. 
    """
    def apply(citations, edits):
        n_edits = 0
        for edit in edits:
            op, _ = edit["edit"].split("_")
            citation = int(edit["citation"])

            if op == "DELETE":
                if citation in citations:
                    citations.remove(citation)
                    n_edits += 1
                else:
                    print(f"DELETE Warning: {citation} to delete not in citations: {citations}")
            elif op == "ADD":
                if citation not in citations:
                    citations.append(citation)
                    n_edits += 1
                else:
                    print(f"ADD Warning: {citation} to add already in citations: {citations}")
            else:
                raise ValueError(f"Invalid op: {edit['edit']}")
        
        return citations, n_edits

    with io.open(ce_output_file) as f:
        ce_output = json.load(f)
        
    data = load_response_output(file_path=response_output_file, skip_data_processing=True if "iter" in response_output_file else False)
    total_n_applied_edits = 0
    total_n_edits = 0

    for idx in range(len(data)):
        sent_info = data[idx]['sent_info']
        sent_id2edits = ce_output[idx]["sent_id2edits"]

        for sid, sinfo in enumerate(sent_info):
            sid = str(sid+1)
            if sid not in sent_id2edits:
                continue
            edited_citations, n_edits = apply(citations=sinfo['citations'], edits=sent_id2edits[sid])
            total_n_applied_edits += n_edits
            total_n_edits += len(sent_id2edits[sid])

            sinfo['citations'] = edited_citations
            citation_seq = ''.join([f'[{cite}]' for cite in edited_citations])
            sinfo['raw_sent'] = sinfo['clean_sent'].strip() + " " + citation_seq
    
    if dump:
        with io.open(new_response_output_file, 'w') as out_f:
            json.dump(data, out_f, indent=2)
    
    print(f"Dump edited response file to: {new_response_output_file}\nTotal #edits applied: {total_n_applied_edits} / {total_n_edits}")


def get_module_output_file(output_prefix, module, version, model_name, eval_output_dir):
    module_signature = f"citeeval_{module}_{version}_{model_name}"
    output_file =  f"{eval_output_dir}/{output_prefix}.{module_signature}.out"
    return output_file


def run_ce(version, model_name, citebench_dev_file, eval_output_dir, max_iter=3):
    initial_response_output_prefix = citebench_dev_file.split("/")[-1][:-len(".json")]
    
    for iter in range(max_iter):
        if iter == 0:
            modules = "ca,ce"  # "cr_itercoe"
            current_response_output_file = citebench_dev_file
        else:
            modules = "ce"
            current_response_output_file =  f"{eval_output_dir}/{initial_response_output_prefix}.iter_{iter}.json"
        
        current_response_output_prefix = current_response_output_file.split("/")[-1][:-len(".json")]
        ce_output_file = get_module_output_file(
            output_prefix=current_response_output_prefix, 
            module="ce",
            version=version, 
            model_name=model_name,
            eval_output_dir=eval_output_dir
        )
        next_response_output_file = f"{eval_output_dir}/{initial_response_output_prefix}.iter_{iter+1}.json"
        
        if not os.path.exists(ce_output_file):
            print(f"Running iteration {iter}...")

            command = [
                "python", "-m", "core.run_citeeval", 
                "--response_output_file", current_response_output_file,
                "--eval_output_dir", eval_output_dir,
                "--modules", modules,
                "--version", version,
                "--model_name", model_name,
                "--n_threads", "24",
            ]
            if iter > 0:
                command.append("--skip_data_processing")

            subprocess.run(command)

            print(f"****** [DONE] Iteration {iter}: Generate citation edits *****")
            print(f"Input: {current_response_output_file}")
            print(f"Output: {ce_output_file}")
        
        else:
            print(f"Found CE output file. Skipping CiteEval for {iter}...")

        if not os.path.exists(next_response_output_file):
            generated_edited_response_output_file(
                response_output_file=current_response_output_file, 
                ce_output_file=ce_output_file, 
                new_response_output_file=next_response_output_file,
                dump=True
            )
            
            print(f"****** [DONE] Iteration {iter}: Apply citation edits *****")
            print(f"Input: {ce_output_file}")
            print(f"Output: {next_response_output_file}")
        else:
            print(f"Found new response output file. Skipping its generation for {iter}...")


def run_cr(version, model_name, citebench_dev_file, eval_output_dir, max_iter=3):
    initial_response_output_prefix = citebench_dev_file.split("/")[-1][:-len(".json")]

    ca_output_file = get_module_output_file(
        output_prefix=initial_response_output_prefix, 
        module="ca", 
        version=version, 
        model_name=model_name, 
        eval_output_dir=eval_output_dir
    )

    for iter in range(max_iter):
        modules = "cr_itercoe,cr_editdist"
        
        if iter == 0:
            current_response_output_file = citebench_dev_file
        else:
            current_response_output_file =  f"{eval_output_dir}/{initial_response_output_prefix}.iter_{iter}.json"
        
        current_response_output_prefix = current_response_output_file.split("/")[-1][:-len(".json")]

        cr_output_file = get_module_output_file(
            output_prefix=current_response_output_prefix, 
            module="cr_itercoe", 
            version=version, 
            model_name=model_name, 
            eval_output_dir=eval_output_dir
        )

        if not os.path.exists(cr_output_file):
            print(f"Running iteration {iter}...")

            command = [
                "python", "-m", "scripts.run_citeeval", 
                "--response_output_file", current_response_output_file,
                "--eval_output_dir", eval_output_dir,
                "--modules", modules,
                "--version", version,
                "--model_name", model_name,
                "--n_threads", "24",
                "--ca_output_file", ca_output_file
            ]
            if iter > 0:
                command.append("--skip_data_processing")

            subprocess.run(command)

            print(f"****** [DONE] Iteration {iter}: Run citation rating *****")
            print(f"Input: {current_response_output_file}")
            print(f"Output: {cr_output_file}")
        
        else:
            print(f"Found CR output file. Skipping CiteEval for {iter}...")


def print_ratings_per_model(version, model_name, citebench_dev_file, eval_output_dir, max_iter=30):
    initial_response_output_prefix = citebench_dev_file.split("/")[-1][:-len(".json")]

    models = ["gpt-4o", "gpt-4o-mini", "llama3_70b", "llama3_8b"]
    
    model2iter_ratings = defaultdict(list)

    def _get_model_name_from_id(id):
        if "gpt-4o-mini" in id:
            return "gpt-4o-mini"
        
        for model in  ["gpt-4o", "llama3_70b", "llama3_8b"]:
            if model in id:
                return model
        
        raise ValueError(f"Invalid id: {id}")

    for iter in range(max_iter):
        if iter == 0:
            current_response_output_file = citebench_dev_file
        else:
            current_response_output_file =  f"{eval_output_dir}/{initial_response_output_prefix}.iter_{iter}.json"
        
        current_response_output_prefix = current_response_output_file.split("/")[-1][:-len(".json")]

        cr_output_file_coe = get_module_output_file(
            output_prefix=current_response_output_prefix, 
            module="cr_itercoe", 
            version=version, 
            model_name=model_name, 
            eval_output_dir=eval_output_dir
        )

        cr_output_file_edit_dist = get_module_output_file(
            output_prefix=current_response_output_prefix, 
            module="cr_editdist", 
            version=version, 
            model_name=model_name, 
            eval_output_dir=eval_output_dir
        )

        assert os.path.exists(cr_output_file_coe), f"Not found: {cr_output_file_coe}"
        assert os.path.exists(cr_output_file_edit_dist), f"Not found: {cr_output_file_edit_dist}"
        
        model2ratings_coe = defaultdict(list)
        with io.open(cr_output_file_coe) as f:
            items = json.load(f)
            for item in items:
                model = _get_model_name_from_id(item["id"])
                model2ratings_coe[model].append(item["answer_rating"])

        model2ratings_dist = defaultdict(list)
        with io.open(cr_output_file_edit_dist) as f:
            items = json.load(f)
            for item in items:
                model = _get_model_name_from_id(item["id"])
                model2ratings_dist[model].append(item["answer_rating"])
        
        for model, ratings in model2ratings_coe.items():
            rating_coe = sum(ratings)/len(ratings)

            ratings_dist = model2ratings_dist[model]
            rating_dist = sum(ratings_dist)/len(ratings_dist)

            rating = (rating_coe + rating_dist) / 2.0
            model2iter_ratings[model].append(rating)

    for model in models:
        print(f"Model: {model}")
        print("\n".join([str(rating) for rating in model2iter_ratings[model]]))


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--citebench_dev_file", type=str, required=True, help="Path to CiteBench dev file.")
    parser.add_argument("--eval_output_dir", type=str, required=True, help="Directory to save evaluation output files.")
    parser.add_argument("--version", type=str, default="", required=False, help="Prompt template version for CiteEval evaluation.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", required=False, help="LLM backbone for CiteEval.")
    parser.add_argument("--max_iter", type=int, default=15, help="Number of iterations to run.")

    args = parser.parse_args()

    run_ce(
        version=args.version,
        model_name=args.model_name,
        citebench_dev_file=args.citebench_dev_file,
        eval_output_dir=args.eval_output_dir,
        max_iter=args.max_iter
    )
    
    run_cr(
        version=args.version,
        model_name=args.model_name,
        citebench_dev_file=args.citebench_dev_file,
        eval_output_dir=args.eval_output_dir,
        max_iter=args.max_iter
    )

    print_ratings_per_model(max_iter=args.max_iter)


if __name__ == "__main__":
    main()