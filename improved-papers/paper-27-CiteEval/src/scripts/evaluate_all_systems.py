"""Print model performance (measured by CiteEval-Auto) on CiteBench.
"""
from pathlib import Path
import io
import json
import argparse
from data.data_loader import load_response_output
from scripts.evaluate_metric import aggregate_sentence_ratings_to_answer_rating

RESPONSE_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "response_outputs"

def load_citeval_ratings(test_file):
    with io.open(test_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    sample_id2sent_ratings = {}

    for ex in preds:
        sample_id2sent_ratings[ex["id"]] = ex["sent_id2rating"]
    
    return sample_id2sent_ratings


def get_citeeval_fp(model, data, split, method, eval_output_dir):
    output_dir = Path(eval_output_dir)
    fname = f"{model}_{data}.{split}.{method}_12272024-simplified-cot_gpt-4o.out"

    return output_dir / fname


def get_sample_id2cited(model, data, split):
    response_output_file= RESPONSE_OUTPUT_DIR / f"{model}/{model}_{data}.{split}.json"
    data = load_response_output(file_path=response_output_file)
    sample_id2cited = {}
    for item in data:
        sample_id2cited[item["id"]] = []
        for sinfo in item["sent_info"]:
            if sinfo["citations"]:
                sample_id2cited[item["id"]].append(True)
            else:
                sample_id2cited[item["id"]].append(False)
    
    return sample_id2cited


def compute_model_sentence_and_response_ratings(model, data, split, method, eval_output_dir, return_all_scores=False, cited_only=False):
    citeeval_fp = get_citeeval_fp(model, data, split, method, eval_output_dir=eval_output_dir)

    sample_id2sent_ratings_model = load_citeval_ratings(test_file=citeeval_fp)
    sample_id2cited = get_sample_id2cited(model=model, data=data, split=split)

    all_sentence_ratings_model = []
    all_response_ratings_model = []
    n_response_sentences = []
    all_none_rating = 1.0

    lengths = []
    densities = []
    
    for sample_idx, sid2model_rating in sample_id2sent_ratings_model.items():
        n_sentences = len(sid2model_rating)
        n_na = list(sid2model_rating.values()).count(None)
        
        if cited_only:
            n_missing = 0
            for sent_idx in range(n_sentences):
                if sample_id2cited[sample_idx][sent_idx] or sid2model_rating[str(sent_idx+1)] is None:
                    continue

                sid2model_rating[str(sent_idx+1)] = None
                n_missing += 1

        n_cited_sentences = sample_id2cited[sample_idx].count(True)
        lengths.append(n_sentences)

        if n_sentences == 0:
            densities.append(0.0)
        else:
            densities.append(n_cited_sentences / n_sentences)
        
        sent_model_ratings = [sid2model_rating[str(sent_idx+1)] for sent_idx in range(n_sentences)]

        response_rating_model = aggregate_sentence_ratings_to_answer_rating(sent_id2rating=sid2model_rating, all_none_rating=all_none_rating)

        if all_none_rating is None and response_rating_model is None:
            continue

        if cited_only and n_missing and n_missing + n_na == n_sentences:
            response_rating_model = 0.0
        
        all_response_ratings_model.append(response_rating_model)
        # replace None with interpolation
        sent_model_ratings = [rating if rating is not None else response_rating_model for rating in sent_model_ratings]
        all_sentence_ratings_model.extend(sent_model_ratings)
        n_response_sentences.append(n_sentences)

    if return_all_scores:
        return all_sentence_ratings_model, all_response_ratings_model, n_response_sentences

    sent_rating = sum(all_sentence_ratings_model) / len(all_sentence_ratings_model)
    response_rating = sum(all_response_ratings_model) / len(all_response_ratings_model)


    length = sum(lengths) / len(lengths)
    density = sum(densities) / len(densities)
    return sent_rating, response_rating, length, density


def print_benchmarking_results(eval_output_dir, split, do_oracle, response_level, model_family, cited_only=False):
    if model_family == "l":
        models = ["llama3_70b", "llama3_8b"]
    elif model_family == "m":
        models = ["mixtral_8_22b_instruct_v0.1", "mixtral_8_7b_instruct_v0.1"]
    elif model_family == "lm":
        models = ["llama3_70b", "llama3_8b", "mixtral_8_22b_instruct_v0.1", "mixtral_8_7b_instruct_v0.1"]
    elif model_family == "g":
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif model_family == "go":
        models = ["gpt-4o", "gpt-4o-mini"]
    elif model_family == "q":
        models = ["qwen2.5_72b_instruct", "qwen2.5_7b_instruct"]
    elif model_family == "lo":
        models = ["longcite_llama3_8b", "longcite_glm4_9b"]
    else:
        raise ValueError(f"Invalid model_family: {model_family}")

    if do_oracle:
        datasets = ["asqa_oracle_10_psgs", "eli5_oracle_10_psgs", "msmarco_oracle_10_psgs"]
    else:
        datasets = ["asqa_gtr_10_psgs", "eli5_bm25_10_psgs", "msmarco_bing_10_psgs", "lfrqa_colbert_10_psgs"]
    
    citeeval_statement = {}
    citeeval_response = {}

    stats_length = {}  # n_statements / n_responses
    stats_density = {}  # n_cited_statements / n_statements
    for dataset in datasets:
        for model in models:
            eval_pair = f"{model}-{dataset}"

            sent_rating_coe, response_rating_coe, length, density = compute_model_sentence_and_response_ratings(
                model=model,
                data=dataset,
                split=split,
                method="citeeval_cr_itercoe", 
                eval_output_dir=eval_output_dir,
                cited_only=cited_only
            )
            
            sent_rating_edit_dist, response_rating_edit_dist, _, __ = compute_model_sentence_and_response_ratings(
                model=model,
                data=dataset,
                split=split, 
                method="citeeval_cr_editdist", 
                eval_output_dir=eval_output_dir, 
                cited_only=cited_only
            )
            
            sent_rating = (sent_rating_coe + sent_rating_edit_dist) / 2.0
            response_rating = (response_rating_coe + response_rating_edit_dist) / 2.0

            citeeval_statement[eval_pair] = sent_rating
            citeeval_response[eval_pair] = response_rating
            stats_length[eval_pair] = length
            stats_density[eval_pair] = density
    
    
    pair2rating = {}
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            eval_pair = f"{model}-{dataset}"
            
            if response_level:
                rating = citeeval_response[eval_pair]
            else:
                rating = citeeval_statement[eval_pair]
            
            print(f"{rating:.3f}")
            pair2rating[eval_pair] = rating
            
    
    dataset2size = {
        "asqa_gtr_10_psgs": 632, 
        "eli5_bm25_10_psgs": 684,
        "msmarco_bing_10_psgs": 684,
        "lfrqa_colbert_10_psgs": 1000
    }


    print(f"***** [Full Test Set] *****")
    for model in models:
        total_rating = 0.0
        total_size = 0.0
        for dataset in datasets:
            eval_pair = f"{model}-{dataset}"
            total_rating += pair2rating[eval_pair] * dataset2size[dataset]
            total_size += dataset2size[dataset]
    
        print(f"{model}: {total_rating / total_size:.3f}")


    print(f"***** [Statistics: Length] *****")
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            eval_pair = f"{model}-{dataset}"
            rating = stats_length[eval_pair]
            
            print(f"{rating:.3f}")
            pair2rating[eval_pair] = rating

    print(f"***** [Statistics: Density] *****")
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            eval_pair = f"{model}-{dataset}"
            rating = stats_density[eval_pair]
            
            print(f"{rating:.3f}")
            pair2rating[eval_pair] = rating




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--eval_output_dir", type=str, required=True, help="Directory to save evaluation output files.")
    parser.add_argument('--response_level', default=False, action='store_true')
    parser.add_argument('--dev', default=False, action='store_true')
    parser.add_argument('--oracle', default=False, action='store_true')
    parser.add_argument('--cited', default=False, action='store_true')
    parser.add_argument('--model', required=False, type=str)
    args = parser.parse_args()

    print_benchmarking_results(
        eval_output_dir=args.eval_output_dir,
        split="dev" if args.dev else "test", 
        do_oracle=args.oracle,
        response_level=args.response_level,
        model_family=args.model,
        cited_only=args.cited
    )
