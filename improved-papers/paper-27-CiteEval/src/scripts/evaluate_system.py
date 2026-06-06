"""Print model performance measured by CiteEval-Auto.
"""
from pathlib import Path
import io
import json
import argparse
from data.data_loader import load_response_output
from scripts.evaluate_metric import aggregate_sentence_ratings_to_answer_rating


def load_citeval_ratings(test_file):
    with io.open(test_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    sample_id2sent_ratings = {}

    for ex in preds:
        sample_id2sent_ratings[ex["id"]] = ex["sent_id2rating"]
    
    return sample_id2sent_ratings


def get_citeeval_metric_eval_output(system_output_file, method, eval_output_dir):
    output_dir = Path(eval_output_dir)
    fname = f"{system_output_file}.citeeval-auto-12272024.{method}.gpt-4o.out"
    return output_dir / fname


def get_sample_id2cited(system_output_file):
    data = load_response_output(file_path=system_output_file)
    sample_id2cited = {}
    for item in data:
        sample_id2cited[item["id"]] = []
        for sinfo in item["sent_info"]:
            if sinfo["citations"]:
                sample_id2cited[item["id"]].append(True)
            else:
                sample_id2cited[item["id"]].append(False)
    
    return sample_id2cited


def compute_model_sentence_and_response_ratings(system_output_file, metric_eval_output_file, cited_only=False):
    sample_id2sent_ratings_model = load_citeval_ratings(test_file=metric_eval_output_file)
    sample_id2cited = get_sample_id2cited(system_output_file)

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

    sent_rating = sum(all_sentence_ratings_model) / len(all_sentence_ratings_model)
    response_rating = sum(all_response_ratings_model) / len(all_response_ratings_model)


    length = sum(lengths) / len(lengths)
    density = sum(densities) / len(densities)
    return sent_rating, response_rating, length, density


def print_benchmarking_results(system_output_file, metric_eval_output_file, cited_only):
    metric_eval_output_files = metric_eval_output_file.split(",")
    all_sent_ratings = []
    all_response_ratings = []
    length = None
    density = None

    for metric_eval_output_file in metric_eval_output_files:
        _sent_rating, _response_rating, length, density = compute_model_sentence_and_response_ratings(
            system_output_file=system_output_file,
            metric_eval_output_file=metric_eval_output_file,
            cited_only=cited_only
        )
        all_sent_ratings.append(_sent_rating)
        all_response_ratings.append(_response_rating)
    
    sent_rating = sum(all_sent_ratings) / len(all_sent_ratings)
    response_rating = sum(all_sent_ratings) / len(all_sent_ratings)

    res = {
        "statement_rating": sent_rating,
        "response_rating": response_rating,
        "length": length,
        "density": density
    }

    print("="*50)
    print(f"CiteEval-Auto Configs")
    print(f"\nSystem output:\n\t{system_output_file}")
    print(f"\nMetric output:\n\t{metric_eval_output_files}")
    print(f"\nCited-only: {cited_only}")
    print
    print("="*50)
    print(f"CiteEval-Auto Results")
    for key in ["statement_rating", "response_rating", "length", "density"]:
        print(f"\n{key}: {res[key]}")
    
        
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--system_output", type=str, required=True, help="System output prediction file.")
    parser.add_argument("--metric_output", type=str, required=True, help="Metric output prediction file.")
    parser.add_argument('--cited', default=False, action='store_true')
    args = parser.parse_args()

    print_benchmarking_results(
        system_output_file=args.system_output,
        metric_eval_output_file=args.metric_output,
        cited_only=args.cited
    )
