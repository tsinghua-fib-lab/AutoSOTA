"""CiteBench: IAA, annotation statistics at each stage and (dataset, model) pair.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from nltk.metrics.distance import binary_distance, interval_distance
from nltk.metrics.agreement import AnnotationTask
from data.data_loader import load_human_annotation_data, load_human_annotation_for_sample


CITEBENCH_DIR = Path(__file__).parent.parent.parent / "data" / "citebench"
CITEBENCH_FILE = CITEBENCH_DIR / "metric_eval" / "citebench.metric_eval.human.out"


def get_model(ex_id):
    if ex_id.startswith("llama3_70b"):
        return "llama3_70b"
    elif ex_id.startswith("llama3_8b"):
        return "llama3_8b"
    elif ex_id.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    elif ex_id.startswith("gpt-4o"):
        return "gpt-4o"
    
    raise ValueError()


def print_response_length_distribution():
    data = load_human_annotation_data(CITEBENCH_FILE)
    count2freq = defaultdict(int)
    model2ones = defaultdict(int)
    nw2freq = defaultdict(int)

    total = 0.0
    def nw2bracket(nw):
        brackets = [50, 30, 20, 10, 0]
        for i in brackets:
            if nw >= i:
                return i

    for sample_idx, ex in data.items():
        nw = len(ex["1"]["prediction"].split(" "))
        nw2freq[nw2bracket(nw)] += 1

        model = get_model(ex_id=sample_idx)
        sent_id2types = load_human_annotation_for_sample(ex)[0]
        n_sents = len(sent_id2types)
        
        if n_sents == 1:
            model2ones[model] += 1
        
        count2freq[n_sents] += 1
        total += 1

    for k, v in sorted(nw2freq.items(), key=lambda x: x[0], reverse=False):
        print(f"{k}: {v/total * 100:.2f}")


def collect_data_statistics():
    data = load_human_annotation_data(CITEBENCH_FILE)
    sample_id2sent_ratings = {}
    sample_id2sent_types = {}
    
    all_sent_ratings_from_all_annotators = []
    all_sent_types_from_all_annotators = []

    n_anotators = 3
    n_annotated_samples = len(data)
    n_annotated_sentences = 0

    all_response_ratings_from_all_annotators = []
    all_edits = []

    refilled_all_sent_ratings_from_all_annotators = []

    for sample_idx, ex in data.items():
        sent_ratings = [[], [], []]

        sent_id2types, sent_id2edits, sent_id2ratings, __ = load_human_annotation_for_sample(ex, binary_ca=False)

        # gather edits
        for edits in sent_id2edits.values():
            total_edits = np.sum(np.array(edits), axis=0)
            all_edits.append(total_edits)

        n_annotated_sentences += len(sent_id2ratings)

        sorted_sent_ids = sorted([int(idx) for idx in sent_id2ratings.keys()])
        for sent_id in sorted_sent_ids:
            sent_id = str(sent_id)
            ratings = sent_id2ratings[sent_id]
            all_sent_types_from_all_annotators.append(sent_id2types[sent_id])
            assert len(ratings) == 3

            all_sent_ratings_from_all_annotators.append(ratings)
            for annot_idx, rating in enumerate(ratings):
                sent_ratings[annot_idx].append(rating)
        
        # get response-level ratings (response rating per annotator -> final response rating)
        response_ratings = []
        for annot_idx, ratings in enumerate(sent_ratings):
            _ratings = [r for r in ratings if r is not None]
            if not _ratings:
                response_rating = 1.0
            else:
                response_rating = sum(_ratings) / float(len(ratings))
            response_ratings.append(response_rating)

            sent_ratings[annot_idx] = [
                response_rating if r is None else r for r in sent_ratings[annot_idx]
            ]

        assert len(response_ratings) == 3
        all_response_ratings_from_all_annotators.append(response_ratings)


        sample_id2sent_ratings[sample_idx] = []

        for sid in range(len(sent_ratings[0])):
            r = [float(sent_ratings[annot_idx][sid]) for annot_idx in range(len(sent_ratings))]
            refilled_all_sent_ratings_from_all_annotators.append(r)
            sample_id2sent_ratings[sample_idx].append(r)

    assert n_annotated_sentences == len(all_sent_ratings_from_all_annotators) == len(all_sent_types_from_all_annotators)

    return {
        "n_annotated_samples": n_annotated_samples,
        "n_annotated_sentences": n_annotated_sentences,
        "n_anotators": n_anotators,
        "sample_id2sent_ratings": sample_id2sent_ratings,
        "sample_id2sent_types": sample_id2sent_types,
        "all_sent_ratings_from_all_annotators": all_sent_ratings_from_all_annotators,
        "all_sent_types_from_all_annotators": all_sent_types_from_all_annotators,
        "all_response_ratings_from_all_annotators": all_response_ratings_from_all_annotators,
        "all_edits": all_edits,
        "refilled_all_sent_ratings_from_all_annotators": refilled_all_sent_ratings_from_all_annotators
    }


def compute_iaa():
    """Compute IAA for human annotation.
    """
    data_stats = collect_data_statistics()

    def _make_task(data, distance_func):
        task_data = []
        for annot_idx in range(data_stats["n_anotators"]):
            for sent_idx, ratings in enumerate(data):
                if distance_func == binary_distance:
                    ratings[annot_idx] = str(ratings[annot_idx])
                task_data.append([annot_idx, str(sent_idx), ratings[annot_idx]])
        
        rating_task = AnnotationTask(data=task_data, distance=distance_func)
        return rating_task

    components = ["Context Attribution", "Citation Rating (Statement)", "Citation Rating (Response)"]
    distance_functions = [binary_distance, interval_distance, interval_distance]

    print(f"*** IAA (Krippendorff’s alpha)***")
    
    for idx, data in enumerate([
        data_stats["all_sent_types_from_all_annotators"], 
        data_stats["refilled_all_sent_ratings_from_all_annotators"], 
        data_stats["all_response_ratings_from_all_annotators"]]
    ):
        task = _make_task(data=data, distance_func=distance_functions[idx])
        print(f"{components[idx]}: {task.alpha():.3f}")


def print_overview():
    data_stats = collect_data_statistics()

    print(f"*** Basic Statistics ***")
    print(f"#Samples: {data_stats['n_annotated_samples']}")
    print(f"#Statements: {data_stats['n_annotated_sentences']}")

    print(f"\n\n*** Statement Type Distribution (%) ***")
    from collections import Counter
    all_types = sum(data_stats["all_sent_types_from_all_annotators"], [])
    context_type_counter =  Counter(all_types)
    for i in range(1, 5):
        print(f"{i}: {100 * context_type_counter[str(i)] /float(len(all_types)):.3f}")


    print(f"\n\n*** Edit Distribution (%) ***")
    edit_counter = np.sum(np.array(data_stats['all_edits']), axis=0)
    for c in edit_counter:
        print(f"{100 * c/float(np.sum(data_stats['all_edits'])):.3f}")


    print(f"\n\n*** Rating Distribution (%) ***")
    all_ratings = sum(data_stats["all_sent_ratings_from_all_annotators"], [])
    rating_counter =  Counter(all_ratings)
    ratings = [0.0, 0.25, 0.5, 0.75, 1.0]
    total = sum([rating_counter[i] for i in ratings])
    for i in ratings:
        print(f"{i}: {100 * rating_counter[i] /float(total):.3f}")


def eval_pairs():
    """Print performance for each (model, dataset) pair.
    """
    data = load_human_annotation_data(CITEBENCH_FILE)
    sample_id2sent_ratings = {}
    n_annotated_sentences = 0

    all_response_ratings_from_all_annotators = []
    all_edits = []

    refilled_all_sent_ratings_from_all_annotators = []

    sample_id2density = {}
    for sample_idx, ex in data.items():
        sent_ratings = [[], [], []]
        _, sent_id2edits, sent_id2ratings, sent_id2citations = load_human_annotation_for_sample(ex, binary_ca=False)

        print(sent_id2citations)
        sample_id2density[sample_idx] = sum([1 if len(c)>0 else 0 for _, c in sent_id2citations.items()]) / float(len(sent_id2citations))

        # gather edits
        for edits in sent_id2edits.values():
            total_edits = np.sum(np.array(edits), axis=0)
            all_edits.append(total_edits)

        n_annotated_sentences += len(sent_id2ratings)

        sorted_sent_ids = sorted([int(idx) for idx in sent_id2ratings.keys()])

        for sent_id in sorted_sent_ids:
            sent_id = str(sent_id)
            ratings = sent_id2ratings[sent_id]
            assert len(ratings) == 3
            for annot_idx, rating in enumerate(ratings):
                sent_ratings[annot_idx].append(rating)
        
        
        # get response-level ratings (response rating per annotator -> final response rating)
        response_ratings = []
        for annot_idx, ratings in enumerate(sent_ratings):
            _ratings = [r for r in ratings if r is not None]
            if not _ratings:
                response_rating = 1.0
            else:
                response_rating = sum(_ratings) / float(len(ratings))
            response_ratings.append(response_rating)

            sent_ratings[annot_idx] = [
                response_rating if r is None else r for r in sent_ratings[annot_idx]
            ]

        assert len(response_ratings) == 3
        all_response_ratings_from_all_annotators.append(response_ratings)


        sample_id2sent_ratings[sample_idx] = []

        for sid in range(len(sent_ratings[0])):
            r = [float(sent_ratings[annot_idx][sid]) for annot_idx in range(len(sent_ratings))]
            refilled_all_sent_ratings_from_all_annotators.append(r)
            sample_id2sent_ratings[sample_idx].append(sum(r) / len(r))


    models = ["gpt-4o", "gpt-4o-mini", "llama3_70b", "llama3_8b"]
    datasets = ["asqa_gtr_10_psgs", "eli5_bm25_10_psgs", "msmarco_bing_10_psgs"]
    
    pair2sentence_ratings = {}
    pair2response_ratings = {}
    
    sample_lengths = []
    sample_ratings = []
    sample_densities = []
    
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            eval_pair = f"{model}_{dataset}"
            pair2sentence_ratings[eval_pair] = []
            pair2response_ratings[eval_pair] = []

    for sample_id, ratings in sample_id2sent_ratings.items():
        for pair in pair2sentence_ratings:
            if sample_id.startswith(pair):
                pair2sentence_ratings[pair].extend(ratings)
                response_rating = sum(ratings)/len(ratings)
                pair2response_ratings[pair].append(response_rating)
                
                sample_ratings.append(response_rating)
                sample_lengths.append(len(ratings))
                sample_densities.append(sample_id2density[sample_id])
                break
    
    print("***** Statement-level performance ******")
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            pair = f"{model}_{dataset}"
            print(f"{sum(pair2sentence_ratings[pair])/len(pair2sentence_ratings[pair])}")

    
    print("***** Response-level performance ******")
    for dataset in datasets:
        print(f"{dataset}: {models}")
        for model in models:
            pair = f"{model}_{dataset}"
            print(f"{sum(pair2response_ratings[pair])/len(pair2response_ratings[pair])}")


    print("***** Response-level correlation between ratings and statistics ******")
    p = pearsonr(sample_lengths, sample_ratings)
    print(f"pearsonr (LR): {p.statistic}, {p.pvalue}")

    p = pearsonr(sample_lengths, sample_densities)
    print(f"pearsonr (LD): {p.statistic}, {p.pvalue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=str)
    args = parser.parse_args()
    
    if args.run == "iaa":
        compute_iaa()
    
    elif args.run == "overview":
        print_overview()
    
    elif args.run == "eval_pairs":
        eval_pairs()

    elif args.run == "response_len":
        print_response_length_distribution()
    
    else:
        raise ValueError(f"Unrecognizable evaluator from file name: {args.test_file}")
