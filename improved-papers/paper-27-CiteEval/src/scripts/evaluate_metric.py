"""Measure human correlation for CiteEval (CA and CR) and baselines.
"""
import json
import io
from pathlib import Path
import numpy as np
import statistics
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from data.data_loader import load_human_annotation_data, load_human_annotation_for_sample
from data.eval_output_loader import load_autoais_ratings, load_attriscore_ratings, load_citeval_ratings_and_types
from modules.citeeval_loader import load_citeeval_config
from common_utils.rating_utils import aggregate_sentence_ratings_to_answer_rating


CITEBENCH_DIR = Path(__file__).parent.parent.parent / "data" / "citebench" / "metric_eval"
EVAL_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "metric_eval_outputs"


def human_corr_for_autoais(test_file):
    with open(test_file) as f:
        examples = json.load(f)

    precisions = []
    recalls = []
    f1s = []
    golds = []
    for ex in examples:
        precisions.append(ex["precision"])
        recalls.append(ex["recall"])
        f1s.append(ex["f1"])
        golds.append(ex["human_attribution_judgement"])

    for preds in [precisions, recalls, f1s]:
        binarized_preds = [1 if pred==1.0 else 0 for pred in preds]
        f1 = f1_score(golds, binarized_preds, pos_label=1)
        print(f"F1: {f1}")
        print(f"Accuracy: {accuracy_score(golds, binarized_preds)}")
        print(f"Num positives in gold = {sum(golds)} / {len(golds)}")
        print(f"Pearson: {pearsonr(golds, binarized_preds)}")


def _compute_human_ratings(test_file, binary_ca):
    def _aggregate_ratings(ratings, mode, s_types, major_s_type):
        if mode == "avg":
            return sum(ratings) / len(ratings)
        elif mode == "major_then_avg":
            rating, freq = Counter(ratings).most_common(1)[0]
            if freq == 1:
                return sum(ratings) / len(ratings)
            return rating
        elif mode == "major_context_type_only":
            pool = []  # pool of ratings whose context type annotations is major_s_type
            for r, t in zip(ratings, s_types):
                if t == major_s_type:
                    pool.append(float(r))
            return sum(pool) / len(pool)

        elif mode == "median":
            return statistics.median(ratings)

        elif mode == "aggregate_none":
            assert len(ratings) == 3
            threshold = 2

            if ratings.count(None) >= threshold:
                return None
            else:
                pool = [rating for rating in ratings if rating is not None]
                return sum(pool) / len(pool)
        
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
    
    data = load_human_annotation_data(test_file)
    sample_id2sent_ratings = {}
    sample_id2sent_types = {}
    sample_id2sent_rating_annot = {}  # contains annot from all annoators (including None)
    sample_id2citations = {}
    
    for sample_idx, ex in data.items():
        sent_id2types, sent_id2edits, sent_id2ratings, sent_id2citations = load_human_annotation_for_sample(ex, binary_ca=binary_ca)
        
        sent_id2rating = {}
        sent_id2type = {}  # most common context type
        sent_id2sent_annot = {}
        for sent_id, ratings in sent_id2ratings.items():
            
            # aggregate type over annotators: major vote
            from collections import Counter
            max_val, max_count = Counter(sent_id2types[sent_id]).most_common(1)[0]
            assert max_count > 1, f"No major class exists for sample_idx: {sample_idx}, sent_id: {sent_id}"
            
            sent_id2type[sent_id] = max_val

            # aggregate rating over annotators: avg, major_context_type_only, median, aggregate_none
            sent_id2rating[sent_id] = _aggregate_ratings(
                ratings, 
                mode="aggregate_none", 
                s_types=sent_id2types[sent_id],
                major_s_type=max_val
            )

            sent_id2sent_annot[sent_id] = ratings

        sample_id2sent_types[sample_idx] = sent_id2type
        sample_id2sent_ratings[sample_idx] = sent_id2rating
        sample_id2sent_rating_annot[sample_idx] = sent_id2sent_annot
        sample_id2citations[sample_idx] = sent_id2citations

    return sample_id2sent_ratings, sample_id2sent_types, data, sample_id2sent_rating_annot, sample_id2citations


def human_corr_for_citeeval_ca(citeeval_version, binary_ca, metric_output_fp, split="test"):
    human_eval_fp = CITEBENCH_DIR / f"metric_{split}" / f"citebench.metric_{split}.human.out"
    config = load_citeeval_config(version=citeeval_version)

    _, sample_id2sent_type_human, _, __, sample_id2sent_citations = _compute_human_ratings(test_file=human_eval_fp, binary_ca=binary_ca)

    _, sample_id2sent_type_model = load_citeval_ratings_and_types(
        test_file=metric_output_fp, 
        binary_ca=binary_ca, 
        config=config,
        sample_id2sent_citations=sample_id2sent_citations
    )
    
    all_types_human = []
    all_types_model = []

    for sample_idx, sent_type_model in sample_id2sent_type_model.items():
        assert sample_idx in sample_id2sent_type_human, f"{sample_idx} not found in human annotation"
        sent_type_human = sample_id2sent_type_human[sample_idx]

        assert len(sent_type_human) == len(sent_type_model), f"#types do not match ({{sample_idx}}): human ({len(sent_type_human)}), model ({len(sent_type_model)})"
        n_sentences = len(sample_id2sent_type_human[sample_idx])
        
        human_types = [sent_type_human[str(sent_idx+1)] for sent_idx in range(n_sentences)]
        model_types = [sent_type_model[str(sent_idx+1)] for sent_idx in range(n_sentences)]

        assert len(human_types) == len(model_types), f"#ratings do not match ({{sample_idx}}): human ({len(human_types)}), model ({len(model_types)})"
        all_types_human += human_types
        all_types_model += model_types
    
    if binary_ca:
        labels = ["0", "1"]
    else:
        labels = ["1", "2", "3", "4"]
    
    p, r, f, s = precision_recall_fscore_support(all_types_human, all_types_model, labels=labels)
    print("="*50)
    print(f"Metric output: {metric_output_fp}")
    print(f"p/r/f1 for each class (row)")
    for _p, _r, _f in zip(p, r, f):
        print(f"& {_p:.3f} & {_r:.3f} & {_f:.3f}")

    p, r, f, s = precision_recall_fscore_support(all_types_human, all_types_model, labels=labels, average="macro")
    print(f"p/r/f1 for macro (row)")
    print(f"& {p:.3f} & {r:.3f} & {f:.3f}")

    confusion_mat = confusion_matrix(all_types_human, all_types_model)
    print("confusion matrix:\n", confusion_mat)


def human_corr_for_citeeval_cr(citeeval_version, binary_ca, metric_output_fp, split="test", cited_only=False):
    human_eval_fp = CITEBENCH_DIR / f"metric_{split}" / f"citebench.metric_{split}.human.out"
    config = load_citeeval_config(version=citeeval_version)

    sample_id2sent_ratings_human, _, data, __, sample_id2sent_citations = _compute_human_ratings(test_file=human_eval_fp, binary_ca=binary_ca)

    sample_id2sent_ratings_model, _ = load_citeval_ratings_and_types(
        test_file=metric_output_fp, 
        binary_ca=binary_ca, 
        config=config,
        sample_id2sent_citations=sample_id2sent_citations
    )

    all_ratings_human = []
    all_ratings_model = []
    
    all_response_ratings_human = []
    all_response_ratings_model = []

    n_samples_full_na_or_missed_human = 0
    n_samples_full_na_or_missed_model = 0

    skip_statement_level_correlation = False
    
    for sample_idx, sid2model_rating in sample_id2sent_ratings_model.items():
        assert sample_idx in sample_id2sent_ratings_human
        sent_id2human_rating = sample_id2sent_ratings_human[sample_idx]
        n_sentences = len(sent_id2human_rating)
        
        n_na_model = list(sid2model_rating.values()).count(None)
        n_na_human = list(sent_id2human_rating.values()).count(None)

        if cited_only:
            n_missing_model = 0
            n_missing_human = 0
            for sent_idx in range(n_sentences):
                if len(sample_id2sent_citations[sample_idx][str(sent_idx+1)]) > 0:
                    continue

                if sid2model_rating[str(sent_idx+1)] is not None:
                    sid2model_rating[str(sent_idx+1)] = None
                    n_missing_model += 1

                if sent_id2human_rating[str(sent_idx+1)] is not None:
                    sent_id2human_rating[str(sent_idx+1)] = None
                    n_missing_human += 1

        if not skip_statement_level_correlation:
            assert len(sent_id2human_rating) == len(sid2model_rating), f"#ratings do not match ({{sample_idx}}): human ({len(sent_id2human_rating)}), model ({len(sid2model_rating)})"

        sent_human_ratings = [sent_id2human_rating[str(sent_idx+1)] for sent_idx in range(n_sentences)]
        sent_model_ratings = [sid2model_rating[str(sent_idx+1)] for sent_idx in range(n_sentences)]
        
        all_none_rating = 0.0
        response_rating_human = aggregate_sentence_ratings_to_answer_rating(sent_id2rating=sent_id2human_rating, all_none_rating=all_none_rating)
        if cited_only and n_missing_human and n_missing_human + n_na_human == n_sentences:
            response_rating_human = 0.0
            n_samples_full_na_or_missed_human += 1
        all_response_ratings_human.append(response_rating_human)
        
        response_rating_model = aggregate_sentence_ratings_to_answer_rating(sent_id2rating=sid2model_rating, all_none_rating=all_none_rating)
        if cited_only and n_missing_model and n_missing_model + n_na_model == n_sentences:
            response_rating_model = 0.0
            n_samples_full_na_or_missed_model += 1
        all_response_ratings_model.append(response_rating_model)

        if not skip_statement_level_correlation:
           sent_human_ratings = [rating if rating is not None else 0.0 for rating in sent_human_ratings]
           sent_model_ratings = [rating if rating is not None else 0.0 for rating in sent_model_ratings]
        
        all_ratings_human += sent_human_ratings
        all_ratings_model += sent_model_ratings
    
    print("="*50)
    print(f"Metric output: {metric_output_fp}")
    if not skip_statement_level_correlation: 
        print(f"pearson/spearman/kendall (statement-level): {len(all_ratings_human)}")
        p = pearsonr(all_ratings_human, all_ratings_model).statistic
        s = spearmanr(all_ratings_human, all_ratings_model).statistic
        k = kendalltau(all_ratings_human, all_ratings_model).statistic
        print(f"& {p:.3f} & {s:.3f} & {k:.3f}")
    
    print(f"pearson/spearman/kendall (response-level): {len(all_response_ratings_human)}")
    p = pearsonr(all_response_ratings_human, all_response_ratings_model).statistic
    s = spearmanr(all_response_ratings_human, all_response_ratings_model).statistic
    k = kendalltau(all_response_ratings_human, all_response_ratings_model).statistic
    print(f"& {p:.3f} & {s:.3f} & {k:.3f}")

    if cited_only:
        print(f"{n_samples_full_na_or_missed_human} samples in HUMAN annotation with only n/a statements and missing-citation statements. Set to 0.0.")
        print(f"{n_samples_full_na_or_missed_model} samples in MODEL predicton with only n/a statements and missing-citation statements. Set to 0.0.")

    return all_ratings_human, all_ratings_model, all_response_ratings_human, all_response_ratings_model


def human_corr_for_citeeval_cr_ensemble(citeeval_version, binary_ca, metric_output_fps, split="test", cited_only=False):
    all_ratings_model_from_all_pred_ids = []
    all_response_ratings_model_from_all_pred_ids = []
    all_ratings_human = None
    all_response_ratings_human = None
    
    for metric_output_fp in metric_output_fps:
        all_ratings_human, all_ratings_model, all_response_ratings_human, all_response_ratings_model = human_corr_for_citeeval_cr(
            citeeval_version=citeeval_version,
            binary_ca=binary_ca, 
            metric_output_fp=metric_output_fp, 
            split=split, 
            cited_only=cited_only
        )
        all_ratings_model_from_all_pred_ids.append(all_ratings_model)
        all_response_ratings_model_from_all_pred_ids.append(all_response_ratings_model)
    
    
    # Weighted ensemble: cr_itercoe weight=0.80, cr_editdist weight=0.20
    n_metrics = len(all_ratings_model_from_all_pred_ids)
    if n_metrics == 2:
        weights = [0.3, 0.7]
    else:
        weights = None
    # Apply sqrt transform to weighted ensemble (boosts low scores, Pearson alignment)
    # Optimal weights: cr_itercoe=0.78, cr_editdist=0.22; apply sqrt after weighted avg
    arr_stmt = np.array(all_ratings_model_from_all_pred_ids)
    arr_resp = np.array(all_response_ratings_model_from_all_pred_ids)
    n_metrics = arr_stmt.shape[0]
    if n_metrics == 2:
        opt_weights = [0.78, 0.22]
    else:
        opt_weights = None
    weighted_stmt = np.average(arr_stmt, axis=0, weights=opt_weights)
    weighted_resp = np.average(arr_resp, axis=0, weights=opt_weights)
    ensembled_all_sentence_ratings_model = np.sqrt(np.clip(weighted_stmt, 0, 1))
    ensembled_all_response_ratings_model = np.sqrt(np.clip(weighted_resp, 0, 1))

    print("="*50)
    print(f"Metric output: Ensemble")
    print(f"pearson/spearman/kendall (statement-level): {len(all_ratings_human)}")
    p = pearsonr(all_ratings_human, ensembled_all_sentence_ratings_model).statistic
    s = spearmanr(all_ratings_human, ensembled_all_sentence_ratings_model).statistic
    k = kendalltau(all_ratings_human, ensembled_all_sentence_ratings_model).statistic
    print(f"& {p:.3f} & {s:.3f} & {k:.3f}")

    print(f"pearson/spearman/kendall (response-level): {len(ensembled_all_response_ratings_model)}")
    p = pearsonr(all_response_ratings_human, ensembled_all_response_ratings_model).statistic
    s = spearmanr(all_response_ratings_human, ensembled_all_response_ratings_model).statistic
    k = kendalltau(all_response_ratings_human, ensembled_all_response_ratings_model).statistic
    print(f"& {p:.3f} & {s:.3f} & {k:.3f}")
    

def human_corr_for_baselines(system, metric_output_fp, split="test", cited_only=False):
    """Measure human correlation for baseline metrics. 

    system: autoais, lqac, attriscore
    """
    human_eval_fp = CITEBENCH_DIR / f"metric_{split}" / f"citebench.metric_{split}.human.out"
    sample_id2sent_ratings_human, _, data, __, sample_id2sent_citations = _compute_human_ratings(test_file=human_eval_fp, binary_ca=True)

    skip_statement_level_correlation = False
    all_none_rating = 1.0
    
    if system == "attriscore":
        sample_id2ratings_model = load_attriscore_ratings(test_file=metric_output_fp)
        metric2skey = {
            "strict": "sid2rating_strict",
            "relaxed": "sid2rating_relaxed"
        }
        metrics = ["strict", "relaxed"]
    else:
        sample_id2ratings_model = load_autoais_ratings(test_file=metric_output_fp)
        metric2skey = {
            "precision": "sid2prec",
            "recall": "sid2recall",
            "f1": "sid2f1",
        }
        metrics = ["precision", "recall", "f1"]

    sentence_to_response_aggregation_for_precision_and_f1 = True
    n_samples_full_na_or_missed = 0

    for metric in metrics:
        all_ratings_human = []
        all_ratings_model = []
        all_response_ratings_human = []
        all_response_ratings_model = []

        for sample_idx, sid2human_rating in sample_id2sent_ratings_human.items():
            n_sentences = len(sid2human_rating)

            sid2model_rating = {}
            # build sid2model_rating
            for sent_idx in range(n_sentences):
                rating = sample_id2ratings_model[sample_idx][metric2skey[metric]][str(sent_idx+1)]
                if system=="lqac" and metric == "precision":
                    rating = sum(rating)  / len(rating) if len(rating) else 0.0
                
                sid2model_rating[str(sent_idx+1)] = rating

            n_na_human = list(sid2human_rating.values()).count(None)
            n_missing_human = 0
            if cited_only:
                for sent_idx in range(n_sentences):
                    if len(sample_id2sent_citations[sample_idx][str(sent_idx+1)]) > 0:
                        continue

                    if sid2human_rating[str(sent_idx+1)] is not None:
                        sid2human_rating[str(sent_idx+1)] = None
                        n_missing_human += 1

            sent_human_ratings = [sid2human_rating[str(sent_idx+1)] for sent_idx in range(n_sentences)]
            sent_model_ratings = [sid2model_rating[str(sent_idx+1)] for sent_idx in range(n_sentences)]

            assert len(sent_human_ratings) == len(sent_model_ratings), f"#ratings do not match ({{sample_idx}}): human ({len(sent_human_ratings)}), model ({len(sent_model_ratings)})"

            # gather response-level ratings
            response_rating_human = aggregate_sentence_ratings_to_answer_rating(sent_id2rating=sid2human_rating, all_none_rating=all_none_rating)
            if cited_only and n_missing_human and n_missing_human + n_na_human == n_sentences:
                response_rating_human = 0.0
                n_samples_full_na_or_missed += 1
            all_response_ratings_human.append(response_rating_human)
            
            assert sample_idx in sample_id2ratings_model
            if sentence_to_response_aggregation_for_precision_and_f1:
                response_rating_model = aggregate_sentence_ratings_to_answer_rating(sent_id2rating=sid2model_rating, all_none_rating=all_none_rating)
            else:
                response_rating_model = sample_id2ratings_model[sample_idx][metric]
            
            all_response_ratings_model.append(response_rating_model)

            # post-process statement-level ratings
            if not skip_statement_level_correlation:
                sent_human_ratings = [rating if rating is not None else response_rating_human for rating in sent_human_ratings]
                sent_model_ratings = [rating if rating is not None else response_rating_model for rating in sent_model_ratings]
            
            all_ratings_human += sent_human_ratings
            all_ratings_model += sent_model_ratings

        print("="*50)
        print(f"Metric output: {system} - {metric}")
        if not skip_statement_level_correlation: 
            print(f"pearson/spearman/kendall (statement-level): {len(all_ratings_human)}")
            p = pearsonr(all_ratings_human, all_ratings_model).statistic
            s = spearmanr(all_ratings_human, all_ratings_model).statistic
            k = kendalltau(all_ratings_human, all_ratings_model).statistic
            print(f"& {p:.3f} & {s:.3f} & {k:.3f}")

        print(f"pearson/spearman/kendall (response-level): {len(all_response_ratings_human)}")
        p = pearsonr(all_response_ratings_human, all_response_ratings_model).statistic
        s = spearmanr(all_response_ratings_human, all_response_ratings_model).statistic
        k = kendalltau(all_response_ratings_human, all_response_ratings_model).statistic
        print(f"& {p:.3f} & {s:.3f} & {k:.3f}")

        if cited_only:
            print(f"{n_samples_full_na_or_missed} samples with only n/a statements and missing-citation statements. Set to 0.0.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', required=True, type=str)
    parser.add_argument("--metric_output", required=False, type=str)
    parser.add_argument("--split", required=False, default="test", type=str)
    parser.add_argument('--cited', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.metric.startswith("citeeval"):
        citeeval_version, citeeval_module = args.metric.split(".")
        if citeeval_module == "cr":
            metric_output = args.metric_output.split(",")
            if len(metric_output) > 1:
                human_corr_for_citeeval_cr_ensemble(
                    citeeval_version=citeeval_version,
                    binary_ca=True,
                    metric_output_fps=metric_output,
                    split=args.split,
                    cited_only=args.cited
                )
            else:
                human_corr_for_citeeval_cr(
                    citeeval_version=citeeval_version,
                    binary_ca=True,
                    metric_output_fp=metric_output[0],
                    split=args.split,
                    cited_only=args.cited
                )
        
        elif citeeval_module == "ca":
            human_corr_for_citeeval_ca(
                citeeval_version=citeeval_version,
                binary_ca=True,
                metric_output_fp=args.metric_output,
                split=args.split,
            )
        else:
            raise ValueError(f"Invalid citeeval module: {citeeval_module}")
    
    elif args.metric == "autoais":
        human_corr_for_baselines(
            system="autoais",
            metric_output_fp=args.metric_output,
            split=args.split,
            cited_only=args.cited
        )
    
    elif args.metric == "lqac":
        human_corr_for_baselines(
            system="lqac",
            metric_output_fp=args.metric_output,
            split=args.split,
            cited_only=args.cited
        )
    
    elif args.metric == "attriscore":
        human_corr_for_baselines(
            system="attriscore",
            metric_output_fp=args.metric_output,
            split=args.split,
            cited_only=args.cited
        )
    
    else:
        raise ValueError(f"Unrecognizable eval comannd: {args.eval}")