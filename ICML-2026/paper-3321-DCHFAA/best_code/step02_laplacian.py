"""
Train classifier using Laplacian operator features for hallucination detection.

Matches the original Lookback-Lens training pipeline:
  - token-level labels (one classification example per new token in the response)
  - sliding-window=1 by default
  - concat order: [context_features, new_tokens_features]
"""

import editdistance as ed
import numpy as np
import json
import pickle
import torch
import transformers
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


jsonl_path_dict = {
    "summary": "dataset/ragtruth/llama-2-7b-chat/anno-Summary-7b.jsonl",
    "qa": "dataset/ragtruth/llama-2-7b-chat/anno-QA-7b.jsonl",
    "data2txt": "dataset/ragtruth/llama-2-7b-chat/anno-Data2txt-7b.jsonl",
}


def min_edit_distance_substring(s1, s2):
    """Find minimum edit distance substring."""
    len_s1 = len(s1)
    min_edit_dist = float('inf')
    best_substring = None
    assert len(s2) >= len_s1, "s2 must be longer than s1"

    for i in range(len(s2) - len_s1 + 1):
        sub_s2 = s2[i:i + len_s1]
        dist = ed.eval(s1, sub_s2)
        if dist < min_edit_dist:
            min_edit_dist = dist
            best_substring = sub_s2
    return best_substring, min_edit_dist


def load_files(anno_file, attn_file, tokenizer_name=None, auth_token=None, ifft_mode=None):
    """Load annotation and attention files, returning per-example tensors with per-token labels."""
    if "mistral" in tokenizer_name.lower():
        tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_name, token=auth_token)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, token=auth_token)

    def manual_offset_mapping(text, token_ids, tokenizer):
        decoded_tokens = []
        for token_id in token_ids:
            decoded_tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))

        offsets = []
        running_start = 0
        cur_text = text
        for t in decoded_tokens:
            t_stripped = t.lstrip() if cur_text.startswith(" ") else t
            t_len = len(t_stripped)
            idx = cur_text.find(t_stripped)
            if idx == -1:
                idx = cur_text.find(t)
                t_len = len(t)
                if idx == -1:
                    offsets.append((running_start, running_start))
                    continue
            start = running_start + idx
            end = start + t_len
            offsets.append((start, end))
            running_start = end
            cur_text = text[running_start:]
        return offsets

    anno_data = []
    attn_data = []

    anno_path = jsonl_path_dict[anno_file]
    attn_path = attn_file

    with open(anno_path, 'r') as f:
        for line in f:
            anno_data.append(json.loads(line))
    attn_data.extend(torch.load(attn_path, weights_only=False))

    # Align by data_index so partial step01 runs match anno entries regardless of order.
    anno_by_index = {a['index']: a for a in anno_data}
    matched_anno, matched_attn = [], []
    for a in attn_data:
        di = a['data_index']
        if di in anno_by_index:
            matched_anno.append(anno_by_index[di])
            matched_attn.append(a)
    anno_data = matched_anno
    attn_data = matched_attn

    if ifft_mode is not None and "laplacian" in ifft_mode and "l2" not in ifft_mode:
        attn_feature_key = 'all_previous_laplacian_score'
        context_key = 'context_laplacian_score'
        new_tokens_key = 'new_tokens_laplacian_score'
    else:
        attn_feature_key = 'context_laplacian_l2_score'
        context_key = 'context_laplacian_l2_score'
        new_tokens_key = 'new_tokens_laplacian_l2_score'

    attn_tensor = []
    new_tokens_tensors = []
    context_tensors = []
    labels = []
    splits = []
    skipped_examples = 0

    for idx in range(len(anno_data)):
        is_hallu = len(anno_data[idx]['labels']) > 0
        this_split = anno_data[idx]['split']

        num_new_tokens = attn_data[idx][attn_feature_key].shape[-1]

        if is_hallu:
            if "mistral" in tokenizer_name.lower():
                tokenized_hallucination = tokenizer(anno_data[idx]['response'])
                hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
                hallucination_token_offsets = manual_offset_mapping(
                    anno_data[idx]['response'], hallucination_text2ids, tokenizer)
            else:
                tokenized_hallucination = tokenizer(
                    anno_data[idx]['response'], return_offsets_mapping=True)
                hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
                hallucination_token_offsets = tokenized_hallucination['offset_mapping'][1:]

            hallucination_attn_ids = attn_data[idx]['model_completion_ids'].tolist()
            if hallucination_attn_ids[-1] == 2:
                hallucination_attn_ids = hallucination_attn_ids[:-1]

            if not hallucination_text2ids == hallucination_attn_ids:
                best_substring, min_edit_dist = min_edit_distance_substring(
                    hallucination_text2ids, hallucination_attn_ids) if len(
                    hallucination_text2ids) < len(hallucination_attn_ids) else min_edit_distance_substring(
                    hallucination_attn_ids, hallucination_text2ids)
                if min_edit_dist >= 5:
                    skipped_examples += 1
                    continue

            hallucinated_spans = anno_data[idx]['problematic_spans']
            hallucinated_spans_token_offsets = []
            for span_text in hallucinated_spans:
                if not span_text in anno_data[idx]['response']:
                    if len(span_text) > len(anno_data[idx]['response']):
                        span_text = anno_data[idx]['response']
                    else:
                        best_substring, _ = min_edit_distance_substring(
                            span_text, anno_data[idx]['response'])
                        span_text = best_substring

                span_start_char_pos = anno_data[idx]['response'].index(span_text)
                span_end_char_pos = span_start_char_pos + len(span_text)

                span_start_token_pos = -1
                span_end_token_pos = -1
                for i, (start_char_pos, end_char_pos) in enumerate(hallucination_token_offsets):
                    if end_char_pos >= span_start_char_pos and span_start_token_pos == -1:
                        span_start_token_pos = i
                    if end_char_pos >= span_end_char_pos and span_end_token_pos == -1:
                        span_end_token_pos = i
                        break

                assert span_start_token_pos != -1 and span_end_token_pos != -1
                hallucinated_spans_token_offsets.append((span_start_token_pos, span_end_token_pos))

            if len(hallucinated_spans_token_offsets) == 0:
                skipped_examples += 1
                continue

            sequential_labels = [1] * num_new_tokens
            for (s, e) in hallucinated_spans_token_offsets:
                e_clipped = min(e, num_new_tokens - 1)
                if s < num_new_tokens:
                    sequential_labels[s:e_clipped + 1] = [0] * (e_clipped - s + 1)

            attn_tensor.append(attn_data[idx][attn_feature_key])
            new_tokens_tensors.append(attn_data[idx][new_tokens_key])
            context_tensors.append(attn_data[idx][context_key])
            labels.append(sequential_labels)
            splits.append([this_split] * num_new_tokens)
        else:
            attn_tensor.append(attn_data[idx][attn_feature_key])
            new_tokens_tensors.append(attn_data[idx][new_tokens_key])
            context_tensors.append(attn_data[idx][context_key])
            labels.append([1] * num_new_tokens)
            splits.append([this_split] * num_new_tokens)

    return attn_tensor, new_tokens_tensors, context_tensors, labels, splits


def convert_to_token_level(attn_tensor, new_tokens_tensors, context_tensors,
                           labels, splits, sliding_window=1, min_pool_target=True):
    """Flatten per-example tensors into per-token (or per-window) classification examples."""
    out_attn, out_new_tokens, out_context = [], [], []
    out_labels, out_splits = [], []

    for i in range(len(attn_tensor)):
        num_layers, num_heads, num_new_tokens = attn_tensor[i].shape
        if sliding_window == 1:
            for j in range(num_new_tokens):
                out_attn.append(attn_tensor[i][:, :, j].unsqueeze(-1))
                out_new_tokens.append(new_tokens_tensors[i][:, :, j].unsqueeze(-1))
                out_context.append(context_tensors[i][:, :, j].unsqueeze(-1))
                out_labels.append(labels[i][j])
                out_splits.append(splits[i][j])
        else:
            for j in range(sliding_window - 1, num_new_tokens):
                lo, hi = j - sliding_window + 1, j + 1
                out_attn.append(attn_tensor[i][:, :, lo:hi])
                out_new_tokens.append(new_tokens_tensors[i][:, :, lo:hi])
                out_context.append(context_tensors[i][:, :, lo:hi])
                if min_pool_target:
                    out_labels.append(min(labels[i][lo:hi]))
                else:
                    out_labels.append(labels[i][j])
                out_splits.append(splits[i][j])

    return out_attn, out_new_tokens, out_context, out_labels, out_splits


def extract_time_series_features(attn_tensor):
    """Extract features from time series."""
    features = []
    num_examples = len(attn_tensor)

    for i in tqdm(range(num_examples)):
        example = attn_tensor[i].clone()
        try:
            example = example.view(-1, example.shape[2])
        except Exception:
            continue
        example = example.transpose(0, 1)
        feature_vector = example.mean(dim=0).numpy()
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(feature_vector)

    return np.array(features)


def calculate_prf_acc_with_threshold(labels, pred_proba, threshold, mode='macro'):
    preds = (pred_proba >= threshold).astype(int)
    precision = precision_score(labels, preds, pos_label=0, average=mode)
    recall = recall_score(labels, preds, pos_label=0, average=mode)
    f1 = f1_score(labels, preds, pos_label=0, average=mode)
    accuracy = accuracy_score(labels, preds)
    return precision, recall, f1, accuracy


def find_best_threshold_on_validation(y_val, y_val_proba, mode='macro', search_step=0.1):
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0, 1.01, search_step):
        preds_threshold = (y_val_proba >= threshold).astype(int)
        f1 = f1_score(y_val, preds_threshold, pos_label=0, average=mode)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def _load_classifier(classifier_path):
    with open(classifier_path, 'rb') as f:
        obj = pickle.load(f)
    return obj['clf'] if isinstance(obj, dict) and 'clf' in obj else obj


def evaluate_loaded_classifier(anno_file, attn_file, classifier_path,
                               tokenizer_name=None, auth_token=None, ifft_mode=None,
                               sliding_window=1, eval_split='test', threshold=None):
    """Load a trained classifier (.pkl) and evaluate it on the requested split."""
    print(f"======== Loading data from {anno_file} and {attn_file}...")
    attn_tensor, new_tokens_tensors, context_tensors, labels, splits = load_files(
        anno_file, attn_file, tokenizer_name=tokenizer_name, auth_token=auth_token,
        ifft_mode=ifft_mode)
    attn_tensor, new_tokens_tensors, context_tensors, labels, splits = convert_to_token_level(
        attn_tensor, new_tokens_tensors, context_tensors, labels, splits,
        sliding_window=sliding_window)
    print(f"Loaded (token-level, sliding_window={sliding_window}): {len(labels)} examples")

    new_features = extract_time_series_features(new_tokens_tensors)
    context_features = extract_time_series_features(context_tensors)
    X = np.concatenate([context_features, new_features], axis=1)
    y = np.asarray(labels)

    keep = [i for i, s in enumerate(splits) if eval_split in s]
    if not keep:
        raise SystemExit(
            f"No examples matched split '{eval_split}'. Available splits: {sorted(set(splits))}")
    X_eval = X[keep]
    y_eval = y[keep]
    print(f"Eval ({eval_split}): {len(y_eval)} examples — class counts: "
          f"0(hallu)={int((y_eval == 0).sum())}, 1(ok)={int((y_eval == 1).sum())}")

    print(f"======== Loading classifier {classifier_path} ...")
    clf = _load_classifier(classifier_path)
    print(f"Classifier: {type(clf).__name__}, n_features_in_={getattr(clf, 'n_features_in_', '?')}, "
          f"classes_={getattr(clf, 'classes_', '?')}")

    if hasattr(clf, 'n_features_in_') and clf.n_features_in_ != X_eval.shape[1]:
        raise SystemExit(
            f"Feature dim mismatch: classifier expects {clf.n_features_in_}, got {X_eval.shape[1]}. "
            "Most likely extraction params (ifft_mode) don't match training.")

    proba_pos = clf.predict_proba(X_eval)[:, 1]
    auroc = roc_auc_score(y_eval, proba_pos)

    if threshold is None:
        threshold = find_best_threshold_on_validation(y_eval, proba_pos, mode='macro', search_step=0.05)
        print(f"(picked F1-best threshold on eval split: {threshold:.2f})")
    precision, recall, f1, accuracy = calculate_prf_acc_with_threshold(
        y_eval, proba_pos, threshold, mode='macro')

    print("\n======== Test Results ========")
    print(f"AUROC: {auroc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    return {'auroc': auroc, 'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'threshold': threshold}


def main(anno_file_1, attn_file_1, anno_file_2, attn_file_2,
         tokenizer_name=None, output_path=None, auth_token=None, ifft_mode=None, f_cutoff=None,
         sliding_window=1, classifier_path=None, eval_split='test', threshold=None):
    if classifier_path is not None:
        return evaluate_loaded_classifier(
            anno_file_1, attn_file_1, classifier_path,
            tokenizer_name=tokenizer_name, auth_token=auth_token, ifft_mode=ifft_mode,
            sliding_window=sliding_window, eval_split=eval_split, threshold=threshold)

    print(f"======== Loading data from {anno_file_1} and {attn_file_1}...")
    attn_tensor, new_tokens_tensors, context_tensors, labels, splits = load_files(
        anno_file_1, attn_file_1, tokenizer_name=tokenizer_name, auth_token=auth_token,
        ifft_mode=ifft_mode)
    attn_tensor, new_tokens_tensors, context_tensors, labels, splits = convert_to_token_level(
        attn_tensor, new_tokens_tensors, context_tensors, labels, splits,
        sliding_window=sliding_window)
    print(f"Loaded (token-level, sliding_window={sliding_window}): {len(labels)} examples")

    new_features = extract_time_series_features(new_tokens_tensors)
    context_features = extract_time_series_features(context_tensors)
    # Match original Lookback-Lens concat order: [context, new_tokens]
    time_series_features = np.concatenate([context_features, new_features], axis=1)

    labels_arr = np.array(labels)

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(time_series_features)):
        if splits[i] == 'train' or "train" in splits[i]:
            X_train.append(time_series_features[i])
            y_train.append(labels_arr[i])
        elif splits[i] == 'test' or "test" in splits[i]:
            X_test.append(time_series_features[i])
            y_test.append(labels_arr[i])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42,
        stratify=y_train if len(set(y_train)) > 1 else None
    )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_tr, y_tr)

    y_val_proba = classifier.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold_on_validation(y_val, y_val_proba, mode='macro')

    y_test_proba = classifier.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_test_proba)
    precision, recall, f1, accuracy = calculate_prf_acc_with_threshold(
        y_test, y_test_proba, best_threshold, mode='macro')

    print("\n======== Test Results ========")
    print(f"AUROC: {auroc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    if anno_file_2 is None or attn_file_2 is None or anno_file_2 == anno_file_1:
        print("\n======== Skipped Transfer Evaluation ========")
        return

    print(f"\n======== Transfer to {anno_file_2} and {attn_file_2}...")
    transfer_attn, transfer_new_tokens, transfer_context, transfer_labels, transfer_splits = load_files(
        anno_file_2, attn_file_2, tokenizer_name=tokenizer_name, auth_token=auth_token,
        ifft_mode=ifft_mode)
    transfer_attn, transfer_new_tokens, transfer_context, transfer_labels, transfer_splits = convert_to_token_level(
        transfer_attn, transfer_new_tokens, transfer_context, transfer_labels, transfer_splits,
        sliding_window=sliding_window)

    transfer_new_features = extract_time_series_features(transfer_new_tokens)
    transfer_context_features = extract_time_series_features(transfer_context)
    transfer_time_series_features = np.concatenate(
        [transfer_context_features, transfer_new_features], axis=1)

    transfer_labels_arr = np.array(transfer_labels)
    transfer_X_test, transfer_y_test = [], []
    for i in range(len(transfer_time_series_features)):
        if 'test' in transfer_splits[i]:
            transfer_X_test.append(transfer_time_series_features[i])
            transfer_y_test.append(transfer_labels_arr[i])

    transfer_X_test = np.array(transfer_X_test)
    transfer_y_test = np.array(transfer_y_test)

    y_pred_proba_transfer = classifier.predict_proba(transfer_X_test)[:, 1]
    transfer_auroc = roc_auc_score(transfer_y_test, y_pred_proba_transfer)
    transfer_precision, transfer_recall, transfer_f1, transfer_accuracy = calculate_prf_acc_with_threshold(
        transfer_y_test, y_pred_proba_transfer, best_threshold, mode='macro')

    print("\n======== Transfer Results ========")
    print(f"AUROC: {transfer_auroc:.4f}")
    print(f"Precision: {transfer_precision:.4f}, Recall: {transfer_recall:.4f}, F1: {transfer_f1:.4f}, Accuracy: {transfer_accuracy:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train classifier using Laplacian features.")

    parser.add_argument('--anno_1', type=str, default='summary')
    parser.add_argument('--attn_1', type=str, default='outputs/attn-features-summary-7b.pt')
    parser.add_argument('--anno_2', type=str, default=None)
    parser.add_argument('--attn_2', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--auth_token', type=str, default=None)
    parser.add_argument('--ifft_mode', type=str, default='new+context')
    parser.add_argument('--f_cutoff', type=float, default=None)
    parser.add_argument('--sliding_window', type=int, default=1)
    parser.add_argument('--classifier', type=str, default=None,
                        help='Path to a trained classifier .pkl. If set, skips training and '
                             'evaluates the loaded classifier on --eval_split of anno_1/attn_1.')
    parser.add_argument('--eval_split', type=str, default='test',
                        help="Split substring to evaluate when --classifier is set (default: 'test').")
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold for class 1 when --classifier is set. '
                             'Omit to pick the F1-best threshold on the eval split.')

    args = parser.parse_args()

    main(
        args.anno_1, args.attn_1, args.anno_2, args.attn_2,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path,
        auth_token=args.auth_token,
        ifft_mode=args.ifft_mode,
        f_cutoff=args.f_cutoff,
        sliding_window=args.sliding_window,
        classifier_path=args.classifier,
        eval_split=args.eval_split,
        threshold=args.threshold,
    )
