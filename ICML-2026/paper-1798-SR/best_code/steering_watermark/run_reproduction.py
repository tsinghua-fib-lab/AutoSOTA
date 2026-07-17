#!/usr/bin/env python3
"""Reproduction script for paper 1798: LLM Self-Recognition on ELI5 with Llama-3.2-1B."""

import os
import sys
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Mock clearml ──────────────────────────────────────────────────────
class MockLogger:
    def report_single_value(self, *args, **kwargs): pass
    def report_scalar(self, *args, **kwargs): pass
    def report_plotly(self, *args, **kwargs): pass
    def report_media(self, *args, **kwargs): pass
    def report_scatter2d(self, *args, **kwargs): pass
    def report_table(self, *args, **kwargs): pass
    def upload_artifact(self, *args, **kwargs): pass

class MockTask:
    _instance = None
    @staticmethod
    def init(*args, **kwargs):
        if MockTask._instance is None:
            MockTask._instance = MockTask()
        return MockTask._instance
    def set_parameters(self, *args, **kwargs): pass
    def get_logger(self): return MockLogger()
    def close(self): pass

# Inject before importing pipeline modules
import clearml
clearml.Task = MockTask

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_wrapper import LLMWrapper, ActivationHook, SteeringHook
from data_processing import (
    cached_function2, split_data_accoring_to_sentence_id2,
    hash_params, params_to_vanilla,
)
from text_generation import generate_noise, get_input_text_dataset_array
from activation_gathering import gather_data as orig_gather_data
from quality_evaluation import evaluate_quality as orig_evaluate_quality
from detection import detect_watermark as orig_detect_watermark
from evaluation import evaluate_detection as orig_evaluate_detection
from ml_model import SimpleMLP, SimpleLDA
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models/Llama-3.2-1B-Instruct"
DATASET_PATH = "/datasets/hc3/all.jsonl"
OUTPUT_DIR = Path("/repo/steering_watermark/reproduction_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUBRIC = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "dataset": "ELI5",
    "n_prompts": 1000,
    "max_sequence_length": 512,
    "alpha": 5,
    "sparsity": 0.003,         # 99.7% sparse
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "steering_layers": [15],
    "split": [0.7, 0.1, 0.2],
    "random_seed": 42,
}

# ── Build params ───────────────────────────────────────────────────────
def build_params():
    return {
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "verbose": True,
        "run_name": "reproduction_eli5",
        "model_arguments": {
            "model_id": MODEL_PATH,
            "load_in_8bit": False,
            "torch_dtype": "torch.bfloat16",
        },
        "generation_arguments": {
            "do_sample": True,
            "temperature": RUBRIC["temperature"],
            "max_new_tokens": RUBRIC["max_sequence_length"],
            "top_p": RUBRIC["top_p"],
            "top_k": 50,
            "repetition_penalty": RUBRIC["repetition_penalty"],
            "generation_batch_size": 32,
        },
        "data_arguments": {
            "input_version": str(DATASET_PATH),  # local jsonl
            "truncate_input_words": None,
            "max_loaded_samples": RUBRIC["n_prompts"],
        },
        "steering_arguments": {
            "noise_seed": RUBRIC["random_seed"],
            "noise_offset": 0,
            "steering_layers": RUBRIC["steering_layers"],
            "noise_max": RUBRIC["alpha"],
            "noise_type": f"sparse_{RUBRIC['sparsity']}",
        },
        "robustness_arguments": {
            "paraphrasing": {"enabled": False},
        },
        "gathering_arguments": {
            "gathering_truncation": None,
            "gathering_layers": RUBRIC["steering_layers"],
            "max_token_seq": RUBRIC["max_sequence_length"],
            "remove_prompt": False,
        },
        "comparison_arguments": {
            "compared_text_type": "human",
        },
        "detection_arguments": {
            "seed": RUBRIC["random_seed"],
            "number_of_bits": 3,
            "number_prompts_truncation": None,
            "max_seq_length": RUBRIC["max_sequence_length"],
            "model_type": "mlp",
            "token_aggregation": False,
            "sentence_array": False,
            "lda_parameters": {
                "shrinkage": None,
                "solver": "lsqr",
            },
            "mlp_parameters": {
                "hidden_dims": [2048, 64, 64, 32],
                "batch_size": 512,
                "num_epochs": 1,
                "learning_rate": 0.001,
            },
        },
        "quality_evaluation_arguments": {
            "log_diversity_calculation": {"n_gram": 4},
            "perplexity_calculation": {
                "model_id": "Qwen/Qwen3-4B",
                "batch_size": 4,
            },
            "quality_classifier_calculation": {
                "model_id": "nvidia/quality-classifier-deberta",
            },
        },
        "evaluation_arguments": {},
    }


# ── Load dataset from local JSONL ──────────────────────────────────────
def load_local_dataset(params):
    """Load HC3/ELI5 dataset from local JSONL file."""
    print(">>> Loading ELI5 dataset from local file...")
    data_path = params["data_arguments"]["input_version"]
    dataset = load_dataset("json", data_files=data_path, split="train")

    max_samples = params["data_arguments"].get("max_loaded_samples", 1000)

    input_questions = []
    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        question = sample.get("question", "").strip()
        if not question:
            continue
        # Skip edited/URL-containing questions
        filtered = ["edit", "url"]
        if any(elem in question.lower() for elem in filtered):
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Write only in plain text, without formatting using * or #."},
            {"role": "user", "content": question}
        ]
        input_questions.append(messages)

        if len(input_questions) >= max_samples:
            break

    print(f"Loaded {len(input_questions)} questions")
    return input_questions


def load_human_texts_from_local(params, llm):
    """Load human-written answers from local dataset for comparison."""
    print(">>> Loading human-written texts from local dataset...")
    data_path = params["data_arguments"]["input_version"]
    dataset = load_dataset("json", data_files=data_path, split="train")

    max_samples = params["data_arguments"]["max_loaded_samples"]
    max_input_tokens = params["data_arguments"].get("max_input_tokens", 512)

    human_texts = []
    classification_label = 0  # Human texts are label 0

    count = 0
    for i in range(len(dataset)):
        if count >= max_samples:
            break
        sample = dataset[i]
        question = sample.get("question", "").strip()
        human_answers = sample.get("human_answers", [])
        if not question or not human_answers:
            continue

        filtered = ["edit", "url"]
        if any(elem in question.lower() for elem in filtered):
            continue

        # Use first human answer
        answer_text = human_answers[0]

        # Truncate to max_input_tokens
        text_token_ids = llm.tokenizer([answer_text], return_tensors="pt")["input_ids"][0]
        truncated_token_ids = text_token_ids[:max_input_tokens]
        truncated_text = llm.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

        human_texts.append({
            "classification_label": classification_label,
            "input_text": "",
            "input_text_id": count,
            "input_token_length": 0,
            "input_token_ids": [],
            "output_text": truncated_text,
            "output_token_ids": truncated_token_ids,
            "output_token_strings": llm.tokenizer.decode(truncated_token_ids),
            "steering_noise": 0,
            "steering_type": "human",
            "steering_layers": params["steering_arguments"]["steering_layers"],
            "key_vector": np.zeros(llm.embedding_dim, dtype=np.float32),
        })
        count += 1

    data = pd.DataFrame(human_texts)
    data["params"] = [params] * len(data)
    print(f"Loaded {len(data)} human texts")
    return data


# ── Generate steered text ──────────────────────────────────────────────
def generate_steered_text(params):
    print(">>> Generating steered text...")
    llm = LLMWrapper(hf_token=params["hf_token"], **params["model_arguments"])

    text_dataset = load_local_dataset(params)

    key_vector = generate_noise(llm.embedding_dim, params).to(llm.device)
    steering_hooks = llm.register_hooks("steering", params["steering_arguments"]["steering_layers"], key_vector)

    formated_gen_args = params["generation_arguments"].copy()
    formated_gen_args.pop("generation_batch_size")
    batch_size = params["generation_arguments"]["generation_batch_size"]

    classification_label = params["steering_arguments"]["noise_seed"]

    generated_text = []
    output_dict = llm(text_dataset, rich_output=True, batch_size=batch_size, **formated_gen_args)

    for i, output_dict_elmt in enumerate(output_dict):
        generated_text.append({
            "classification_label": classification_label,
            "input_text": output_dict_elmt["input_text"],
            "input_text_id": i,
            "input_token_length": output_dict_elmt["input_lengths"],
            "input_token_ids": output_dict_elmt.get("encoded_inputs", []),
            "output_text": output_dict_elmt["generated_texts"],
            "output_token_ids": output_dict_elmt.get("encoded_outputs", []),
            "output_token_strings": output_dict_elmt["output_token_strings"],
            "steering_noise": params["steering_arguments"]["noise_max"],
            "steering_type": params["steering_arguments"]["noise_type"],
            "steering_layers": params["steering_arguments"]["steering_layers"],
            "key_vector": key_vector.float().detach().cpu().numpy(),
        })

    for hook in steering_hooks:
        hook.remove()

    data = pd.DataFrame(generated_text)
    data["params"] = [params] * len(data)
    print(f"Generated {len(data)} steered texts")

    return data, llm


# ── Gather activations ─────────────────────────────────────────────────
def gather_activations(df, params, llm):
    print(">>> Gathering activations...")
    saving_hooks = llm.register_hooks("gather", params["gathering_arguments"]["gathering_layers"])

    gathering_kwargs = params["generation_arguments"].copy()
    gathering_kwargs.pop("generation_batch_size")
    gathering_kwargs["max_new_tokens"] = 1

    if params["gathering_arguments"].get("remove_prompt", True):
        gathering_text_list = df["output_text"].tolist()
    else:
        gathering_text_list = (df["input_text"].fillna("") + " " + df["output_text"]).tolist()

    max_token_seq = params["gathering_arguments"].get("max_token_seq", None)

    expended_data = []
    for i, generated_text in enumerate(tqdm(gathering_text_list, desc="Gathering")):
        _ = llm.gathering_forward([generated_text], **gathering_kwargs)

        activations = {}
        for hook in saving_hooks:
            for layer in params["gathering_arguments"]["gathering_layers"]:
                if hook.layer_name == layer:
                    activations[layer] = hook.activations

        if max_token_seq is not None:
            for layer in activations:
                trimmed_size = min(max_token_seq, activations[layer].shape[0])
                activations[layer] = activations[layer][:trimmed_size]

        expended_data.append({"activations": activations})

    for hook in saving_hooks:
        hook.remove()

    df = pd.concat([df, pd.DataFrame(expended_data)], axis=1)
    df["params"] = [params] * len(df)

    return df


# ── Add dummy quality columns ──────────────────────────────────────────
def add_dummy_quality(df, params):
    print(">>> Adding placeholder quality metrics...")
    n = len(df)
    df["perplexity"] = [1.0] * n
    df["log_diversity"] = [0.0] * n
    df["quality"] = [[0.5]] * n  # List of 1 element to match expected format
    return df


# ── Detection ──────────────────────────────────────────────────────────
def run_detection(df_all, params):
    print(">>> Running detection...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = params["detection_arguments"].get("seed", 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Format labels to be consecutive
    unique_labels = sorted(df_all["classification_label"].unique())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    df_all["classification_label"] = df_all["classification_label"].map(label_map)

    df_train, df_val, df_test, split_list = split_data_accoring_to_sentence_id2(
        df_all,
        val_size=0.1,
        test_size=0.2,
        seed=0,
        token_aggregation=params["detection_arguments"]["token_aggregation"],
        sentence_array=params["detection_arguments"]["sentence_array"],
        max_token_seq=params["detection_arguments"].get("max_seq_length", None),
        split_labels=None,
    )

    X_train = df_train["fwd_data"].values
    Y_train = df_train["classification_label"].values.astype(np.int64)
    X_val = df_val["fwd_data"].values
    Y_val = df_val["classification_label"].values.astype(np.int64)
    X_test = df_test["fwd_data"].values
    Y_test = df_test["classification_label"].values.astype(np.int64)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Y_train counts: {np.bincount(Y_train)}")
    print(f"Y_val counts: {np.bincount(Y_val)}")
    print(f"Y_test counts: {np.bincount(Y_test)}")

    # Stack tensors
    X_train = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_train]).to(device)
    X_val = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_val]).to(device)
    X_test = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(device) for x in X_test]).to(device)

    output_dim = len(np.unique(Y_train))
    token_dim = X_train[0].shape[0]
    print(f"Token dim: {token_dim}, Output dim: {output_dim}")

    model_params = params["detection_arguments"]["mlp_parameters"]
    model = SimpleMLP(
        input_dim=token_dim,
        hidden_dims=model_params["hidden_dims"],
        output_dim=output_dim,
        device=device,
    ).to(device)

    print("Training MLP classifier...")
    loss_memory, train_accuracy, validation_accuracy, validation_loss = model.fit(
        train_data=X_train,
        train_labels=Y_train,
        val_data=X_val,
        val_labels=Y_val,
        epochs=model_params.get("num_epochs", 1),
        batch_size=model_params.get("batch_size", 512),
        learning_rate=model_params.get("learning_rate", 0.001),
        verbose=True,
    )

    test_accuracy, test_predictions, test_probabilities = model.evaluate(
        X_test, Y_test, batch_size=model_params.get("batch_size", 512)
    )

    result_dict = {
        "train_loss": loss_memory,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "test_accuracy": test_accuracy,
        "test_ground_truth": Y_test,
        "test_sentence_ids": df_test["input_text_id"].values,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "test_token_ids": df_test["token_id"].values,
        "split_list": split_list,
        "model": model,
    }

    return result_dict, df_all


# ── Evaluation ─────────────────────────────────────────────────────────
def run_evaluation(df_all, params, detection_dict):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    test_predictions = np.array(detection_dict["test_predictions"])
    test_ground_truth = np.array(detection_dict["test_ground_truth"])
    test_token_ids = np.array(detection_dict["test_token_ids"])
    test_sentence_ids = np.array(detection_dict["test_sentence_ids"])
    test_probabilities = np.array(detection_dict["test_probabilities"])

    # Token-level metrics
    token_acc = accuracy_score(test_ground_truth, test_predictions)
    token_f1 = f1_score(test_ground_truth, test_predictions, average='binary')
    token_precision = precision_score(test_ground_truth, test_predictions, average='binary')
    token_recall = recall_score(test_ground_truth, test_predictions, average='binary')

    print(f"\nToken-level Results:")
    print(f"  Accuracy:  {token_acc:.4f}")
    print(f"  F1 Score:  {token_f1:.4f}")
    print(f"  Precision: {token_precision:.4f}")
    print(f"  Recall:    {token_recall:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(test_ground_truth, test_predictions)}")

    # Text-level metrics via majority voting
    unique_sentence_ids = np.unique(test_sentence_ids)
    sentence_preds = []
    sentence_labels = []

    for sid in unique_sentence_ids:
        mask = test_sentence_ids == sid
        s_preds = test_predictions[mask]
        s_labels = test_ground_truth[mask]
        if len(s_preds) == 0:
            continue
        # Majority vote
        majority = np.bincount(s_preds).argmax()
        sentence_preds.append(majority)
        sentence_labels.append(s_labels[0])  # All tokens in a sentence have the same label

    sentence_preds = np.array(sentence_preds)
    sentence_labels = np.array(sentence_labels)

    text_acc = accuracy_score(sentence_labels, sentence_preds)
    text_f1 = f1_score(sentence_labels, sentence_preds, average='binary')
    text_precision = precision_score(sentence_labels, sentence_preds, average='binary')
    text_recall = recall_score(sentence_labels, sentence_preds, average='binary')

    print(f"\nText-level Results (majority voting):")
    print(f"  Accuracy:  {text_acc:.4f}")
    print(f"  F1 Score:  {text_f1:.4f}")
    print(f"  Precision: {text_precision:.4f}")
    print(f"  Recall:    {text_recall:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(sentence_labels, sentence_preds)}")

    results = {
        "token_level": {
            "accuracy": float(token_acc),
            "f1": float(token_f1),
            "precision": float(token_precision),
            "recall": float(token_recall),
        },
        "text_level": {
            "accuracy": float(text_acc),
            "f1": float(text_f1),
            "precision": float(text_precision),
            "recall": float(text_recall),
        },
        "config": RUBRIC,
    }

    # Save results
    result_path = OUTPUT_DIR / "reproduction_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {result_path}")

    return results


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("="*80)
    print("PAPER 1798 REPRODUCTION: LLM Self-Recognition on ELI5")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Config: {json.dumps(RUBRIC, indent=2)}")
    print("="*80)

    params = build_params()

    # Step 1: Generate steered text
    print("\n" + "="*40)
    print("STEP 1: Generate steered text")
    print("="*40)
    df_steered, llm = generate_steered_text(params)

    # Step 2: Add dummy quality metrics
    df_steered = add_dummy_quality(df_steered, params)

    # Step 3: Gather activations for steered text
    print("\n" + "="*40)
    print("STEP 2: Gather activations (steered)")
    print("="*40)
    df_steered = gather_activations(df_steered, params, llm)

    # Step 4: Load human texts and gather activations
    print("\n" + "="*40)
    print("STEP 3: Load human texts and gather activations")
    print("="*40)
    human_params = copy.deepcopy(params)
    human_params["steering_arguments"]["noise_max"] = 0.0
    human_params["steering_arguments"]["noise_type"] = "human"
    human_params["steering_arguments"]["steering_layers"] = []

    df_human = load_human_texts_from_local(params, llm)
    df_human = add_dummy_quality(df_human, human_params)
    df_human = gather_activations(df_human, human_params, llm)

    # Step 5: Combine and run detection
    print("\n" + "="*40)
    print("STEP 4: Detection")
    print("="*40)
    df_all = pd.concat([df_steered, df_human], ignore_index=True)
    print(f"Combined dataset: {len(df_all)} rows")
    print(f"Steered texts: {len(df_steered)}, Human texts: {len(df_human)}")

    detection_dict, df_all = run_detection(df_all, params)

    # Step 6: Evaluate
    print("\n" + "="*40)
    print("STEP 5: Evaluation")
    print("="*40)
    results = run_evaluation(df_all, params, detection_dict)

    # Clean up
    del llm
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("REPRODUCTION COMPLETE")
    print("="*80)
    return results


if __name__ == "__main__":
    main()
