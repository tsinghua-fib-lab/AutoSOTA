import argparse
from datasets import load_from_disk
from jaxcld import ASRModel, CVXNNLangDetectHead, NNLangDetectHead, SklearnLangDetectHead, SVMLangDetectHead
from sklearn.metrics import classification_report
import evaluate
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on a dataset for language detection and transcription.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory (e.g., 'data/en_hi/').")
    parser.add_argument("--model_name", type=str, default="models/whisper-small-enhi-out", help="Name of huggingface model.")
    parser.add_argument("--model_path", type=str, default="models/whisper-small-enhi-out", help="Path to the model directory.")
    parser.add_argument("--cld_path", type=str, default="models/en_hi_nn", help="Path to the language classifier model directory.")
    parser.add_argument(
        "--cld_type",
        type=str,
        default="nn",
        choices=["nn", "cvx", "linear_svm", "kernel_svm", "knn", "vanilla"],
        help="Detection head architecture.",
    )
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of languages to evaluate on")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()

def main():
    args = parse_args()

    wb = None
    if not args.no_wandb:
        wb = wandb
        wb.init()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    ds = load_from_disk(args.dataset_path)

    asr_model = ASRModel.from_pretrained(args.model_name, config={"languages": args.languages.split(",")})
    lang_detect_head = None
    if args.cld_type == 'nn':
        lang_detect_head = NNLangDetectHead.load(args.cld_path, asr_model)
    elif args.cld_type == 'cvx':
        lang_detect_head = CVXNNLangDetectHead.load(args.cld_path, asr_model)
    elif args.cld_type == 'linear_svm':
        lang_detect_head = SVMLangDetectHead.load(args.cld_path, asr_model)
    elif args.cld_type in {'kernel_svm', 'knn'}:
        lang_detect_head = SklearnLangDetectHead.load(args.cld_path, asr_model)
    asr_model.set_lang_detect_head(lang_detect_head)

    # Use test split consistently
    test_ds = ds["test"]
    true_language_ids = [x["lang"] for x in test_ds]
    true_texts = [x["text"] for x in test_ds]

    pred_language_ids = []
    pred_texts = []
    
    batch_temp = []
    for i, sample in enumerate(test_ds):
        batch_temp.append(sample["audio"]["array"])
        if len(batch_temp) == args.batch_size or i == len(test_ds) - 1:
            language_ids_batch, texts_batch = asr_model.predict(batch_temp)
            pred_language_ids.extend(language_ids_batch)
            pred_texts.extend(texts_batch)
            batch_temp = []

    wer = 100.0 * wer_metric.compute(predictions=pred_texts, references=true_texts) 
    cer = 100.0 * cer_metric.compute(predictions=pred_texts, references=true_texts)

    # Get classification report as dict
    report = classification_report(true_language_ids, pred_language_ids, output_dict=True)

    # Log WER and CER
    if wb is not None:
        wb.log({
            "eval/wer": wer,
            "eval/cer": cer,
        })

    # Log overall metrics
    if wb is not None:
        wb.log({
            "eval/accuracy": report['accuracy'],
            "macro_precision": report['macro avg']['precision'],
            "macro_recall": report['macro avg']['recall'],
            "macro_f1": report['macro avg']['f1-score'],
            "macro_support": report['macro avg']['support'],
            "weighted_precision": report['weighted avg']['precision'],
            "weighted_recall": report['weighted avg']['recall'],
            "weighted_f1": report['weighted avg']['f1-score'],
            "weighted_support": report['weighted avg']['support'],
        })

    # Log per-class metrics
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    if wb is not None:
        for lang in langs:
            if lang in report:
                wb.log({
                    f"{lang}_precision": report[lang]['precision'],
                    f"{lang}_recall": report[lang]['recall'],
                    f"{lang}_f1": report[lang]['f1-score'],
                    f"{lang}_support": report[lang]['support'],
                })

    # Create a wandb.Table for the classification report
    data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
        elif label == 'accuracy':
            data.append([label, None, None, metrics, None])  # Accuracy is a single value

    columns = ["label", "precision", "recall", "f1-score", "support"]
    table = None if wb is None else wb.Table(data=data, columns=columns)

    # Log the table as an artifact
    if wb is not None and table is not None:
        artifact = wb.Artifact("classification_report", type="report")
        artifact.add(table, "classification_report_table")
        wb.log_artifact(artifact)

    # Per-sample predictions artifact: true language, predicted language, accent code
    per_sample_columns = ["true_lang", "pred_lang", "accent"]
    if 'accent' in test_ds.features:
        accents = [x.get("accent", None) for x in test_ds]
    else:
        accents = [None] * len(test_ds)

    per_sample_data = [
        [t_lang, p_lang, acc]
        for t_lang, p_lang, acc in zip(true_language_ids, pred_language_ids, accents)
    ]
    if wb is not None:
        per_sample_table = wb.Table(data=per_sample_data, columns=per_sample_columns)
        preds_artifact = wb.Artifact("per_sample_predictions", type="predictions")
        preds_artifact.add(per_sample_table, "per_sample_predictions_table")
        wb.log_artifact(preds_artifact)

    # Per-accent analysis
    if 'accent' in test_ds.features:
        true_accents = [x["lang"]+"-"+x["accent"] for x in test_ds]
        unique_accents = set(true_accents)
        
        per_accent_data = []
        
        for accent in sorted(unique_accents):
            mask = [a == accent for a in true_accents]
            
            filtered_true_lang = [t for t, m in zip(true_language_ids, mask) if m]
            filtered_pred_lang = [p for p, m in zip(pred_language_ids, mask) if m]
            filtered_true_text = [t for t, m in zip(true_texts, mask) if m]
            filtered_pred_text = [p for p, m in zip(pred_texts, mask) if m]
            
            if len(filtered_true_lang) == 0:
                continue
            
            acc_wer = 100.0 * wer_metric.compute(predictions=filtered_pred_text, references=filtered_true_text)
            acc_cer = 100.0 * cer_metric.compute(predictions=filtered_pred_text, references=filtered_true_text)
            acc_report = classification_report(filtered_true_lang, filtered_pred_lang, output_dict=True)
            
            if wb is not None:
                wb.log({
                    f"eval/accent/{accent}/wer": acc_wer,
                    f"eval/accent/{accent}/cer": acc_cer,
                    f"eval/accent/{accent}/accuracy": acc_report['accuracy'],
                    f"eval/accent/{accent}/macro_precision": acc_report['macro avg']['precision'],
                    f"eval/accent/{accent}/macro_recall": acc_report['macro avg']['recall'],
                    f"eval/accent/{accent}/macro_f1": acc_report['macro avg']['f1-score'],
                    f"eval/accent/{accent}/macro_support": acc_report['macro avg']['support'],
                })
            
            # Per-class per-accent (optional, but for completeness)
            for lang in langs:
                if lang in acc_report:
                    if wb is not None:
                        wb.log({
                            f"eval/accent/{accent}/{lang}_precision": acc_report[lang]['precision'],
                            f"eval/accent/{accent}/{lang}_recall": acc_report[lang]['recall'],
                            f"eval/accent/{accent}/{lang}_f1": acc_report[lang]['f1-score'],
                            f"eval/accent/{accent}/{lang}_support": acc_report[lang]['support'],
                        })
            
            # Collect for table
            per_accent_data.append([accent, acc_wer, acc_cer, acc_report['accuracy'], acc_report['macro avg']['f1-score'], acc_report['macro avg']['support']])
        
        if per_accent_data:
            per_accent_columns = ["accent", "wer", "cer", "accuracy", "macro_f1", "support"]
            if wb is not None:
                per_accent_table = wb.Table(data=per_accent_data, columns=per_accent_columns)
                wb.log({"per_accent_metrics": per_accent_table})

    # Optional: Print for local output
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print("\nLanguage Classification Report:")
    print(classification_report(true_language_ids, pred_language_ids))

    if wb is not None:
        wb.finish()

if __name__ == "__main__":
    main()