import os

import jsonlines
import nltk
import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.stats import hmean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from dataset import TextDatasetQA, TextIdkDatasetQA, SafeAlignEvalDataset, custom_data_collator, get_batch_loss


def read_jsonline(file_path):
    data = []
    with open(file_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data


def token_entropy(tokenizer, gen_texts, normalize=True):
    return {
        "token_entropy": [
            compute_token_entropy(tokenizer, txt, normalize) for txt in gen_texts
        ]
    }


def compute_token_entropy(tokenizer, sentence, normalize=True):
    # get n-gram dist
    tokens = tokenizer.tokenize(sentence)
    ngrams = nltk.ngrams(tokens, 1)
    fdist = nltk.FreqDist(ngrams)
    # get n-gram freq
    freqs = np.array([freq for _, freq in fdist.items()])
    freqs = freqs / freqs.sum()

    entropy = np.sum(-freqs * np.log(freqs) / np.log(2))

    num_ngrams = len(tokens)
    if num_ngrams <= 1:
        return 0  # If there are not enough n-grams, entropy is 0
    max_entropy = np.log2(num_ngrams)

    # Normalize entropy
    normalized_entropy = entropy / max_entropy

    return normalized_entropy if normalize else entropy


def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask = batch
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {
            "input_ids": perturb_input_ids.view(bsz * seq_len, -1),
            "labels": perturb_labels.view(bsz * seq_len, -1),
            "attention_mask": perturb_attention_mask.view(bsz * seq_len, -1),
        }

        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        perturb_loss = get_batch_loss(
            perturb_outputs.logits, perturb_batch["labels"]
        ).view(bsz, seq_len)

        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_perturb = (
            (perturb_batch["labels"] != -100).view(bsz, seq_len, -1).sum(-1)
        )

        eval_logs["average_perturb_loss"] = (
            eval_logs.get("average_perturb_loss", [])
            + (perturb_loss / num_token_perturb).tolist()
        )
        eval_logs["avg_paraphrased_loss"] = (
            eval_logs.get("avg_paraphrased_loss", [])
            + (gt_loss / num_token_gt).float().cpu().numpy().tolist()
        )

        eval_logs["paraphrased_loss"] = (
            eval_logs.get("paraphrased_loss", []) + gt_loss.tolist()
        )
        eval_logs["perturb_loss"] = (
            eval_logs.get("perturb_loss", []) + perturb_loss.tolist()
        )

        eval_logs["num_token_paraphrased"] = (
            eval_logs.get("num_token_paraphrased", []) + num_token_gt.tolist()
        )
        eval_logs["num_token_perturb"] = (
            eval_logs.get("num_token_perturb", []) + num_token_perturb.tolist()
        )

    return eval_logs


def get_dataloader(
    cfg,
    eval_task,
    tokenizer,
    folder,
    split,
    question_key,
    answer_key,
    base_answer_key,
    perturbed_answer_key,
):
    torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
        split=split,
        question_key=question_key,
        answer_key=answer_key,
    )
    base_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
        split=split,
        question_key=question_key,
        answer_key=base_answer_key,
    )
    perturb_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
        split=split,
        question_key=question_key,
        answer_key=perturbed_answer_key,
    )
    idk_torch_format_dataset = TextIdkDatasetQA(
        folder,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
        split=split,
        question_key=question_key,
        answer_key=answer_key,
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(
            range(min(cfg.ds_size, len(torch_format_dataset.data)))
        )
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(
            range(min(cfg.ds_size, len(base_torch_format_dataset.data)))
        )
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(
            range(min(cfg.ds_size, len(perturb_torch_format_dataset.data)))
        )

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset,
        batch_size=cfg.batch_size // 4,
        collate_fn=custom_data_collator,
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset,
        batch_size=cfg.batch_size // 4,
        collate_fn=custom_data_collator,
    )
    idk_dataloader = torch.utils.data.DataLoader(
        idk_torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader, idk_dataloader


def get_safe_align_dataloader(
    cfg,
    tokenizer,
    folder,
    task
):
    dataset = SafeAlignEvalDataset(
        data_path=os.path.join(folder, "evaluation", task+".json"),
        template_path=os.path.join(folder, "alpaca.json"),
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=cfg.generation.max_length,
    )

    if cfg.ds_size:
        dataset.data = dataset.data.select(
            range(min(cfg.ds_size, len(dataset.data)))
            )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator
    )

    return dataloader


def get_all_evals(
    cfg,
    model,
    tokenizer,
    folder,
    split,
    eval_task,
    eval_dataloader,
    base_eval_dataloader,
    perturb_dataloader,
    idk_dataloader,
    tofu,
):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    idk_ground_truths = []
    input_strings = []

    for batch, idk_batch in tqdm(zip(eval_dataloader, idk_dataloader), total=len(eval_dataloader)):
        input_ids, labels, attention_mask = batch
        idk_input_ids, idk_labels, idk_attention_mask = idk_batch
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        idk_batch = {
            "input_ids": idk_input_ids,
            "labels": idk_labels,
            "attention_mask": idk_attention_mask,
        }

        # send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
            idk_batch[k] = idk_batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt, idk_gt = run_generation(
                cfg, batch, idk_batch, model, tokenizer=tokenizer
            )
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            idk_ground_truths.extend(idk_gt)
            input_strings.extend(input_string)

        # GT loss
        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        num_token_gt = (batch["labels"] != -100).sum(-1)  # bs
        eval_logs["avg_gt_loss"] = (
            eval_logs.get("avg_gt_loss", [])
            + (gt_loss / num_token_gt).float().cpu().numpy().tolist()
        )
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["num_token_gt"] = (
                eval_logs.get("num_token_gt", []) + num_token_gt.tolist()
        )

        if "forget" in eval_task:
            # IDK loss
            idk_loss = get_batch_loss(outputs.logits, idk_batch["labels"])
            num_roken_idk_gt = (idk_batch["labels"] != -100).sum(-1)  # bs
            eval_logs["avg_idk_loss"] = (
                eval_logs.get("avg_idk_loss", [])
                + (idk_loss / num_roken_idk_gt).float().cpu().numpy().tolist()
            )
            eval_logs["idk_loss"] = eval_logs.get("idk_loss", []) + idk_loss.tolist()
            eval_logs["num_token_idk"] = (
                    eval_logs.get("num_token_idk", []) + num_roken_idk_gt.tolist()
            )

    rouge_cores = eval_rouge_recall(gen_outputs, ground_truths)
    eval_logs.update(rouge_cores)
    eval_logs.update(
        eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model)
    )

    if "forget" in eval_task:
        # eval IDK rouge
        idk_rouge_cores = eval_rouge_recall(gen_outputs, idk_ground_truths)
        idk_rouge_cores = {"idk_"+k: v for k, v in idk_rouge_cores.items()}
        eval_logs.update(idk_rouge_cores)

    es_pipe = pipeline(
        "text-classification",
        model="sileod/deberta-v3-base-tasksource-nli",
        # model="sileod/deberta-v3-base-tasksource-nli",
        device=torch.device("cuda"),
    )
    # eval_real_author_wo_options, eval_real_world_wo
    if "real_author" in eval_task:
        real_author_data = read_jsonline(
            os.path.join(folder, "real_authors_perturbed.json")
        )
        real_author_org_answer = [i["original_answer"] for i in real_author_data]
        eval_logs["generated_text"] = list(
            zip(input_strings, gen_outputs, ground_truths, real_author_org_answer)
        )
        eval_logs.update(eval_cosine_similarity(gen_outputs, real_author_org_answer))
        eval_logs.update(
            get_entailment_results(
                es_pipe,
                gen_outputs,
                real_author_org_answer,
                eval_task,
                rouge_cores["rougeL_recall"],
                bs=30,
                tofu=tofu,
            )
        )
    elif "real_world" in eval_task:
        real_world_data = read_jsonline(
            os.path.join(folder, "world_facts_perturbed.json")
        )
        real_world_org_answer = [i["original_answer"] for i in real_world_data]
        eval_logs["generated_text"] = list(
            zip(input_strings, gen_outputs, ground_truths, real_world_org_answer)
        )
        eval_logs.update(eval_cosine_similarity(gen_outputs, real_world_org_answer))
        eval_logs.update(
            get_entailment_results(
                es_pipe,
                gen_outputs,
                real_world_org_answer,
                eval_task,
                rouge_cores["rougeL_recall"],
                bs=30,
                tofu=tofu,
            )
        )
    else:
        # for real world, use org answer rather than golden answer for cos similarity
        org_answer = [
            i["answer"] for i in read_jsonline(os.path.join(folder, split + ".json"))
        ]
        eval_logs["generated_text"] = list(
            zip(input_strings, gen_outputs, ground_truths, org_answer)
        )

        eval_logs.update(eval_cosine_similarity(gen_outputs, org_answer))
        eval_logs.update(
            get_entailment_results(
                es_pipe,
                gen_outputs,
                ground_truths,
                eval_task,
                rouge_cores["rougeL_recall"],
                bs=30,
                tofu=tofu,
            )
        )
        if "forget" in eval_task:
            # eval idk cosine similarity and entailment results
            # idk_cs = eval_cosine_similarity(gen_outputs, idk_ground_truths)
            idk_es = get_entailment_results(
                es_pipe,
                gen_outputs,
                idk_ground_truths,
                eval_task,
                idk_rouge_cores["idk_rougeL_recall"],
                bs=30,
                tofu=tofu,
            )
            # idk_cs = {"idk_"+k: v for k, v in idk_cs.items()}
            idk_es = {"idk_"+k: v for k, v in idk_es.items()}
            # eval_logs.update(idk_cs)
            eval_logs.update(idk_es)

    # import pdb;pdb.set_trace()
    eval_logs.update(token_entropy(tokenizer, gen_outputs, normalize=True))

    return eval_logs


"""
eval_result_dict = {
    'eval_log.json': {},
    'eval_log_forget.json': {}
}
"""


def get_eval_results(eval_result_dict):
    eval_task_dict = {
        "eval_real_author_wo_options.json": "Real Authors",
        "eval_real_world_wo_options.json": "Real World",
        "eval_log.json": "Retain",
        "eval_log_forget.json": "Forget",
    }

    eval_tasks = list(eval_task_dict.keys())
    metrics = [
        "ROUGE",
        "Probability",
        "Truth Ratio",
        "Token Entropy",
        "Cosine Similarity",
        "Entailment Score",
    ]
    output_result = {}
    for eval_task in eval_tasks:
        if eval_task in eval_result_dict.keys():
            for metric in metrics:
                output_result[eval_task_dict[eval_task] + " " + metric] = []
                if eval_task_dict[eval_task] == "Forget" and metric in ["Token Entropy", "Entailment Score"]:
                    output_result[eval_task_dict[eval_task] + " IDK " + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if "eval_log" in k:
            gt_probs = np.exp(-1 * np.array(eval_result_dict[k]["avg_gt_loss"]))
            avg_gt_prob = np.mean(gt_probs)
            # if "forget" in k:
            #     idk_probs = np.exp(-1 * np.array(eval_result_dict[k]["avg_idk_loss"]))
            #     avg_idk_prob = np.mean(idk_probs)
            #     output_result[f"{eval_task_dict[k]} IDK Probability"] = avg_idk_prob
        else:
            avg_true_prob = np.exp(-1 * np.array(eval_result_dict[k]["avg_gt_loss"]))
            avg_false_prob = np.exp(
                -1 * np.array(eval_result_dict[k]["average_perturb_loss"])
            )
            avg_all_prob = np.concatenate(
                [np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1
            ).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob / avg_all_prob)
        output_result[f"{eval_task_dict[k]} Probability"] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(eval_result_dict[k]["rougeL_recall"]).mean()
        output_result[f"{eval_task_dict[k]} ROUGE"] = avg_rouge
        # if "forget" in k:
        #     avg_idk_rouge = np.array(eval_result_dict[k]["idk_rougeL_recall"]).mean()
        #     output_result[f"{eval_task_dict[k]} IDK ROUGE"] = avg_idk_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(eval_result_dict[k]["avg_paraphrased_loss"])
        avg_perturbed_np_values = np.array(eval_result_dict[k]["average_perturb_loss"])
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 = np.exp(avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if "forget" in k:
            paraphrased_perturb_ratio = 1 - np.mean(
                np.minimum(curr_stat_1, 1 / curr_stat_1)
            )
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1 / curr_stat_1))

        output_result[f"{eval_task_dict[k]} Truth Ratio"] = paraphrased_perturb_ratio
        output_result[f"{eval_task_dict[k]} Token Entropy"] = np.array(
            eval_result_dict[k]["token_entropy"]
        ).mean()
        output_result[f"{eval_task_dict[k]} Cosine Similarity"] = np.array(
            eval_result_dict[k]["cosine_similarity"]
        ).mean()
        output_result[f"{eval_task_dict[k]} Entailment Score"] = get_entailment_score(
            eval_result_dict[k]["entailment_labels"]
        )
        if "forget" in k:
            output_result[f"{eval_task_dict[k]} IDK Token Entropy"] = np.array(
                eval_result_dict[k]["token_entropy"]
            ).mean()
            # output_result[f"{eval_task_dict[k]} IDK Cosine Similarity"] = np.array(
            #     eval_result_dict[k]["idk_cosine_similarity"]
            # ).mean()
            output_result[f"{eval_task_dict[k]} IDK Entailment Score"] = get_entailment_score(
                eval_result_dict[k]["idk_entailment_labels"]
            )

    # print(output_result)
    model_utility_retain_cands = []
    model_utility_cands = []
    forget_efficacy_cands = []
    targeted_forget_efficacy_cands = []
    for k, v in output_result.items():
        # all six metrics
        if "Forget" not in k:
            # model utlity
            model_utility_cands.append(v)
            if "Retain" in k:
                # only consider the metrics on retain/neighbor set
                model_utility_retain_cands.append(v)
        else:
            # forget_efficacy
            if "Entropy" not in k and "IDK" not in k:  # exclude the token entropy
                forget_efficacy_cands.append(v)
            if "IDK" in k:
                targeted_forget_efficacy_cands.append(v)

    output_result["Model Utility Retain"] = hmean(model_utility_retain_cands)
    output_result["Model Utility"] = hmean(model_utility_cands)
    # The larger the value, the worse the performance on Forget Set.
    output_result["Untargeted Forget Efficacy"] = 1.0 - np.mean(forget_efficacy_cands)
    output_result["Targeted Forget Efficacy"] = hmean(targeted_forget_efficacy_cands)

    return output_result


def run_generation(cfg, batch, idk_batch, model, tokenizer):
    # generate outputs based on question
    input_ids = batch["input_ids"]
    idk_input_ids = idk_batch["input_ids"]

    if cfg.model_family[:3] == "phi":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|end|>\n<|assistant|>\n"
        ground_truth = []
        for s in input_strings:
            # Split the string into user and assistant parts
            parts = s.split(split_symbol)
            if len(parts) > 1:
                # Extract the assistant's response and remove trailing <|endoftext|> tokens
                assistant_response = parts[1].split("<|endoftext|>")[0]
                ground_truth.append(assistant_response.strip())
            else:
                # If no assistant response is found, append an empty string
                ground_truth.append("")
        input_strings = [s.split(split_symbol)[0] + split_symbol for s in input_strings]

        # record the ground truth for idk
        idk_input_strings = tokenizer.batch_decode(idk_input_ids, skip_special_tokens=False)
        idk_ground_truth = []
        for s in idk_input_strings:
            # Split the string into user and assistant parts
            parts = s.split(split_symbol)
            if len(parts) > 1:
                # Extract the assistant's response and remove trailing <|endoftext|> tokens
                assistant_response = parts[1].split("<|endoftext|>")[0]
                idk_ground_truth.append(assistant_response.strip())
            else:
                # If no assistant response is found, append an empty string
                idk_ground_truth.append("")
        # import pdb;pdb.set_trace()
    else:
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        split_symbol = "Answer: "
        ground_truth = [s.split(split_symbol)[1] for s in input_strings]
        input_strings = [s.split(split_symbol)[0] for s in input_strings]
        input_strings = [s + split_symbol for s in input_strings]

        # record the ground truth for idk
        idk_input_strings = tokenizer.batch_decode(idk_input_ids, skip_special_tokens=True)
        idk_ground_truth = [s.split(split_symbol)[1] for s in idk_input_strings]

    # now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to(model.device)

    # now generate
    torch.manual_seed(0)
    out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=cfg.generation.max_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.eos_token_id,
    )

    # outputs
    strs = left_pad_tokenizer.batch_decode(
        out[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return input_strings, strs, ground_truth, idk_ground_truth


def run_safe_align_generation(cfg, batch, model, tokenizer):
    # generate outputs based on question
    input_ids = batch["input_ids"]

    if cfg.model_family[:3] == "phi":
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        split_symbol = "<|end|>\n<|assistant|>\n"
        ground_truth = []
        for s in input_strings:
            # Split the string into user and assistant parts
            parts = s.split(split_symbol)
            if len(parts) > 1:
                # Extract the assistant's response and remove trailing <|endoftext|> tokens
                assistant_response = parts[1].split("<|endoftext|>")[0]
                ground_truth.append(assistant_response.strip())
            else:
                # If no assistant response is found, append an empty string
                ground_truth.append("")
        input_strings = [s.split(split_symbol)[0] + split_symbol for s in input_strings]
        input_strings = [s.split(split_symbol)[0] for s in input_strings]
        input_strings_clean = [s.replace("<|user|>\n", "") for s in input_strings]
        input_strings_clean = [s.replace(split_symbol, "") for s in input_strings_clean]
    else:
        input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        split_symbol = "Answer: "
        ground_truth = [s.split(split_symbol)[1] for s in input_strings]
        input_strings = [s.split(split_symbol)[0] for s in input_strings]
        input_strings = [s + split_symbol for s in input_strings]
        input_strings_clean = [s.replace("Question: ", "") for s in input_strings]
        input_strings_clean = [s.replace(split_symbol, "") for s in input_strings_clean]

    # now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to(model.device)

    # now generate
    torch.manual_seed(0)
    out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=cfg.generation.max_length,
        max_new_tokens=cfg.generation.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.eos_token_id,
    )

    # outputs
    strs = left_pad_tokenizer.batch_decode(
        out[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return input_strings_clean, strs


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def eval_cosine_similarity(gen_outputs, ground_truths):
    scores = []
    # model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device=torch.device("cuda"))
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device=torch.device("cuda"))
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            gen_embedding = model.encode(gen, show_progress_bar=False)
            gt_embedding = model.encode(gt, show_progress_bar=False)
            cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
            scores.append(float(max(0, cosine_sim)))

    return {"cosine_similarity": scores}


def get_entailment_results(
    pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True
):
    results = []
    for i in range(0, len(gen_outputs), bs):
        targets = ground_truths[i : i + bs]
        outputs = gen_outputs[i : i + bs]
        data_list = []
        # 能否从answer推断output
        for i in range(len(targets)):
            # For real world scenarios
            if not tofu:
                # for foget set & retain set
                data_list.append({"text": outputs[i], "text_pair": targets[i]})
            # For TOFU
            else:
                if "forget" in eval_task:
                    # for foget set
                    data_list.append({"text": outputs[i], "text_pair": targets[i]})
                else:
                    # for foget set & retain set & real author & real world
                    data_list.append({"text": targets[i], "text_pair": outputs[i]})
        results.extend(pipe(data_list))

    entailment_labels = []
    for i, result in enumerate(results):
        # If ROUGE is less than 0.1, we consider the output is factually incorrect.
        if rouge_scores[i] < 0.1:
            label = "none"
        else:
            label = result["label"]
        entailment_labels.append(label)
    return {"entailment_labels": entailment_labels}


def get_entailment_score(entailment_labels):
    correct = 0
    for label in entailment_labels:
        if label == "entailment":
            correct += 1
    return correct / len(entailment_labels)
