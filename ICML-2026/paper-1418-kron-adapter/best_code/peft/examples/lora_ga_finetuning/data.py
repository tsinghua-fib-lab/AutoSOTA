from datasets import load_dataset, Dataset
import typing as tp
import functools
import os
import pickle
import logging

log = logging.getLogger(__name__)

def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    log.info(f"Loading cached data for {func.__name__}")
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
                log.info(f"Cached data for {func.__name__}")
            return result

        return wrapper_cache

    return decorator_cache

@cache_to_disk("data_cache")
def load_emo():
    dataset = load_dataset("emo")
    label_map = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    instruction = "classify the emotion of the text: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["text"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    test_set = dataset["test"]
    return train_set, test_set, test_set

@cache_to_disk("data_cache")
def load_sst2():
    dataset = load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_cola():
    dataset = load_dataset("glue", "cola")
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_qqp():
    dataset = load_dataset("glue", "qqp")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "duplicate", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question1"]}\n{e["question2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_mrpc():
    dataset = load_dataset("glue", "mrpc")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_mnli():
    dataset = load_dataset("glue", "mnli")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation_matched"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_squad():
    dataset = load_dataset("rajpurkar/squad")
    instruction = "answer the question: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\ncontext: {e["context"]}\nresult: ',
            "y": ", ".join(e["answers"]["text"]),
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_qnli():
    dataset = load_dataset("glue", "qnli")
    instruction = "classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    return train_set, validation_set, test_set


template_with_input = '''### Instruction:
{instruction}

### Input:
{input}

### Response:
'''

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

@cache_to_disk("data_cache")
def load_alpaca():
    dataset = load_dataset("tatsu-lab/alpaca")
    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}
    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)['train']
    validation_set = dataset["train"].train_test_split(test_size=0.1)['test']
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_gsm8k():
    dataset = load_dataset("gsm8k", "main")
    #x = "Q: " + x[0] + "\n" + "A:"
    dataset = dataset.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_alpaca_gpt4():
    dataset = load_dataset("tatsu-lab/alpaca")
    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}
    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # we sample 10% of the training set as validation set
    train_set = dataset["train"].train_test_split(test_size=0.1)['train']
    validation_set = dataset["train"].train_test_split(test_size=0.1)['test']
    return train_set, validation_set, validation_set

@cache_to_disk("data_cache")
def load_flan():
    dataset = load_dataset("Muennighoff/flan", split='train', streaming=True)
    def preprocess(data):
        return {
            "x": template_wo_input.format(instruction=data['inputs']),
            "y": data['targets'],
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(buffer_size=5000, seed=42)
    from tqdm import tqdm
    for sample in tqdm(dataset, total=110000):
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_meta_math(max_tokens=512):
    dataset = load_dataset("meta-math/MetaMathQA", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens or "GSM" not in sample["type"]:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_flan_v2(max_tokens=512):
    dataset = load_dataset("SirNeural/flan_v2", split='train', streaming=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        return {
            "x": data['inputs'],
            "y": data['targets'],
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(buffer_size=5000, seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=1024):
    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        y = data['answer']
        y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set

@cache_to_disk("data_cache")
def load_wizardlm(max_tokens=1024):
    dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def preprocess(data):
        y = data['output']
        return {
            "x": template_wo_input.format(
                instruction=data['instruction']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=70000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp['y'].lower() or "as an ai" in temp['y'].lower():
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < 52000:
            train_samples.append(processed_sample)
        elif 52000 <= count < 70000:
            eval_samples.append(processed_sample)
        elif count >= 70000:  # Stop processing after collecting enough samples
            break
        count += 1
    # convert to hf dataset
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


DATASET_MAP = {
    "sst2": load_sst2,
    "cola": load_cola,
    "qqp": load_qqp,
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "emo": load_emo,
    "squad": load_squad,
    "alpaca": load_alpaca,
    "qnli": load_qnli,
    "gsm8k": load_gsm8k,
    "alpaca_gpt4": load_alpaca_gpt4,
    "flan": load_flan,
    "flan_v2": load_flan_v2,
    "meta_math": load_meta_math,
    "codefeedback": load_codefeedback,
    "wizard_lm": load_wizardlm,
}


if __name__ == "__main__":
    # for dataset in [load_emo, load_sst2, load_cola, load_qqp, load_mrpc, load_mnli]:
    #     train_set, val_set, test_set = dataset()
    #     print(train_set[0])
    #     print(val_set[0])
    #     print(test_set[0])
    #     print()
    # print(load_alpaca())
    # for name, dataset in DATASET_MAP.items():
    #     train_set, val_set, test_set = dataset()
    #     print(name)
    #     print(train_set[0])
    #     print(val_set[0])
    #     print(test_set[0])
    #     print()
    x, r, _ = load_wizardlm()
    print(x[0]['x'])
    print(x[0]['y'])
    print(len(x))
    print(len(r))
