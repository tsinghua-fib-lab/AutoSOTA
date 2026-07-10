import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# ==============================
# Data Loading & Preprocessing
# ==============================
data = load_dataset("TIGER-Lab/MMLU-Pro", keep_in_memory=True)

old_domain = "biology"
new_domain = "health"

old_domain_data = data["test"].filter(lambda x: x["category"] == old_domain, load_from_cache_file=False)
new_domain_data = data["test"].filter(lambda x: x["category"] == new_domain, load_from_cache_file=False)

print(f"Old Domain ({old_domain}): {len(old_domain_data)} examples")
print(f"New Domain ({new_domain}): {len(new_domain_data)} examples")

# ==============================
# Load Models
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

logit_model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-6B")
logit_model.eval()

embed_tokenizer = AutoTokenizer.from_pretrained("gpt2")
embed_tokenizer.pad_token = embed_tokenizer.eos_token
embed_model = AutoModel.from_pretrained("gpt2")
embed_model.eval()


def build_prompt(question, options):
    alphabet_options = [f"({chr(65 + i)}) {option}." for i, option in enumerate(options)]
    options_str = " ".join(alphabet_options)
    prompt = (
        f"Carefully analyze the following multiple-choice question "
        f"and think through your reasoning step-by-step. Only one option is correct. "
        f"You only need to output the option. The question is: {question} "
        f"The options are: {options_str} After thoughtful consideration, the correct answer is option:"
    )
    return prompt


def compute_multi_choice_probs(question, options):
    """
    Computes the probability distribution over candidate answers using the
    last token's log probabilities in one model forward pass.
    """
    # Build the prompt without appending any candidate letter.
    prompt = build_prompt(question, options)

    # Tokenize the prompt.
    input_ids = tokenizer.encode(prompt, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = logit_model(input_ids)
    logits = outputs.logits

    # Extract the logits for the last token.
    last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]

    # Compute log softmax to convert logits into log probabilities.
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)

    # Precompute the candidate token IDs for option letters ('A', 'B', 'C', etc.)
    candidate_token_ids = [
        tokenizer.encode(chr(65 + i), add_special_tokens=False)[-1] for i in range(len(options))
    ]

    # Extract the log probability for each candidate answer.
    option_log_probs = [log_probs[token_id].item() for token_id in candidate_token_ids]

    # For numerical stability, subtract the maximum log probability.
    option_log_probs = np.array(option_log_probs)
    option_log_probs -= np.max(option_log_probs)

    # Convert log probabilities into actual probabilities.
    probs = np.exp(option_log_probs)
    probs = probs / np.sum(probs)
    return probs


def nonconformity_score(example):
    """
    Computes the nonconformity score as the negative log probability of the correct answer.
    """
    question = example["question"]
    options = example["options"]
    correct = example["answer_index"]
    probs = compute_multi_choice_probs(question, options)
    # Add a small constant to avoid log(0)
    return -np.log(probs[correct] + 1e-9)


# ==============================
# Sentence Embedding Extraction
# ==============================
def get_embeddings(text_list):
    encoded = embed_tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embed_model(**encoded)
    last_hidden_states = outputs.last_hidden_state
    embedding = last_hidden_states.mean(dim=1)
    embedding_array = embedding.squeeze().cpu().numpy()
    return embedding_array


# ==============================
# Domain Adaptation using Logistic Regression with PCA
# ==============================
old_list = list(old_domain_data)
new_list = list(new_domain_data)
domain_labels = [0] * len(old_list) + [1] * len(new_list)

X_domain = [get_embeddings(ex["question"]) for ex in (old_list + new_list)]
pca = PCA(n_components=50)
X_domain_reduced = pca.fit_transform(X_domain)

log_reg = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000)
log_reg.fit(X_domain_reduced, np.array(domain_labels))


def density_ratio(texts, clip_min=1e-3, clip_max=1 - 1e-3):
    emb = get_embeddings(texts)
    emb_reduced = pca.transform(emb.reshape(1, -1))
    probs = log_reg.predict_proba(emb_reduced)
    p_new = np.clip(probs[:, 1], clip_min, clip_max)
    return p_new / (1 - p_new)


# ==============================
# Weighted Conformal Calibration on Old Domain
# ==============================
cal_weights = []
cal_scores = []
old_correct = 0
for ex in old_list:
    w_val = density_ratio([ex["question"]])[0]
    probs = compute_multi_choice_probs(ex["question"], ex["options"])
    s_val = -np.log(probs[ex["answer_index"]] + 1e-9)
    cal_weights.append(w_val)
    cal_scores.append(s_val)
    pred = np.argmax(probs)
    if pred == ex["answer_index"]:
        old_correct += 1

cal_weights = np.array(cal_weights)
cal_scores = np.array(cal_scores)


def weighted_quantile(values, weights, quantile):
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum_weights = np.cumsum(sorted_weights)
    cutoff = quantile * np.sum(sorted_weights)
    return sorted_values[np.searchsorted(cumsum_weights, cutoff)]


alpha = 0.1
target = 1 - alpha
cal_weights_normalized = cal_weights * len(cal_weights) / np.sum(cal_weights)
weighted_thresh = weighted_quantile(cal_scores, cal_weights_normalized, target)
standard_thresh = np.percentile(cal_scores, target * 100)
print(f"Weighted Conformal threshold: {weighted_thresh:.4f}")
print(f"Standard Conformal threshold: {standard_thresh:.4f}")

# ==============================
# Evaluate on New Domain & Compute Prediction Set Sizes and Accuracy
# ==============================
# Precompute probability distributions for new domain examples.
new_probs_list = [compute_multi_choice_probs(ex["question"], ex["options"]) for ex in new_list]

# Compute nonconformity scores for the correct answers.
new_scores = [-np.log(probs[ex["answer_index"]] + 1e-9) for probs, ex in zip(new_probs_list, new_list)]

weighted_correct = sum(score <= weighted_thresh for score in new_scores)
standard_correct = sum(score <= standard_thresh for score in new_scores)
weighted_coverage = weighted_correct / len(new_list)
standard_coverage = standard_correct / len(new_list)
print(f"Weighted Conformal Coverage: {weighted_coverage:.2f}")
print(f"Standard Conformal Coverage: {standard_coverage:.2f}")

# Compute prediction set sizes.
weighted_set_sizes = [np.sum(-np.log(probs + 1e-9) <= weighted_thresh) for probs in new_probs_list]
standard_set_sizes = [np.sum(-np.log(probs + 1e-9) <= standard_thresh) for probs in new_probs_list]
print("Average weighted prediction set size:", np.mean(weighted_set_sizes))
print("Average standard prediction set size:", np.mean(standard_set_sizes))

# Compute prediction accuracy for new domain.
new_accuracy = np.mean([np.argmax(probs) == ex["answer_index"] for probs, ex in zip(new_probs_list, new_list)])
print("Prediction Accuracy on New Domain:", new_accuracy)

# Compute prediction accuracy for old domain.
old_accuracy = old_correct / len(old_list)
print("Prediction Accuracy on Old Domain:", old_accuracy)