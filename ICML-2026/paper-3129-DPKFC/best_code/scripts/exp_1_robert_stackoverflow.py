import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from opacus.accountants.utils import get_noise_multiplier
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
import itertools
from tqdm import tqdm
from datasets import load_dataset
import random
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class StackOverflowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


class KFACRecorder:
    """Captures activations and backprops for Linear layers only (LayerNorm excluded)."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.activations = {}
        self.backprops = {}
        
        # Handle Opacus wrapper if present
        target_model = model._module if hasattr(model, "_module") else model

        for name, module in target_model.named_modules():
            # Check for Linear layers only
            if isinstance(module, nn.Linear) and not module.weight.requires_grad is False:
                self.handles.append(
                    module.register_forward_hook(self._save_activation(name))
                )
                self.handles.append(
                    module.register_full_backward_hook(self._save_backprop(name))
                )
    
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.activations = {}
        self.backprops = {}

    def _save_activation(self, name):
        def hook(module, inputs, outputs):
            if getattr(self, "enabled", True): 
                self.activations[name] = inputs[0].detach()
        return hook
    
    def _save_backprop(self, name):
        def hook(module, grad_input, grad_output):
            if getattr(self, "enabled", True): 
                self.backprops[name] = grad_output[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def compute_covariances(model, acts, grads):
    """Computes covariances for Linear layers only."""
    cov_A, cov_G = {}, {}
    eps = 1e-5

    target_model = model._module if hasattr(model, "_module") else model
    name_to_module = dict(target_model.named_modules())

    for name in grads:
        if name not in name_to_module:
            continue
            
        module = name_to_module[name]
        
        if isinstance(module, nn.Linear):
            # 1. Gradient Covariance (G)
            G = grads[name]
            if G.dim() > 2:
                G = G.reshape(-1, G.shape[-1])
            
            cov_G[name] = (G.T @ G) / G.size(0) + eps * torch.eye(
                G.size(1), device=G.device
            )

            # 2. Activation Covariance (A)
            A = acts[name]
            if A.dim() > 2:
                A = A.reshape(-1, A.shape[-1])
            
            # Add bias term to activations: A = [A, 1]
            A_with_bias = torch.cat([A, torch.ones_like(A[:, :1])], dim=1)
            cov_A[name] = (A_with_bias.T @ A_with_bias) / A.size(0) + eps * torch.eye(
                A_with_bias.size(1), device=A.device
            )

    return cov_A, cov_G


def precondition_per_sample_gradients(model, cov_A, cov_G):
    eps = 1e-3
    target_model = model._module if hasattr(model, "_module") else model

    for name, module in target_model.named_modules():
        if (
            isinstance(module, nn.Linear) 
            and hasattr(module, "weight")
            and module.weight.grad_sample is not None
            and name in cov_G
            and name in cov_A
        ):
            # --- Compute Inverse Sqrt of G ---
            G = cov_G[name]
            eva_G, evc_G = torch.linalg.eigh(
                G + eps * torch.eye(G.size(0), device=G.device)
            )
            inv_sqrt_G = evc_G @ torch.diag(eva_G.clamp(min=1e-6).rsqrt()) @ evc_G.T

            # --- Compute Inverse Sqrt of A ---
            A = cov_A[name]
            eva_A, evc_A = torch.linalg.eigh(
                A + eps * torch.eye(A.size(0), device=A.device)
            )
            inv_sqrt_A = evc_A @ torch.diag(eva_A.clamp(min=1e-6).rsqrt()) @ evc_A.T

            g_sample_w = module.weight.grad_sample
            g_sample_b = module.bias.grad_sample
            
            if isinstance(g_sample_w, list): g_sample_w = g_sample_w[-1]
            if isinstance(g_sample_b, list): g_sample_b = g_sample_b[-1]

            # [B, out, in+1]
            g_sample_augmented = torch.cat(
                [g_sample_w, g_sample_b.unsqueeze(2)], dim=2
            )

            # KFAC projection
            temp = torch.einsum("oj, bjk -> bok", inv_sqrt_G, g_sample_augmented)
            preconditioned_g_sample = torch.einsum("bok, ki -> boi", temp, inv_sqrt_A)

            module.weight.grad_sample = preconditioned_g_sample[:, :, :-1]
            module.bias.grad_sample = preconditioned_g_sample[:, :, -1]


def apply_dp(model, noise_multiplier, max_grad_norm, batch_size):
    total_norm_sq = None
    params_with_grad_sample = []
    
    for p in model.parameters():
        if hasattr(p, "grad_sample") and p.grad_sample is not None:
            params_with_grad_sample.append(p)

    for p in params_with_grad_sample:
        grad_sample = p.grad_sample
        if isinstance(grad_sample, list):
            grad_sample = grad_sample[-1]
        grad_sample = grad_sample.contiguous().view(batch_size, -1)
        norms_sq = grad_sample.norm(2, dim=1).pow(2)

        if total_norm_sq is None:
            total_norm_sq = norms_sq
        else:
            total_norm_sq += norms_sq

    total_norm = (
        total_norm_sq.sqrt() if total_norm_sq is not None else torch.tensor(0.0)
    )
    clip_factors = (max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

    for p in params_with_grad_sample:
        grad_sample = p.grad_sample
        if isinstance(grad_sample, list):
            grad_sample = grad_sample[-1]
        grad_sample = grad_sample.contiguous().view(batch_size, -1)
        
        grad_sample_clipped = grad_sample * clip_factors.unsqueeze(1)
        summed_grad = torch.sum(grad_sample_clipped, dim=0)
        
        noise = torch.randn_like(summed_grad) * noise_multiplier * max_grad_norm
        
        p.grad = ((summed_grad + noise) / batch_size).view_as(p.grad)
        p.grad_sample = None


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            model_to_eval = model._module if hasattr(model, "_module") else model
            
            outputs = model_to_eval(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    # print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    return accuracy, avg_loss

def generate_synthetic_bert_batch(batch_size, max_len, vocab_size, device):
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    seq_lens = torch.randint(5, max_len, (batch_size,), device=device)
    random_tokens = torch.randint(999, vocab_size, (batch_size, max_len), device=device)
    
    range_matrix = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = range_matrix < seq_lens.unsqueeze(1)
    
    input_ids = input_ids + (random_tokens * mask.long())
    input_ids[:, 0] = 101 # CLS
    
    sep_indices = (seq_lens - 1).unsqueeze(1)
    input_ids.scatter_(1, sep_indices, 102) # SEP

    attention_mask = mask.long()
    labels = torch.randint(0, 2, (batch_size,), device=device)

    return input_ids, attention_mask, labels


def freeze_except_classifier(model):
    real_model = model._module if hasattr(model, "_module") else model
    for name, param in real_model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def train_plain_sgd(
    model,
    train_loader,
    optimizer,
    num_epochs
):
    """
    Baseline training: No DP, No Clipping, No Noise. 
    Uses Adam as requested.
    """
    for epoch in range(num_epochs):
        model.train()
        # pbar = tqdm(train_loader, desc=f"Plain Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

def train_dp_sgd(
    model,
    private_loader,
    public_loader,
    optimizer,
    noise_multiplier,
    sample_rate,
    max_grad_norm,
    num_epochs,
    target_delta=1e-5,
    kfac=False,
    public_data=True,
):
    accountant = RDPAccountant()
    
    recorder = None
    public_iter = None
    if kfac:
        recorder = KFACRecorder(model)
        public_iter = iter(itertools.cycle(public_loader))

    for epoch in range(num_epochs):
        model.train()
        
        idx = 0
        # pbar = tqdm(private_loader, desc=f"DP Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in private_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            current_batch_size = input_ids.size(0)

            model.zero_grad(set_to_none=True)

            if kfac and idx % len(private_loader) == 0:
                recorder.enable()
                
                if public_data:
                    public_batch = next(public_iter)
                    pub_input_ids = public_batch["input_ids"].to(DEVICE)
                    pub_mask = public_batch["attention_mask"].to(DEVICE)
                    pub_labels = public_batch["labels"].to(DEVICE)
                    
                    public_outputs = model(input_ids=pub_input_ids, attention_mask=pub_mask)
                    public_loss = F.cross_entropy(public_outputs, pub_labels) 
                    public_loss.backward()
                else:
                    vocab_size = model._module.bert.config.vocab_size
                    seq_len = input_ids.size(1)
                    
                    noise_input_ids, noise_mask, noise_labels = generate_synthetic_bert_batch(
                        batch_size=current_batch_size,
                        max_len=seq_len,
                        vocab_size=vocab_size,
                        device=DEVICE
                    )
                    noise_outputs = model(input_ids=noise_input_ids, attention_mask=noise_mask)
                    noise_loss = F.cross_entropy(noise_outputs, noise_labels)
                    noise_loss.backward()

                cov_A, cov_G = compute_covariances(
                    model,
                    recorder.activations,
                    recorder.backprops,
                )
                recorder.disable()
                model.zero_grad(set_to_none=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels, reduction='sum')

            loss.backward()

            if kfac:
                precondition_per_sample_gradients(model, cov_A, cov_G)

            idx += 1
            apply_dp(model, noise_multiplier, max_grad_norm, current_batch_size)
            optimizer.step()
            
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        if kfac and recorder:
            recorder.activations.clear()
            recorder.backprops.clear()

def prepare_stackoverflow_data():
    print("Loading Stack Overflow dataset...")
    so_dataset = load_dataset("mteb/stackoverflowdupquestions-reranking")

    private_texts = []
    private_labels = []

    # Process training data
    for item in so_dataset["train"]:
        text = f"Question 1: {item['query']} Question 2: {item['positive']}"
        private_texts.append(text)
        private_labels.append(1)  # duplicate

        text = f"Question 1: {item['query']} Question 2: {item['negative']}"
        private_texts.append(text)
        private_labels.append(0)  # not duplicate

    max_samples = 5000
    indices = random.sample(
        range(len(private_texts)), min(max_samples, len(private_texts))
    )
    private_texts = [private_texts[i] for i in indices]
    private_labels = [private_labels[i] for i in indices]

    print("Loading AG News dataset for public data...")
    ag_dataset = load_dataset("ag_news")

    public_texts = []
    public_labels = []

    for item in ag_dataset["train"]:
        text = item["text"]
        label = item["label"]
        binary_label = 0 if label < 2 else 1
        public_texts.append(text)
        public_labels.append(binary_label)
        if len(public_texts) >= 5000:
            break

    test_texts = []
    test_labels = []
    for item in so_dataset["test"]:
        text = f"Question 1: {item['query']} Question 2: {item['positive']}"
        test_texts.append(text)
        test_labels.append(1)

        text = f"Question 1: {item['query']} Question 2: {item['negative']}"
        test_texts.append(text)
        test_labels.append(0)

    return (
        private_texts,
        private_labels,
        public_texts,
        public_labels,
        test_texts,
        test_labels,
    )


if __name__ == "__main__":
    # --- Experiment Settings ---
    EPSILONS = [0.5, 1.0, 2, 3, 5.0, 7.5, 10]
    SEEDS = [42, 43, 44, 45, 46]
    
    EPOCHS = 5
    LR = 2e-4
    TARGET_DELTA = 1e-6
    MAX_GRAD_NORM = 1.0
    BATCH_SIZE = 64 
    MAX_LENGTH = 128

    # Prepare data once
    (
        private_texts,
        private_labels,
        public_texts,
        public_labels,
        test_texts,
        test_labels,
    ) = prepare_stackoverflow_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    private_dataset = StackOverflowDataset(private_texts, private_labels, tokenizer, MAX_LENGTH)
    public_dataset = StackOverflowDataset(public_texts, public_labels, tokenizer, MAX_LENGTH)
    test_dataset = StackOverflowDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    private_loader = DataLoader(private_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    public_loader = DataLoader(public_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    PRIVATE_DATA_SIZE = len(private_dataset)
    SAMPLE_RATE = BATCH_SIZE / PRIVATE_DATA_SIZE
    TOTAL_STEPS = EPOCHS * (PRIVATE_DATA_SIZE // BATCH_SIZE)

    # Store results: list of dicts
    experiment_results = []

    print("\n" + "="*60)
    print("STARTING EXPERIMENTAL SUITE")
    print(f"Epsilons: {EPSILONS}")
    print(f"Seeds: {SEEDS}")
    print("="*60 + "\n")

    # --- 1. Baseline: Plain SGD (Run once per seed, epsilon irrelevant) ---
    for seed in SEEDS:
        print(f"Running Baseline Plain SGD (Adam) | Seed: {seed}")
        set_seed(seed)
        
        model = BERTClassifier(num_classes=2).to(DEVICE)
        freeze_except_classifier(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=0.0)
        
        train_plain_sgd(model, private_loader, optimizer, EPOCHS)
        acc, loss = evaluate(model, test_loader)
        
        experiment_results.append({
            "Method": "Plain SGD",
            "Epsilon": "N/A",
            "Seed": seed,
            "Accuracy": acc,
            "Loss": loss
        })
        print(f" -> Result: Acc={acc*100:.2f}%")

    # --- 2. DP Methods Loop ---
    for eps in EPSILONS:
        # Calculate noise for this specific epsilon
        noise_multiplier = get_noise_multiplier(
            target_epsilon=eps,
            target_delta=TARGET_DELTA,
            sample_rate=SAMPLE_RATE,
            steps=TOTAL_STEPS,
            accountant="rdp",
        )
        print(f"\n--- Configuration: Target ε = {eps} (Noise σ = {noise_multiplier:.2f}) ---")

        for seed in SEEDS:
            print(f" Processing Seed {seed}...")

            # A) Standard DP-SGD
            set_seed(seed)
            model = BERTClassifier(num_classes=2).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            freeze_except_classifier(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=0.0)

            train_dp_sgd(
                model, private_loader, public_loader, optimizer,
                noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS,
                TARGET_DELTA, kfac=False
            )
            acc, loss = evaluate(model, test_loader)
            experiment_results.append({
                "Method": "Standard DP-SGD",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss
            })

            # B) Public KFAC DP-SGD
            set_seed(seed)
            model = BERTClassifier(num_classes=2).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            freeze_except_classifier(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=0.0)

            train_dp_sgd(
                model, private_loader, public_loader, optimizer,
                noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS,
                TARGET_DELTA, kfac=True, public_data=True
            )
            acc, loss = evaluate(model, test_loader)
            experiment_results.append({
                "Method": "KFAC (Public)",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss
            })

            # C) Noise KFAC DP-SGD
            set_seed(seed)
            model = BERTClassifier(num_classes=2).to(DEVICE)
            model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
            freeze_except_classifier(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=0.0)

            train_dp_sgd(
                model, private_loader, public_loader, optimizer,
                noise_multiplier, SAMPLE_RATE, MAX_GRAD_NORM, EPOCHS,
                TARGET_DELTA, kfac=True, public_data=False
            )
            acc, loss = evaluate(model, test_loader)
            experiment_results.append({
                "Method": "KFAC (Noise)",
                "Epsilon": eps,
                "Seed": seed,
                "Accuracy": acc,
                "Loss": loss
            })
    # save results to CSV
    results_df = pd.DataFrame(experiment_results)
    results_df.to_csv("results_stackoverflow/stackoverflow_dp_kfac_results.csv", index=False)
    # --- Final Report ---
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Epsilon':<10} | {'Seed':<5} | {'Accuracy (%)':<12} | {'Loss':<10}")
    print("-" * 80)
    for res in experiment_results:
        print(f"{res['Method']:<20} | {str(res['Epsilon']):<10} | {res['Seed']:<5} | {res['Accuracy']*100:05.2f}%       | {res['Loss']:.4f}")
    print("="*80)