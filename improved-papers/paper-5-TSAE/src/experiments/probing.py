from dotenv import load_dotenv
load_dotenv()

import os
from nnsight import LanguageModel
from dictionary_learning.utils import load_dictionary
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import load_activations_and_labels, load_activations_and_labels_sequences

def quiver_arrows(labels, SAE_acts):
    num_classes = np.unique(labels).size
    K = 10

    top_indices = set()
    for cls in range(num_classes):
        class_mask = labels == cls
        class_acts = SAE_acts[class_mask]           # Shape: (N_class, num_latents)
        rest_acts = SAE_acts[~class_mask]
        # Option 1: Use "difference to mean"
        mean_class = np.mean(class_acts, axis=0)
        mean_rest = np.mean(rest_acts, axis=0)
        mean_diff = mean_class - mean_rest
        top10 = np.argsort(mean_diff)[-K:]
        top_indices.update(top10)

    return np.array(list(top_indices))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--run_name", type=str)
    
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="finefineweb")
    parser.add_argument("--chunk", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading SAEs from {args.checkpoints_path}")


    device = "cuda:0"
    model_name =  args.model # can be any Huggingface model

    model = LanguageModel(
        model_name,
        device_map=device,
        dispatch=True
    )

    if hasattr(model, "model"):
        submodule = model.model.layers[args.layer] # gemma
    elif hasattr(model, "gpt_neox"):
        submodule = model.gpt_neox.layers[args.layer] 
    else:
        print("model doesnt have .model or .gpt_neox attr, check how to access layers")
        exit(1)

    if args.run_name == "baseline_model":
        sae=None
    else:
        sae, trainer_cfg = load_dictionary(os.path.join(args.checkpoints_path, f"{args.run_name}/trainer_0"), device, checkpoint_path=args.checkpoint)
        sae.temporal = False
        sae.eval()
        for param in sae.parameters():
            param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    domains = ["christianity", "law", "literature", "economics", "food", "drama_and_film", "health", "mathematics", "medical", "history"]
    features_total, pos, domains = load_activations_and_labels(model, submodule, sae, args.dataset, chunk=args.chunk, domains=domains)
    features_seq, sequences = load_activations_and_labels_sequences(model, submodule, sae, args.dataset, chunk=args.chunk, domains=domains)
    print(features_total.shape)
    print(len(pos), len(domains), len(sequences))
    domain_indices = np.unique(domains, return_inverse=True)[1]
    pos_indices = np.unique(pos, return_inverse=True)[1]
    N = sae.group_sizes[0].cpu().numpy()

    # Train Test Split
    arr = np.arange(len(features_total))
    np.random.shuffle(arr)
    train_features = features_total[arr[:3000]]
    train_labels = domain_indices[arr[:3000]]
    test_features = features_total[arr[3000:]]
    test_labels = domain_indices[arr[3000:]]

    top_indices = quiver_arrows(train_labels, train_features)
    print(len(top_indices), max(top_indices), top_indices)
    train_features = train_features[:, top_indices]
    test_features  = test_features[:, top_indices]

    ### SEMANTICS ###
    print(args)
    print(train_features.shape, test_features.shape)
    print("fitting probe: semantics")
    probe = LogisticRegression()
    probe.fit(train_features, train_labels)

    weights = probe.coef_
    first_chunk_mask = top_indices < N
    rest_mask = top_indices >= N
    first_chunk_weights = weights[:, first_chunk_mask]
    rest_weights = weights[:, rest_mask]

    # Mean across all classes and all matching features
    mean_first_chunk = np.abs(first_chunk_weights).mean()
    mean_rest = np.abs(rest_weights).mean()

    train_accuracy = probe.score(train_features, train_labels)
    accuracy = probe.score(test_features, test_labels)
    print(f"Probe train accuracy: {train_accuracy}")
    print(f"Probe test accuracy: {accuracy}")

    ### SYNTAX ###
    print(args)
    train_features = features_total[arr[:3000]]
    test_features = features_total[arr[3000:]]
    train_labels = pos_indices[arr[:3000]]
    test_labels = pos_indices[arr[3000:]]

    top_indices = quiver_arrows(train_labels, train_features)
    print(len(top_indices), max(top_indices), top_indices)
    train_features = train_features[:, top_indices]
    test_features  = test_features[:, top_indices]

    print(train_features.shape)
    print(test_features.shape)
    print("fitting probe: syntax")
    probe = LogisticRegression()
    probe.fit(train_features, train_labels)
    weights = probe.coef_
    first_chunk_mask = top_indices < N
    rest_mask = top_indices >= N
    first_chunk_weights = weights[:, first_chunk_mask]
    rest_weights = weights[:, rest_mask]

    # Mean across all classes and all matching features
    mean_first_chunk = np.abs(first_chunk_weights).mean()
    mean_rest = np.abs(rest_weights).mean()

    train_accuracy = probe.score(train_features, train_labels)
    print(f"Probe train accuracy: {train_accuracy}")

    accuracy = probe.score(test_features, test_labels)
    print(f"Probe test accuracy: {accuracy}")


    ### CONTEXT ###
    train_labels, test_labels = sequences[arr[:3000]], sequences[arr[3000:]]
    train_features, test_features = features_seq[arr[:3000]], features_seq[arr[3000:]]

    top_indices = quiver_arrows(train_labels, train_features)
    print(len(top_indices), max(top_indices), top_indices)
    train_features = train_features[:, top_indices]
    test_features  = test_features[:, top_indices]

    print(args)
    print(train_features.shape, test_features.shape)
    print("fitting probe: context")
    probe = LogisticRegression()
    probe.fit(train_features, train_labels)

    weights = probe.coef_
    first_chunk_mask = top_indices < N
    rest_mask = top_indices >= N
    first_chunk_weights = weights[:, first_chunk_mask]
    rest_weights = weights[:, rest_mask]

    # Mean across all classes and all matching features
    mean_first_chunk = np.abs(first_chunk_weights).mean()
    mean_rest = np.abs(rest_weights).mean()

    train_accuracy = probe.score(train_features, train_labels)
    accuracy = probe.score(test_features, test_labels)
    print(f"Probe train accuracy: {train_accuracy}")
    print(f"Probe test accuracy: {accuracy}")
