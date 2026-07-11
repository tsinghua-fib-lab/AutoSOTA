import argparse
import numpy as np
import os
import json
import pandas as pd
from src.utils import load_data, preprocess_text, create_stopwords
import matplotlib.pyplot as plt

# Scikit-learn imports for classification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--data_name', type=str, default='gsm8k')
    return parser

def preprocess(data):
    all_texts = []
    labels = []  # 1 for ill-posed, 0 for well-posed

    # Collect thinking content and assign labels
    for row in data:
        # Only add if the content is not empty after preprocessing
        well_content = preprocess_text(row.get("reasoning_well_posed", ""))
        if well_content:
            all_texts.append(well_content)
            labels.append(0)  # With context

        ill_content = preprocess_text(row.get("reasoning_ill_posed", ""))
        if ill_content:
            all_texts.append(ill_content)
            labels.append(1)  # Without context

    if not all_texts or len(set(labels)) < 2:
        return {
            "classification_accuracy": "N/A (Insufficient data or only one class present)",
            "classification_report": "N/A",
            "top_feature_importances": []
        }

    stopwords = create_stopwords()
    vectorizer = CountVectorizer(ngram_range=(1, 4), stop_words=list(stopwords), min_df=5)
    X = vectorizer.fit_transform(all_texts)

    y = np.array(labels)
    feature_names = vectorizer.get_feature_names_out()
        
    return X, y, feature_names

def thin_grams_negation_parity(
    X,
    feature_names,
    small: int,
    large: int,
    threshold: float = 0.85,
    top_k: int = 5,
    neg_tokens=("not", "no", "nt", "n't"),
):
    """
    Negation-parity thinning:

    Let neg(g) be whether gram g contains any negation token.

    For each small-gram s and each containing large-gram L:
      - If neg(s) == neg(L) and cooccurrence is high => DROP L (keep s).
      - If neg(s) == False and neg(L) == True and cooccurrence is high => DROP s.

    Cooccurrence score is P(L | s) estimated by overlap_docs(s, L) / docs(s).
    Aggregation is sum of top_k scores over eligible containers.

    Returns keep_idx, dropped_small_idx_set, dropped_large_idx_set.
    """

    if not (small > 0 and large > 0 and small < large):
        raise ValueError("Require integers with 0 < small < large.")

    feat = np.asarray(feature_names)
    n = len(feat)

    # Tokenize features and compute token-lengths
    feat_toks = [tuple(f.split()) for f in feat]
    n_tokens = np.fromiter((len(t) for t in feat_toks), dtype=int, count=n)

    idx_small = np.where(n_tokens == small)[0]
    idx_large = np.where(n_tokens == large)[0]
    if len(idx_small) == 0 or len(idx_large) == 0:
        return np.arange(n), set(), set()

    neg_set = set(neg_tokens)

    # negation status per feature
    has_neg = np.fromiter((any(tok in neg_set for tok in toks) for toks in feat_toks),
                          dtype=bool, count=n)

    # Binary presence + DF
    X_bool = (X > 0)
    df = np.asarray(X_bool.sum(axis=0)).ravel()

    # Map token-tuple -> feature index
    toks2idx = {t: i for i, t in enumerate(feat_toks)}

    # Build containment maps per small i:
    # same_parity_containers[i] = [j_large,...] where neg parity matches
    # neg_mismatch_containers[i] = [j_large,...] where small non-neg but large neg
    same_parity_containers = {}
    neg_mismatch_containers = {}

    for j in idx_large:
        toksL = feat_toks[j]
        L_neg = has_neg[j]

        for start in range(0, len(toksL) - small + 1):
            sub = toksL[start:start + small]
            i = toks2idx.get(sub)
            if i is None or n_tokens[i] != small:
                continue

            s_neg = has_neg[i]

            if s_neg == L_neg:
                same_parity_containers.setdefault(i, []).append(j)
            else:
                # The only mismatch that can happen under containment is:
                # small has no neg, large has neg (neg token outside the subgram).
                if (not s_neg) and L_neg:
                    neg_mismatch_containers.setdefault(i, []).append(j)

    dropped_small = set()
    dropped_large = set()

    def cond_scores(i_small, large_indices):
        """Compute (score, j) list sorted desc, score = P(L|s)."""
        s_docs = df[i_small]
        if s_docs == 0 or not large_indices:
            return []
        a = X_bool[:, i_small]
        out = []
        for j in large_indices:
            overlap = a.multiply(X_bool[:, j]).getnnz()
            out.append((overlap / s_docs, j))
        out.sort(reverse=True, key=lambda x: x[0])
        return out

    # Iterate small grams and apply the two rules
    for i in idx_small:
        if df[i] == 0:
            continue

        # Case B: mismatch (small non-neg, large neg) => potentially DROP SMALL
        mismatch = neg_mismatch_containers.get(i, [])
        if mismatch:
            scores = cond_scores(i, mismatch)
            top_sum = sum(s for s, _ in scores[:min(top_k, len(scores))])
            if top_sum >= threshold:
                dropped_small.add(i)
                # print(f"ADDED SMALL {feat[i]}, example larges: {[feat[j] for _, j in scores[:min(3, len(scores))]]}")
                continue  # once small dropped, no need to drop larges for it

        # Case A: same parity => potentially DROP LARGE(s)
        same = same_parity_containers.get(i, [])
        if same:
            scores = cond_scores(i, same)
            top = scores[:min(top_k, len(scores))]
            top_sum = sum(s for s, _ in top)
            if top_sum >= threshold:
                for _, j_drop in top:
                    dropped_large.add(j_drop)
                    # print(f"ADDED LARGE {feat[i]}, {feat[j_drop]}")

    to_drop = dropped_small | dropped_large
    keep_idx = np.array(sorted(set(range(n)) - to_drop), dtype=int)
    return keep_idx


def prune_redundant_subngrams(vectorizer: CountVectorizer, X):
    """
    Drop n-grams that are contiguous sub-ngrams of a longer n-gram
    and have identical document frequency (i.e., never appear without the longer n-gram).

    Parameters
    ----------
    vectorizer : fitted CountVectorizer
    X : sparse matrix from vectorizer.transform(...) or fit_transform(...)

    Returns
    -------
    pruned_vocab : dict[str, int]
        Vocabulary mapping after pruning.
    removed : set[str]
        Removed feature strings.
    """
    # document frequency for each feature
    df = np.asarray((X > 0).sum(axis=0)).ravel()

    # feature strings indexed by column
    try:
        feats = vectorizer.get_feature_names_out()
    except AttributeError:
        feats = np.array(vectorizer.get_feature_names())

    # tokenize feature into tuple of tokens (CountVectorizer joins tokens by spaces)
    feat_tokens = [tuple(f.split()) for f in feats]
    tok2idx = {toks: i for i, toks in enumerate(feat_tokens)}

    # For each longer n-gram, mark its sub-ngrams as removable if DF matches
    removable = set()

    # Process longer-to-shorter so a short gram can be removed by any longer container
    lengths = sorted({len(t) for t in feat_tokens}, reverse=True)
    for L in lengths:
        if L <= 1:
            continue
        # consider all features of length L
        for toks in (t for t in feat_tokens if len(t) == L):
            i_long = tok2idx[toks]
            df_long = df[i_long]
            if df_long == 0:
                continue

            # generate all strict contiguous sub-ngrams
            for subL in range(1, L):
                for start in range(0, L - subL + 1):
                    sub = toks[start:start+subL]
                    j = tok2idx.get(sub)
                    if j is None:
                        continue
                    if df[j] == df_long:
                        removable.add(feats[j])
                        #print(f"Removing sub-gram {feats[j]} (sub of {feats[i_long]})")

    # Build pruned vocabulary (re-indexed)
    kept = [f for f in feats if f not in removable]
    pruned_vocab = {f: k for k, f in enumerate(kept)}
    return pruned_vocab, removable


def classify_thinking_content(data):
    """
    Classifies thinking content as with or without context using n-grams and Random Forest.
    
    Args:
        data (list): A list of dictionaries, each containing 'thinking_content'
                     and 'reasoning_ill_posed'.
    
    Returns:
        dict: A dictionary containing classification accuracy, report, confusion matrix,
              and top feature importances.
    """
    all_texts = []
    ill_texts = []
    labels = []  # 1 for ill-posed, 0 for well-posed
    for row in data:
        # Only add if the content is not empty after preprocessing
        well_content = preprocess_text(row.get("reasoning_well_posed", ""))
        if well_content:
            all_texts.append(well_content)
            labels.append(0)  # With context

        ill_content = preprocess_text(row.get("reasoning_ill_posed", ""))
        if ill_content:
            all_texts.append(ill_content)
            ill_texts.append(ill_content)
            labels.append(1)  # Without context

    stopwords = create_stopwords()
    vectorizer = CountVectorizer(ngram_range=(2, 4), stop_words=list(stopwords), min_df=5)
    X = vectorizer.fit_transform(ill_texts)
    print(f"Number of features before sub-gram pruning: {X.shape[1]}")
    pruned_vocab, _ = prune_redundant_subngrams(vectorizer, X)
    print(f"Number of features after sub-gram pruning: {len(pruned_vocab)}")
    
    vectorizer = CountVectorizer(ngram_range=(2, 4), stop_words=list(stopwords), vocabulary=pruned_vocab)
    X = vectorizer.fit_transform(ill_texts)
    feat = vectorizer.get_feature_names_out()

    for small in [3, 2]:
        for large in np.arange(4, small, -1):
            keep_idx = thin_grams_negation_parity(X, feat, small=small, large=large, threshold=0.8)
            X = X[:, keep_idx]
            feat = feat[keep_idx]
    
    pruned_vocab = {f: k for k, f in enumerate(feat)}
    print(f"Number of features after negation-parity thinning: {len(pruned_vocab)}")
    
    vectorizer = CountVectorizer(ngram_range=(2, 4), stop_words=list(stopwords), vocabulary=pruned_vocab)
    X = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()
    y = np.array(labels)
    
    # Initialize stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_importances = []
    accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # train stumps
        rf = RandomForestClassifier(
            n_estimators=500, 
            random_state=42, 
            class_weight='balanced',
        )
        rf.fit(X_train, y_train)

        # Collect feature importances
        feature_importances.append(rf.feature_importances_)

        # Evaluate
        y_pred = rf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # Average performance
    accuracy = np.mean(accuracies)

    # Collect top features from each fold
    top_features_per_fold = []
    for idx, fold_importances in enumerate(feature_importances):
        # Ensure feature_names and fold_importances have the same length
        if len(feature_names) != len(fold_importances):
            raise ValueError("Mismatch between feature names and importances lengths.")
        
        # Create DataFrame for the current fold
        fold_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': fold_importances
        }).sort_values(by='importance', ascending=False)

        # choose the index whose importance is 3 std dev above the mean
        mean_importance = fold_importance_df['importance'].mean()
        threshold = mean_importance
        k = fold_importance_df[fold_importance_df['importance'] >= threshold].shape[0]
        print(f"k for fold {idx + 1}: {k}")
        selected = fold_importance_df.head(k)["feature"].tolist()

        top_features_per_fold.append(set(selected))
    
    # Take the union of top features across all folds
    top_features = list(set.intersection(*top_features_per_fold))
    print(f"Number of top features after union: {len(top_features)}")


    return {
        "classification_accuracy": accuracy,
        "top_features": top_features
    }


class SoftUpperbound:
    """
    Class to compute a soft upper bound on classification power using n-grams. 
    To measure to power, we control the false positive rate at alpha, then measure the true positive rate.
    We use 2-fold CV to estimate this.
    """
    def __init__(self, data):
        self.data = data
        X, y, _ = preprocess(data)

        # Initialize stratified 2-fold CV
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        self.tpr_list = []
        self.fpr_list = []
        feature_importances = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf = RandomForestClassifier(
                n_estimators=500, 
                random_state=42, 
                class_weight='balanced'
            )
            rf.fit(X_train, y_train)

            # Collect feature importances
            feature_importances.append(rf.feature_importances_)

            # Evaluate
            y_scores = rf.predict_proba(X_test)[:, 1]  # probability for class '1' (Without Context)

            fpr, tpr, _ = roc_curve(y_test, y_scores)
            self.fpr_list.append(fpr)
            self.tpr_list.append(tpr)
        

    def get_soft_upperbound_at_alpha(self, alpha=0.05):
        """
        Get estimated true positive rate (power) at given false positive rate (alpha) as soft upper bound.
        """
        tpr_at_alpha = []
        for fpr, tpr in zip(self.fpr_list, self.tpr_list):
            # Find the TPR where FPR <= alpha
            idxs = np.where(fpr <= alpha)[0]
            if len(idxs) > 0:
                tpr_at_alpha.append(tpr[idxs[-1]])
            else:
                tpr_at_alpha.append(0.0)
        power = np.mean(tpr_at_alpha)
        return power
    

    def plot_roc_curve(self, save_path):
        plt.figure()
        for i, (fpr, tpr) in enumerate(zip(self.fpr_list, self.tpr_list)):
            plt.plot(fpr, tpr, label=f'Fold {i+1}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        # draw a line at alpha = 0.05
        plt.axvline(x=0.05, color='r', linestyle='--', label='Alpha = 0.05')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

def main():
    """Main function to run the classification."""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Ensure NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords')

    # Load data using your existing utility function
    # Note: For this script to run as a standalone, `load_data` must be defined
    # or mocked to return data in the expected format.
    train_data = []
    for model in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                   "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"]:
        train_data += load_data(model, args.data_name)

    print("\n--- Running Classification Analysis ---")
    classification_results = classify_thinking_content(train_data)
    print("--- Classification Analysis Complete ---")

    # save results to a file
    output_dir = f"classification_results/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.data_name}_classification_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(classification_results, f, indent=4)
    


if __name__ == "__main__":
    results = main()