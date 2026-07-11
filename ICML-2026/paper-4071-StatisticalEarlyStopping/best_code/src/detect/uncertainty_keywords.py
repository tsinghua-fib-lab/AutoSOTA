from audioop import add
import json

# ---------------------------------------------------------------------
# Substring patterns for each category
# (ordered by diagnostic strength; category precedence handled later)
# ---------------------------------------------------------------------

# read features from file classification_results/gsm8k_classification_results.jsonl

with open("classification_results/gsm8k_classification_results.json", "r") as f:
    # Each line is a JSON object with a "features" field
    data = json.load(f)
    features = data["top_features"]
    

KEYWORDS_DICT = {
    "Impossibility": {
        "core": [
            "not possible", "impossible", "cannot", "hard", 
        ],
        "keywords": []
    },
    "Speculation": {
        "core": [
            "guess", "assum", "forgot", "intend"
        ],
        "keywords": []
    },
    "Insufficiency": {
        "core": [
            "missing", "insufficient", "incomplete", "without", "additional", "lack", 
            "no data", "no info", "not helpful",
            "not give", "not specif", "not recall", "not provide", "not include", "not enough", "not access",
            'absence', 'ambiguous', 'vague'
        ],
        "keywords": []
    },
}

ABLATION_DICT = {
    "Epistemic Uncertainty": {
        "core": [
            "maybe", "perhaps"
        ],
        "keywords": []
    },
    "Transition": {
        "core": [
            "alternatively", "wait"
        ],
        "keywords": []
    },
}

# organize the keywords into categories and create a dictionary of "category to keywords"
for kw in features:
    for key, pattern in KEYWORDS_DICT.items():
        if any(tok in kw for tok in pattern["core"]):
                pattern["keywords"].append(kw)

for kw in features:
    for key, pattern in ABLATION_DICT.items():
        if any(tok in kw for tok in pattern["core"]):
                pattern["keywords"].append(kw)

def get_uncertainty_keywords(ablation = "none"):
    '''
    Docstring for get_uncertainty_keywords
    
    :param ablation: str, one of ["none", "loo"]
        - "none": return the full set of keywords from all categories
        - "loo": return a list of keyword sets, each excluding one category
    :return: list of tuples (keyword_set, category)
    '''
    if ablation == "none":
        # merge all keywords from all categories
        keywords_list = []
        for category in KEYWORDS_DICT.values():
            keywords_list.extend(category["keywords"])
        return list(set(keywords_list))
    elif ablation == "loo":
        # return a list of keyword sets, each excluding one category
        result = []
        for key in KEYWORDS_DICT.keys():
            keywords_list = []
            for k, category in KEYWORDS_DICT.items():
                if k != key:
                    keywords_list.extend(category["keywords"])
            result.append((list(set(keywords_list)), key))
        return result
    else:
        raise ValueError(f"Unknown ablation type: {ablation}")

if __name__ == "__main__":
    for key, category in KEYWORDS_DICT.items():
        print(f"Category: {key}")
        # sample 10 random keywords with seed 42
        import random
        random.seed(42)
        sampled_keywords = random.sample(category["keywords"], min(10, len(category["keywords"])))
        print(f"Sampled Keywords: {sampled_keywords}\n")
    
    print("Full keyword set:")
    full_keywords = get_uncertainty_keywords(ablation="none")
    # print the coverage of keyword set as a percentage of total features
    print(f"Total keywords: {len(full_keywords)} / {len(features)} ({len(full_keywords) / len(features) * 100:.2f}%)")

    # print five examples of keywords not covered
    uncovered_keywords = list(set(features) - set(full_keywords))

    print(f"Uncovered Keywords (sample 5): {random.sample(uncovered_keywords, min(5, len(uncovered_keywords)))}\n")

    print(f"Uncovered Keywords: {uncovered_keywords}")