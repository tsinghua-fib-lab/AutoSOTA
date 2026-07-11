# read jsonl files
import json
import os
from typing import List
from sklearn.model_selection import train_test_split
import re
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer

def read_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def load_data(model_name, data_name, split = 'train', results_path='results', indices = False):
    """Load data from the specified path."""
    path = f"{results_path}/{model_name}/{data_name}_results.jsonl"
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        return None
    data = read_jsonl(path)

    if split in ['train', 'test', 'both']:
        train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)
        if indices:
            train_indices = [data.index(item) for item in train_data]
            test_indices = [data.index(item) for item in test_data]
            if split == 'train':
                return (train_data, train_indices)
            elif split == 'test':
                return (test_data, test_indices)
            else:
                return (train_data, train_indices, test_data, test_indices)
        if split == 'train':
            return train_data
        elif split == 'test':
            return test_data
        else:
            return train_data, test_data
    elif split == 'full':
        if indices:
            return data, [data.index(item) for item in data]
        return data
    else:
        raise ValueError(f"Unknown split: {split}")
    
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("’", "'")
    # map can't and couldn't to cannot
    text = re.sub(r"\b(can't|couldn't)\b", " cannot ", text, flags=re.IGNORECASE)
    # map other negations to not
    text = re.sub(
        r"\b(don't|doesn't|didn't|won't|wouldn't|shouldn't|isn't|aren't|weren't|wasn't|hasn't|haven't|hadn't|mightn't|mustn't|needn't)\b",
        " not ",
        text,
        flags=re.IGNORECASE,
    )
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def create_stopwords():
    """Create a comprehensive set of stopwords."""
    # Ensure to import nltk and download stopwords if not already done
    stopwords = set()
    
    # remove common English stopwords
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        nltk_words = set(nltk_stopwords.words('english'))
        # remove negation words from nltk stopwords
        stopwords.update(nltk_words)
    except ImportError:
        print("NLTK not installed. Skipping NLTK stopwords.")
    except LookupError:
        print("NLTK stopwords not found. Skipping NLTK stopwords.")
    # Numbers
    stopwords.update({str(i) for i in range(1000)})
    # remove negation words from stopwords
    negation_words = {"no", "not", "never", "none"}
    stopwords.difference_update(negation_words)
    # remove any words in stopwords that contains n't
    stopwords = {word for word in stopwords if "n't" not in word}
    return stopwords

def find_keyword_positions(text, keywords, debug=False):
    """
    Finds starting word indices of keyword occurrences (multi-word supported).
    Greedy longest match; skips ahead, so no overlaps in output.
    """

    stopwords = create_stopwords()
    # --- preprocess & tokenize ---
    clean_text = preprocess_text(text).strip()
    analyzer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", ngram_range=(1,1), stop_words=list(stopwords),).build_analyzer()
    clean_text = ' '.join(analyzer(clean_text))
    tokens = clean_text.split()

    # Prepare keyword sequences
    seqs = []
    for kw in keywords or []:
        s = preprocess_text(kw).strip()
        if s:
            seq = s.split()
            if seq:
                seqs.append(seq)
    
    # remove duplicate sequences by converting to set and back to list
    seqs = list(map(list, set(tuple(seq) for seq in seqs)))

    # --- build a token trie ---
    # node structure: {"_end": bool, "_ch": {token: child_node}}
    root = {"_end": False, "_ch": {}}

    def insert(seq):
        node = root
        for t in seq:
            node = node["_ch"].setdefault(t, {"_end": False, "_ch": {}})
        node["_end"] = True

    for seq in seqs:
        insert(seq)

    # --- single pass scan: longest match at each i, then jump ---
    positions = []
    n = len(tokens)
    i = 0
    while i < n:
        node = root
        j = i
        last_match_len = 0

        while j < n:
            t = tokens[j]
            nxt = node["_ch"].get(t)
            if not nxt:
                break
            node = nxt
            if node["_end"]:
                last_match_len = j - i + 1  # longest-so-far
            j += 1

        if last_match_len > 0:
            positions.append(i)
            GAP = 5
            i += max(last_match_len, GAP)   # skip matched span (no overlaps)
        else:
            i += 1
            
    if debug:
        # print all matches and their positions
        for pos in positions:
            print(f"Match at position {pos}: {' '.join(tokens[pos:min(pos + 5, n)])}")
        print("\n")

    return positions, len(tokens)

def tokenize_string(string: str, tokenizer: AutoTokenizer):
    """
    Tokenize the given string using the specified model's tokenizer.
    :param string: The string to tokenize.
    :param model_name: The name of the model to use for tokenization.
    :return: A list of token IDs.
    """
    encoded = tokenizer(string, truncation=False, add_special_tokens=False)
    return encoded['input_ids']

def count_tokens(text, model_name):
    """Count the number of tokens in a text string using the specified tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenize_string(text, tokenizer)
    return len(tokens)

def save_jsonl_for_simulation(file_path, new_data):
    """
    Save simulation data to a JSONL file by simply appending.
    
    Args:
        file_path: Path to the JSONL file
        new_data: New data to save (dict)
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Append new entry
    with open(file_path, 'a') as f:
        f.write(json.dumps(new_data) + '\n')

if __name__ == "__main__":
    # test tokenize_string
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    long_string = "Hello world " * 5000
    tokens = tokenize_string(long_string, tokenizer)
    print(len(tokens))   # same total length as full_tokens, still flat list

    