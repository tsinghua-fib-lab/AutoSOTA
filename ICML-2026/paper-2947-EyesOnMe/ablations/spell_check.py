import re
from spellchecker import SpellChecker
import json
import tqdm
import random

def analyze_document_spelling(text):
    """Compute OOV rate and symbol ratio for a given text."""
    spell = SpellChecker()

    # 1. Symbol ratio (Symbol-to-Word Ratio)
    symbols = re.findall(r'[^a-zA-Z0-9\s]', text)
    symbol_count = len(symbols)

    # 2. Extract and normalize words (pure ASCII alphabetic tokens, lowercased)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_words = len(words)

    if total_words == 0:
        return {"error": "No valid words found in document."}

    # 3. Spell check on unique words for efficiency
    unique_words = set(words)
    misspelled_unique = spell.unknown(unique_words)

    # Count total occurrences of misspelled words (weighted by frequency)
    misspelled_total_count = sum(1 for word in words if word in misspelled_unique)

    # 4. Compute metrics
    oov_rate = (misspelled_total_count / total_words) * 100
    symbol_ratio = (symbol_count / total_words) * 100 if total_words > 0 else 0

    return {
        "Total Words": total_words,
        "Total Symbols": symbol_count,
        "Misspelled Count": misspelled_total_count,
        "OOV Rate (%)": round(oov_rate, 2),
        "Symbol Ratio (%)": round(symbol_ratio, 2),
        "Sample Misspelled Words": list(misspelled_unique)[:10],
    }


if __name__ == "__main__":
    filepath = "../data/passages/marco_human_passages.json"
    with open(filepath, 'r') as f:
        documents = random.sample(json.load(f), 100)

    OOV_rates = []
    symbol_ratios = []
    avg_word_cnt = []
    for document in tqdm.tqdm(documents):
        stats = analyze_document_spelling(document)
        OOV_rates.append(stats["OOV Rate (%)"])
        symbol_ratios.append(stats["Symbol Ratio (%)"])
        avg_word_cnt.append(stats["Total Words"])

    print("Average OOV Rate: ", sum(OOV_rates) / len(OOV_rates))
    print("Average Symbol Ratio: ", sum(symbol_ratios) / len(symbol_ratios))
    print("Average Word Count: ", sum(avg_word_cnt) / len(avg_word_cnt))

    exp_filepath = "../results/attention_passage_list_llm+ours.json"
    with open(exp_filepath, 'r') as f:
        experiments = json.load(f)

    OOV_rates = []
    symbol_ratios = []
    avg_word_cnt = []
    for experiment in tqdm.tqdm(experiments):
        stats = analyze_document_spelling(experiment["passage"])
        OOV_rates.append(stats["OOV Rate (%)"])
        symbol_ratios.append(stats["Symbol Ratio (%)"])
        avg_word_cnt.append(stats["Total Words"])

    print("Average OOV Rate: ", sum(OOV_rates) / len(OOV_rates))
    print("Average Symbol Ratio: ", sum(symbol_ratios) / len(symbol_ratios))
    print("Average Word Count: ", sum(avg_word_cnt) / len(avg_word_cnt))
