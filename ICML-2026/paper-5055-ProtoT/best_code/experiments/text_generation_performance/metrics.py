import math
from collections import Counter

def _get_ngrams(segment, max_order):
    """Return n-gram counts up to `max_order`."""
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def calculate_bleu(reference, hypothesis, max_order=4):
    """Calculate a BLEU score for one hypothesis/reference pair."""
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    if len(hypothesis_tokens) == 0:
        return 0.0

    clipped_ngram_counts = Counter()
    for order in range(1, max_order + 1):
        ref_ngrams = _get_ngrams(reference_tokens, order)
        hyp_ngrams = _get_ngrams(hypothesis_tokens, order)
        
        for ngram, count in hyp_ngrams.items():
            clipped_ngram_counts[ngram] += min(count, ref_ngrams[ngram])

    def _ngram_precision(order):
        hyp_ngrams = _get_ngrams(hypothesis_tokens, order)
        total_count = sum(hyp_ngrams.values())
        if total_count == 0:
            return 0.0
        
        clipped_count = 0
        for ngram, count in hyp_ngrams.items():
            if len(ngram) == order:
                clipped_count += min(count, _get_ngrams(reference_tokens, order)[ngram])
        
        return clipped_count / total_count

    precisions = [_ngram_precision(order) for order in range(1, max_order + 1)]
    
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    return bp * geo_mean

def calculate_rouge_l(reference, hypothesis):
    """Calculate ROUGE-L for one hypothesis/reference pair."""
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    if len(reference_tokens) == 0 or len(hypothesis_tokens) == 0:
        return 0.0

    m = len(reference_tokens)
    n = len(hypothesis_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference_tokens[i-1] == hypothesis_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    
    beta = 1.2
    f1 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)
    
    return f1
