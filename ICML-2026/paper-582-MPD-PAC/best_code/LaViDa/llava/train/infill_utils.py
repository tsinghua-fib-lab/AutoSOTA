
import numpy as np

INFILL_TOKEN = '<|reserved_token_1|>' # 126085
FILL_TOKEN = '<|reserved_token_2|>' # 126086

def count_num_words(s):
    return len(s.split())
def insert_infill_substrings(s, N=4, K=5):
    words = s.split()
    n = np.random.randint(1, N + 1)  # number of insertions

    if len(words) < 2 or n == 0:
        return s

    insert_positions = sorted(np.random.choice(
        range(1, len(words)), size=min(n, len(words) - 1), replace=False
    ))

    result = []
    for i, word in enumerate(words):
        result.append(word)
        if i + 1 in insert_positions:
            k = np.random.randint(0, K + 1)
            if k > 0:
                tokens = ''.join([FILL_TOKEN] * k + [INFILL_TOKEN])
                result.append(tokens)
            else:
                result.append(INFILL_TOKEN)

    return ' '.join(result)

# Example usage

if __name__ == '__main__':
    # np.random.seed(0)
    s = "The quick brown fox jumps over the lazy dog"
    print(insert_infill_substrings(s, N=3, K=4))
    print(count_num_words(s))