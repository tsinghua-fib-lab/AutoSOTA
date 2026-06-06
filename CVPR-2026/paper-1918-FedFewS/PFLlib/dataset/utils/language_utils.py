"""Utils for language models."""

import re
import numpy as np
import json
from collections import Counter

# Try to import torchtext (optional, for backward compatibility)
try:
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    TORCHTEXT_AVAILABLE = True
except (ImportError, OSError):
    # OSError: handle Windows DLL loading issues
    TORCHTEXT_AVAILABLE = False


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def letter_to_index(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    print('num of letters (classes):', NUM_LETTERS)
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec

def basic_english_tokenize(text):
    """
    Basic English tokenizer (mimics torchtext's basic_english).

    Behavior:
    - Lowercase
    - Split on whitespace and punctuation
    - Keep contractions (e.g., "don't" stays as one token)
    """
    text = text.lower()
    # Split on whitespace but keep words and punctuation separate
    # This regex matches: words (with optional apostrophes), or single punctuation
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
    return tokens


def tokenizer_without_torchtext(text, max_len, max_tokens=32000):
    """
    Tokenize text and build vocabulary without torchtext dependency.

    This function mimics the behavior of torchtext's tokenizer but uses
    only standard Python libraries to avoid DLL compatibility issues.

    Args:
        text: List of text strings
        max_len: Maximum sequence length
        max_tokens: Maximum vocabulary size

    Returns:
        vocab: Dictionary mapping words to indices
        text_list: List of tokenized and padded sequences
    """
    # Special tokens
    specials = ['<pad>', '<cls>', '<unk>', '<eos>']

    # Tokenize all texts and count word frequencies
    all_tokens = []
    for t in text:
        tokens = basic_english_tokenize(t)
        all_tokens.extend(tokens)

    # Build vocabulary from most common tokens
    word_counts = Counter(all_tokens)
    # Reserve space for special tokens
    most_common = word_counts.most_common(max_tokens - len(specials))

    # Build vocab dictionary: word -> index
    vocab = {word: idx for idx, word in enumerate(specials)}
    for word, _ in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)

    # Special token indices
    pad_idx = vocab['<pad>']
    cls_idx = vocab['<cls>']
    unk_idx = vocab['<unk>']

    # Convert texts to token indices
    text_list = []
    for t in text:
        tokens = basic_english_tokenize(t)
        # Convert tokens to indices
        token_ids = [cls_idx] + [vocab.get(token, unk_idx) for token in tokens]

        # Pad or truncate to max_len
        if len(token_ids) < max_len:
            # Pad with pad_idx
            token_ids.extend([pad_idx] * (max_len - len(token_ids)))
        else:
            # Truncate
            token_ids = token_ids[:max_len]

        text_list.append(token_ids)

    return vocab, text_list


def tokenizer(text, max_len, max_tokens=32000):
    """
    Tokenize text with torchtext if available, otherwise use custom implementation.
    """
    if TORCHTEXT_AVAILABLE:
        tokenizer_fn = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(
            map(tokenizer_fn, iter(text)),
            specials = ['<pad>', '<cls>', '<unk>', '<eos>'],
            special_first = True,
            max_tokens = max_tokens
        )
        vocab.set_default_index(vocab['<unk>'])
        text_pipeline = lambda x: vocab(tokenizer_fn(x))

        text_list = []
        for t in text:
            tokens = [vocab['<cls>']] + text_pipeline(t)
            padding = [0 for i in range(max_len - len(tokens))]
            tokens.extend(padding)
            text_list.append(tokens[:max_len])
        return vocab, text_list
    else:
        # Fallback to custom implementation
        return tokenizer_without_torchtext(text, max_len, max_tokens)
