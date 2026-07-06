"""Shared utilities for the flashtrace project.

This module contains common constants, NLP pipeline initialization,
and sentence processing functions used across multiple modules.
"""

import re
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import torch

# Common constants
DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context:{context}\n\n\nQuery: {query}"

# Sentence detector - NLP pipeline initialization
try:
    nlp = spacy.load("en_core_web_sm")
    _newline_pipe_position = {"before": "parser"}
except OSError:
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _newline_pipe_position = {"after": "sentencizer"} if "sentencizer" in nlp.pipe_names else {"last": True}


@Language.component("newline_cap_split")
def newline_cap_split(doc: Doc) -> Doc:
    """Custom component to split on capitalized words after newline."""
    for i, token in enumerate(doc):
        if token.is_title and i > 0:
            prev_token = doc[i - 1]
            if "\n" in prev_token.text or (prev_token.is_space and "\n" in prev_token.text):
                token.is_sent_start = True
    return doc


# Add to pipeline
nlp.add_pipe("newline_cap_split", **_newline_pipe_position)


def create_sentences(text, tokenizer, return_indices=False, show=False) -> list[str]:
    """Split text into sentences and return the sentences.

    Args:
        text: The text to split into sentences.
        tokenizer: The tokenizer to use for EOS token handling.
        return_indices: If True, return both sentences and their indices.
        show: Unused, kept for backward compatibility.

    Returns:
        A list of sentences, or a tuple of (sentences, indices) if return_indices is True.
    """
    sentences = []
    separators = []
    indices = []

    # Process the text with spacy
    doc = nlp(text)

    # Extract sentences
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    # extract separators
    cur_start = 0
    for sentence in sentences:
        indices.append(cur_start)
        cur_end = text.find(sentence, cur_start)
        separator = text[cur_start:cur_end]
        separators.append(separator)
        cur_start = cur_end + len(sentence)

    # combine the separators with the sentences properly
    for i in range(len(sentences)):
        if separators[i] == "\n":
            sentences[i] = sentences[i] + separators[i]
        else:
            sentences[i] = separators[i] + sentences[i]

    # if the text had an eos token (generated text) it will be missed
    # and attached on the last sentence, so we manually handle it
    if tokenizer is not None:
        eos = tokenizer.eos_token
        if eos and sentences and eos in sentences[-1]:
            sentences[-1] = sentences[-1].replace(eos, "")
            indices.append(len("".join(sentences)))
            sentences.append(eos)

    indices.append(len(text))

    if return_indices:
        return sentences, indices
    else:
        return sentences


def create_sentences_fallback(text, tokenizer=None) -> list[str]:
    """Very naive fallback sentence splitter for when spacy is unavailable.

    Split by newline first, then by simple punctuation boundaries.
    """
    parts = []
    for block in text.split("\n"):
        xs = re.split(r"(?<=[.!?])\s+", block.strip()) if block.strip() else []
        parts.extend([x for x in xs if x])
    return parts or ([text] if text else [])


def create_sentence_masks(tokens, sentences, show=False) -> torch.Tensor:
    """Create a binary mask of shape [sentences, tokens].

    Each row has a 1 where a token is in the represented sentence.

    Args:
        tokens: List of tokens.
        sentences: List of sentences.
        show: Unused, kept for backward compatibility.

    Returns:
        A tensor mask of shape [len(sentences), len(tokens)].
    """
    mask = torch.zeros((len(sentences), len(tokens)))

    sentence_idx = 0
    sent_pointer = 0

    for token_idx, token in enumerate(tokens):
        current_sentence = sentences[sentence_idx]

        mask[sentence_idx, token_idx] = 1

        if '\n' in token:
            sent_pointer += len(token) + 1
        else:
            sent_pointer += len(token)

        if sent_pointer >= len(current_sentence):
            sentence_idx += 1
            sent_pointer = 0

        if sentence_idx >= len(sentences):
            break

    return mask
