import gc
import random
import re
import time
import traceback
import ast

import numpy as np
import torch
import transformers

import modules.shared as shared
from modules.callbacks import (Iteratorize, Stream, _StopEverythingStoppingCriteria)

import matplotlib.pyplot as plt
import hashlib

import spacy

# Load the English language model
nlp = shared.nlp


def split_into_sentences(text):
    """
    Splits the input text into sentences using SpaCy's sentence segmentation.

    Parameters:
    - text (str): The text to split.

    Returns:
    - List[str]: A list of sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def get_words_in_sentence(sentence):
    """
    Extracts words from a sentence, excluding punctuation and spaces.

    Parameters:
    - sentence (str): The sentence to process.

    Returns:
    - List[str]: A list of words.
    """
    doc = nlp(sentence)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    return words


def secure_hash_for_word(word, range_min, range_max):
    """
    Generates a secure hash for a word and maps it to a number within a specified range.

    Parameters:
    - word (str): The word to hash.
    - range_min (int): The minimum value of the range.
    - range_max (int): The maximum value of the range.

    Returns:
    - int: The mapped number within [range_min, range_max].
    """
    hashed_word_bytes = hashlib.sha256(word.encode()).digest()
    hashed_word_int = int.from_bytes(hashed_word_bytes[:4], byteorder='big')
    mapped_number = (hashed_word_int % (range_max - range_min + 1)) + range_min
    return mapped_number


def secure_hash_for_sentence(sentence, range_min, range_max):
    """
    Generates a secure hash for a sentence (lemmatized, excluding stopwords and punctuation)
    and maps it to a number within a specified range.

    Parameters:
    - sentence (str): The sentence to hash.
    - range_min (int): The minimum value of the range.
    - range_max (int): The maximum value of the range.

    Returns:
    - int: The mapped number within [range_min, range_max].
    """
    doc = nlp(sentence)
    core_sentence = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    core_sentence_str = " ".join(core_sentence)

    hashed_sentence_bytes = hashlib.sha256(core_sentence_str.encode()).digest()
    hashed_sentence_int = int.from_bytes(hashed_sentence_bytes[:4], byteorder='big')
    mapped_number = (hashed_sentence_int % (range_max - range_min + 1)) + range_min
    return mapped_number


def secure_hash_to_numbers(last_sentence=None, last_word=None, range_list=[(0, 10), (0, 25)]):
    """
    Generates a list of numbers based on the last sentence and/or last word.
    - The first number is based on the sensorimotor class (word-based hashing).
    - The second number is based on the acrostic letter (sentence-based hashing).

    Parameters:
    - last_sentence (str or None): The last sentence if updating acrostic.
    - last_word (str or None): The last word if updating sensorimotor.
    - range_list (List[Tuple[int, int]]): List of ranges for sensorimotor and acrostic.

    Returns:
    - List[int]: A list containing [sensorimotor_class, acrostic_letter].
    """
    result_numbers = []

    # Sensorimotor class based on last word
    if last_word is not None:
        sensorimotor_hash = secure_hash_for_word(last_word, range_list[0][0], range_list[0][1])
    else:
        sensorimotor_hash = shared.secret_key[0]  # Reuse previous sensorimotor class if not updating
    result_numbers.append(sensorimotor_hash)

    # Acrostic letter based on last sentence
    if last_sentence is not None:
        acrostic_hash = secure_hash_for_sentence(last_sentence, range_list[1][0], range_list[1][1])
    else:
        acrostic_hash = shared.secret_key[1]  # Reuse previous acrostic letter if not updating
    result_numbers.append(acrostic_hash)

    return result_numbers


def get_last_word(reply):
    """
    Retrieves the last word from the current reply.

    Parameters:
    - reply (str): The cumulative reply text.

    Returns:
    - str or None: The last word if available, else None.
    """
    words = get_words_in_sentence(reply)
    return words[-1] if words else None


def get_last_sentence(reply):
    """
    Retrieves the last sentence from the current reply.

    Parameters:
    - reply (str): The cumulative reply text.

    Returns:
    - str or None: The last sentence if available, else None.
    """
    sentences = split_into_sentences(reply)
    return sentences[-1] if sentences else None


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply


def get_max_prompt_length(state):
    max_length = state['truncation_length'] - state['max_new_tokens']
    return max_length


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=True)

    if hasattr(shared.tokenizer, 'bos_token_id') and shared.tokenizer.bos_token_id is not None:
        if add_bos_token:
            if (len(input_ids[0]) > 0 and input_ids[0][0] != shared.tokenizer.bos_token_id) or len(input_ids[0]) == 0:
                # Add a missing bos token (it may not have been added due to faulty model metadata)
                bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]]).cuda()
                input_ids = torch.cat((bos_tensor, input_ids.cuda()), 1)

            # Prevent double bos token due to jinja templates with <s> somewhere
            while len(input_ids[0]) > 1 and input_ids[0][0] == shared.tokenizer.bos_token_id and input_ids[0][1] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
        else:
            # Remove any bos token that may have been added
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    return input_ids.cuda()


def decode(output_ids, skip_special_tokens=False):
    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)


def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def calc_greenlist_mask(scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
    # TODO lets see if we can lose this loop
    green_tokens_mask = torch.zeros_like(scores)
    for b_idx in range(len(greenlist_token_ids)):
        green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
    final_mask = green_tokens_mask.bool()
    return final_mask


def bias_greenlist_logits(scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
    scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
    return scores


def get_greenlist_ids(input_ids: torch.LongTensor) -> torch.Tensor:
    vocab_size = len(shared.vocab)
    vocab_permutation2 = torch.zeros(vocab_size, device='cuda:0')

    if shared.new_sentence:
        shared.new_sentence = False

        # ASCII values 65 to 90 represent uppercase letters (A to Z)
        alphabetic_characters = [chr(i) for i in range(65, 91)]

        target_char = alphabetic_characters[shared.secret_key[1]]

        # Create a list of booleans indicating which vocab items start with the target character
        matches = [1 if token.startswith(f' {target_char}') else 0 for token in shared.vocab_decode]

        # Create tensor from matches
        matches_tensor = torch.tensor(matches, device='cuda:0', dtype=torch.float32)

        # Multiply matches_tensor by shared.delta_acro
        vocab_permutation2 += matches_tensor * shared.delta_acro
    else:
        # Process sensorimotor tokens
        token_stripped_upper = [token.strip().upper() for token in shared.vocab_decode]

        matches = []
        for token in token_stripped_upper:
            if token in shared.sensorimotor:
                if shared.sensorimotor[token]["Dominant.sensorimotor"] == shared.classes[shared.secret_key[0]]:
                    matches.append(1)
                else:
                    matches.append(0)
            else:
                matches.append(0)

        matches_tensor = torch.tensor(matches, device='cuda:0', dtype=torch.float32)
        vocab_permutation2 += matches_tensor * shared.delta_senso

    # Watermarking feature added here
    # Apply watermarking delta_redgreen
    last_token = input_ids[-1].item()

    # Define the hash function
    def default_hash_fn(token_id):
        return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

    # Seed the random generator based on the last token
    seed = default_hash_fn(last_token)
    generator = torch.Generator(device='cuda:0').manual_seed(seed)

    # Generate a random permutation of the vocabulary
    gli = torch.randperm(vocab_size, generator=generator, device='cuda:0')

    # Define the proportion of the green list (gamma)
    gamma = 0.5  # Adjust gamma as needed
    gls = int(gamma * vocab_size)  # Green list size

    # Get the indices of the green list tokens
    green_list_indices = gli[:gls]

    # Adjust delta values for the green list tokens using delta_redgreen
    vocab_permutation2[green_list_indices] += shared.delta_redgreen

    return vocab_permutation2


def boost_tokens_with_a(input_ids, scores, **kwargs):
    vocab_size = len(shared.vocab)
    batch_size = input_ids.shape[0]

    # Initialize a tensor of zeros directly on GPU
    permutation_tensor = torch.zeros((batch_size, vocab_size), device='cuda:0')

    # Only calculate get_greenlist_ids for the latest batch index
    latest_b_idx = batch_size - 1
    greenlist_ids = get_greenlist_ids(input_ids[latest_b_idx])

    # Set the last batch greenlist
    permutation_tensor[latest_b_idx] = greenlist_ids

    # Add the permutation tensor to the scores
    scores[:] = scores[:] + permutation_tensor[:]

    return scores


def generate_reply(question, state, eos_token=None, stopping_strings=None):
    clear_torch_cache()
    done = False
    seed = set_manual_seed(state['seed'])
    generate_params = {}

    original_question = question

    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'smoothing_factor', 'smoothing_curve', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'dry_multiplier', 'dry_base', 'dry_allowed_length', 'dry_sequence_breakers']:
        if k in state:
            generate_params[k] = state[k]

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))

    original_input_ids = input_ids

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids

    # Add the encoded tokens to generate_params
    generate_params.update({'inputs': input_ids})

    # Create the StoppingCriteriaList with the stopping strings
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    generate_params.update({"logits_processor": transformers.LogitsProcessorList([boost_tokens_with_a])})

    # Adjust shared vocab and decode
    shared.vocab = list(range(len(shared.tokenizer.get_vocab())))

    shared.vocab_decode = []
    for word in shared.vocab:
        decoded_token = shared.tokenizer.decode(word, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        shared.vocab_decode.append(decoded_token)

    reply = ""
    t0 = time.time()
    try:
        stream = True
        if not stream:
            # Generate the entire reply at once.
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                output = output.cuda()

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            reply = get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        else:
            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs, pad_token_id=shared.tokenizer.pad_token_id)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        done = True
                        break
                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)

                    # Detect if a sentence has ended
                    decoded_token = decode(output[-1], skip_special_tokens=True)

                    if decoded_token in {".", "!", "?"}:
                        shared.new_sentence = True
                        last_sentence = get_last_sentence(reply)

                        # Update the secret key based on the last sentence
                        if last_sentence:
                            shared.secret_key = secure_hash_to_numbers(last_sentence=last_sentence, last_word=None, range_list=[(0, 10), (0, 25)])

                    # Update the secret key based on the last word
                    last_word = get_last_word(reply)
                    if last_word:
                        shared.secret_key = secure_hash_to_numbers(last_sentence=None, last_word=last_word, range_list=[(0, 10), (0, 25)])

                    # Handle special Unicode character (if any)
                    if chr(0xfffd) in new_content:
                        continue

                    reply += new_content
                    starting_from = len(output)

    except Exception:
        done = True
        traceback.print_exc()

    finally:
        # Find the stopping strings
        all_stop_strings = []
        for st in (stopping_strings, state['custom_stopping_strings']):
            if type(st) is str:
                st = ast.literal_eval(f"[{st}]")

            if type(st) is list and len(st) > 0:
                all_stop_strings += st

        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)

        shared.acrostic = 0
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - original_tokens
        return reply, done
