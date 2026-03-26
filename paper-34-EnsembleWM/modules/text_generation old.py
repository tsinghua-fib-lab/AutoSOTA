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
from modules.callbacks import (Iteratorize, Stream,_StopEverythingStoppingCriteria)


import matplotlib.pyplot as plt
import hashlib

import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")
start = 0


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
    # if shared.model_type == 'HF_seq2seq':
    #     reply = decode(output_ids, state['skip_special_tokens'])
    # else:
    #     new_tokens = len(output_ids) - len(input_ids[0])

        

    #     reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])

    #     print(reply)

        # Prevent LlamaTokenizer from skipping a space
        # if type(shared.tokenizer) is transformers.LlamaTokenizer and len(output_ids) > 0:
        #     if shared.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
        #         reply = ' ' + reply


    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    #print(reply)
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
                bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]])
                input_ids = torch.cat((bos_tensor, input_ids), 1)

            # Prevent double bos token due to jinja templates with <s> somewhere
            while len(input_ids[0]) > 1 and input_ids[0][0] == shared.tokenizer.bos_token_id and input_ids[0][1] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
        else:
            # Remove any bos token that may have been added
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # This is a hack for making replies more creative.
    #if not add_bos_token and input_ids[0][0] == shared.tokenizer.bos_token_id:
    #    input_ids = input_ids[:, 1:]
    


    # Llama adds this extra token when the first character is '\n', and this
    # compromises the stopping criteria, so we just remove it
    #if type(shared.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
    #    input_ids = input_ids[:, 1:]

    #///////////////////////////////////////////////////////////////////////// i commented this out for testng llama 3, dont forget

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    return input_ids.cuda()


def decode(output_ids, skip_special_tokens=False):
    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens,clean_up_tokenization_spaces=False)




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

        #scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias[greenlist_mask]
        return scores


def get_greenlist_ids(input_ids: torch.LongTensor) -> list[float]:


        vocab_permutation = list(range(len(shared.vocab)))
        greenlist_size = 0

        vocab_permutation2 = [0] * len(shared.vocab) #[(n, 0) for n in range(len(shared.vocab))]
        #vocab_dict = {value: 0 for value in shared.vocab} 

        # Check if input contains special characters
        # special_tokens = [
        #     shared.tokenizer.pad_token_id,  # Padding token
        #     shared.tokenizer.eos_token_id,  # End-of-sequence token
        #     shared.tokenizer.bos_token_id   # Start-of-sequence token
        # ]
        # Decode input tokens to check if special characters are present
        #TODO: is this really required? this is supposed to not change any logits if current token is special on
        #for token_id in input_ids:
        #    decoded_token = shared.tokenizer.decode([token_id], skip_special_tokens=False,clean_up_tokenization_spaces=False)
        #    # If it's a special token, skip the function and return an empty list
        #    if token_id in special_tokens or decoded_token in ['▁', 'Ġ']:  # Add more as needed
        #        return [0] * len(shared.vocab)  # Return empty or default vocab_permutation2


        if(shared.new_sentence == True):
            
            shared.new_sentence = False

            #ASCII values 65 to 90 represent uppercase letters (A to Z), and values 97 to 122 represent lowercase letters (a to z)
            alphabetic_characters = [chr(i) for i in range(65, 91)] #+ [chr(i) for i in range(97, 123)]

            #print(f'''//////////// {alphabetic_characters[shared.secret_key[1]]}''')
            i = 0
            for word in shared.vocab:
                # decoded_token = shared.vocab_decode[i]
                # if decoded_token.startswith('▁') or decoded_token.startswith('Ġ'):
                #     decoded_token = decoded_token[1:]

                
                # if shared.vocab_decode[i].startswith(alphabetic_characters[shared.secret_key[1]]):
                #the new sentence should have a whitespace as first token....
                if shared.vocab_decode[i].startswith(f' {alphabetic_characters[shared.secret_key[1]]}'):                    
                    vocab_permutation[greenlist_size] = word
                    vocab_permutation2[i] = shared.delta_acro
                    greenlist_size += 1
                i += 1
                
            #print(f'''acro size {greenlist_size}''')
        else:
            i = 0
            for word in shared.vocab:
                if shared.vocab_decode[i].strip().upper() in shared.sensorimotor:

                    #this is old behaviour with mean values
                    #if (shared.sensorimotor[shared.vocab_decode[i].upper()][shared.classes[shared.secret_key[0]]] > 2.0):
                    if (shared.sensorimotor[shared.vocab_decode[i].strip().upper()]["Dominant.sensorimotor"] == shared.classes[shared.secret_key[0]]):
                        vocab_permutation[greenlist_size] = word

                        vocab_permutation2[i] = shared.delta_senso
                        greenlist_size += 1

                                    
                i += 1
            #print(f'''greenlist size {greenlist_size}''')
            
        # Watermarking feature added here
        # Apply watermarking delta_redgreen
        last_token = input_ids[-1].item()

        # Define the hash function
        def default_hash_fn(token_id):
            return int(hashlib.sha256(str(token_id).encode('utf-8')).hexdigest(), 16) % (10 ** 8)

        # Seed the random generator based on the last token
        seed = default_hash_fn(last_token)
        generator = torch.Generator().manual_seed(seed)

        # Generate a random permutation of the vocabulary
        vocab_size = len(shared.vocab)
        gli = torch.randperm(vocab_size, generator=generator)

        # Define the proportion of the green list (gamma)
        gamma = 0.5  # Adjust gamma as needed
        gls = int(gamma * vocab_size)  # Green list size

        # Get the indices of the green list tokens
        green_list_indices = gli[:gls]

        # Adjust delta values for the green list tokens using delta_redgreen
        for idx in green_list_indices:
            vocab_permutation2[idx] += shared.delta_redgreen

        #print(f'''watermark greenlist size {gls}''')

        return vocab_permutation2
        #return greenlist_ids 
        

def boost_tokens_with_a(input_ids, scores, **kwargs):
   
    # batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

    # for b_idx in range(input_ids.shape[0]):
    #         #greenlist_ids = get_greenlist_ids(input_ids[b_idx])


    #         greenlist_ids = get_greenlist_ids(input_ids[b_idx])
    #         batched_greenlist_ids[b_idx] = greenlist_ids



    # permutation_tensor = torch.as_tensor(batched_greenlist_ids)
    # permutation_tensor = permutation_tensor.to("cuda:0")

    # print(input_ids.shape)
    # scores[:] = scores[:] + permutation_tensor[:] 
    # return scores
        # Initialize all batched_greenlist_ids with [0] * len(shared.vocab)
    vocab_size = len(shared.vocab)
    batched_greenlist_ids = [[0] * vocab_size for _ in range(input_ids.shape[0])]

    # Only calculate get_greenlist_ids for the latest b_idx (the last one)
    latest_b_idx = input_ids.shape[0] - 1
    greenlist_ids = get_greenlist_ids(input_ids[latest_b_idx])
    
    # Set the last batch greenlist
    batched_greenlist_ids[latest_b_idx] = greenlist_ids

    # Convert batched_greenlist_ids to a tensor and move to GPU
    permutation_tensor = torch.as_tensor(batched_greenlist_ids).to("cuda:0")

    # Add the permutation tensor to the scores
    scores[:] = scores[:] + permutation_tensor[:]

    return scores

#max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, 
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
    #output = input_ids[0]

    # Find the eos tokens
    #eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    #if eos_token is not None:
    #    eos_token_ids.append(int(encode(eos_token)[0][-1]))
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids

    # Add the encoded tokens to generate_params
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})


    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    # stopping_criteria_list = transformers.StoppingCriteriaList()

    # for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
    #     if type(st) is list and len(st) > 0:
    #         sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
    #         stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
    #         break
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())
    
    ##logits code
    generate_params.update({"logits_processor": transformers.LogitsProcessorList([boost_tokens_with_a])})

    #this does not work anymore since rust tokenizer implementations do not order their vocab by token id, fix is to instead create a range() of length of vocab
    #shared.vocab=list(shared.tokenizer.get_vocab().values())
    shared.vocab = list(range(len(shared.tokenizer.get_vocab())))
    
    shared.vocab_decode = []
    for word in shared.vocab:
        decoded_token = shared.tokenizer.decode(word, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        #better do this inside the logits function
        #  if decoded_token.startswith('▁') or decoded_token.startswith('Ġ'):
        #             decoded_token = decoded_token[1:]
                    
        shared.vocab_decode.append(decoded_token)#,state['skip_special_tokens']))

  
   
    reply = ""
    t0 = time.time()
    try:
        stream = True
        if not stream:
            # Generate the entire reply at once.
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                output = output.cuda()

            #reply = get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=True)
            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            reply = get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:
            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs, pad_token_id=shared.tokenizer.pad_token_id)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''#is reply in our case
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    #print((decode(output[-1],state['skip_special_tokens'])))
                    if output[-1] in eos_token_ids:
                        done = True
                        break
                    #reply = get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=True)
                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                        

                    # Detect if a sentence has ended
                    decoded_token = decode(output[-1], skip_special_tokens=True)
                    
                    if decoded_token in {".", "!", "?"}:
                        shared.new_sentence = True
                        last_sentence = get_last_sentence(reply)
                        #print("------------------found end of sentence, last sentence is:")
                        #print(last_sentence)


                        # Update the secret key based on the last sentence
                        if last_sentence:
                            shared.secret_key = secure_hash_to_numbers(last_sentence=last_sentence, last_word=None, range_list=[(0, 10), (0, 25)])
                            #print(f"Updated secret_key based on sentence: {shared.secret_key}")
            

                    # Update the secret key based on the last word
                    last_word = get_last_word(reply)
                    if last_word:
                        shared.secret_key = secure_hash_to_numbers(last_sentence=None, last_word=last_word, range_list=[(0, 10), (0, 25)])
                        #print(last_word)
                        #print(f"Updated secret_key based on word: {shared.secret_key}")
                    
                    # Handle special Unicode character (if any)
                    if chr(0xfffd) in new_content:
                        continue

                    reply += new_content
                    starting_from = len(output)
                    
                    

    except Exception:
        done = True
        traceback.print_exc()

    finally:  
        #new_tokens = len(output) - len(input_ids[0])
        #reply = decode(output[-new_tokens:], state['skip_special_tokens'])
        #print(reply)

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
        #print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        cleaned_string = reply.replace("\n", "").rstrip()
        return reply, done

