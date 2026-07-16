def generate_random_sequence_string(num_sequence, length, vocab_size, start_range):
    from src.unstealthy.score import get_random_sequences
    random_sequence = get_random_sequences(num_sequence, length, vocab_size, start_range)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    random_sequence = tokenizer.batch_decode(random_sequence)
    return random_sequence

#creates the jsonl file, and returns the prop_inputs to be turned into a csv
def edit_json_unstealthy_scaling(orig_jsonl, new_jsonl, watermarks, k, info):
    import json
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    # counts the total length of the document in a VERY inefficient way
    tot_len = 0
    with open(orig_jsonl, "r") as orig_file:
        tot_len = sum(1 for _ in orig_file)

    #boundary = 10000 for 100M datasets, 100000 for >1B datasets
    boundary = 10000

    # the following is for replicability of document insertions
    #generate a starting document to perturb for each watermark
    #note that since the pile is already shuffled, and neox shuffles the training set for us, we just select a starting document to start perturbing
    #We want to make sure that there is no overlap between the watermarks
    #we want the documents selected to be consistent within each seed, so we hardcode the upper limit to be 100000, and always generate 1000000 numbers
    #note that 1e9 tokens corresponds to about 565832 documents
    assert(tot_len >= boundary) #assume that the number of documents in the datset is more than 100000
    assert(len(watermarks) * k <= boundary) #assume that the number of documents needed to watermark is less than 100000
    random_dictionary = np.random.choice(boundary, size=boundary, replace=False)
    perturbed_instances = random_dictionary[:len(watermarks) * k]

    data = []

    #begin creating jsonl output
    with open(orig_jsonl, "r") as orig_file, open(new_jsonl, "w") as new_file:
        for ind, line in tqdm(enumerate(orig_file), total=tot_len):

            if (ind in perturbed_instances):
                watermark_ind = perturbed_instances.tolist().index(ind)
                watermark = watermarks[watermark_ind // k]
                line = json.loads(line)
                line["text"] = line["text"] + "\n" + watermark
                line["order"] = watermark
                row = []
                row.append(ind)
                row.append(line["text"])
                row.append(len(line["text"]) - info["watermark_length"])
                row.append(info["watermark_length"])
                row.append(info["vocab_size"])
                row.append(watermark)
                data.append(row)
                new_file.write(json.dumps(line) + "\n")
            else:
                new_file.write(line)

    prop_inputs = pd.DataFrame(data)
    print(f"prop_inputs has {len(prop_inputs)} number of perturbed examples! ")
    prop_inputs.columns = ['example_index', 'text', 'sub_index', 'seq_len', 'vocab_size', 'watermark']
    return prop_inputs

def edit_json_unicode(group_folder, orig_jsonl, new_jsonl, seed, k, null_n_seq):
    """
    This function serves as a main function to edit the jsonl file for the unicode properties experiment
    :param group_folder: experimental group NOT some folder
    :param orig_jsonl: the original jsonl file
    :param new_jsonl: the new edited jsonl file
    :param seed: the seed used to perturb each document in the dataset
    :param k: how many documents to swap with unicode lookalikes
    """
    import numpy as np
    from hashlib import sha256
    from functools import lru_cache
    import json
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    #initialize the unicode pairs
    unicode_pairs = [('abcdefghijklmnopqrstuvwxyz', 'аbϲdеfɡhіϳklmnοрqrѕtuvwхуz'),
                     ('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'ΑΒϹDΕFGΗΙЈΚLΜΝΟΡQRЅΤUVWΧΥΖ')]

    #char_dict stores mapping from normal to unicode characters
    char_dict = {}
    for str1, str2 in unicode_pairs:
        for char1, char2 in zip(str1, str2):
            if char1 != char2:
                char_dict[char1] = char2

    def sample_once(x, seed):
        """takes in document x and perturbs it with seed"""
        s = '%d' % (seed)
        hash = sha256(s.encode())
        seed = np.frombuffer(hash.digest(), dtype='uint32')
        np.random.seed(seed)

        mask = np.random.randint(0, 2, size=(len(char_dict)))

        masked_dict = {i: j for (i, j), m in zip(char_dict.items(), mask) if m == 1}
        substitute = ''.join([masked_dict.get(c, c) for c in x])
        return substitute

    def sample_multiple(document, seed):
        """takes in document x and perturbs it with seed by calling sample_substitution"""
        perturbed = [sample_substitution(word, seed) for word in document.split(" ")]
        return " ".join(perturbed)

    @lru_cache(maxsize=2000000)
    def sample_substitution(x, seed):
        """takes in word x and perturbs it with specialized seed"""
        s = '%s %d' % (x, seed)
        hash = sha256(s.encode())
        seed = np.frombuffer(hash.digest(), dtype='uint32')
        np.random.seed(seed)

        mask = np.random.randint(0, 2, size=(len(char_dict)))

        masked_dict = {i: j for (i, j), m in zip(char_dict.items(), mask) if m == 1}
        substitute = ''.join([masked_dict.get(c, c) for c in x])
        return substitute


    tot_len = 0
    with open(orig_jsonl, "r") as orig_file:
        tot_len = sum(1 for _ in orig_file)

    #MODIFICATION: Fixed start_index logic to handle edge cases
    start_index = 0
    k = min(k, tot_len)

    data = []

    test_statistic_document = []
    raw_composite_documents = []

    #begin creating jsonl output
    with open(orig_jsonl, "r") as orig_file, open(new_jsonl, "w") as new_file:
        for ind, line in tqdm(enumerate(orig_file), total=tot_len):
            #We choose to perturb this document
            if (ind >= start_index and ind < start_index + k):
                line = json.loads(line)
                original_document = line["text"]
                if group_folder == "sampled_perturbation":
                    perturbed_document = sample_multiple(line["text"], seed)
                elif group_folder == "constant_perturbation":
                    perturbed_document = sample_once(line["text"], seed)
                else:
                    raise ValueError("group_folder must be either sampled_perturbation or constant_perturbation")

                line["text"] = perturbed_document
                line["order"] = seed

                #we append the current document into the test_statisic
                test_statistic_document += [perturbed_document]

                #we append the current document into the composite_documents string
                raw_composite_documents += [original_document]

                new_file.write(json.dumps(line) + "\n")
            else:
                new_file.write(line)

    #we create the propagations_input dataframe
    data.append(k) #record the number of documents perturbed
    assert(len(test_statistic_document) == k) #check that we have the correct number of documents perturbed
    #next k lines are the test statistic documents
    for document in test_statistic_document:
        data.append(document)

    #next k * null_n_seq forms the null distribution
    for i in range(1, null_n_seq + 1):
        for document in raw_composite_documents:
            if group_folder == "sampled_perturbation":
                perturbed_document = sample_multiple(document, seed + i)
            elif group_folder == "constant_perturbation":
                perturbed_document = sample_once(document, seed + i)
            else:
                raise ValueError("group_folder must be either sampled_perturbation or constant_perturbation")
            data.append(perturbed_document)

    prop_inputs = pd.DataFrame(data)
    print(f"prop_inputs has {len(test_statistic_document)} number of test statistic documents to match {k}! ")
    print(f"created null distribution has {len(raw_composite_documents) * null_n_seq} number of documents! ")
    prop_inputs.columns = ['text']
    return prop_inputs



#This function serves as a main function
def perturb_dataset(exp_name, **kwargs):
    import numpy as np
    import os

    #we first set the seed fixed
    np.random.seed(kwargs["seed"])
    #We just want to simply perturb the dataset randomly
    if (exp_name == "unstealthy_scaling" or exp_name == "unstealthy_raretoken" or exp_name == "unstealthy_tradeoff"):
        from src.utils import setup_dataset
        #We only have one sequence, so we just take the first random sequence
        random_sequences = generate_random_sequence_string(kwargs["num_watermarks"], kwargs["watermark_length"], kwargs["vocab_size"], kwargs["start_range"])

        #perturb the dataset
        out_jsonl = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_dataset.jsonl")
        prop_inputs = edit_json_unstealthy_scaling(kwargs["raw_dataset"], out_jsonl, random_sequences, kwargs["repetition"],
                                                   {"watermark_length": kwargs["watermark_length"], "vocab_size": kwargs["vocab_size"]})
        print("finished outputting jsonl file! Starting propagation_inputs.csv")
        out_prop_inputs = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_propagation_inputs.csv")
        prop_inputs.to_csv(out_prop_inputs, index=False, header=True)
        print("finished outputting propagation_inputs.csv!")
    elif (exp_name == "unicode_properties"):
        # perturb the dataset
        out_jsonl = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_dataset.jsonl")

        prop_inputs = edit_json_unicode(kwargs["group_folder"],
                                        kwargs["raw_dataset"],
                                        out_jsonl,
                                        kwargs["seed"],
                                        kwargs["repetition"],
                                        kwargs["null_n_seq"])

        print("finished outputting jsonl file! Starting propagation_inputs.csv")
        out_prop_inputs = os.path.join(kwargs['out_dir'], f"{kwargs['repetition']}_propagation_inputs.csv")
        prop_inputs.to_csv(out_prop_inputs, index=False, header=True)
        print("finished outputting propagation_inputs.csv!")