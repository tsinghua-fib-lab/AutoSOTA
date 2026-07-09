import random
import torch
import os
from datasets import load_dataset

def get_wikitext2(tokenizer, nsamples, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader

    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testloader = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

        return testloader


def get_ptb_new(tokenizer, nsamples, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader

    else:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=True)
        testloader = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

        return testloader


def get_c4(tokenizer, nsamples, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        return trainloader
    
    else:
        testdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        random.seed(0)
        testenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(testdata) - 1)
                tmp = tokenizer(testdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            testenc.append(tmp.input_ids[:, i:j])
        testenc = torch.hstack(testenc)

        return testenc 
    

def get_c4_new(tokenizer, nsamples, seqlen, mode="train"):
    if mode == "train":
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        return trainloader
    
    else:
        testdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]

        return testenc 


def get_loaders(tokenizer, name, nsamples, seqlen, seed, mode="train"):
    random.seed(seed)

    if name == "wikitext2":
        return get_wikitext2(tokenizer, nsamples, seqlen, mode)
    elif name == "ptb-new":
        return get_ptb_new(tokenizer, nsamples, seqlen, mode)
    elif name == "c4":
        return get_c4(tokenizer, nsamples, seqlen, mode)
    elif name == "c4-new":
        return get_c4_new(tokenizer, nsamples, seqlen, mode)
    else:
        raise NotImplementedError(f"{name} dataset loader is NOT implemented.")
    

def get_weight_quant_data(args):
    cache_calib_data_path = f'{args.cache_dir}/calib_{args.model_type}_{args.calib_data}_{args.nsamples}_{args.seqlen}_{args.seed}.cache'

    if os.path.exists(cache_calib_data_path):
        # Cache files store full objects (e.g., tokenizer BatchEncoding); PyTorch 2.6+
        # defaults to weights_only=True, so we disable that to load our own cache.
        calib_data = torch.load(cache_calib_data_path, weights_only=False)
        print(f"load calibration from {cache_calib_data_path}")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        calib_data = get_loaders(tokenizer, args.calib_data, args.nsamples, args.seqlen, args.seed, mode="train")
        torch.save(calib_data, cache_calib_data_path)

    return calib_data


def get_testdata(testset, args):
    cache_test_data_path = f'{args.cache_dir}/testloader_{args.model_type}_{testset}.cache'

    if os.path.exists(cache_test_data_path):
        # Same reasoning as calibration cache: allow full object loading.
        testdata = torch.load(cache_test_data_path, weights_only=False)
        print(f"load testdata from {cache_test_data_path}")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        testdata = get_loaders(tokenizer, testset, args.nsamples, args.seqlen, args.seed, mode="test")
        torch.save(testdata, cache_test_data_path)

    return testdata