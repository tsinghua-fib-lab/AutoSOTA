
def plm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from tokenizers import AddedToken
    import torch, os

    model_id = "facebook/Perception-LM-1B"
    save_dir = "facebook/Perception-LM-1B-MT256x2"

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*2)] + [MT_END_TOKEN]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)

def qwen25vl():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from tokenizers import AddedToken
    import torch, os

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    save_dir = "Qwen/Qwen2.5-VL-7B-MT256x2"

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*2)] + [MT_END_TOKEN]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)

def qwen3vl():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from tokenizers import AddedToken
    import torch, os

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    save_dir = "Qwen/Qwen3-VL-8B-MT256x2"

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*2)] + [MT_END_TOKEN]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)


if __name__ == "__main__":
    plm()

    # qwen25vl()

    # qwen3vl()


