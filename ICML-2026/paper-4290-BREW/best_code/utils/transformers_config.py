class TransformersConfig:

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)
