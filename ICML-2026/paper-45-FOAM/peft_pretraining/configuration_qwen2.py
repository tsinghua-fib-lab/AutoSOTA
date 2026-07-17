from transformers.configuration_utils import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        rope_theta=1000000.0,
        use_mrope=False,
        use_sliding_window=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_mrope = use_mrope
        self.use_sliding_window = use_sliding_window
        self.tie_word_embeddings = tie_word_embeddings
