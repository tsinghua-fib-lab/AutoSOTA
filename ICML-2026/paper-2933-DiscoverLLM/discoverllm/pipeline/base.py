class LLMPipeline:
    """
    Shared init for the stateful pipeline classes (UserSimulator,
    AssistantSimulator, Updater, Abstractor).

    All four subclasses share the same four LLM-call attributes
    (model name, temperature, max tokens, verbose flag) and override
    ``__init__`` to add their own state on top.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
