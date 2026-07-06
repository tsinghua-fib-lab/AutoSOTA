"""
Official templates module
Uses each model's official chat template, on top of which we add our caveats, structured instructions, etc.
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


# Llama-2 official default System Prompt
LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


# Llama-3.1 official default System Prompt (Llama-3.1 usually uses a more concise system prompt)
LLAMA31_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""


# Qwen official default System Prompt
QWEN_DEFAULT_SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""


class OfficialTemplateBuilder:
    """
    Official template builder
    Uses the official chat template format for different models
    """

    def __init__(self, model_name: str, tokenizer_path: str):
        """
        Args:
            model_name: Model name (used to identify the model type)
            tokenizer_path: Tokenizer path
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Identify the model type (prefer the official apply_chat_template; the type is mainly used for fallback)
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            # Distinguish between Llama-2 and Llama-3.1
            # Note: need to match "3.1" or "3-" precisely (to avoid "13b" being misidentified as "3")
            if "3.1" in model_name_lower or (model_name_lower.startswith("llama-3") or "llama3" in model_name_lower):
                self.model_type = "llama31"
                self.default_system_prompt = LLAMA31_DEFAULT_SYSTEM_PROMPT
            else:
                # Llama-2 series (including Llama-2-7b, Llama-2-13b, etc.)
                self.model_type = "llama2"
                self.default_system_prompt = LLAMA2_DEFAULT_SYSTEM_PROMPT
        elif "qwen" in model_name_lower:
            self.model_type = "qwen"
            self.default_system_prompt = QWEN_DEFAULT_SYSTEM_PROMPT
        else:
            logger.warning(f"Unknown model type: {model_name}, using the default Qwen template")
            self.model_type = "qwen"
            self.default_system_prompt = QWEN_DEFAULT_SYSTEM_PROMPT

        # Verify whether an official chat_template exists
        if not hasattr(self.tokenizer, 'chat_template') or not self.tokenizer.chat_template:
            logger.warning(f"⚠️  Model {model_name} has no official chat_template definition; will use manual fallback construction")
        else:
            logger.info(f"✅ Model {model_name} has an official chat_template definition; will use the 100% official template")

        logger.info(f"Initializing official template builder: {model_name} (type: {self.model_type})")
    
    def build_prompt(
        self,
        user_message: str,
        task_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        add_format_constraints: bool = True,
    ) -> str:
        """
        Build the full prompt using the official template format

        Args:
            user_message: User message (question)
            task_type: Question type (multiple_choice, factual, code, reasoning)
            history: List of conversation history, format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: Custom system prompt; if None, use the default
            add_format_constraints: Whether to add format constraints (per task_type)

        Returns:
            The full prompt string
        """
        # Use the default system prompt
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        # Add our caveats and structured instructions to the system prompt
        enhanced_system_prompt = self._enhance_system_prompt(system_prompt, task_type, add_format_constraints)

        # Build the message list
        messages = []

        # Add the system message
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # Add the conversation history
        if history:
            for turn in history:
                if "role" in turn and "content" in turn:
                    messages.append({"role": turn["role"], "content": turn["content"]})

        # Add the current user message (may include format constraints)
        current_user_message = user_message
        if add_format_constraints and task_type:
            format_constraint = self._get_format_constraint(task_type)
            if format_constraint:
                current_user_message = f"{user_message}\n\n{format_constraint}"

        messages.append({"role": "user", "content": current_user_message})

        # Generate the prompt using the official apply_chat_template (100% official template, no deviation)
        try:
            # This is the official method provided by the transformers library, using the model's built-in chat_template
            # Ensures 100% use of the official format, without any custom modifications
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"✅ Generated prompt using the official apply_chat_template (model: {self.model_name})")
            return prompt
        except Exception as e:
            logger.error(f"❌ Failed to use the official template: {e}, falling back to manual construction (this should not happen; please check the model configuration)")
            logger.error(f"   Model: {self.model_name}, path: {self.tokenizer}")
            return self._fallback_build_prompt(messages)
    
    def _enhance_system_prompt(
        self,
        base_system_prompt: str,
        task_type: Optional[str] = None,
        add_format_constraints: bool = True
    ) -> str:
        """
        Enhance the system prompt by adding our caveats

        Args:
            base_system_prompt: Base system prompt
            task_type: Question type
            add_format_constraints: Whether to add format constraints

        Returns:
            The enhanced system prompt
        """
        enhanced = base_system_prompt

        # Add general caveats (emphasizing no filler, using stronger language)
        enhanced += "\n\nCRITICAL: You MUST provide direct, concise answers. FORBIDDEN: greetings, introductory phrases like 'Sure!', 'Okay!', 'Here's', 'Got it!', 'Sure thing!', etc. STRICTLY PROHIBITED: any phrases before the actual answer. You MUST start directly with the answer. You MUST provide an answer even if uncertain."
        
        # Add task-specific constraints based on task_type (these are also added to the user message; here they serve as a supplement)
        if add_format_constraints and task_type:
            if task_type == "multiple_choice":
                enhanced += "\n\nFor multiple-choice questions, provide ONLY the letter (A, B, C, D, etc.) as your answer."
            elif task_type == "factual":
                enhanced += "\n\nFor factual questions, provide ONLY the key noun, phrase, or short answer."
            elif task_type == "code":
                enhanced += "\n\nFor code questions, provide ONLY the code without markdown, comments, or explanations."
            elif task_type == "reasoning":
                enhanced += "\n\nFor reasoning questions, provide a direct and concise answer."
        
        return enhanced
    
    def _get_format_constraint(self, task_type: str) -> Optional[str]:
        """
        Return the format constraint based on question type (added to the user message)

        Args:
            task_type: Question type

        Returns:
            Format constraint string
        """
        if task_type == 'multiple_choice':
            return """CRITICAL: This is a multiple-choice question.
You MUST provide ONLY a single letter (A, B, C, D, etc.) as your answer.
FORBIDDEN: explanations, thinking process, sentences, additional content.
STRICTLY PROHIBITED: greetings, "Sure!", "Okay!", "Here's", "Got it!", "Sure thing!", or ANY introductory phrases.
You MUST start directly with the letter. NO text before the letter. NO text after the letter.
You MUST provide an answer. Even if uncertain, provide your best guess.
Answer:"""
        
        elif task_type == 'factual':
            return """CRITICAL: This is a FACTUAL question.
Provide ONLY the key noun, phrase, or short answer.
FORBIDDEN: complete sentences, explanations, thinking process, additional content.
STRICTLY PROHIBITED: greetings, "Sure!", "Okay!", "Here's", "Got it!", "Sure thing!", or ANY introductory phrases.
ONLY the essential information. Start directly with the answer. NO text before the answer.
You MUST provide an answer. Even if uncertain, provide your best guess.
Answer:"""
        
        elif task_type == 'code':
            return """CRITICAL: This is a CODE question.
Provide ONLY the code.
NO markdown. NO comments. NO explanations. NO thinking process. NO additional content.
ONLY the code itself. Be concise and precise.
You MUST provide an answer. Even if you are uncertain, provide your best attempt.
Code:"""
        
        elif task_type == 'reasoning':
            return """CRITICAL: This is a REASONING question.
Provide a direct and concise answer to the question.
FORBIDDEN: lengthy explanations, reasoning steps, thinking process, additional content.
STRICTLY PROHIBITED: greetings, "Sure!", "Okay!", "Here's", "Got it!", "Sure thing!", or ANY introductory phrases.
ONLY the answer itself. Start directly with the answer. NO text before the answer.
You MUST provide an answer. Even if uncertain, provide your best guess.
Answer:"""
        
        else:
            return """CRITICAL: Provide ONLY the answer.
NO explanations. NO thinking process. NO additional content.
You MUST provide an answer. Even if you are uncertain, provide your best guess.
Answer:"""
    
    def _fallback_build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback option: manually build the prompt (if the official template fails)

        Args:
            messages: List of messages

        Returns:
            Prompt string
        """
        # Manual fallback construction (used only when the official template fails)
        if self.model_type == "llama2":
            return self._build_llama2_prompt_manual(messages)
        elif self.model_type == "llama31":
            return self._build_llama31_prompt_manual(messages)
        else:  # qwen
            return self._build_qwen_prompt_manual(messages)
    
    def _build_llama2_prompt_manual(self, messages: List[Dict[str, str]]) -> str:
        """
        Manually build a prompt in the Llama-2 format
        Format: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg} [/INST]
        """
        prompt_parts = []
        system_content = None
        user_assistant_pairs = []

        # Separate the system message from the other messages
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_assistant_pairs.append(msg)

        # Build the prompt
        if system_content:
            prompt_parts.append(f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n")
        else:
            prompt_parts.append("<s>[INST] ")

        # Process user-assistant pairs
        for i, msg in enumerate(user_assistant_pairs):
            if msg["role"] == "user":
                if i > 0 or system_content:
                    prompt_parts.append(f"{msg['content']} [/INST]")
                else:
                    prompt_parts[0] = prompt_parts[0].rstrip() + f"{msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                prompt_parts.append(f" {msg['content']} </s><s>[INST] ")

        # Remove the trailing [INST] if there is no assistant reply yet
        result = "".join(prompt_parts)
        if result.endswith(" [INST] "):
            result = result[:-9]  # Remove " [INST] "

        return result
    
    def _build_llama31_prompt_manual(self, messages: List[Dict[str, str]]) -> str:
        """
        Manually build a prompt in the Llama-3.1 format.
        Format: <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>...
        followed by an empty assistant header to start generation.
        """
        prompt_parts = ["<|begin_of_text|>"]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        # Add the assistant start marker (for generation)
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(prompt_parts)

    def _build_qwen_prompt_manual(self, messages: List[Dict[str, str]]) -> str:
        """
        Manually build a prompt in the Qwen format
        Format: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n
        """
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        # Add the assistant start marker (for generation)
        prompt_parts.append("<|im_start|>assistant\n")

        return "".join(prompt_parts)

