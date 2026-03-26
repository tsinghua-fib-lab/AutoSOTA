"""A utility class that contains system prompts, header and step format and other tokens needed for generation"""
import warnings
from typing import Callable, Optional, Sequence, TypeVar, Dict, Any, Protocol, Union

import transformers
from torch import TensorType
from transformers.utils import PaddingStrategy

T = TypeVar("T")


class FormattingBase:
    workers: Sequence[str]
    worker_prompts: Sequence[str]
    step_separator: str
    incomplete_step: str
    s1_collab_message: str
    s1_finisher_suffix: str
    current_step_header: str
    current_worker_header: str

    @property
    def sep(self): return self.step_separator

    def get_step_prefix(self, worker: str, index: Any) -> str: """get a prefix for a given step, e.g. "Alice [5]:"""
    def is_end_of_step(self, worker_tokens: Sequence[int]) -> bool: """Check if a worker finished their step"""
    def apply_chat_template(self, problem: str) -> str: """Add system prompt and formatting to a given problem"""
    def get_final_answer(self, response: str) -> Optional[T]: """Extract the final answer or None if no answer given"""


class CommonFormatting(FormattingBase):
    step_separator = '\n\n'
    history_header = "### Past steps".strip()
    work_in_progress_others = "### Work in progress (others)".strip()
    work_in_progress_self = "### Work in progress (own)".strip()
    incomplete_step = "<...>".strip()

    generation_prefix = f"\n{history_header}{step_separator}"
    current_step_header = work_in_progress_others + step_separator
    current_worker_header = incomplete_step + step_separator + work_in_progress_self + step_separator

    s1_collab_message = "Quick check: am I doing redundant work? (yes/no): "
    s1_finisher_suffix = (f"{step_separator}Wait, given the limited time, I have to give an answer right now. "
                          "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{")
    end_of_step_chars = ['.', '?', '!', '。', '۔', '؟', '।', '॥', '…', '‽', '།', '᠃', '։', '჻', '¶', '❧', '！', '？']  # before SEP
    block_borders = ["```", "~~~", "$$"]

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 workers: Sequence[str] = ("Alice", "Bob"),
                 **kwargs):
        self.tokenizer, self.workers = tokenizer, tuple(workers)
        self.worker_prompts = [
            f"""{self.get_step_prefix(workers[0], 1)}Hi, I'm {workers[0]}. Here's how we can""".strip(),
            f"""{self.get_step_prefix(workers[1], 1)}Hi, I'm {workers[1]}.""".strip()
        ]
        self.system_prompt_kwargs = kwargs
        _sep_token_index, = self.tokenizer.encode(self.sep, add_special_tokens=False)
        _sep_internal_str = {i: t for t, i in tokenizer.vocab.items()}[_sep_token_index]
        self.tokens_containing_sep = {i for t, i in self.tokenizer.vocab.items() if _sep_internal_str in t}

    def get_step_prefix(self, worker: str, index: Any): return f"**{worker} [{index}]:** "

    def is_end_of_step(self, worker_tokens: Sequence[int]) -> bool:
        if worker_tokens[-1] not in self.tokens_containing_sep:
            return False
        step_string = self.tokenizer.decode(worker_tokens)
        if any(step_string.count(b) % 2 != 0 for b in self.block_borders):  # note: str.count is non-overlapping
            return False  # unfinished code block - do not finish step
        step_string = step_string[:step_string.rindex(self.sep)].strip()
        return any(step_string.endswith(t) for t in self.end_of_step_chars)

    def apply_chat_template(self, problem: str, **kwargs) -> str:
        """Wrap a given task into a model input with system prompt; applies chat template"""
        return self._apply_chat_template_batched(problem, **dict(self.system_prompt_kwargs, **kwargs))

    def _apply_chat_template_batched(
        self,
        problem_or_problems: Union[str, Sequence[str]],
        tokenize: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **formatting_kwargs
    ):
        if return_assistant_tokens_mask:
            raise NotImplementedError("Hogwild chat template does not implement return_assistant_tokens_mask for now")
        is_batched = not isinstance(problem_or_problems, str)
        problems = problem_or_problems if is_batched else [problem_or_problems]
        rendered = [self._apply_chat_template_once(problem, **formatting_kwargs) for problem in problems]
        rendered = rendered[0] if not is_batched else rendered
        assert tokenize or not return_dict, "`return_dict=True` is incompatible with `tokenize=False`"
        tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs is not None else {}
        if tokenize:
            out = self.tokenizer(
                rendered, padding=padding, truncation=truncation, max_length=max_length, add_special_tokens=False,
                return_tensors=return_tensors, **tokenizer_kwargs,
            )
            return out if return_dict else out["input_ids"]
        else:
            return rendered

    def _apply_chat_template_once(
        self,
        problem: str,
        pass_system_prompt_as_user_message: bool = True,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        add_suggestions_on_collaborating: bool = True,
        generation_prefix: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a system prompt for 2 workers with rules"""
        assert isinstance(problem, str)
        if continue_final_message or not add_generation_prompt:
            raise NotImplementedError("Hogwild! apply_chat_template only implements generation prompt for now")
        system_prompt = _make_system_prompt_math_2workers(
            self,
            add_suggestions_on_collaborating=add_suggestions_on_collaborating,
        )
        if pass_system_prompt_as_user_message:
            conversation = [dict(role='user', content=system_prompt + self.sep + problem)]
        else:
            conversation = [dict(role='system', content=system_prompt), dict(role='user', content=problem)]
        full_prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, continue_final_message=False, **kwargs
        )
        generation_prefix = generation_prefix if generation_prefix is not None else self.generation_prefix
        if generation_prefix is not None:
            full_prompt += generation_prefix
        return full_prompt

    def get_final_answer(self, response: str) -> Any:
        return find_last_valid_result(response, prefix="\\boxed{", suffix="}", extract_result=lambda x: x)


class MathFormatting(CommonFormatting):
    s1_finisher_suffix = (f"\n\nWait, given the limited time, I have to give an answer right now. "
                          "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{")

    def __init__(self, *args, extract_result: callable = float, **kwargs):
        super().__init__(*args, **kwargs)
        self.extract_result = extract_result

    def get_final_answer(self, response: str) -> Optional[float]:
        return find_last_valid_result(response, prefix="\\boxed{", suffix="}", extract_result=self.extract_result)


class CodeFormatting(MathFormatting):
    s1_finisher_suffix = (f"\n\nWait, given the limited time, I have to write the final code right now. "
                          "Considering all my previous attempts, the final solution code is:\n\n```python")

    def get_final_answer(self, response: str) -> str:
        return find_last_valid_result(response, prefix="```python", suffix="```", extract_result=check_python_code)


def check_python_code(excerpt: str):
    """Check if a given code snippet (without backticks) is a """
    if len(excerpt.strip()) == 0:
        raise ValueError()
    compile(excerpt, '<string>', 'exec')  # check if code compiles (w/o running); if it doesn't, this will throw error
    return excerpt


def find_last_valid_result(
    response: str, prefix: str = "\\boxed{", suffix: str = "}", extract_result: Callable[[str], T] = int
) -> Optional[T]:
    """
    Find the rightmost entry between prefix and suffix where exract_result does not fail, defaults to \\boxed{x}
    :param response: full de-tokenized response text
    :param prefix: this substring must come directly before the answer
    :param suffix: this substring must come directly after the answer
    :param extract_result: this is called on the substring before prefix and suffix (not including either)
        If extract_result succeeds, the answer is returned; if it throws any exception, try next answer to the left;
    :returns: answer (the output of extract_result) or None of no answer could be found
    """
    while True:
        try:
            start = response.rindex(prefix)
            try:
                end = response.index(suffix, start + 1)
                return extract_result(response[start + len(prefix):end])
            except Exception:  # missing suffix or extract_result failed
                response = response[:start]
        except ValueError:
            return None


def _make_system_prompt_math_2workers(
        fmt: FormattingBase, *,
        add_suggestions_on_collaborating: bool,
) -> str:
    """Create a system prompt for 2 workers with rules"""
    return f"""
# Collaborative Reasoning

{RULES(fmt)}

{(f'''
# How to collaborate

{SUGGESTIONS_ON_COLLABORATING(fmt)}
 '''.strip()).strip() + fmt.sep if add_suggestions_on_collaborating else ""
}# Solve the following problem

{f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}, you will now solve the next problem together. Keep track of who does what work and communicate to avoid doing the same work twice.
""".strip()


RULES=lambda fmt: f"""
You will collaborate on this problem with another assistant. You will write your thoughts simultaneously with them and collaborate without redundant work. You can collaborate by doing different parts of the problem, double-checking each other's results, trying different approaches, or any other means.

There are {len(fmt.workers)} assistants, including yourself. You will refer to each other as {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}.

You will solve the problem together, writing your thoughts in parallel. You will be able to see each other's past and current thoughts as we write them. You will see each other's previous steps as {fmt.get_step_prefix('AssistantName', 'step')}{fmt.incomplete_step} .

In the '{fmt.history_header}' section, the automated system will gather the thoughts of {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'} as you write them.

After the '{fmt.work_in_progress_others}' section, you will see the other assistants' unfinished steps. They will write those steps concurrently with you. You will take into account what they are doing. If another assistant gives you suggestions, you should address them.

You will always see *other* assistants' incomplete thoughts first, and then, after '{fmt.work_in_progress_self}', your own current step. Other assistants will continue writing their thoughts in the background while you will continue writing your own.

Since you and others both write your thoughts in parallel, you will initially see only partial (unfinished) thoughts that others will continue in parallel, while you write yours. Others' thoughts will appear at the end of their unfinished step, near {fmt.incomplete_step}. Other assistants may write new thoughts while you are writing yours.

You will use these partial thoughts to decide how best to collaborate without doing the same work twice. You will periodically check what other assistants are doing and you should adjust your actions based on what they are doing so you collaborate efficiently with them.

If what you are currently doing is the same thing that another assistant has already done or is in process of doing, you will stop (e.g. {fmt.workers[0]} may say 'Wait, I was doing the same as {fmt.workers[1]} ...') and change to a different task right away, so as to avoid doing redundant work. 
""".strip()

SUGGESTIONS_ON_COLLABORATING=lambda fmt: f"""
You will take into account what the other assistant is doing and change your actions accordingly. Here is how you can collaborate with them:

- **1. Strategizing:** you should think on how best to divide work between us (e.g. if {fmt.workers[0]} writes: {fmt.workers[1]}, please do this, then {fmt.workers[1]} should take this into account). If assistants disagree about what to do, you should both default to {fmt.workers[0]}'s version.
- **2. Splitting:** you can split the problem into subtasks (simplify one equation or the other equation) and split the tasks between us. Prioritize subtasks that are not redundant (i.e. do not verify minor calculation done by another worker if there is another calculation that wasn't attempted yet).
- **3. Alternatives:** you can each try to solve a problem with different methods (e.g. calculate a mathematical expression with brute force vs mathematical derivations) and see which approach is faster.
- **4. Communicating:** you can look at each other's thoughts, ask each other questions (e.g. '{fmt.workers[0]}, which of these should I do first?'), give each other suggestions or corrections (e.g. 'Hey, {fmt.workers[1]}! You have a mistake in step 3 ...')
- **5. Announcing:** you can announce what you will do next (e.g. 'Let me try x=5 next' or 'I will double-check {fmt.workers[0]}'s result from step 5'). If another assistant says this, you should take it into consideration to avoid redundancy.
- **6. Reacting:** if you notice that another assistant is doing the same thing as you do, you should stop and think what else can you do to avoid redundancy. If you are ahead of the other assistant, you will instead ask them to change task problem (e.g. '{fmt.workers[1]}, please do something else, I am already solving that').
- **7. Pivoting:** if you notice that what you are doing is no longer useful after change in circumstances, you will stop mid-sentence and pivot to another direction (e.g. '... computing p^4 | Wait, {fmt.workers[0]} is already on it, I should switch to adding up the results.')

You can also collaborate in any different way. You can invent new ways that would help you arrive at the correct solution faster together.

To decide how best to collaborate, you will periodically, every few steps or more often, think what you are doing and if you are contributing or doing redundant work. If it is the latter, you will stop and do something else to better contribute to solving the problem together.
""".strip()



def _make_example_fewshot(fmt, question: str, answer: str, use_chat_template: bool, **kwargs):
    if use_chat_template:
        return "<example>\n\n" + fmt.tokenizer.apply_chat_template(
            [dict(role='user', content=question)],
            tokenize=False, add_generation_prompt=True, **kwargs
        ) + answer + "\n\n</example>"
    return f"<example>\n\n{question}\n\n{answer}\n\n</example>"


class CallableMakeFewShotExample(Protocol):
    def __call__(self, fmt: FormattingBase, **kwargs: Any) -> str: ...


def get_default_options_for_model(model: transformers.PreTrainedModel) -> Dict[str, Any]:
    opts = DEFAULT_FORMATTING_OPTIONS_BY_MODEL_TYPE.get(model.config.get_text_config().model_type, None)
    if opts is None:
        warnings.warn(f"Untested model type {model.config.get_text_config().model_type}, using global defaults")
        return dict()
    return opts


DEFAULT_FORMATTING_OPTIONS_BY_MODEL_TYPE = dict(  # comments indicate intended models
    qwen2=dict(),  # based on Qwen/QwQ-32B, all default parameters
    qwen3=dict(generation_prefix="<think>" + CommonFormatting.generation_prefix),  # Qwen/Qwen3-32B
    qwen3_moe=dict(generation_prefix="<think>" + CommonFormatting.generation_prefix),  # Qwen/Qwen3-235B-A22B
    phi3=dict(generation_prefix="<think>" + CommonFormatting.generation_prefix),  # microsoft/Phi-4-reasoning-plus
    llama=dict(),  # meta-llama/Llama-3.3-70B-Instruct
)