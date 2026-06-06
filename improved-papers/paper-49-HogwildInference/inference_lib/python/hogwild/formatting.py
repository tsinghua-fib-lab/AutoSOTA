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
    end_of_step_chars = ['.', '?', '!', '。', '۔', '؟', '।', '॥', '…', '‽', '།', '᠃', '։', '჻', '¶', '❧']  # before SEP
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
        """Wrap a given task into a model input with system prompt and few-shot examples; applies chat template"""
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
        prompt_style: Optional[str] = "second_person",
        add_suggestions_on_collaborating: bool = True,
        add_examples: bool = False,
        wrap_shots_with_chat_template: bool = True,
        generation_prefix: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a system prompt for 2 workers with rules and optional few-shot examples"""
        assert isinstance(problem, str)
        if continue_final_message or not add_generation_prompt:
            raise NotImplementedError("Hogwild! apply_chat_template only implements generation prompt for now")
        if prompt_style is None:
            conversation = [dict(role='user', content=problem)]
        elif prompt_style in {"first_person", "second_person"}:
            system_prompt = _make_system_prompt_math_2workers(
                self,
                prompt_style=prompt_style,
                add_suggestions_on_collaborating=add_suggestions_on_collaborating,
                add_examples=add_examples, wrap_shots_with_chat_template=wrap_shots_with_chat_template,
            )
            if pass_system_prompt_as_user_message:
                conversation = [dict(role='user', content=system_prompt + self.sep + problem)]
            else:
                conversation = [dict(role='system', content=system_prompt), dict(role='user', content=problem)]
        else:
            raise ValueError(f"Unexpected {prompt_style=}")
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
        fmt: FormattingBase, *, prompt_style: str,
        add_suggestions_on_collaborating: bool, add_examples: bool, wrap_shots_with_chat_template: bool,
) -> str:
    """Create a system prompt for 2 workers with rules and optional few-shot examples"""
    return f"""
# Collaborative Reasoning

{RULES_VARIANTS[prompt_style](fmt)}{(fmt.sep + f'''
# Examples

## 1. Basic example of collaborating within one step

{EXAMPLES["example_first_step"](fmt, use_chat_template=wrap_shots_with_chat_template)}

## 2. Full example

{EXAMPLES["example_full_equation"](fmt, use_chat_template=wrap_shots_with_chat_template)}
'''.strip()) if add_examples else ""}

{(f'''
# How to collaborate

{SUGGESTIONS_ON_COLLABORATING_VARIANTS[prompt_style](fmt)}
 '''.strip() + (fmt.sep + f'''
**Strategizing:**

{EXAMPLES["example_medium_calc"](fmt, use_chat_template=wrap_shots_with_chat_template)}

**Communicating:**

{EXAMPLES["example_debate_geom"](fmt, use_chat_template=wrap_shots_with_chat_template)}

**Detecting redundant work and changing task:**

{EXAMPLES["example_step_avoid_redundancy_1"](fmt)}

{EXAMPLES["example_step_avoid_redundancy_2"](fmt)}

{EXAMPLES["example_step_avoid_redundancy_3"](fmt)}
'''.strip()) if add_examples else ""
).strip() + fmt.sep if add_suggestions_on_collaborating else ""
}# Solve the following problem

{f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}, you will now solve the next problem together. Keep track of who does what work and communicate to avoid doing the same work twice.
""".strip()


RULES_VARIANTS: Dict[str, Callable[[FormattingBase], str]] = dict(
    first_person=lambda fmt: f"""
I will collaborate on this problem with another assistant. We will write our thoughts simultaneously and collaborate without redundant work. We can collaborate by doing different parts of the problem, double-checking each other's results, trying different approaches, or any other means.

There are {len(fmt.workers)} assistants, including myself. We will refer to each other as {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}.

We will solve the problem together, writing our thoughts in parallel. We will be able to see each other's past and current thoughts as we write them. We will see each other's previous steps as {fmt.get_step_prefix('AssistantName', 'step')}<...> .

In the '{fmt.history_header}' section, the automated system will gather the thoughts of {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'} as we write them.

After the '{fmt.work_in_progress_others}' section, I will see the other assistants' unfinished steps. They will write those steps concurrently with me. I will take into account what they are doing. If another assistant gives me suggestions, I will address them.

I will always see *other* assistants' incomplete thoughts first, and then, after '{fmt.work_in_progress_self}', my own step. Other assistants will continue writing their thoughts in their thoughts, while I will continue writing my own.

Since we both write our thoughts in parallel, I will initially see only partial (unfinished) thoughts that will be continued in parallel with mine. Others' thoughts will appear at the end of their unfinished step, near {fmt.incomplete_step}. Other assistants may write new thoughts while I am writing mine.

I will use these partial thoughts to decide how best to collaborate without doing the same work twice. I will periodically check what other assistants are doing and I should adjust my actions based on what they are doing so we collaborate efficiently.

If what I am currently doing is the same that another assistant is doing or has already done, I will stop (e.g. {fmt.workers[0]} may say 'Wait, I was doing the same as {fmt.workers[1]} ...') and change to a different task right away.
""".strip(),
    second_person=lambda fmt: f"""
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
)

SUGGESTIONS_ON_COLLABORATING_VARIANTS: Dict[str, Callable[[FormattingBase], str]] = dict(
    first_person=lambda fmt: f"""
I will take into account what the other assistant is doing and change my actions accordingly. Here is how we can collaborate:

- **1. Strategizing:** we should think on how best to divide work between us (e.g. if {fmt.workers[0]} writes: {fmt.workers[1]}, please do this, then {fmt.workers[1]} should take this into account). If we disagree about what to do, we will default to {fmt.workers[0]}'s version.
- **2. Splitting:** we can split the problem into subtasks (simplify one equation or the other) and split the tasks between us. Prioritize subtasks that are not redundant (i.e. do not verify minor calculation done by another worker if there is another calculation that wasn't attempted yet).
- **3. Alternatives:** we can each try to solve a problem with different methods (e.g. calculate a mathematical expression by brute force vs mathematical derivations) and see which approach is faster.
- **4. Communicating:** we can look at each other's thoughts, ask each other questions (e.g. '{fmt.workers[0]}, which of these should I do first?'), give each other suggestions or corrections (e.g. 'Hey, {fmt.workers[1]}! You have a mistake in step 3 ...')
- **5. Announcing:** I can announce what I will do next (e.g. 'Let me try x=5 next' or 'I will double-check {fmt.workers[0]}'s result from step 5'). If another assistant says this, I will take it into consideration and do something else to avoid redundancy.
- **6. Reacting:** if I notice that another assistant is doing the same thing as I do, I should stop and think what else can I do to avoid redundancy. If I am ahead of the other assistant, I will instead ask them to change task problem.
- **7. Pivoting:** if I notice that what I am doing is no longer useful after change in circumstances, I will stop mid-sentence and pivot to another direction (e.g. '... computing p^4 | Wait, {fmt.workers[0]} is already on it, I should switch to adding up the results.')

We can also collaborate in any different way. We can invent new ways that would help us arrive at the correct solution faster.

To decide how best to collaborate, I will periodically, every few steps or more often, think what I am doing and if I am contributing or doing redundant work. If it is the latter, I will stop and choose something else to do to better contribute to solving the problem.
""".strip(),
    second_person=lambda fmt: f"""
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
)


def _make_example_fewshot(fmt, question: str, answer: str, use_chat_template: bool, **kwargs):
    if use_chat_template:
        return "<example>\n\n" + fmt.tokenizer.apply_chat_template(
            [dict(role='user', content=question)],
            tokenize=False, add_generation_prompt=True, **kwargs
        ) + answer + "\n\n</example>"
    return f"<example>\n\n{question}\n\n{answer}\n\n</example>"


class CallableMakeFewShotExample(Protocol):
    def __call__(self, fmt: FormattingBase, **kwargs: Any) -> str: ...


EXAMPLES: Dict[str, CallableMakeFewShotExample] = dict(
    example_first_step=lambda fmt, **kwargs: _make_example_fewshot(
        question=f"""
Solve two problems. Ann has 2 apples, Mark has 5 apples. How many apples do they have? Also, solve the equation x + y = 4, if y = 5. Write both answers in coma-separated format.
            """.strip(),
        answer=f"""
{fmt.history_header}{fmt.sep}{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[0], 1)}I will solve the first problem. Ann has 2 apples, Mark has Wait, {fmt.workers[1]}, I am doing that already, please switch to the equation. If Mark has 5 apples, they have 2 +{fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[1], 1)}Ann has 2 apples, Mark has 5 apples. How many... Wait, {fmt.workers[0]} is already solving that problem. I will instead solve x + y = 4, if y = 5: substituting y with 5 gives x + 5 = 4. The answer to the equation is
""".strip(), fmt=fmt, **kwargs),
    example_medium_calc=lambda fmt, **kwargs: _make_example_fewshot(
        question=f"""
Calculate S(x) = x + x^2 + x^3 + x^4 + x^5 for x = 1..10.
""".strip(),
        answer=f"""
{fmt.history_header}

{fmt.get_step_prefix(fmt.workers[0], 1)}Here's how we should collaborate: I'll handle the odd numbers from 1 to 10, and you ({fmt.workers[1]}) can handle the even numbers. That way, we can split the work evenly. Let me start with x=1, 3, 5, 7, 9.

{fmt.get_step_prefix(fmt.workers[1], 1)}Here's how we should collaborate: Wait, {fmt.workers[0]} is already suggesting so I will listen. Let me wait for {fmt.workers[0]} to finish . . . . . . . Okay, {fmt.workers[0]} wants me to do even numbers. I will do x=2, 4, 6, 8, 10.

{fmt.get_step_prefix(fmt.workers[0], 2)}For x=1: S(1) = 1 + 1 + 1 + 1 + 1 = 5. That's straightforward. I will do x=2 next. Wait, we agreed that {fmt.workers[1]} does even numbers and {fmt.workers[1]} is already doing x=2. I'll do x=3 instead.

{fmt.get_step_prefix(fmt.workers[1], 2)}Calculating S(2): 2 + 4 + 8 + 16 + 32. Let me add them step by step. 2+4=6, 6+8=14, 14+16=30, 30+32=62. So S(2)=62.

{fmt.get_step_prefix(fmt.workers[1], 3)}Calculating S(4): Let's compute each term: 4, 16, 64, 256, 1024. Adding them up: 4+16=20; 20+64=84; 84+256=340; 340+1024=1364. So S(4)=1364.

{fmt.get_step_prefix(fmt.workers[0], 3)}Next, x=3. Let's compute each term step by step to avoid mistakes. 3 + 9 (3²) is 12, plus 27 (3³) gives 39, plus 81 (3⁴) is 120, then add 243 (3⁵) to get 363. Wait, let me check again: 3 + 9 =12; 12+27=39; 39+81=120; 120+243=363. Yes, that's correct. Let me look to {fmt.workers[1]}'s progress to see how our computations are going. Looks like he computed x=2 and is currently computing x=4 in the incomplete step. Since we agreed that {fmt.workers[1]} handles even numbers, this checks out. We should continue as planned. 

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[1], 4)}Calculating S(6): 6 + 36 + 216 + 1296 + 7776. Let me add step by step. 6+36=42; 42+216=258; 258+1296=1554; 1554+7776=9330. Wait, let me check again. 6^1=6, 6^2=36, 6^ {fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[0], 4)}Next, x=5
""".strip(), fmt=fmt, **kwargs),
    example_debate_geom=lambda fmt, **kwargs: _make_example_fewshot(
        question=f"""
Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?
""".strip(),
        answer=f"""
{fmt.history_header}

{fmt.get_step_prefix(fmt.workers[1], 1)}Hey! Let me suggest splitting the problem into two parts. {fmt.workers[0]}, you can work on the first logarithmic equation to find possible relationships between x and y, while I can work on the second equation involving K. Once you have some relationships, I can substitute them into my part to solve for K. Let me know if you need help.

{fmt.get_step_prefix(fmt.workers[0], 1)}Here's how we should collaborate: Let me start by trying to find vectors between the points to see if I can find edge lengths or face diagonals. Since it's a cube, all edges are equal and the angles between edges are 90 degrees. Maybe we can compute the distances between P, Q, R and see which pairs are edges, face diagonals, or space diagonals. Let me compute the distances PQ, QR, and RP first.

{fmt.get_step_prefix(fmt.workers[0], 2)}Let's compute the distances between each pair of points to see the possible edge lengths.

{fmt.get_step_prefix(fmt.workers[0], 3)}Let me compute PQ first. The distance between P(7,12,10) and Q(8,8,1):

{fmt.get_step_prefix(fmt.workers[1], 2)}Alternatively, since three vertices are given, perhaps they form a triangle, and we can use coordinates to find edge lengths. Let me start by calculating the distance between P=(7,12,10) and Q=(8,8,1). The distance formula in 3D is sqrt[(x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2]. Wait, {fmt.workers[0]} has a suggestion.

{fmt.get_step_prefix(fmt.workers[0], 4)}{fmt.workers[1]}, we are both doing PQ. Let me proceed.

{fmt.get_step_prefix(fmt.workers[1], 3)}Okay, I'll do QR then. QR is between Q=(8,8,1) and R=(11,3,9).

{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[1], 4)}The differences are 3 in x, -5 in y, 8 in z. So distance squared is 3{fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[0], 5)}I will compute PQ now: x difference is 8-7=1, y difference 8-12=-4, z
""".strip(), fmt=fmt, **kwargs),
    example_full_equation=lambda fmt, **kwargs: _make_example_fewshot(
            question=f"""
For each ordered pair of real numbers $(x,y)$ satisfying \\[\\log_2(2x+y) = \\log_4(x^2+xy+7y^2)\\] there is a real number $K$ such that \\[\\log_3(3x+y) = \\log_9(3x^2+4xy+Ky^2).\\] Find the product of all possible values of $K$ .
        """.strip(),
            answer=f"""
{fmt.history_header}

{fmt.get_step_prefix(fmt.workers[0], 1)}Hey! Here's how we should collaborate: Let's first tackle the first equation to find relationships between x and y, and then use those to find K in the second equation. I'll work on the first equation, and {fmt.workers[1]} can handle the second equation once we have expressions for x and y.

{fmt.get_step_prefix(fmt.workers[1], 1)}Hey! Let me suggest splitting the problem into two parts. {fmt.workers[0]}, you can work on the first logarithmic equation to find possible relationships between x and y, while I can work on the second equation involving K. Once you have some relationships, I can substitute them into my part to solve for K. Let me know if you need help.

{fmt.get_step_prefix(fmt.workers[0], 2)}Okay, starting with the first equation: log₂(2x + y) = log₄(x² + xy + 7y²). Hmm, since the logs have different bases, maybe I can convert them to the same base. Remember that log₄(z) is equal to (log₂z)/(log₂4) which is (log₂z)/2. So the equation becomes log₂(2x + y) = (1/2) log₂(x² + xy +7y²). Let me write that down:

{fmt.get_step_prefix(fmt.workers[1], 2)}Alright, I'll focus on the second equation. Let me write down what's given: log₃(3x + y) = log₉(3x² +4xy + Ky²). I know that log base 9 can be expressed as log base 3 squared. So log₉(w) = (log₃w)/2. Therefore, the equation becomes log₃(3x + y) = (1/2) log₃(3x² +4xy + Ky²). Multiply both sides by 2 to eliminate the fraction: 2 log₃(3x + y) = log₃(3x² +4xy + Ky²). Using logarithm properties, that's log₃((3x + y)^2) = log₃(3x² +4xy + Ky²). So the arguments must be equal: (3x + y)^2 = 3x² +4xy + Ky². Let me expand the left side: 9x² +6xy + y² = 3x² +4xy + Ky². Subtract the right side from the left: (9x² -3x²) + (6xy -4xy) + (y² - Ky²) =0 → 6x² +2xy + (1 - K)y² =0. So 6x² +2xy + (1 - K)y² =0. Hmm, so this equation must hold for the same x and y that satisfy the first equation. So I need to find K such that this holds given the constraints from the first equation. 

{fmt.get_step_prefix(fmt.workers[0], 3)}So, starting from log₂(2x + y) = (1/2) log₂(x² + xy +7y²). Let me let A = log₂(2x + y), so the equation becomes A = (1/2) log₂(...) which implies 2A = log₂(...) so 2^(2A) = x² + xy +7y². But since A is log₂(2x + y), 2^A = 2x + y. So 2^(2A) is (2x + y)^2. Therefore, (2x + y)^2 = x² + xy +7y². Let me expand the left side: 4x² +4xy + y² = x² + xy +7y². Subtract the right side from the left: 3x² +3xy -6y² =0 → simplifying: divide by 3: x² + xy -2y² =0. So x² +xy -2y² =0. Let me factor this quadratic in x: x² +xy -2y². Let me see: factors of -2y² that add to +y. Hmm, (x + 2y)(x - y) = x² +xy -2y². Yes! So (x + 2y)(x - y) =0. Therefore, either x = -2y or x = y. So possible solutions are x = -2y or x = y. 

{fmt.get_step_prefix(fmt.workers[0], 4)}So, the first equation gives x = -2y or x = y. Now, but we also need to ensure that the original logarithm arguments are valid. For log₂(2x + y) to be defined, 2x + y >0. Similarly, the argument of log₄ must be positive: x² +xy +7y² >0. Let me check for both cases:

{fmt.get_step_prefix(fmt.workers[0], 5)}Now, for validity: If x = -2y, then 2x + y = 2*(-2y) + y = -4y + y = -3y. But log₂(-3y) must be defined, so -3y >0 → y <0. Also, the other log's argument: x² +xy +7y². Plugging x=-2y: (4y²) + (-2y)(y) +7y² =4y² -2y² +7y² =9y². Since y is real, 9y² is always non-negative, but must be positive. Since y≠0 (since x=-2y, if y=0 then x=0, but then 2x+y=0 which is invalid). So y≠0 and y<0. 

{fmt.get_step_prefix(fmt.workers[1], 3)}Okay, so I have 6x² +2xy + (1 - K)y² =0. But from {fmt.workers[0]}'s first equation, there might be a relation between x and y that I can substitute here. Let me see what {fmt.workers[0]} found. She's working on the first equation. Let me check her steps. She arrived at x² +xy -2y² =0, which factors into (x + 2y)(x - y)=0, so x = -2y or x = y. So, these are the possible cases. Therefore, in each case, I can substitute x in terms of y into the equation I have (6x² +2xy + (1 - K)y² =0) and solve for K. Let me do that. Since there are two cases, maybe I should handle them separately. Since {fmt.workers[0]} is done with the first part, I can proceed with substitution. Let me start with the first case: Case 1: x = -2y. Substitute into my equation: 6*(-2y)^2 +2*(-2y)y + (1 - K)y² =0. Compute each term: 6*(4y²) =24y², 2*(-2y)y =-4y², so total: 24y² -4y² + (1 - K)y² =0 → (24 -4 +1 - K)y² = (21 - K)y² =0. Since y is real and in this case x=-2y, but y can't be zero (as earlier), so y≠0, so the coefficient must be zero: 21 - K =0 → K=21. 

{fmt.get_step_prefix(fmt.workers[0], 6)}For the case x = y: Substitute into 2x + y = 3x, which must be positive, so 3x >0 → x>0. Also, the other argument: x² +x*x +7x² = x² +x² +7x² =9x²>0, so x≠0. So x>0. So both cases are valid as long as the conditions hold. Now, moving on, since {fmt.workers[1]} is handling K, perhaps he can proceed. 

{fmt.get_step_prefix(fmt.workers[1], 4)}For Case 1 (x=-2y): So K=21. Now, moving to Case 2: x=y. Substitute x=y into my equation: 6x² +2x*x + (1 - K)x² =0 → 6x² +2x² + (1 - K)x² = (8 +1 - K)x² = (9 - K)x² =0. Again, x≠0 (since x=y and in this case x must be positive from my earlier validity check). So 9 - K =0 → K=9. Thus, possible K values are 21 and 9. Therefore, the product is 21*9=189. 

{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[0], 7)}Wait, {fmt.workers[1]}, let me confirm your calculation for the second case. When x=y, substituting into 6x² +2xy + (1 - K)y²: since x=y, that becomes 6x² +2x² + (1 - K)x² = (6+2+1 - K)x² = (9 - K)x². So yes, so 9 - K =0 ⇒ K=9. So the two K values are 21 and 9, so product is 189. Hmm, but let me double-check the first case. For x = -2y{fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[1], 5)}Exactly, so 6+2=8 plus 1-K gives 9-K. Therefore, to have (9-K)x²=0 with x≠0, K must be 9. Therefore, the possible K values are 9 and 21. Their product is 189. Hey, {fmt.workers[0]}, what should I do next?
""".strip(), fmt=fmt, **kwargs),
    example_step_avoid_redundancy_1=lambda fmt: f"""
<example>{fmt.sep}(previous steps omitted)

{fmt.get_step_prefix(fmt.workers[1], 7)}So the equation becomes: The right side is (a +30)(a +k) = a² +a k +30a +30k.

{fmt.get_step_prefix(fmt.workers[0], 5)}: Right. The right side is (a +30)(a +k) = a² +a k +30a +30k. So set equal to left side:

{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[0], 6)}Let me expand the left side: (a +2k)^2 = a² +4ak +4k². The right side is (a +30)(a +k) = a² + a k +30a +30k. Subtract right side from left side: (a² +4ak +4k²) - (a² +a k +30a +30k) =0 ⇒ 3ak +4k² -30a -30k =0. Let me factor terms: 3ak -30a +4k² -30k =0 ⇒ 3a(k -10) +k(4k -30)=0. Hmm, so {fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[1], 8)}Wait, am I solving the same equation as {fmt.workers[0]}? Yes, it seems I am. {fmt.workers[0]} seems to have made more progress than me. Since I should not do the equation, what can I do instead? There is another case where a-d=-30 instead. Let me solve that part{fmt.sep}</example>
""".strip(),
    example_step_avoid_redundancy_2=lambda fmt: f"""
<example>{fmt.sep}(previous steps omitted)

{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[0], 6)}Let's compute how many days each direction is used. Since 40 days divided by 4 directions gives 10 days per direction. So each direction (E,N,W,S) is used exactly 10 times. So for East, days 1,5,9,...,37 (step of 4). The same for the others. So we can compute the total East displacement as the sum over{fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[1], 6)}Exactly. So for East, the days are 1,5,9,...,37 (total 10 days). Similarly for the others. Therefore, the total displacement in the East ... Wait, {fmt.workers[0]} is already doing that! I should compute N-S{fmt.sep}</example>
""".strip(),
    example_step_avoid_redundancy_3=lambda fmt: f"""
<example>{fmt.sep}(previous steps omitted)

{fmt.get_step_prefix(fmt.workers[0], 3)}Yes, expanding (a +2k)^2 gives a² +4ak +4k². Subtract a(a +k) which is a² +ak. So numerator is 3ak +4k². So:

{fmt.work_in_progress_others}

{fmt.get_step_prefix(fmt.workers[1], 3)}So equation: [k(3a +4k)] / (a +k) =30 ⇒ k(3a +4k) =30(a +k). Let me rearrange terms: 3a k +4k² =30a +30k. Let me bring all terms to left side: 3a k +4k² -30a -30k =0. Let me factor terms with a and terms{fmt.incomplete_step}

{fmt.work_in_progress_self}

{fmt.get_step_prefix(fmt.workers[0], 4)}Wait, {fmt.workers[1]} has already started that equation before me and seems ahead of me, so I am doing redundant work. What can I do in the meantime?{fmt.sep}</example>
""".strip()
)


def get_default_options_for_model(model: transformers.PreTrainedModel) -> Dict[str, Any]:
    opts = DEFAULT_FORMATTING_OPTIONS_BY_MODEL_TYPE.get(model.config.get_text_config().model_type, None)
    if opts is None:
        warnings.warn(f"Untested model type {model.config.get_text_config().model_type}, using global defaults")
        return dict()
    return opts


DEFAULT_FORMATTING_OPTIONS_BY_MODEL_TYPE = dict(  # comments indicate intended models
    qwen2=dict(add_examples=False),  # based on Qwen/QwQ-32B, all default parameters
    qwen3=dict(add_examples=False, generation_prefix="<think>" + CommonFormatting.generation_prefix),  # Qwen/Qwen3-32B
    qwen3_moe=dict(add_examples=False, generation_prefix="<think>" + CommonFormatting.generation_prefix),  # Qwen/Qwen3-235B-A22B
    phi3=dict(add_examples=False, generation_prefix="<think>" + CommonFormatting.generation_prefix),  # microsoft/Phi-4-reasoning-plus
    llama=dict(add_examples=False, wrap_shots_with_chat_template=False),  # meta-llama/Llama-3.3-70B-Instruct
)
