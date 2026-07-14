import inspect
import re
from typing import Optional, Self

from mcp import StdioServerParameters, stdio_client
from strands.agent import AgentResult
from strands.telemetry import EventLoopMetrics, metrics_to_string
from strands.tools import tool
from strands.tools.mcp import MCPClient

from bitween.analyzer import verify_with_timeout
from bitween.config import Config, Method, MILPSolver
from bitween.main import infer_property_with_timeout
from bitween.miscs import getLogger
from bitween.sampler import Distribution, Domain

config = Config()
log = getLogger(__name__, config.logger_level)


def create_infer_property_tool(
    domain: Domain,
    distribution: Distribution,
    functions: list[callable],
    trace_file: str,
    preconditions: dict[str, callable] = None,
    timeout_sec: float = 600.0,
):
    """Create an inference property tool that can be used by any agent."""

    @tool
    def infer_property_tool(
        exprs: list[str],
        max_degree: int = 2,
        n: int = 30,
        epsilon: float = 0.001,
        milp: Optional[MILPSolver] = None,
        var_bound: int = 20,
        method: Method = Method.MULTIPLE_REGRESSION,
    ) -> tuple[dict, dict, dict, str]:
        """
        Infer properties containing `exprs` as terms using linear regression.

        Use this tool when you do not know a closed-form expression of property
        in order to try and find some properties that are based on `exprs`.

        This tool uses a data-driven approach, which means that it generates
        up to `n` concrete samples of the provided `exprs` randomly and then
        tries to find properties based on the `exprs` that fit the data using
        the provided `method`.

        In order to find properties, the tool combines together all the `exprs`
        up to `max_degree`. For example:

        if max_degree = 2, and exprs = ['f(x)', 'f(x-y)'], then
         degree 1: ['f(x)', 'f(x-y)', '1']
         degree 2: ['f(x)', 'f(x-y)', 'f(x)*f(x)', 'f(x)*f(x-y)', 'f(x-y)*f(x-y)', '1']

        Notes:
            - The `exprs` should contain only the names of the functions that
              are defined in the tool's context and described in the docstring.

            - The defined functions are the implementations of the function
              symbols that are used for the sample generation.

        Example:
            Defined functions:
                def f(x, c=5):
                    return c * x

            >> infer_property_tool(
                    exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
                    max_degree=2,
                    n=10,
                    epsilon=0.001,
                    milp=None,
                    var_bound=20,
                    method=Method.MULTIPLE_REGRESSION,
                )

            >> (
                {
                    'vtrace1': [
                        Eq(f(x) - f(x-y) - f(y), 0),
                        Eq(f(x+y) - f(x-y) - 2*f(y), 0)
                    ]
                },

                {
                    'vtrace1': 1.33e-13
                },

                {
                    'vtrace1': 9.27
                }
              )

        Defined functions:\n{functions_spec}

        Args:
            exprs: List of strings representing functional terms. The function
                   symbol should only be one of the defined functions. These
                   terms are very important in the success of the inference.

            max_degree: The maximum degree that the term combination should
                        reach. Usually it should not be too large.

            n: The number of samples that need to be generated. Usually, more
               samples means more accuracy, but there are many cases that
               few samples, like 20, are sufficient enough.

            epsilon: Tolerance for the mean squared error. The default value is
                     good enough, but sometimes increasing the tolerance is
                     beneficial, as more properties can be found.

            milp: The solver to be used. You can experiment with the different
                  solvers.

            var_bound: The variable bound for the MILP algorithm. It specifies
                       that the variables would be in the range
                       [`-var_bound`, `var_bound`]. The default value
                       works well.

            method: The available MILP method to be used. If it is left `None`,
                    then no MILP will be performed and the tool will fallback
                    to `Method.MULTIPLE_REGRESSION`.

        Returns:
            A tuple with four items:
                Three dictionaries all having identifiers as keys:
                    - The first dictionary is for the found equations.
                      These are sympy expressions or equalities.
                    - The second dictionary is for the equations' mean error
                    - The third dictionary is for the equations' sample complexity

                    Pay attention to the second dictionary value with the mean
                    error, because ideally it needs to be very close to zero.

                An error message as the last item of the tuple, which when
                present can provide useful information when something went
                wrong.
        """
        return infer_property_with_timeout(
            domain=domain,
            distribution=distribution,
            exprs=exprs,
            template=exprs,
            functions=functions,
            max_degree=max_degree,
            n=n,
            epsilon=epsilon,
            preconditions=preconditions,
            milp=milp,
            var_bound=var_bound,
            method=method,
            trace_file=trace_file,
            timeout_sec=timeout_sec,
        )

    def format_func(func, indent=8):
        prefix = indent * " "
        func_str = inspect.getsource(func)
        source_lines = func_str.splitlines()
        return "\n".join(map(lambda s: prefix + s, source_lines))

    functions_spec = "\n\n".join(map(format_func, functions))
    docstr = infer_property_tool.__doc__
    infer_property_tool.__doc__ = docstr.replace("{functions_spec}", functions_spec)

    return infer_property_tool


def create_symbolic_verify_tool(
    functions: list[callable],
    domain: Domain = Domain.Real,
    constants: dict = None,
    timeout_sec: float = 2.0,
):
    """Create a symbolic verification tool that can be used by any agent."""

    @tool
    def symbolic_verify_tool(exprs: list[str]) -> list[tuple[bool, str]]:
        """
        Verify sympy expressions symbolically using mathematical derivations.

        Use this tool when you need to verify that `exprs` hold.
        There are two cases in which this can happen:
         1) an `expr` is an equality with the right-hand-side being zero, or
         2) an `expr` is an expression that should equal to zero
        The first case can be converted to the second if we keep only the
        left-hand-side of the equation.

        This tool utilizes the sympy package, in order to parse each `expr`
        and then symbolically simplify it. Verification is successful when the
        simplification leads to zero.

        Notes:
            - Each provided `expr` should contain the symbolic functions
              defined in sympy's context for this tool. These are described
              in the docstring of the tool.
            - If you want to use in an `expr` any other function or constant
              defined in sympy, you should use it only by name and not prefix
              the namespace `sympy` before. For instance, instead of
              `sympy.Max` use `Max`.

        Example:
            Defined functions:
              def f(x, c=5):
                  return c * x

            >> symbolic_verify_tool(["Eq(f(x) + f(y) - f(x+y), 0)"])
            >> [(True, "")]

            >> symbolic_verify_tool(["f(x) + f(y) - f(x+y)"])
            >> [(True, "")]

        Defined functions:\n{functions_spec}

        Args:
            expr: A list of string representations of sympy expressions.
                  Each string in the list if parsed by sympy, should lead to
                  `sympy.Expr` or `sympy.Eq` that represents equality to zero.

        Returns:
            A list of tuples of (status, reason) that correspond to each
            provided expression:
                - status is True if the `expr` at that index is verified
                  or False otherwise
                - reason states why verification failed for that expression,
                  which could be either error or simplification to zero failed,
                  reason is empty only when status is True

            Pay attention to the reason part of each output tuple as
            it conveys useful information about what went wrong.
        """
        return [
            verify_with_timeout(
                expr=expr,
                functions=functions,
                domain=domain,
                constants=constants,
                timeout_sec=timeout_sec,
            )
            for expr in exprs
        ]

    def format_func(func, indent=8):
        prefix = indent * " "
        old_name = func.__name__
        new_name = old_name.removeprefix("_sp_")
        func_str = inspect.getsource(func).replace(old_name, new_name)
        source_lines = func_str.splitlines()
        return "\n".join(map(lambda s: prefix + s, source_lines))

    functions_spec = "\n\n".join(map(format_func, functions))
    docstr = symbolic_verify_tool.__doc__
    symbolic_verify_tool.__doc__ = docstr.replace("{functions_spec}", functions_spec)

    return symbolic_verify_tool


class AgentResponse:
    def __init__(
        self,
        agent_result: AgentResult,
        session_messages: list[dict],
        agent_name: str = "Agent",
    ):
        self.agent_name = agent_name

        self.stop_reason = str(agent_result.stop_reason)
        self.metrics = agent_result.metrics
        self.tool_use_ids_names = {}

        self.answers = self._extract_answers(str(agent_result))

        self.trace = ""
        self.user_trace = ""
        self.assistant_trace = ""

        for message in session_messages:
            role = message.get("role", "No role")

            start_sep, end_sep = self._get_separators(role)
            formatted_msg = self._format_message(message)

            formatted_msg_with_seps = f"{start_sep}\n\n{formatted_msg}\n\n{end_sep}"

            self.trace += f"{formatted_msg_with_seps}\n\n"
            if role == "user":
                self.user_trace += f"{formatted_msg}\n\n"
            elif role == "assistant":
                self.assistant_trace += f"{formatted_msg}\n\n"
            else:
                log.warning(f"Found message with different role: {message}")

    @staticmethod
    def create_empty_response(
        content: str = "-",
        agent_name: str = "Agent",
    ) -> Self:
        message = {
            "content": [{"text": f"No response received: {content}"}],
            "role": "assistant",
        }

        return AgentResponse(
            agent_result=AgentResult(
                stop_reason="end_turn",
                message=message,
                metrics=EventLoopMetrics(),
                state=None,
            ),
            session_messages=[message],
            agent_name=agent_name,
        )

    def _extract_answers(self, response: str):
        answers = []

        # find shortest match with ?
        answer_pattern = r"<answer>(.*?)</answer>"

        # find longest match without ?
        zero_eq_pattern = r"Eq\(.*, 0\)"

        # Search only inside answer tags
        log.info("Extracting answers between answer tags")
        for xml_match in re.findall(answer_pattern, response, re.DOTALL):
            for equation in re.findall(zero_eq_pattern, xml_match):
                answers.append(equation)

        # If no answers found, search in whole response
        if not answers:
            log.info("Extracting answers from the whole response")
            for equation in re.findall(zero_eq_pattern, response):
                answers.append(equation)

        return answers

    def _get_separators(
        self,
        title: str,
        sep: str = "=",
        max_sep_len: int = 100,
    ) -> tuple[str, str]:
        start_pre = f"{10 * sep} {title}: start "
        start_post = (max_sep_len - len(start_pre)) * sep
        start_sep = start_pre + start_post

        end_pre = f"{10 * sep} {title}: end "
        end_post = (max_sep_len - len(end_pre)) * sep
        end_sep = end_pre + end_post

        return start_sep, end_sep

    def _format_message(self, message: dict) -> str:
        try:
            return "\n\n".join(
                map(self._format_message_content_dct, message["content"])
            )

        except Exception as e:
            msg = f"Error formatting message:\n {message}\n {e}"
            log.error(msg)
            return msg

    def _format_message_content_dct(
        self,
        content_dct: dict,
        indent=4,
    ) -> str:
        if "text" in content_dct:
            return content_dct["text"]

        elif "toolUse" in content_dct:
            prefix = indent * " "
            name = content_dct["toolUse"]["name"]
            input_dct = content_dct["toolUse"]["input"]

            toolUseId = content_dct["toolUse"]["toolUseId"]
            self.tool_use_ids_names[toolUseId] = name

            msg = f"Tool Use `{name}` with args:"
            for arg, val in input_dct.items():
                msg += f"\n{prefix}{arg}: {val}"

            return msg

        elif "toolResult" in content_dct:
            prefix = indent * " "
            toolUseId = content_dct["toolResult"]["toolUseId"]
            name = self.tool_use_ids_names.get(toolUseId, "???")
            status = content_dct["toolResult"]["status"]
            content_dcts = content_dct["toolResult"]["content"]

            msg = f"Tool Result `{name}`:"
            msg += f"\n{prefix}status: {status}"
            msg += f"\n{prefix}content:"
            for inner_dct in content_dcts:
                inner_content = self._format_message_content_dct(
                    content_dct=inner_dct,
                    indent=2 * indent,
                )
                msg += f"\n{prefix}{prefix}{inner_content}"

            return msg

        elif "reasoningContent" in content_dct:
            start_sep, end_sep = self._get_separators(
                title="thinking",
                sep="-",
            )
            content = content_dct["reasoningContent"]["reasoningText"]["text"]
            return f"{start_sep}\n\n{content}\n\n{end_sep}"

        else:
            start_sep, end_sep = self._get_separators(
                title="unformatted",
                sep="-",
            )
            return f"{start_sep}\n\n{content_dct}\n\n{end_sep}"

    def to_string(self, with_trace: bool = True):
        answer_str = "\n".join(self.answers)
        metrics_str = metrics_to_string(self.metrics)
        trace_str = self.trace if with_trace else "... redacted ..."

        return (
            f"{self.agent_name} Response:\n\n"
            f"===Stop Reason===\n{self.stop_reason}\n\n"
            f"===Metrics===\n{metrics_str}\n\n"
            f"===Trace===\n{trace_str}\n\n"
            f"===Answers===\n{answer_str}\n\n"
        )

    def get_trace(self) -> str:
        return self.trace

    def get_user_trace(self) -> str:
        return self.user_trace

    def get_assistant_trace(self) -> str:
        return self.assistant_trace

    def get_answers(self) -> list[str]:
        return self.answers

    def get_stop_reason(self) -> str:
        return self.stop_reason

    def get_metrics(self) -> dict:
        return self.metrics


class BaseAgent:
    mcp_clients = {
        "sequential_thinking": MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-sequential-thinking",
                    ],
                )
            )
        )
    }

    _system_prompt = """\
    You are exceptional at mathematics and at finding randomized
    self-reductions (RSRs) for functions. An RSR is a powerful property where
    a function f(x) can be computed by evaluating f at random correlated
    points. You have deep mathematical knowledge spanning algebra, analysis,
    group theory, and computational mathematics. Your goal is to discover
    these reductions that reveal the hidden mathematical structure of functions
    - showing how f(x) relates to f(x+r), f(x-r), f(r) for random r. These
    properties enable self-correction, instance hiding, and other applications.

    Think deeply: Each function has hidden symmetries and patterns. Your role
    is not just to find properties mechanically, but to understand WHY they
    exist. When you discover a property, it's a window into the function's
    soul - use it to guide your next exploration. Your mathematical insight
    and intuition are crucial - use them to guide your exploration and
    recognize elegant patterns.

    You are allowed to respond only in the following format and do not forget
    to include all opening and closing XML tags in your response:
        <reasoning>
        Provide detailed mathematical reasoning. When you discover a property,
        explain WHY it holds based on the function's nature. Connect properties
        to show how they relate. If something fails verification, explain what
        you learned from it.
        </reasoning>
        <answer>
        Only include properties that you have verified or have strong
        mathematical confidence in. Quality matters more than quantity.
        </answer>
        """

    custom_tool_prompts = {
        "infer_property_tool": (
            """\
        - Try to reason on your own and find properties that you can
        verify using the `symbolic_verify_tool`. Use this tool in a batched
        fashion in order to preserve tool calls. For instance, if you want
        to verify 5 properties, use one tool call with all those properties
        in its argument.

        - If the `symbolic_verify_tool` indicates that the verification
        failed, then it will show you the simplified expression as well.
        You can keep that expression and add it to your property under-test
        in order to form a new correct property. This tactic, however, should
        be used on non-trivial properties."""
        ),
        "symbolic_verify_tool": (
            """\
        - You can use the `infer_property_tool`, whose most important
        parameters are `exprs` and `max_degree`. Remember that the tool will
        use the `exprs` to create polynomials up to `max_degree`. This means
        that your already found solutions might come up again, but also
        new ones might appear. You can use this tool multiple times with
        different parameters. However, note that it does not make sense to
        call this tool many times with the same `exprs` and different
        `max_degree`. Instead, call it once with the greatest `max_degree`
        that you want."""
        ),
    }

    def create_prompt_from_functions(
        self,
        functions: list[callable],
        custom_tool_names: list[str],
    ):
        def prompt_from_tool_names(tool_names):
            if not custom_tool_names:
                return ""

            available_tools = list(self.custom_tool_prompts.keys())
            tool_prompts = ""

            for name in tool_names:
                if name not in available_tools:
                    log.warning(f"Provided unavailable tool: {name}")
                else:
                    tool_prompts += "\n\n" + self.custom_tool_prompts[name]

            if not tool_prompts:
                return ""

            return f"""
        <tool_usage>
        You are given a series of tools, which you can use to aid
        your search. You can call them as many times as you want.
        Perform the search according to the following instructions:
        {tool_prompts}
        </tool_usage>
        """

        def format_func(func, indent=8):
            prefix = indent * " "
            func_str = inspect.getsource(func)
            source_lines = func_str.splitlines()
            return "\n".join(map(lambda s: prefix + s, source_lines))

        functions_spec = "\n\n".join(map(format_func, functions))

        return f"""
        <instructions>
        You are given some functions implemented in python. You need to
        understand their implementation and note their names. Your goal is to
        discover randomized self-reductions (RSRs) for these functions.

        An RSR is a property that allows computing f(x) by evaluating f at
        random correlated points. Formally, an RSR consists of:
        - Query functions: q_i(x,r) that generate correlated evaluation points
        - Recovery function: p that combines results to recover f(x)
        Such that: f(x) = p(x, r, f(q_1(x,r)), ..., f(q_k(x,r)))

        For a single function f(x), you should explore properties containing
        terms like: f(x+r), f(x-r), f(r), f(2x+r), etc., where r represents
        randomness. These properties reveal how f(x) can be computed from f
        evaluated at other correlated points, exposing the function's hidden
        mathematical structure.

        RSRs enable powerful applications:
        - Self-correction: Fix errors by using multiple evaluations
        - Instance hiding: Compute f(x) without revealing x
        - Computational efficiency: Sometimes simpler to compute at related points

        <rsr_thinking>
        When searching for RSRs, think systematically:
        1. **Query Functions**: What transformations of x make sense?
           - Additive: x+r, x-r, x+2r, etc.
           - Multiplicative: x*r, x/r (if applicable)
           - Compositions: f(g(x+r)) where g is related to f

        2. **Recovery Function**: How do the queried values combine?
           - Linear combinations: a*f(x+r) + b*f(x-r) + c*f(r)
           - Products: f(x+r)*f(x-r)*...
           - Rational expressions: numerator/denominator forms

        3. **Mathematical Structure**: What drives the relationship?
           - Symmetries (even, odd, periodic)
           - Algebraic identities (addition formulas, etc.)
           - Analytic properties (derivatives, series expansions)
        </rsr_thinking>

        <mathematical_approach>
        Think of this as mathematical detective work. Each property you find
        is a clue about the function's deeper structure. When selecting
        expressions to explore, each choice is a hypothesis about what
        relationships might exist.

        Draw upon your mathematical knowledge - consider what you know about
        similar functions, algebraic structures, and mathematical patterns.
        Your understanding of mathematics can guide you to discover non-obvious
        relationships that pure computation might miss.

        **Deep Exploration Strategy**:
        - When you find a property, ask: "Why does this hold? What does it tell
          me about the function's structure?"
        - Look for patterns: for instance, if f(2*x) and f(3*x) have a specific
                             pattern, try to generalize it, like f(cx).
        - Consider special values: What happens at x=0, x=π/c, x=π/2c?
        - Explore symmetries: If you find one symmetry, are there related ones?
        - Connect properties: How do discovered properties relate to each other?
        - Form conjectures: Based on patterns, hypothesize new relationships
        </mathematical_approach>

        {prompt_from_tool_names(custom_tool_names)}

        <reflection>
        As you discover properties, consider: What mathematical structure do
        they reveal? Properties often form families - if you find one, related
        ones may exist.

        Trust your mathematical intuition. Sometimes the most elegant
        properties come from recognizing deeper patterns that connect to
        fundamental mathematical concepts you already understand. Let your
        knowledge guide your exploration beyond mechanical search.

        **Quality over Quantity**:
        - Always verify discovered properties before including them in your answer
        - If a property fails verification, analyze why - it might lead to an insight
        - Look for the most general form of a property
        - Consider edge cases and domain restrictions

        **Systematic Exploration**:
        - Start with known mathematical identities, then generalize
        - If you find f(2x) relations, systematically check f(3x), f(4x), etc.
        - Explore both additive (f(x+y)) and multiplicative (f(xy)) structures
        - Consider compositions: if g(f(x)) has properties, what about f(g(x))?
        </reflection>
        </instructions>

        <formatting>
        Format the found properties as zero equalities in the format of sympy,
        one property per line. Contain only these properties inside the
        <answer>, </answer> tags. All of the other words should be inside the
        <reason>, </reason> tags.

        Here is an example of the correct output format:
        <example>
        Provided functions:
            def f(x, c=5):
                return c * x

        Your potential answer:
            <reasoning>
                All your reasoning...
            </reasoning>
            <answer>
                Eq(f(x) - f(x-y) - f(y), 0)
                Eq(f(x+y) - f(x-y) - 2*f(y), 0)
                ...
            </answer>
        </example>
        </formatting>

        <instructions>
        These are the provided functions:\n{functions_spec}

        Try to find as many properties in sympy format as possible and do
        not forget to format your response using the XML tags.
        </instructions>
        """

    def query(
        self,
        prompt: str,
        tools: list = None,
        timeout_sec: float = None,
        *kwargs,
    ) -> AgentResponse:
        return AgentResponse.create_empty_response()
