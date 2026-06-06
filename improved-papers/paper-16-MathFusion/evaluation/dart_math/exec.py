import json
import os
import traceback

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io

# %% ../nbs/06_exec.ipynb 0
TOOL_CONFIG_ID2DICT = {
    "python": {
        "input_begin": "```python",
        "input_end": "```",
        "output_begin": "```output",
        "output_end": "```",
    }
}


class CodeExecCfg:
    """Configuration for code execution.

    Parameters
    ----------
    input_begin : str, default: "```python"
    input_end : str, default: "```"
    output_code_prefix : str, default: "print("
        Prefix of code that will be executed to display the output.
    output_begin : str, default: "```output"
    output_end : str, default: "```"
    timeout : int, default: 5
        Timeout in seconds for code execution.
    max_n_workers : int, default: 4
        The maximum number of CPU core workers to execute the code with multi-processing.
    max_n_calls : int, default: None
        The maximum number of calls to the code execution function.
        This could be large because there is token length limit already.
        `None` / Non-positive values mean no limit.
    trunc_len : tuple[int, int], default: None
        The maximum lengths to truncate the output into the beginning and end.
        `None` / double non-positive values like `(0, 0)` mean no truncation.
    elipsis : str, default: "..."
        The elipsis to use when truncating the output.
    """

    def __init__(
        self,
        input_begin: str = "```python",
        input_end: str = "```",
        output_code_prefix: str = "print(",
        output_begin: str = "```output",
        output_end: str = "```",
        timeout: int = 5,
        max_n_workers: int = 4,
        max_n_calls: int = None,
        trunc_len: tuple[int, int] = None,
        elipsis: str = "...",
    ):
        self.input_begin = input_begin
        self.input_end = input_end
        self.output_code_prefix = output_code_prefix
        self.output_begin = output_begin
        self.output_end = output_end
        self.timeout = timeout
        self.max_n_workers = max_n_workers
        self.max_n_calls = max_n_calls
        self.trunc_len = trunc_len
        self.elipsis = elipsis

    @staticmethod
    def load_from_id_or_path(tool_config: str = "python") -> "CodeExecCfg":
        """Load the configuration from the ID or path.

        Parameters
        ----------
        tool_config : str, default: "python"
            ID / Path to file of the code executeion configuration.

        Returns
        -------
        CodeExecCfg
            The code execution configuration object.
        """
        if tool_config in TOOL_CONFIG_ID2DICT:
            return CodeExecCfg(**TOOL_CONFIG_ID2DICT[tool_config])
        elif isinstance(tool_config, str) and os.path.exists(tool_config):
            with open(tool_config, "r") as f:
                tool_config = json.loads(f.read())
            return CodeExecCfg(**tool_config)
        else:
            return CodeExecCfg()  # Default: "python"

    def no_cells_todo(self, context: str) -> int:
        """Judge if there are no code cells to execute.

        Parameters
        ----------
        context : str
            The whole context containing all the code cells.

        Returns
        -------
        int
            0: Normal
            1: No code cells to execute
            2: Output cells are more than input cells
        """
        input_cnt = context.count(self.input_begin)
        output_cnt = context.count(self.output_begin)
        if input_cnt == output_cnt:
            return 1
        elif output_cnt > input_cnt:
            return 2  # Must be an error
        else:
            return 0  # Normal

    def extract_cells(self, text: str) -> list[str]:
        """Extract code cells from the text.

        Parameters
        ----------
        text : str
            The text to extract code cells from.

        Returns
        -------
        list[str]
            The extracted code cells.
        """
        cells = [
            snippet.split(self.input_end, 1)[0].strip()
            for snippet in text.split(self.input_begin)
            if self.input_end in snippet
        ]
        cells = [cell for cell in cells if cell]  # Remove empty cells
        return cells

    def wrap_output(self, output: str) -> str:
        """Return `f"{self.output_begin}\\n{output}\\n{self.output_end}"`"""
        return f"{self.output_begin}\n{output}\n{self.output_end}"


NB_OUTPUT_PROMPT = "Out[1]: "


def exec_cells(cells: list[str]) -> str:
    """Execute the code cells like a notebook and return the stdout and stderr of the last cell.
    Modified from
    - https://github.com/Kipok/NeMo-Skills/blob/6a909ec0974340b02a1083dce90e79bea30ecb60/nemo_skills/code_execution/sandbox.py#L168-L233
    - https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/infer/run_tool_integrated_eval.py#L163-L180

    Parameters
    ----------
    cells : list[str]
        The code cells to execute.
    """
    try:
        shell = InteractiveShell()
        shell.run_cell(
            """
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'
"""
        )
        for cell in cells:
            with io.capture_output(display=False) as captured:
                shell.run_cell(cell)
                # serializing to str to make sure things like Rational can be converted to json
                stdout = captured.stdout.replace(NB_OUTPUT_PROMPT, "")
                stderr = captured.stderr.replace(NB_OUTPUT_PROMPT, "")
    except Exception:
        # removing useless prefix from traceback
        stdout = ""
        stderr = traceback.format_exc()

    return stdout.strip(), stderr.strip()
