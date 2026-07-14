import os
import re

import sympy

from bitween.config import Config
from bitween.miscs import getLogger

"""
This module provides a function to fuzz a C function with random inputs and
check assertions.
"""

config = Config()
log = getLogger(__name__, config.logger_level)


def get_is_close_code():
    return f"""
#define EPSILON {config.epsilon}  // Define the precision threshold

int is_close(double a, double b) {{
    return fabs(a - b) < EPSILON;  // Use fabs to get the absolute difference
}}
"""


def comment_out_assertions(file_path, failed_lines):
    """
    Comment out assertions in the C code based on the failed line numbers and append a specific comment
    just after the assertion ends.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Regular expression to find 'assert(' and match until the closing parenthesis
    assert_pattern = re.compile(r"\bassert\((.*)\);")

    with open(file_path, "w") as file:
        for i, line in enumerate(lines, 1):
            if i in failed_lines:
                # Find and modify the assertion
                def replace_assert(match):
                    # Capture the entire assertion expression, adding comments after it ends
                    full_assertion = match.group(0)  # The entire assertion statement
                    return f"// {full_assertion} -- Not Valid!"

                # Replace assertion in the line using the replace function
                line = re.sub(assert_pattern, replace_assert, line)

            file.write(line)


def replace_vtrace_with_equations(file_path, out_path, trace_equations):
    """
    Replace the vtrace variables in the C code with the inferred equations.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Regular expression to find 'assert(' and match until the closing parenthesis
    pattern = re.compile(r"(\s*)vtrace(\d*).*?;")

    # merge all the equations into a single string by trace locations
    merged_equations = {}
    for trace, equations in trace_equations.items():
        merged_equations[trace] = [f"assert({equation});" for equation in equations]

    with open(out_path, "w") as file:
        for i, line in enumerate(lines, 1):
            match = pattern.findall(line)
            if match:
                leading_whitespace = match[0][0]
                trace_number = "vtrace" + match[0][1]
                if trace_number in merged_equations:
                    for assertion in merged_equations[trace_number]:
                        file.write(f"{leading_whitespace}{assertion}\n")
            else:
                file.write(line)


def decorate_assertions(file_path, func_name, out_path, trace_equations):
    """
    Decorates the assertions in the given C code with the inferred equations.
    """
    for trace, equations in trace_equations.items():
        # trace_equations[trace] = [sympy.ccode(equation) for equation in equations]
        for i, equation in enumerate(equations):
            if isinstance(equation, sympy.Eq):
                trace_equations[trace][i] = sympy.ccode(equation.lhs)
            elif isinstance(equation, str):
                trace_equations[trace][i] = equation
            else:
                trace_equations[trace][i] = sympy.ccode(equation)

    # remove the .c extension from the file_path
    folder_path = os.path.dirname(file_path)

    replace_vtrace_with_equations(file_path, out_path, trace_equations)

    # TODO: Add the is_close function to the code if the type is floating point


if __name__ == "__main__":
    # Example usage

    # This should be the path to your C file
    file_path = "./benchmarks/bitween/dig/bresenham.c"
    func_name = "bresenham"
    out_path = "./benchmarks/bitween/dig/bresenham.out.c"

    decorate_assertions(
        file_path,
        func_name,
        out_path,
        {
            "vtrace1": [
                sympy.parse_expr("X + 2*X*y - 2*Y - 2*Y*x + v == 0", evaluate=False)
            ],
            "vtrace2": [
                sympy.parse_expr("2*Y*x - 2*X*y - X + 2*Y - v == 0", evaluate=False),
                sympy.parse_expr("X - x + 1 == 0", evaluate=False),
            ],
        },
    )

    print("\n")

    file_path = "./benchmarks/bitween/dig/cohencu.c"
    func_name = "cohencu"
    out_path = "./benchmarks/bitween/dig/cohencu.out.c"

    decorate_assertions(
        file_path,
        func_name,
        out_path,
        {
            "vtrace1": [
                sympy.parse_expr("6*n - z + 6 == 0", evaluate=False),
                sympy.parse_expr("12*y - z**2 + 6*z - 12 == 0", evaluate=False),
            ],
            "vtrace2": [
                sympy.parse_expr("6*a - z + 12 == 0", evaluate=False),
                sympy.parse_expr("6*n - z + 6 == 0", evaluate=False),
                sympy.parse_expr("12*y - z**2 + 6*z - 12 == 0", evaluate=False),
            ],
        },
    )
