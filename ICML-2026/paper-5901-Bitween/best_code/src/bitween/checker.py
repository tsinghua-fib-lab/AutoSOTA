import os
import re
import subprocess

import sympy
from pycparser import c_ast, c_generator, c_parser

from bitween import c_types, miscs
from bitween.config import Config

"""
This module provides a function to fuzz a C function with random inputs and
check assertions.
"""

config = Config()
log = miscs.getLogger(__name__, config.logger_level)


def read_c_file(file_path):
    try:
        with open(file_path, "r") as file:
            c_code = file.read()
        return c_code
    except FileNotFoundError:
        print(f"Error: The file does not exist: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_is_close_code():
    return f"""
#define EPSILON {config.epsilon}  // Define the precision threshold

int is_close(double a, double b) {{
    return fabs(a - b) < EPSILON;  // Use fabs to get the absolute difference
}}
"""


def preprocess_c_code(c_code):
    """
    Preprocesses the C code by removing comments, vtrace function definitions, and adding stdio.h if not included.
    """
    # Extract preprocessor directives and store them
    preprocessor_directives = re.findall(r"^\s*#.*$", c_code, flags=re.MULTILINE)
    # Add stdio.h if not included
    stdio_included = any(
        "#include <stdio.h>" in directive for directive in preprocessor_directives
    )
    if not stdio_included:
        preprocessor_directives.append("#include <stdio.h>")

    # Add assert.h if not included
    assert_included = any(
        "#include <assert.h>" in directive for directive in preprocessor_directives
    )
    if not assert_included:
        preprocessor_directives.append("#include <assert.h>")

    # Add math.h if not included
    math_included = any(
        "#include <math.h>" in directive for directive in preprocessor_directives
    )
    if not math_included:
        preprocessor_directives.append("#include <math.h>")

    # Add stdlib.h if not included
    stdlib_included = any(
        "#include <stdlib.h>" in directive for directive in preprocessor_directives
    )
    if not stdlib_included:
        preprocessor_directives.append("#include <stdlib.h>")

    # Remove directives and vtrace function definitions from the code
    preprocessed_code = re.sub(
        r"^\s*#.*$|^void vtrace[0-9]+\(.*\)\s*\{\s*\}", "", c_code, flags=re.MULTILINE
    )

    # Pattern to match single-line and multi-line comments
    comments_pattern = r"//.*?$|/\*.*?\*/"
    # Pattern to remove specific function definitions
    functions_pattern = r"^void vtrace[0-9]+\s*\(.*\)\s*\{.*?\}"

    # Remove comments
    preprocessed_code = re.sub(
        comments_pattern, "", preprocessed_code, flags=re.DOTALL | re.MULTILINE
    )
    # Remove vtrace function definitions
    preprocessed_code = re.sub(
        functions_pattern, "", preprocessed_code, flags=re.MULTILINE | re.DOTALL
    )

    return preprocessed_code.strip(), preprocessor_directives


class TransformFuncForAssertions(c_ast.NodeVisitor):
    def __init__(self, func_name, trace_equations):
        self.func_name = func_name
        # Dictionary of trace locations to equations
        self.trace_equations = trace_equations
        # Added to capture the entry function's signature
        self.params = []
        self.return_type = None  # Added to capture the function's return type
        self.distr = {}  # Dictionary to store distribution intervals
        self.float_assertions = False  # assertions involve floating point numbers

    def visit_FuncDef(self, node):
        # Extract the function's return type
        if node.decl.name == self.func_name:
            self.return_type = (
                node.decl.type.type.type.names[0]
                if isinstance(node.decl.type, c_ast.FuncDecl)
                else "int"
            )

        # Collect function parameter types
        if node.decl.type.args is not None:
            params = node.decl.type.args.params
            if node.decl.name == self.func_name:
                for param in params:
                    type_name = param.type.type.names[0]
                    if type_name in c_types.float_types:
                        self.float_assertions = True
                    self.params.append((param.name, type_name))

        # if block_items is None, the function is defined but not implemented
        if node.body.block_items is None:
            return

        if node.decl.name == self.func_name:
            # Traverse and modify the function body if it matches the specified function name
            self.replace_vtrace_calls(node.body)

    def replace_vtrace_calls(self, node):
        if isinstance(node, c_ast.Compound):
            new_items = []
            for item in node.block_items:
                if (
                    isinstance(item, c_ast.FuncCall)
                    and item.name.name in self.trace_equations
                ):
                    # Replace vtrace calls with assert statements
                    for equation in self.trace_equations[item.name.name]:
                        assert_stmt = self.create_assert_statement(equation)
                        new_items.append(assert_stmt)
                elif isinstance(item, c_ast.FuncCall) and item.name.name == "vdistr":
                    self.handle_vdistr_call(item)
                    continue
                elif isinstance(item, c_ast.FuncCall) and (
                    item.name.name == "assume" or item.name.name == "vassume"
                ):
                    # Transform assume calls into if conditions leading to return
                    new_if_statement = self.handle_assume_call(item)
                    new_items.append(new_if_statement)
                else:
                    new_items.append(item)
            node.block_items = new_items

        # Recursively process child nodes
        for child in node:
            self.replace_vtrace_calls(child)

    def create_assert_statement(self, equation):
        # Create an assert statement for the given equation
        if self.float_assertions and config.is_close_for_float:
            # remove the " == 0" part from the equation
            equation = equation.replace(" == 0", "")  # TODO: generalize this
            return c_ast.FuncCall(
                name=c_ast.ID(name="assert"),
                args=c_ast.ExprList(
                    exprs=[
                        c_ast.FuncCall(
                            name=c_ast.ID(name="is_close"),
                            args=c_ast.ExprList(
                                exprs=[
                                    c_ast.Constant(type="double", value=equation),
                                    c_ast.Constant(type="double", value="0"),
                                ]
                            ),
                        )
                    ]
                ),
            )
        else:  # Integer assertions
            return c_ast.FuncCall(
                name=c_ast.ID(name="assert"),
                args=c_ast.ExprList(exprs=[c_ast.Constant(type="int", value=equation)]),
            )

    def handle_assume_call(self, item):
        # The original condition is directly used as the first argument to the assume function.
        condition_expr = item.args.exprs[0]

        # Create a call to assume(condition, "Assumption violated: condition failed.");
        assume_call = c_ast.FuncCall(
            name=c_ast.ID(name="assume"),
            args=c_ast.ExprList(
                exprs=[
                    condition_expr,  # Use the direct condition, not the negated
                ]
            ),
        )

        return assume_call

    # def handle_assume_call(self, item):
    #     # Transform assume call into an if statement that checks condition and aborts if false
    #     negated_condition = c_ast.UnaryOp(op="!", expr=item.args.exprs[0])
    #     # Create a call to abort() function
    #     abort_call = c_ast.FuncCall(
    #         name=c_ast.ID(name="abort"), args=None  # abort does not take any arguments
    #     )
    #     # Use the abort call in a compound statement (i.e., a block of code)
    #     abort_statement = c_ast.Compound(block_items=[abort_call])

    #     # Create an if statement that calls abort if the negated condition is true
    #     if_statement = c_ast.If(
    #         cond=negated_condition, iftrue=abort_statement, iffalse=None
    #     )

    #     return if_statement

    def handle_vdistr_call(self, item):
        # Assuming the vdistr call looks like vdistr(var, min, max)
        if len(item.args.exprs) == 3:
            var_name = item.args.exprs[0].name
            min_val = self.extract_value(item.args.exprs[1])
            max_val = self.extract_value(item.args.exprs[2])
            if var_name not in [param[0] for param in self.params]:
                print(f"Variable {var_name} not found in function parameters.")
                return
            self.distr[var_name] = [min_val, max_val]  # Store the distribution interval
        else:
            print("Invalid vdistr call:", item)

    def extract_value(self, expr):
        # This method extracts the value from the expression,
        # whether it is a direct constant or a negated value.
        if isinstance(expr, c_ast.Constant):
            type = expr.type
            value = expr.value
            if type in c_types.int_types:
                return int(value)
            elif type in c_types.float_types:
                return float(value.strip("f"))
        elif isinstance(expr, c_ast.UnaryOp) and expr.op == "-":
            # Handle negation
            value = self.extract_value(expr.expr)
            return -value
        else:
            raise ValueError(f"Unsupported expression type for bounds: {expr}")


class RemoveMainVisitor(c_ast.NodeVisitor):
    """
    A class to remove the main function from the C code.
    """

    def visit_FuncDef(self, node):
        # Check if the function name is 'main' and remove it
        if node.decl.name == "main":
            # Remove the node by returning None (effectively removing it)
            return None
        # Continue traversing the AST
        self.generic_visit(node)


class RemoveAssumeVisitor(c_ast.NodeVisitor):
    """
    A class to remove the assume function from the C code.
    """

    def visit_FuncDef(self, node):
        # Check if the function name is 'assume' and remove it
        if node.decl.name == "assume" or node.decl.name == "vassume":
            # Remove the node by returning None (effectively removing it)
            return None
        # Continue traversing the AST
        self.generic_visit(node)


def generate_assume_code():
    code = """
void assume(int cond) {
    if (!cond) {
        fprintf(stderr, "Assumption violated.\\n");
        abort();
    }
}"""
    return code


def create_test_driver(func_name, params, return_type):
    """
    Generates a main function to test the given C function with command-line arguments.
    """
    # String template for the main function
    main_function = "int main(int argc, char** argv) {\n"
    main_function += f"    if (argc != {len(params) + 1}) {{\n"
    main_function += '        printf("Usage: %s '
    for param_name, param_type in params:
        main_function += f"<{param_name}:{param_type}> "
    main_function += '\\n", argv[0]);\n        return 1;\n    }\n\n'
    for i, (param_name, param_type) in enumerate(params, start=1):
        conversion_function = "atoi" if param_type in c_types.int_types else "atof"
        main_function += (
            f"    {param_type} {param_name} = {conversion_function}(argv[{i}]);\n"
        )
    param_names = ", ".join([name for name, _ in params])
    if return_type != "void":
        main_function += f"    {return_type} result = {func_name}({param_names});\n"
        print_statement = (
            'printf("%d\\n", result);'
            if return_type in c_types.int_types
            else 'printf("%f\\n", result);'
        )
        main_function += f"    {print_statement}\n"
    else:
        main_function += f"    {func_name}({param_names});\n"
    main_function += "    return 0;\n}\n"
    return main_function


def compile_code(source_file):
    # Assuming gcc is installed and available in the path
    executable = source_file.replace(".checker.c", ".checker")
    compilation_command = ["gcc", source_file, "-o", executable, "-lm"]
    try:
        subprocess.run(compilation_command, check=True)
        log.debug(f"Compilation successful: {executable} created.")
        return executable
    except subprocess.CalledProcessError:
        log.error("Compilation failed.")
        raise SystemExit


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


def random_value(param_type, distr=None):
    """
    Generates a random value based on the parameter type.
    """
    func, default_range = c_types.random_functions.get(
        param_type, (lambda *args: 0, None)
    )

    if distr:
        return func(float(distr[0]), float(distr[1]))
    elif default_range:
        return func(*default_range)
    else:
        return 0


def fuzz_function_to_check_assertions(executable, iterations, params, distributions):
    results = []
    failed_lines = set()  # To store line numbers of failed assertions

    for _ in range(iterations):
        # Generate test inputs based on the specified distributions or simply random within a range
        test_inputs = []
        for param in params:
            param_name = param[0]
            param_type = param[1]
            test_inputs.append(
                str(random_value(param_type, distributions.get(param_name, None)))
            )

        # Convert list to command-line arguments
        input_str = " ".join(test_inputs)
        timeout = config.fuzz_timeout
        try:
            # Run the executable with the generated inputs and a timeout
            res = subprocess.run(
                [executable] + test_inputs,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Store the result along with the input data
            results.append((input_str, res.stdout, res.stderr, res.returncode))
        except subprocess.TimeoutExpired as e:
            results.append((input_str, "", f"Timeout after {timeout} seconds", -1))

    # Analyze the results
    for input_data, output, error, return_code in results:
        if return_code != 0:
            print(f"Test Failed | Input: {input_data} | {error.strip()}")
            # Parse the error for line number
            line_match = re.search(r"line (\d+)", error)
            if line_match:
                failed_lines.add(int(line_match.group(1)))
        else:
            print(f"Test Passed | Input: {input_data} | {output.strip()}")

    return failed_lines


def fuzz_and_check(file_path, func_name, trace_equations, iterations):
    """
    Fuzzes the given C function by calling it with random inputs and checks the assertions.
    """
    for trace, equations in trace_equations.items():
        trace_equations[trace] = [sympy.ccode(equation) for equation in equations]

    # remove the .c extension from the file_path
    folder_path = os.path.dirname(file_path)

    c_code = read_c_file(file_path)

    if c_code is None:
        print("Error reading the C file.")
        exit()

    # NOTE: Preprocessing part

    # Preprocess the C code and extract preprocessor directives
    # CParser does not handle preprocessor directives, so we need to extract and add them back later
    preprocessed_code, preprocessor_directives = preprocess_c_code(c_code)

    # NOTE: Instrument assertions part

    # Parse the code using pycparser
    parser = c_parser.CParser()
    ast = parser.parse(preprocessed_code.strip())

    # remove the main function from the C code
    main_remover = RemoveMainVisitor()
    main_remover.visit(ast)

    # remove the assume function from the C code
    assume_remover = RemoveAssumeVisitor()
    assume_remover.visit(ast)

    # Visit and process the function definition
    transformer = TransformFuncForAssertions(func_name, trace_equations)
    transformer.visit(ast)

    # Generate the modified C code
    generator = c_generator.CGenerator()
    modified_code = generator.visit(ast)

    # Add the is_close function to the code
    if transformer.float_assertions:
        modified_code = get_is_close_code() + "\n" + modified_code

    # Add our version of assume function back to the code
    modified_code = generate_assume_code() + "\n\n" + modified_code

    # NOTE: Instrument Test Driver

    # Generate the new main function
    params = transformer.params
    return_type = transformer.return_type
    main_code = create_test_driver(func_name, params, return_type)

    # Combine modified code with the new main function
    modified_code = modified_code + main_code

    # Add back the preprocessor directives
    final_code = "\n".join(preprocessor_directives) + "\n" + modified_code
    # print(final_code)

    # write the final code to a file
    test_driver_file = f"{folder_path}/{func_name}.checker.c"
    with open(test_driver_file, "w") as file:
        file.write(final_code)

    # NOTE: Compilation part

    # construct the executable
    executable = compile_code(test_driver_file)

    # NOTE: Fuzzing part

    print(f"Fuzzing {func_name} function with {iterations} random inputs...")
    print(f"Params: {transformer.params}")
    if transformer.distr:
        print(f"Distributions: {transformer.distr}")

    # Fuzz the function with random inputs
    falied_line_number_of_assertions = fuzz_function_to_check_assertions(
        executable, iterations, transformer.params, transformer.distr
    )

    # remove the failed assertions from the code
    # remove_failed_assertions(test_driver_file, falied_line_number_of_assertions)
    comment_out_assertions(test_driver_file, falied_line_number_of_assertions)

    print(f"Test driver: {test_driver_file}")


if __name__ == "__main__":
    # Example usage

    # This should be the path to your C file
    file_path = "./benchmarks/bitween/dig/bresenham.c"
    func_name = "bresenham"

    fuzz_and_check(
        file_path,
        func_name,
        {
            "vtrace1": [
                sympy.parse_expr("X + 2*X*y - 2*Y - 2*Y*x + v == 0", evaluate=False)
            ],
            "vtrace2": [
                sympy.parse_expr("2*Y*x - 2*X*y - X + 2*Y - v == 0", evaluate=False),
                sympy.parse_expr("X - x + 1 == 0", evaluate=False),
            ],
        },
        5,
    )

    print("\n")

    file_path = "./benchmarks/bitween/dig/cohencu.c"
    func_name = "cohencu"

    fuzz_and_check(
        file_path,
        func_name,
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
        5,
    )
