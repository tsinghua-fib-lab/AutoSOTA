import os
import re
import subprocess
import time

from pycparser import c_ast, c_generator, c_parser

from bitween import c_types
from bitween.config import Config
from bitween.miscs import getLogger

"""
This module provides a function to fuzz a C function by replacing vtrace calls with
printf calls and handling vassume calls.
"""

config = Config()
log = getLogger(__name__, config.logger_level)


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


def preprocess_c_code(c_code):
    """
    Preprocesses the C code by removing comments, vtrace function definitions, and adding stdio.h if not included.
    """
    # Extract preprocessor directives and store them
    preprocessor_directives = re.findall(r"^\s*#.*$", c_code, flags=re.MULTILINE)
    # Check if stdio.h is included
    stdio_included = any(
        "#include <stdio.h>" in directive for directive in preprocessor_directives
    )
    # Add stdio.h if not included
    if not stdio_included:
        preprocessor_directives.append("#include <stdio.h>")

    assert_included = any(
        "#include <assert.h>" in directive for directive in preprocessor_directives
    )
    # Add assert.h if not included
    if not assert_included:
        preprocessor_directives.append("#include <assert.h>")

    # Add math.h if not included
    math_included = any(
        "#include <math.h>" in directive for directive in preprocessor_directives
    )
    # Add math.h if not included
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


class TransformFunc(c_ast.NodeVisitor):
    """
    A class to transform a C function by replacing vtrace calls with printf calls and handling vassume calls.
    """

    def __init__(self, func_name):
        self.func_name = func_name
        self.trace_headers = {}
        self.func_name_to_return_type = {}
        # Added to capture the entry function's signature
        self.params = []
        self.return_type = "void"  # Added to capture the function's return type
        self.distr = {}  # Dictionary to store distribution intervals

    def visit_FuncDef(self, node):
        self.curr_params = []
        self.curr_variables = {}
        self.curr_return_type = "void"

        # Extract the function's return type
        self.curr_return_type = (
            node.decl.type.type.type.names[0]
            if isinstance(node.decl.type, c_ast.FuncDecl)
            else "int"
        )
        if node.decl.name == self.func_name:
            self.return_type = self.curr_return_type
        else:
            self.func_name_to_return_type[node.decl.name] = self.curr_return_type

        # Collect function parameter types
        if node.decl.type.args is not None:
            params = node.decl.type.args.params
            for param in params:
                if isinstance(param.type, c_ast.PtrDecl):
                    log.warning("Pointer parameters are not supported.")
                    continue
                elif isinstance(param.type, c_ast.ArrayDecl):
                    log.warning("Array parameters are not supported.")
                    continue
                type_name = param.type.type.names[0]
                self.curr_variables[param.name] = type_name
                self.curr_params.append((param.name, type_name))
                if node.decl.name == self.func_name:
                    self.params.append((param.name, type_name))

        # if block_items is None, the function is defined but not implemented
        if node.body.block_items is None:
            return

        # Collect local variable types from declarations
        for decl in node.body.block_items:
            if isinstance(decl, c_ast.Decl) and (
                not isinstance(decl.type, c_ast.ArrayDecl)
                and not isinstance(decl.type, c_ast.PtrDecl)
            ):
                self.curr_variables[decl.name] = decl.type.type.names[0]

        print(f"Function Name: {node.decl.name}")
        print(f"Parameters: {self.curr_params}")
        print(f"Return Type: {self.curr_return_type}")
        print(f"Types: {self.curr_variables}")

        # Traverse and modify the function body
        self.replace_vtrace_calls(node.body)

    def replace_vtrace_calls(self, node):
        if isinstance(node, c_ast.Compound):
            items_to_replace = []
            for item in node.block_items:
                if isinstance(item, c_ast.FuncCall):
                    if item.name.name.startswith("vtrace"):
                        new_printf_call, fflush_call = self.handle_vtrace_call(item)
                        items_to_replace.append((item, (new_printf_call, fflush_call)))
                    elif item.name.name == "vassume" or item.name.name == "assume":
                        new_if_statement = self.handle_assume_call(item)
                        items_to_replace.append((item, new_if_statement))
                    elif item.name.name == "vdistr":
                        self.handle_vdistr_call(item)
                        items_to_replace.append((item, None))

            for old_item, new_item in items_to_replace:
                index = node.block_items.index(old_item)
                if isinstance(new_item, tuple):  # This is from vtrace handling
                    node.block_items[index] = new_item[0]  # Replace with printf call
                    node.block_items.insert(index + 1, new_item[1])  # Insert fflush
                else:  # This is from vassume and vdistr handling
                    if new_item is not None:
                        node.block_items[index] = new_item  # Replace with if statement
                    else:
                        node.block_items.remove(old_item)

        # Recursively process child nodes
        for child in node:
            self.replace_vtrace_calls(child)

    def handle_vtrace_call(self, item):
        class FuncCallVisitor(c_ast.NodeVisitor):
            def __init__(self):
                self.func_call = ""
                self.func_name = ""

            def visit_FuncCall(self, node):
                self.func_name = self.resolve_expr(node.name)
                args = self.resolve_args(node.args)
                self.func_call = f"{self.func_name}({args})"
                print(self.func_call)  # Print or collect the formatted function call

            def resolve_expr(self, node):
                # This function recursively resolves any expression to a string
                if isinstance(node, c_ast.ID):
                    return node.name
                elif isinstance(node, c_ast.Constant):
                    return node.value
                elif isinstance(node, c_ast.BinaryOp):
                    left = self.resolve_expr(node.left)
                    right = self.resolve_expr(node.right)
                    return f"({left} {node.op} {right})"
                elif isinstance(node, c_ast.UnaryOp):
                    expr = self.resolve_expr(node.expr)
                    return f"{node.op}{expr}"
                else:
                    return ""

            def resolve_args(self, node):
                # This function resolves the arguments of the function call
                if isinstance(node, c_ast.ExprList):
                    args = [self.resolve_expr(expr) for expr in node.exprs]
                    return ", ".join(args)
                else:
                    return ""

        arg_names = []
        arg_format_specifiers = []
        for arg in item.args.exprs:
            if isinstance(arg, c_ast.FuncCall):
                visitor = FuncCallVisitor()
                visitor.visit(arg)
                arg_names.append(visitor.func_call)
                func_name = visitor.func_name
                if func_name in self.func_name_to_return_type:
                    return_type = self.func_name_to_return_type[func_name]
                else:
                    return_type = c_types.math_functions.get(func_name, "double")
                arg_format_specifiers.append(
                    c_types.format_specifiers.get(return_type, "%lf")
                )
            else:
                arg_names.append(arg.name)
                arg_format_specifiers.append(self.format_specifier(arg.name))

        args_str = f"{item.name.name}; " + "; ".join(arg_format_specifiers)

        self.trace_headers[f"{item.name.name}"] = f"{item.name.name}; " + "; ".join(
            [f"I {name}" for name in arg_names]
        )
        new_printf_call = c_ast.FuncCall(
            name=c_ast.ID(name="printf"),
            args=c_ast.ExprList(
                exprs=[
                    c_ast.Constant(type="string", value=f'"{args_str}\\n"'),
                    *item.args.exprs,
                ]
            ),
        )
        fflush_call = c_ast.FuncCall(
            name=c_ast.ID(name="fflush"),
            args=c_ast.ExprList(exprs=[c_ast.ID(name="stdout")]),
        )
        return new_printf_call, fflush_call

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

    # def handle_vassume_call(self, item):
    #     # Negate the condition and create an if statement
    #     negated_condition = c_ast.UnaryOp(op="!", expr=item.args.exprs[0])
    #     return_statement = c_ast.Return(expr=c_ast.Constant(type="int", value="0"))
    #     if_statement = c_ast.If(
    #         cond=negated_condition,
    #         iftrue=c_ast.Compound(block_items=[return_statement]),
    #         iffalse=None,
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

    def format_specifier(self, variable_name):
        # Get the type name from the current variables, default to "int" if not found.
        type_name = self.curr_variables.get(variable_name, "int")
        # Return the format specifier for the type name, defaulting to "%d" if type not found.
        return c_types.format_specifiers.get(type_name, "%d")


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
    executable = source_file.replace(".fuzzer.c", ".fuzzer")
    compilation_command = ["gcc", source_file, "-o", executable, "-lm"]
    try:
        subprocess.run(compilation_command, check=True)
        log.debug(f"Compilation successful: {executable} created.")
        return executable
    except subprocess.CalledProcessError:
        log.error("Compilation failed.")
        raise SystemExit


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


def fuzz_function_to_collect_traces(
    executable, trace_path, iterations, params, distributions
):
    """
    Fuzzes the given C function by calling it with random inputs and writes the output
    to a trace file named after the function, while keeping other outputs such as status
    messages and errors in the console.
    """

    # extract the file name from the path
    file_name = os.path.basename(executable)
    file_name = os.path.splitext(file_name)[0]

    with open(trace_path, "w+") as trace_file:
        results = []
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
                print(f"Run Failed | Input: {input_data} | {error.strip()}")
            else:
                out = ""
                # Process each line in the output
                for line in output.splitlines():
                    if line.startswith("vtrace"):
                        trace_file.write(
                            line + "\n"
                        )  # Write trace lines to the trace file
                    else:
                        out += line + "; "

                # extract the trace markers from the output (starts with 'vtraceN;')
                print(f"Run Passed | Input: {input_data} | {out.strip()}")


# Sort the trace file by trace markers
def sort_file_by_trace_marker(trace_path, trace_headers):
    """
    Sorts the lines in a file based on the trace markers (e.g., vtrace1, vtrace2, etc.).
    """
    with open(trace_path, "r") as file:
        lines = file.readlines()

    # Create a dictionary to hold lines based on their trace marker
    trace_dict = {}
    for line in lines:
        # Assume line format starts with 'vtraceN;' where N is a number
        marker = line.split(";")[0]
        if marker in trace_dict:
            trace_dict[marker].append(line)
        else:
            trace_dict[marker] = [line]

    # Sort dictionary keys to ensure vtrace1, vtrace2,... order
    sorted_keys = sorted(trace_dict.keys(), key=lambda x: (x[:-1], int(x[-1])))

    # Write sorted lines back to file
    with open(trace_path, "w") as file:
        for key in sorted_keys:
            file.write(trace_headers[key] + "\n")
            file.writelines(trace_dict[key])


def fuzz_and_trace(file_path: str, func_name: str, iterations: int):
    """
    Fuzzes the given C function by calling it with random inputs and writes the output as a trace file.
    """

    # remove the .c extension from the file_path
    folder_path = os.path.dirname(file_path)

    # NOTE: Load the C code to be fuzzed
    start_time = time.time()

    c_code = read_c_file(file_path)

    if c_code is None:
        print("Error reading the C file.")
        exit()

    # NOTE: Preprocessing part

    # Preprocess the C code and extract preprocessor directives
    # CParser does not handle preprocessor directives, so we need to extract and add them back later
    preprocessed_code, preprocessor_directives = preprocess_c_code(c_code)

    # NOTE: Parsing and instrumentation part

    # Parse the code using pycparser
    parser = c_parser.CParser()
    ast = parser.parse(preprocessed_code.strip())

    # remove the assume function from the C code
    assume_remover = RemoveAssumeVisitor()
    assume_remover.visit(ast)

    # Visit and process the function definition
    transformer = TransformFunc(func_name)
    transformer.visit(ast)

    # Generate the modified C code
    generator = c_generator.CGenerator()
    modified_code = generator.visit(ast)

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
    test_driver_file = f"{folder_path}/{func_name}.fuzzer.c"
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

    trace_path = f"{folder_path}/{func_name}.trace.csv"

    # Fuzz the function with random inputs
    fuzz_function_to_collect_traces(
        executable, trace_path, iterations, transformer.params, transformer.distr
    )

    # NOTE: Post-processing part

    # Sort the trace file by trace markers
    sort_file_by_trace_marker(trace_path, transformer.trace_headers)

    print(f"Fuzzing time: {time.time() - start_time:.2f} seconds.")

    print(f"Trace file written to: {trace_path}")
    print(f"Fuzzing file written to: {test_driver_file}")

    return trace_path


if __name__ == "__main__":
    # Example usage

    # This should be the path to your C file
    file_path = "./benchmarks/bitween/dig/bresenham.c"
    func_name = "bresenham"  # This should be the name of the function you want to fuzz

    # file_path = "./benchmarks/bitween/dig/cohencu.c"
    # func_name = "cohencu"

    # file_path = "./benchmarks/bitween/dig/cohendiv.c"
    # func_name = "cohendiv"

    # file_path = "./benchmarks/bitween/dig/dijkstra.c"
    # func_name = "dijkstra"

    # file_path = "./benchmarks/bitween/dig/divbin.c"
    # func_name = "divbin"

    # file_path = "./benchmarks/bitween/dig/egcd.c"
    # func_name = "egcd"

    # file_path = "./benchmarks/bitween/dig/egcd2.c"
    # func_name = "egcd2"

    # file_path = "./benchmarks/bitween/dig/egcd3.c"
    # func_name = "egcd3"

    # file_path = "./benchmarks/bitween/dig/fermat1.c"
    # func_name = "fermat1"

    # file_path = "./benchmarks/bitween/dig/fermat2.c"
    # func_name = "fermat2"

    # file_path = "./benchmarks/bitween/dig/freire1_int.c"
    # func_name = "freire1_int"

    # file_path = "./benchmarks/bitween/dig/freire1.c"
    # func_name = "freire1"

    # file_path = "./benchmarks/bitween/dig/freire2.c"
    # func_name = "freire2"

    # file_path = "./benchmarks/bitween/dig/geo1.c"
    # func_name = "geo1"

    # file_path = "./benchmarks/bitween/dig/geo2.c"
    # func_name = "geo2"

    # file_path = "./benchmarks/bitween/dig/geo3.c"
    # func_name = "geo3"

    # file_path = "./benchmarks/bitween/dig/hard.c"
    # func_name = "hard"

    # file_path = "./benchmarks/bitween/dig/knuth.c"
    # func_name = "knuth"

    # file_path = "./benchmarks/bitween/dig/lcm1.c"
    # func_name = "lcm1"

    # file_path = "./benchmarks/bitween/dig/lcm2.c"
    # func_name = "lcm2"

    # file_path = "./benchmarks/bitween/dig/mannadiv.c"
    # func_name = "mannadiv"

    # file_path = "./benchmarks/bitween/fpcore/rosa.c"
    # func_name = "doppler1"

    # file_path = "./benchmarks/bitween/fpcore/salsa.c"
    # func_name = "Odometry"
    # func_name = "PID"
    # func_name = "Runge_Kutta_4"

    iterations = 30  # Number of times to call the function with random inputs

    fuzz_and_trace(file_path, func_name, iterations)
