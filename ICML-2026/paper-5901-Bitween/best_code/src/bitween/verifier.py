import os
import re
import subprocess

import sympy
from pycparser import c_ast, c_generator, c_parser
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import precedence

from bitween import c_types, miscs
from bitween.config import Config

config = Config()
log = miscs.getLogger(__name__, config.logger_level)

"""
This module provides a function to verify a C function with civl.
https://vsl.cis.udel.edu/trac/civl/wiki/Introduction
"""


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


# https://github.com/sympy/sympy/blob/master/sympy/printing/c.py
# https://stackoverflow.com/questions/65534432/generate-c-code-with-sympy-replace-powx-2-by-xx
class CivlCCodePrinter(C99CodePrinter):
    """
    Due to some pecularities of CIVL's pow function assumption,
    which enforces the base should be positive.
    """

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == 2:
            # TODO: exhaustively check for all possible cases
            return "{0}*{0}".format(self.parenthesize(expr.base, PREC))
        else:
            return super()._print_Pow(expr)


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

            if node.decl.type.args is not None:
                # Collect function parameter types
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

            # Process and Remove vdistr calls from the function body
            items = []
            for item in node.body.block_items:
                if isinstance(item, c_ast.FuncCall) and item.name.name == "vdistr":
                    self.handle_vdistr_call(item)
                else:
                    items.append(item)
            node.body.block_items = items

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
                name=c_ast.ID(name="$assert"),
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
                name=c_ast.ID(name="$assert"),
                args=c_ast.ExprList(exprs=[c_ast.Constant(type="int", value=equation)]),
            )

    def handle_assume_call(self, item):
        # The original condition is directly used as the first argument to the assume function.
        condition_expr = item.args.exprs[0]

        # Create a call to assume(condition, "Assumption violated: condition failed.");
        assume_call = c_ast.FuncCall(
            name=c_ast.ID(name="$assume"),
            args=c_ast.ExprList(
                exprs=[
                    condition_expr,  # Use the direct condition, not the negated
                ]
            ),
        )

        return assume_call

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


class FunctionRenamer(c_ast.NodeVisitor):
    def __init__(self, func_name):
        self.func_name = func_name

    def visit_FuncDef(self, node):
        # Change the function's name to "main"
        if node.decl.name == self.func_name:
            node.decl.name = "main"
            # Ensure the return type is correctly updated to int
            if isinstance(node.decl.type, c_ast.FuncDecl):
                if isinstance(node.decl.type.type, c_ast.TypeDecl):
                    node.decl.type.type.declname = "main"
                    node.decl.type.type.type = c_ast.IdentifierType(names=["int"])
                # Set parameters to None (no arguments)
                node.decl.type.args = None


def create_extern_function_prototype(func_name, params, return_type):
    # Create the parameter string by joining type and name for each parameter
    param_string = ", ".join(
        f"{param_type} {param_name}" for param_name, param_type in params
    )
    # Construct the full function prototype
    return f"extern {return_type} {func_name}({param_string});"


class CustomDecl:
    """Class to hold a custom declaration with additional attributes."""

    def __init__(self, declaration, custom_attribute, upper_bound=None):
        self.declaration = declaration
        self.custom_attribute = custom_attribute
        self.upper_bound = upper_bound

    def children(self):
        """Yield the child nodes, required by pycparser for visiting the AST."""
        return [("declaration", self.declaration)]


class InputDeclarationAdder(c_ast.NodeVisitor):
    def __init__(self, params, upper_bounds=None):
        self.params = params
        self.upper_bounds = upper_bounds or {}

    def visit_FileAST(self, node):
        new_decls = []
        for param_name, param_type in self.params:
            # Determine the initial value from upper_bounds if available
            interval = self.upper_bounds.get(param_name)
            if interval is not None:
                # Use the upper bound as the initial value
                init_value = interval[1]
            else:
                if param_type in c_types.int_types:
                    init_value = c_types.int_upper_bound_civl
                elif param_type in c_types.float_types:
                    init_value = c_types.float_upper_bound_civl
            # Create constant initializer if an initial value is determined
            init = (
                c_ast.Constant(type=param_type, value=str(init_value))
                if init_value is not None
                else None
            )
            # Create the declaration
            decl = c_ast.Decl(
                name=param_name,
                quals=[],
                storage=[],
                funcspec=[],
                align=None,
                type=c_ast.TypeDecl(
                    declname=param_name,
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(names=[param_type]),
                ),
                init=init,  # Set the initial value if specified
                bitsize=None,
            )
            custom_decl = CustomDecl(decl, "$input", init_value)
            new_decls.append(custom_decl)

        node.ext = new_decls + node.ext if node.ext else new_decls


class CustomCGenerator(c_generator.CGenerator):
    def visit_CustomDecl(self, node):
        custom_attribute = node.custom_attribute
        decl = super().visit_Decl(node.declaration)
        return f"{custom_attribute} {decl}"


def verify_w_civl(file_path):

    civl_path = os.path.join(config.project_dir, config.civl_path)
    if not os.path.exists(civl_path):
        log.error("Error: The CIVL JAR file does not exist.")
        raise FileNotFoundError(
            "The CIVL JAR file does not exist. Please provide it in the config file."
        )

    # Construct the command to run the Java process
    command = ["java", "-jar", civl_path, "verify", file_path]

    try:
        # Execute the command and capture the output and error
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Check if there was an error
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
        else:
            print(result.stdout)

    except Exception as e:
        # Return any exceptions raised during the execution as errors
        return {"error": str(e)}


def fuzz_and_verify(file_path, func_name, trace_equations):
    """
    Fuzzes the given C function by calling it with random inputs and checks the assertions.
    """
    for trace, equations in trace_equations.items():
        trace_equations[trace] = [sympy.ccode(equation) for equation in equations]
        # NOTE: Civl team fixed the bug in the pow function assumption
        # trace_equations[trace] = [
        #     CivlCCodePrinter().doprint(equation) for equation in equations
        # ]

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

    # Rename the function to 'main'
    renamer = FunctionRenamer(func_name)
    renamer.visit(ast)

    # Add input declarations to the main function
    input_adder = InputDeclarationAdder(transformer.params, transformer.distr)
    input_adder.visit(ast)

    # Generate the modified C code
    generator = CustomCGenerator()
    modified_code = generator.visit(ast)

    prototype = create_extern_function_prototype(
        func_name, transformer.params, transformer.return_type
    )

    # Add the is_close function to the code
    if transformer.float_assertions:
        modified_code = get_is_close_code() + "\n" + modified_code

    # Add back the preprocessor directives and the function prototype
    final_code = (
        "\n".join(preprocessor_directives) + "\n" + prototype + "\n" + modified_code
    )
    # print(final_code)

    # write the final code to a file
    verifier_file = f"{folder_path}/{func_name}.cvl"
    with open(verifier_file, "w") as file:
        file.write(final_code)

    # NOTE: Verification part

    print(f"Verifying {func_name} function with symbolic execution...")
    print(f"Params: {transformer.params}")
    if transformer.distr:
        print(f"Distributions: {transformer.distr}")

    print(f"Verifier file: {verifier_file}")

    # verify the C code with civl
    verify_w_civl(verifier_file)

    print(f"verifier file: {verifier_file}")


if __name__ == "__main__":
    # Example usage

    # This should be the path to your C file
    file_path = "./benchmarks/bitween/dig/bresenham.c"
    func_name = "bresenham"

    fuzz_and_verify(
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
    )

    print("\n")

    file_path = "./benchmarks/bitween/dig/cohencu.c"
    func_name = "cohencu"

    fuzz_and_verify(
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
    )

    print("\n")

    file_path = "./benchmarks/bitween/dig/knuth.c"
    func_name = "knuth"

    fuzz_and_verify(
        file_path,
        func_name,
        {
            "vtrace1": [
                sympy.parse_expr("d*k - d*t - a*k + a*t == 0", evaluate=False),
                sympy.parse_expr("k*t == t * t", evaluate=False),
            ],
            "vtrace2": [
                sympy.parse_expr("a*k*r - a*r*t - d*k*r + d*r*t == 0", evaluate=False),
            ],
        },
    )
