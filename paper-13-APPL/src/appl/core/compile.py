from __future__ import annotations

import ast
import inspect
import linecache
import os
import re
import sys
import textwrap
import traceback
from ast import (
    AST,
    Assign,
    Attribute,
    Call,
    Constant,
    Expr,
    FormattedValue,
    FunctionDef,
    JoinedStr,
    Load,
    Name,
    NamedExpr,
    NodeTransformer,
    Store,
    With,
    stmt,
)
from types import CodeType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import libcst as cst
from loguru import logger

from .context import PromptContext

GLOBALS_KEYWORD = ast.keyword(
    arg="_globals",
    value=Call(func=Name(id="globals", ctx=Load()), args=[], keywords=[]),
)
LOCALS_KEYWORD = ast.keyword(
    arg="_locals",
    value=Call(func=Name(id="locals", ctx=Load()), args=[], keywords=[]),
)
CTX_KEYWORD = ast.keyword(arg="_ctx", value=Name(id="_ctx", ctx=Load()))
CTX_ARG = ast.arg(arg="_ctx", annotation=Name(id="PromptContext", ctx=Load()))


def _has_arg(args: Union[List[ast.arg], List[ast.keyword]], name: str) -> bool:
    return any(
        isinstance(arg, (ast.arg, ast.keyword)) and arg.arg == name for arg in args
    )


class ApplNodeTransformer(NodeTransformer):
    """A base class for AST node transformers in APPL."""

    def __init__(self, compile_info: Dict, *args: Any, **kwargs: Any) -> None:
        """Initialize the transformer with compile info."""
        super().__init__(*args, **kwargs)
        self._compile_info = compile_info

    def _raise_syntax_error(self, lineno: int, col_offset: int, msg: str) -> None:
        file = self._compile_info["sourcefile"]
        lineno = lineno + self._compile_info["lineno"] - 1
        text = linecache.getline(file, lineno)
        raise SyntaxError(msg, (file, lineno, col_offset, text))


class RemoveApplDecorator(ApplNodeTransformer):
    """An AST node transformer that removes the ppl decorator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the transformer with the outmost flag."""
        super().__init__(*args, **kwargs)
        self._outmost = True

    def _is_ppl_decorator(self, decorator: AST) -> bool:
        if isinstance(decorator, Name):
            return decorator.id == "ppl"
        elif isinstance(decorator, Call):
            if isinstance(func := decorator.func, Name):
                return func.id == "ppl"
        return False  # pragma: no cover

    def visit_FunctionDef(self, node):
        """Remove the ppl decorator from the function definition."""
        if node.decorator_list:
            for decorator in node.decorator_list:
                if self._is_ppl_decorator(decorator):
                    if not self._outmost:
                        self._raise_syntax_error(
                            decorator.lineno,
                            decorator.col_offset,
                            "Nested ppl decorator is not allowed yet for APPL.",
                        )
            # all decorators should be removed
            node.decorator_list = []
        if self._outmost:
            self._outmost = False
        self.generic_visit(node)
        return node


class SplitString(ApplNodeTransformer):
    """An AST node transformer that splits the f-string into multiple parts."""

    def _add_formatted_value(self, node: FormattedValue) -> Iterable[stmt]:
        format_args = [node.value]
        if node.format_spec is not None:
            format_args.append(node.format_spec)
        format_keywords = []
        if node.conversion != -1:
            # add conversion
            format_keywords.append(
                ast.keyword(arg="conversion", value=Constant(node.conversion))
            )
        expr = Call(
            func=Attribute(
                value=Name(id="appl", ctx=Load()),
                attr="format",
                ctx=Load(),
            ),
            args=format_args,
            keywords=format_keywords,
        )
        # converted to `appl.format(value, format_spec)`
        default_result: Iterable[stmt] = [Expr(expr)]
        if isinstance(node.value, NamedExpr):
            return default_result

        if spec := node.format_spec:
            spec_str = ast.unparse(spec)
            # logger.debug(spec_str)
            if spec_str and spec_str[2] == "=":  # f"= ..."
                self._raise_syntax_error(
                    node.lineno,
                    node.col_offset,
                    "Not supported format of named expression inside f-string. "
                    "To use named expression, please add brackets around "
                    f"`{ast.unparse(node)[3:-2]}`.",
                )
        return default_result

    def visit_Expr(self, node: Expr) -> stmt:
        """Split the f-string into multiple parts, so that we can add appl.execute wrapper to each part."""
        if isinstance(node.value, JoinedStr):
            fstring = node.value
            # logger.debug(f"For joined string: {fstring}")
            body: List[stmt] = []
            for value in fstring.values:
                if isinstance(value, Constant):
                    body.append(Expr(value))
                elif isinstance(value, FormattedValue):
                    body.extend(self._add_formatted_value(value))
                else:
                    raise ValueError(
                        f"Unknown value type in a JoinedStr: {type(value)}"
                    )
            if len(body) == 0:  # empty string
                return node
            if len(body) == 1:  # single string
                return body[0]
            return With(
                items=[
                    ast.withitem(
                        context_expr=Call(
                            func=Attribute(
                                value=Name(id="appl", ctx=Load()),
                                attr="Str",
                                ctx=Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    )
                ],
                body=body,
            )
        return node


class CallWithContext(ApplNodeTransformer):
    """An AST node transformer provides the context to function calls."""

    def visit_Call(self, node: Call) -> Call:
        """Provide context (_ctx) to function calls that needs ctx."""
        self.generic_visit(node)
        # logger.debug(f"visit Call: {ast.dump(node, indent=4)}")
        # * use appl.with_ctx as wrapper for all functions,
        # * pass _ctx to the function annotated with @need_ctx
        new_node = Call(
            Attribute(
                value=Name(id="appl", ctx=Load()),
                attr="with_ctx",
                ctx=Load(),
            ),
            node.args,
            node.keywords,
        )
        new_node.keywords.append(ast.keyword(arg="_func", value=node.func))
        # add _ctx to kwargs if not present
        if not _has_arg(node.keywords, "_ctx"):
            new_node.keywords.append(CTX_KEYWORD)
        new_node.keywords.append(GLOBALS_KEYWORD)
        new_node.keywords.append(LOCALS_KEYWORD)
        return new_node


class AddCtxToArgs(ApplNodeTransformer):
    """An AST node transformer that adds _ctx to the function arguments."""

    def visit_FunctionDef(self, node: FunctionDef) -> FunctionDef:
        """Add _ctx to the function arguments if not present."""
        # ! only add _ctx to outermost function def, not call generic_visit here.
        # self.generic_visit(node) # !! do not call
        args = node.args
        # add _ctx to kwargs if not present
        if not _has_arg(args.args, "_ctx") and not _has_arg(args.kwonlyargs, "_ctx"):
            args.kwonlyargs.append(CTX_ARG)
            # PromptContext() as default
            args.kw_defaults.append(
                Call(func=Name(id="PromptContext", ctx=Load()), args=[], keywords=[])
            )
        for var in self._compile_info["freevars"]:
            args.kwonlyargs.append(ast.arg(arg=var))
            args.kw_defaults.append(ast.Name(id=var, ctx=Load()))
            logger.debug(f"add freevar {var} to function {node.name} args.")
        return node


class AddExecuteWrapper(ApplNodeTransformer):
    """An AST node transformer that adds the appl.execute wrapper to expression statements."""

    def visit_Expr(self, node: Expr) -> Expr:
        """Add appl.execute wrapper to expression statements."""
        return Expr(
            Call(
                func=Attribute(
                    value=Name(id="appl", ctx=Load()),
                    attr="execute",
                    ctx=Load(),
                ),
                args=[node.value],
                keywords=[CTX_KEYWORD],  # , GLOBALS_KEYWORD, LOCALS_KEYWORD],
            )
        )


class DedentTripleQuotedString(cst.CSTTransformer):
    """A CST transformer that dedents triple-quoted strings in the source code."""

    def visit_Module(self, node: cst.Module) -> bool:
        """Store the module node for later use."""
        self.module = node
        return True

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.SimpleString:
        """Dedent triple-quoted strings in the source code."""
        value = original_node.value
        delim = value[-3:]
        # Check if the string is wrapped by triple quotes (""" or ''')
        if delim in ['"""', "'''"]:
            has_prefix_r = False
            if value.startswith("r" + delim):  # deal with raw string
                has_prefix_r = True
                value = value[1:]

            assert value[:3] == delim
            # Remove the quotes to process the inner string content
            inner_string = value[3:-3]

            # Use the standard of cleandoc to dedent the inner string
            modified_inner_string = inspect.cleandoc(inner_string)

            # Add back the triple quotes to the modified string
            new_value = f"{delim}{modified_inner_string}{delim}"
            if has_prefix_r:
                new_value = "r" + new_value

            logger.debug(
                f"multiline string dedented\n"
                f"before:\n{original_node.value}\n"
                f"after:\n{new_value}"
            )

            # Return the updated SimpleString node with the modified value
            return updated_node.with_changes(value=new_value)

        return updated_node

    # format string
    def leave_FormattedString(
        self, original_node: cst.FormattedString, updated_node: cst.FormattedString
    ) -> cst.FormattedString:
        """Dedent triple-quoted formatted strings in the source code."""
        if len(original_node.end) != 3:
            return updated_node

        def get_num_leading_whitespace(line: str) -> int:
            """Compute the leading whitespace of this line."""
            if g := re.match(r"^[ \t]*", line):
                return len(g.group())
            return 0

        def dedent_str(text: str) -> str:
            cleaned = inspect.cleandoc(text)
            lines = cleaned.splitlines()
            margin: Optional[int] = None
            for line in lines:
                n = get_num_leading_whitespace(line)
                # try to fix a bug in `libcst.code_for_node``
                # the leading whitespace of '}' is not recovered correctly
                # as well as the contents within `{` and `}` (TODO)
                if len(line) > n and line[n] != "}":
                    if margin is None:
                        margin = n
                    else:
                        margin = min(margin, n)
            if margin is not None and margin > 0:
                new_lines = []
                for line in lines:
                    n = get_num_leading_whitespace(line)
                    new_lines.append(line[min(n, margin) :])
                cleaned = os.linesep.join(new_lines)
                cleaned = inspect.cleandoc(cleaned)  # clean again
                logger.warning(
                    f"unusual multiline formatted string detected, "
                    f"this is a bug related to libcst.code_for_node\n"
                    f"The string after manual fix is:\n{cleaned}"
                )
            logger.debug(
                f"multiline formmated string dedented\n"
                f"before:\n{text}\n"
                f"after:\n{cleaned}"
            )
            return cleaned

        # example: f"""    foo\n    bar
        original_code = self.module.code_for_node(original_node)
        # example: start = f""", end = """
        simple_str = original_code[len(original_node.start) : -len(original_node.end)]
        # example: simple_str = foo\n    bar
        format_str = f"{original_node.start}{dedent_str(simple_str)}{original_node.end}"
        # example: format_str = f"""foo\nbar"""
        parsed_cst = cst.parse_expression(format_str)
        # replace the original node with the parsed cst
        if isinstance(parsed_cst, cst.FormattedString):
            return parsed_cst
        raise RuntimeError(
            f"Failed to parse modified formatted string as libcst.FormattedString: {format_str}"
        )


def dedent_triple_quoted_string(code: str) -> str:
    """Automatically dedent triple-quoted strings with in the code (with inspect.cleandoc)."""
    # Parse the source code into a CST
    cst_module = cst.parse_module(code)
    # Apply the transformer to dedent triple-quoted strings
    cst_transformer = DedentTripleQuotedString()
    modified_cst_module = cst_module.visit(cst_transformer)
    # return the modified code
    return modified_cst_module.code


def _get_docstring_quote_count(code: str) -> Optional[int]:
    cst_tree = cst.parse_module(code)
    for node in cst_tree.body:
        if isinstance(node, cst.FunctionDef):
            # Only detect the first func
            statement = node.body.body[0]
            if isinstance(statement, cst.SimpleStatementLine):
                if isinstance(statement.body[0], cst.Expr):
                    s = statement.body[0].value
                    if isinstance(s, cst.SimpleString):
                        if s.value.startswith('"""') or s.value.startswith("'''"):
                            return 3
                        else:
                            return 1
            return None
    assert False, "no function detected"  # should not reach here


class APPLCompiled:
    """A compiled APPL function that can be called with context."""

    def __init__(
        self, code: CodeType, ast: AST, original_func: Callable, compile_info: Dict
    ):
        """Initialize the compiled function.

        Args:
            code: The compiled code object.
            ast: The AST of the compiled code.
            original_func: The original function.
            compile_info: The compile information.
        """
        self._code = code
        self._ast = ast
        self._name = original_func.__name__
        self._original_func = original_func
        self._compile_info = compile_info
        self._docstring_quote_count = _get_docstring_quote_count(compile_info["source"])

    @property
    def freevars(self) -> Tuple[str, ...]:
        """Get the free variables of the compiled function."""
        if "freevars" in self._compile_info:
            return self._compile_info["freevars"]
        return self._original_func.__code__.co_freevars

    def __call__(
        self,
        *args: Any,
        _globals: Optional[Dict] = None,
        _locals: Dict = {},
        **kwargs: Any,
    ) -> Any:
        """Call the compiled function."""
        _globals = _globals or self._original_func.__globals__
        if "appl" not in _globals:
            _globals["appl"] = sys.modules["appl"]
        local_vars = {"PromptContext": PromptContext}
        # get closure variables from locals
        for name in self.freevars:
            if name in _locals:
                # set the closure variables to local_vars
                local_vars[name] = _locals[name]
            elif name == "__class__":
                # attempt for super() workaround
                # def super_(t: Any = None, obj: Any = None):
                #     t = t or self_.__class__
                #     obj = obj or self_
                #     return super(t, obj)
                local_vars["__class__"] = args[0].__class__
            else:
                raise ValueError(
                    f"Freevar '{name}' not found. If you are using closure variables, "
                    "please provide their values in the _locals argument. "
                    "For example, assume the function is `func`, use `func(..., _locals=locals())`. "
                    "Alternatively, you can first use the `appl.as_func` to convert the "
                    "function within the current scope (automatically feeding the locals)."
                )

        exec(self._code, _globals, local_vars)  # TODO: use closure argument
        func = local_vars[self._name]
        return func(*args, **kwargs)

    def __repr__(self):
        return f"APPLCompiled({self._name})"


def appl_dedent(source: str) -> str:
    """Dedent the source code."""
    source = textwrap.dedent(source)
    try:
        ast.parse(source)
    except Exception:  # the dedent failed due to multiline string
        logger.warning(
            "The source code contains multiline string that cannot be dedented. "
            "It is recommended to dedent the multiline string aligning with the function, "
            "where APPL will automatically dedent the multiline string "
            "in the same way as cleaning docstring."
        )
        # Compute the dedent and remove the leading whitespace
        leading_whitespace = re.compile("(^[ \t]*)(?:[^ \t\n])", re.MULTILINE)
        indents = leading_whitespace.findall(source)
        margin = indents[0]  # use the first line as the standard
        source = re.sub(r"(?m)^" + margin, "", source)
        logger.warning(f"The source code after workaround:\n{source}")
    return source


def appl_compile(func: Callable) -> APPLCompiled:
    """Compile an APPL function."""
    sourcefile = inspect.getsourcefile(func)
    lines, lineno = inspect.getsourcelines(func)
    source = appl_dedent(inspect.getsource(func))
    key = f"<appl-compiled:{sourcefile}:{lineno}>"
    linecache.cache[key] = (
        len(source),
        None,
        [line + os.linesep for line in source.splitlines()],
        key,
    )

    source = dedent_triple_quoted_string(source)
    parsed_ast = ast.parse(source)
    logger.debug(
        f"\n{'-'*20} code BEFORE appl compile {'-'*20}\n{ast.unparse(parsed_ast)}"
    )

    transformers = [
        RemoveApplDecorator,
        SplitString,
        CallWithContext,
        AddCtxToArgs,
        AddExecuteWrapper,
    ]
    compile_info = {
        "source": source,
        "sourcefile": sourcefile,
        "lineno": lineno,
        "func_name": func.__name__,
        "freevars": func.__code__.co_freevars,
        "docstring": func.__doc__,
    }
    for transformer in transformers:
        parsed_ast = transformer(compile_info).visit(parsed_ast)

    parsed_ast = ast.fix_missing_locations(parsed_ast)
    compiled_ast = compile(parsed_ast, filename=key, mode="exec")
    logger.debug(
        f"\n{'-'*20} code AFTER appl compile {'-'*20}\n{ast.unparse(parsed_ast)}"
    )

    return APPLCompiled(compiled_ast, parsed_ast, func, compile_info)
