"""Lexical guards and static Python certificates for post-generation routing."""

from __future__ import annotations

import ast
import builtins
import re
import textwrap
from dataclasses import dataclass

import torch

OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "=",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    ":",
    ",",
    ".",
    ";",
    "->",
    "+=",
    "-=",
    "*=",
    "/=",
}
STOP_MARKERS = ("[DONE]", "\nclass", "\ndef", "\n#", "\nif __name__", "\nprint(")


def token_piece(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return ""
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def normalized_piece(piece: str) -> str:
    return piece.strip()


def local_lexical_risk(pieces: list[str]) -> int:
    """Count only high-precision local lexical defects."""
    tokens = [normalized_piece(piece) for piece in pieces]
    tokens = [token for token in tokens if token]
    risk = 0
    for left, right in zip(tokens, tokens[1:]):
        if left == "=" and right == "=":
            risk += 3
        if left in {".", ",", ":"} and right in {".", ",", ":"}:
            risk += 2
        if left in {"(", "[", "{"} and right in {")", "]", "}"}:
            continue
        if left in OPERATORS and right in OPERATORS:
            allowed = {
                ("=", "="),
                ("!", "="),
                ("<", "="),
                (">", "="),
                ("*", "*"),
                ("/", "/"),
                ("-", ">"),
                (":", "="),
            }
            if (left, right) not in allowed:
                risk += 1
    return risk


def filter_local_safe_selection(
    input_ids: torch.Tensor,
    repair_mask: torch.Tensor,
    proposal: torch.Tensor,
    tokenizer,
    window: int = 4,
) -> torch.Tensor:
    """Reject proposed token edits that increase obvious local lexical risk."""
    if tokenizer is None:
        return repair_mask.clone()
    kept = repair_mask.clone()
    length = input_ids.shape[1]
    for batch in range(input_ids.shape[0]):
        for position in torch.nonzero(repair_mask[batch], as_tuple=False).flatten().tolist():
            start, end = (max(0, position - window), min(length, position + window + 1))
            before_ids = input_ids[batch, start:end].detach().cpu().tolist()
            after_ids = list(before_ids)
            after_ids[position - start] = int(proposal[batch, position].item())
            before = [token_piece(tokenizer, token) for token in before_ids]
            after = [token_piece(tokenizer, token) for token in after_ids]
            if local_lexical_risk(after) > local_lexical_risk(before):
                kept[batch, position] = False
    return kept


class _NameAudit(ast.NodeVisitor):
    def __init__(self):
        self.defined = set(dir(builtins))
        self.loaded = []
        self.argument_names = set()
        self.calls_to_arguments = 0

    def visit_arg(self, node):
        self.defined.add(node.arg)
        self.argument_names.add(node.arg)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            self.defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.loaded.append(node.id)

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.argument_names:
            self.calls_to_arguments += 1
        self.generic_visit(node)


@dataclass(frozen=True, order=True)
class StaticCertificate:
    parse_ok: int
    negative_undefined: int
    negative_argument_calls: int
    negative_lexical_risk: int


def _certificate_for_text(text: str) -> StaticCertificate:
    lexical = local_lexical_risk(re.findall("\\w+|==|!=|<=|>=|:=|->|\\S", text))
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return StaticCertificate(0, -10000, -10000, -lexical)
    audit = _NameAudit()
    audit.visit(tree)
    undefined = len({name for name in audit.loaded if name not in audit.defined})
    return StaticCertificate(1, -undefined, -audit.calls_to_arguments, -lexical)


def _program_candidates(input_ids, prompt_length: int, tokenizer) -> list[str]:
    ids = input_ids.detach().cpu().tolist()
    full = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    generated = tokenizer.decode(
        ids[prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    texts = [full, generated, textwrap.dedent(generated)]
    for text in list(texts):
        for marker in STOP_MARKERS:
            offset = text.find(marker, 1)
            if offset > 0:
                texts.append(text[:offset])
    return [text for text in texts if text.strip()]


def python_certificate(input_ids, prompt_length: int, tokenizer) -> StaticCertificate:
    if tokenizer is None:
        return StaticCertificate(0, -10000, -10000, 0)
    certificates = [
        _certificate_for_text(text)
        for text in _program_candidates(input_ids, prompt_length, tokenizer)
    ]
    return max(certificates, default=StaticCertificate(0, -10000, -10000, 0))


def transactional_finalize(
    current: torch.Tensor,
    rollback: torch.Tensor,
    touched: torch.Tensor,
    prompt_length: int,
    tokenizer,
) -> tuple[torch.Tensor, int]:
    """Rollback touched tokens if the completed program certificate worsens."""
    output = current.clone()
    rolled_back = 0
    for batch in range(current.shape[0]):
        if not touched[batch].any():
            continue
        alternative = current[batch].clone()
        alternative[touched[batch]] = rollback[batch, touched[batch]]
        current_cert = python_certificate(current[batch], prompt_length, tokenizer)
        rollback_cert = python_certificate(alternative, prompt_length, tokenizer)
        if rollback_cert > current_cert:
            output[batch] = alternative
            rolled_back += int(touched[batch].sum().item())
    return (output, rolled_back)
