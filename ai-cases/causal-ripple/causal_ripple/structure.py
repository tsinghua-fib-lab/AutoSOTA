"""Identifier and delimiter closure for the Structural-CORE expert."""

from __future__ import annotations

import ast

import torch

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {value: key for key, value in OPEN_TO_CLOSE.items()}


def _piece(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def _identifier_piece(text):
    stripped = text.strip()
    return bool(stripped) and all((char.isalnum() or char == "_" for char in stripped))


def expand_structural_mask(input_ids, seed_mask, tokenizer, valid_mask=None, max_total=6):
    if tokenizer is None or max_total <= 1:
        return seed_mask.clone()
    if valid_mask is None:
        valid_mask = torch.ones_like(seed_mask, dtype=torch.bool)
    expanded = seed_mask.clone()
    for batch in range(input_ids.shape[0]):
        pieces = [_piece(tokenizer, token) for token in input_ids[batch].detach().cpu().tolist()]
        valid = valid_mask[batch].detach().cpu().tolist()
        stacks = {opening: [] for opening in OPEN_TO_CLOSE}
        pairs = {}
        for position, text in enumerate(pieces):
            if not valid[position]:
                continue
            symbol = text.strip()
            if symbol in OPEN_TO_CLOSE:
                stacks[symbol].append(position)
            elif symbol in CLOSE_TO_OPEN and stacks[CLOSE_TO_OPEN[symbol]]:
                left = stacks[CLOSE_TO_OPEN[symbol]].pop()
                pairs[left], pairs[position] = (position, left)
        selected = set(torch.nonzero(seed_mask[batch], as_tuple=False).flatten().tolist())
        proposals = []
        for seed in list(selected):
            if seed in pairs:
                proposals.append((abs(pairs[seed] - seed), pairs[seed]))
            if _identifier_piece(pieces[seed]):
                for direction in (-1, 1):
                    position = seed + direction
                    while 0 <= position < len(pieces) and valid[position]:
                        if not _identifier_piece(pieces[position]):
                            break
                        if direction == 1 and pieces[position][:1].isspace():
                            break
                        if direction == -1 and pieces[position + 1][:1].isspace():
                            break
                        proposals.append((abs(position - seed), position))
                        position += direction
        for _, position in sorted(set(proposals)):
            if len(selected) >= max_total:
                break
            selected.add(position)
        if selected:
            positions = torch.tensor(sorted(selected), device=expanded.device)
            expanded[batch, positions] = True
    return expanded


def syntax_error_seed(token_ids, tokenizer, offset=0):
    """Locate the token nearest Python's first syntax error."""
    pieces = [_piece(tokenizer, token_id) for token_id in token_ids]
    clean = ["" if piece.startswith("<|") else piece for piece in pieces]
    text = "".join(clean)
    try:
        ast.parse(text)
        return (None, None)
    except SyntaxError as error:
        lines = text.splitlines(keepends=True)
        line = max(1, error.lineno or 1)
        column = max(1, error.offset or 1)
        char_index = sum((len(value) for value in lines[: line - 1])) + column - 1
        cursor = 0
        nearest = 0
        for position, piece in enumerate(clean):
            next_cursor = cursor + len(piece)
            nearest = position
            if cursor <= char_index < max(cursor + 1, next_cursor):
                return (offset + position, error.msg)
            cursor = next_cursor
        return (offset + nearest, error.msg)
