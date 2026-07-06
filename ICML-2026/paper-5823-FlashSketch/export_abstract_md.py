from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from pathlib import Path
import re
from typing import Dict, Tuple

from abstract_md_config import AbstractMdConfig, DEFAULT_CONFIG


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def strip_comments(text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    lines = []
    for line in text.splitlines():
        index = 0
        cleaned = []
        while index < len(line):
            char = line[index]
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                break
            cleaned.append(char)
            index += 1
        lines.append("".join(cleaned))
    return "\n".join(lines)


def extract_abstract(text: str) -> str:
    """Extract the contents of the abstract environment."""
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def skip_whitespace(text: str, index: int) -> int:
    """Advance index past whitespace."""
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def parse_braced(text: str, start_index: int) -> Tuple[str, int]:
    """Parse a braced group and return its content and next index."""
    if start_index >= len(text) or text[start_index] != "{":
        raise ValueError("Expected opening brace.")
    depth = 0
    content_start = start_index + 1
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            if depth == 1:
                content_start = index + 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index + 1
        index += 1
    raise ValueError("Unmatched brace.")


def parse_newcommands(text: str) -> Dict[str, Tuple[int, str]]:
    """Parse \\newcommand definitions into a macro map."""
    macros: Dict[str, Tuple[int, str]] = {}
    index = 0
    while True:
        index = text.find("\\newcommand", index)
        if index == -1:
            break
        index += len("\\newcommand")
        index = skip_whitespace(text, index)
        if index >= len(text) or text[index] != "{":
            continue
        name_content, index = parse_braced(text, index)
        name = name_content.strip()
        if name.startswith("\\"):
            name = name[1:]
        index = skip_whitespace(text, index)
        arg_count = 0
        if index < len(text) and text[index] == "[":
            end_index = text.find("]", index + 1)
            if end_index == -1:
                raise ValueError("Unmatched bracket in macro definition.")
            arg_count = int(text[index + 1 : end_index])
            index = end_index + 1
        index = skip_whitespace(text, index)
        if index >= len(text) or text[index] != "{":
            continue
        body, index = parse_braced(text, index)
        macros[name] = (arg_count, body)
    return macros


def expand_macros_once(text: str, macros: Dict[str, Tuple[int, str]]) -> Tuple[str, bool]:
    """Expand macros a single pass."""
    output = []
    index = 0
    changed = False
    while index < len(text):
        if text[index] != "\\":
            output.append(text[index])
            index += 1
            continue
        name_start = index + 1
        name_end = name_start
        while name_end < len(text) and text[name_end].isalpha():
            name_end += 1
        if name_end == name_start:
            output.append(text[index])
            index += 1
            continue
        name = text[name_start:name_end]
        if name not in macros:
            output.append(text[index:name_end])
            index = name_end
            continue
        arg_count, body = macros[name]
        cursor = name_end
        args = []
        for _ in range(arg_count):
            cursor = skip_whitespace(text, cursor)
            if cursor >= len(text) or text[cursor] != "{":
                break
            arg_text, cursor = parse_braced(text, cursor)
            args.append(arg_text)
        if len(args) != arg_count:
            output.append(text[index:name_end])
            index = name_end
            continue
        expanded = body
        for arg_index, arg_text in enumerate(args, start=1):
            expanded = expanded.replace(f"#{arg_index}", arg_text)
        if arg_count == 0:
            cursor = skip_whitespace(text, cursor)
            if cursor < len(text) and text[cursor] == "{":
                empty_text, next_cursor = parse_braced(text, cursor)
                if empty_text.strip() == "":
                    cursor = next_cursor
        output.append(expanded)
        index = cursor
        changed = True
    return "".join(output), changed


def expand_macros(text: str, macros: Dict[str, Tuple[int, str]]) -> str:
    """Expand macros with multiple passes to resolve nesting."""
    current = text
    for _ in range(10):
        expanded, changed = expand_macros_once(current, macros)
        current = expanded
        if not changed:
            break
    return current


def strip_commands(text: str, commands: Tuple[str, ...]) -> str:
    """Strip commands that take one braced argument."""
    output = []
    index = 0
    command_set = set(commands)
    while index < len(text):
        if text[index] != "\\":
            output.append(text[index])
            index += 1
            continue
        name_start = index + 1
        name_end = name_start
        while name_end < len(text) and text[name_end].isalpha():
            name_end += 1
        name = text[name_start:name_end]
        if name in command_set:
            cursor = skip_whitespace(text, name_end)
            if cursor < len(text) and text[cursor] == "{":
                _, cursor = parse_braced(text, cursor)
                index = cursor
                continue
        output.append(text[index])
        index += 1
    return "".join(output)


def unescape_latex(text: str) -> str:
    """Convert common LaTeX escapes to literal characters."""
    replacements = {
        r"\%": "%",
        r"\&": "&",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for latex, value in replacements.items():
        text = text.replace(latex, value)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace into Markdown-friendly paragraphs."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = []
    for paragraph in text.split("\n\n"):
        compact = re.sub(r"[ \t]+", " ", paragraph.replace("\n", " ")).strip()
        if compact:
            paragraphs.append(compact)
    normalized = "\n\n".join(paragraphs).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized + "\n"


def latex_to_markdown(text: str) -> str:
    """Convert a LaTeX fragment into Markdown."""
    command_macros = {
        "textbf": (1, "**#1**"),
        "emph": (1, "*#1*"),
        "textit": (1, "*#1*"),
        "textsc": (1, "#1"),
        "textcolor": (2, "#2"),
    }
    text = expand_macros(text, command_macros)
    text = text.replace(r"\xspace", " ")
    text = strip_commands(text, ("cite", "Cref", "cref", "ref", "label", "footnote"))
    text = text.replace("\\\\", "\n")
    text = text.replace("~", " ")
    text = unescape_latex(text)
    text = text.replace("``", "\"").replace("''", "\"")
    text = text.replace("---", "—").replace("--", "–")
    return text.strip()


def build_markdown(config: AbstractMdConfig) -> str:
    """Build the Markdown abstract string from LaTeX sources."""
    abstract_tex = strip_comments(read_text(config.abstract_tex_path))
    abstract_body = extract_abstract(abstract_tex)
    macros_tex = strip_comments(read_text(config.macros_tex_path))
    macros = parse_newcommands(macros_tex)
    expanded = expand_macros(abstract_body, macros)
    markdown = latex_to_markdown(expanded)
    if config.normalize_whitespace:
        markdown = normalize_whitespace(markdown)
    return markdown


def main() -> None:
    """Entry point for exporting the abstract."""
    markdown = build_markdown(DEFAULT_CONFIG)
    write_text(DEFAULT_CONFIG.output_md_path, markdown)


if __name__ == "__main__":
    main()
