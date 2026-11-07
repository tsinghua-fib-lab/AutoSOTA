"""DS-1000 public-context preparation and deterministic snippet cleanup."""

import ast
import builtins
import hashlib
import re

VERSION = "benchmark-contract-v2-marker-cleanup"
DEMO = "# Python 3. Complete each requested solution; stop at # END SOLUTION.\n# Demonstration: sum only the positive values of an existing list.\nvalues = [-3, 2, 5]\n# SOLUTION START\npositive_total = sum(value for value in values if value > 0)\n# END SOLUTION\n# Demonstration: implement a function without printing its return value.\ndef demo_ascii_lower(text):\n    return text.lower()\n# END SOLUTION\n"
LIBRARY_RULES = {
    "Pandas": "Use the supplied DataFrame and its actual column names; preserve indices unless the task asks otherwise.",
    "Numpy": "Use the supplied arrays; respect shapes, axes, broadcasting and requested dtype.",
    "Matplotlib": "Modify the current plot using the supplied data; preserve requested artists, labels, axes and styles.",
    "Scipy": "Use the supplied numerical data and the requested SciPy API; preserve requested return shapes.",
    "Sklearn": "Use scikit-learn 1.4 APIs; respect precomputed distances, estimator parameters and requested outputs.",
    "Tensorflow": "Use TensorFlow 2.16 APIs and supplied tensors; do not construct replacement example data.",
    "Pytorch": "Use the supplied tensors, their dimensions, dtype and device; do not replace the input data.",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def adapt_ds1000(public):
    """The caller passes an explicit whitelist, not an evaluation problem row."""
    assert (
        not {
            "solutions",
            "canonical_solution",
            "code_context",
            "base_input",
            "plus_input",
            "test",
            "input_output",
        }
        & public.keys()
    )
    task_id = public["task_id"]
    contract = dict(
        benchmark="ds1000",
        task_id=task_id,
        entry_point=public.get("entry_point"),
        output_names=[],
        provided_names=[],
        examples=[],
        assertions=[],
        doctest_text="",
    )
    prefix = ""
    official_prefix = ""
    raw = public["prompt"]
    library = public["metadata"]["library"]
    contract["library"] = library
    assignments = re.findall("(?m)^([^\\n]+?)\\s*=\\s*\\.\\.\\.\\s*# put solution", raw)
    if assignments:
        lhs = assignments[-1].strip()
        try:
            tree = ast.parse(lhs + " = None")
            contract["output_names"] = sorted(
                {
                    n.id
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
            )
            prefix = lhs + " ="
        except SyntaxError:
            pass
    blocks = re.findall("<code>\\s*\\n(.*?)</code>", raw, re.S)
    setup = blocks[-1].strip() if blocks else raw.split("# SOLUTION START", 1)[0].strip()
    question = raw.split("\nA:", 1)[0] if blocks else raw.split("# SOLUTION START", 1)[0]
    try:
        contract["provided_names"] = sorted(
            {
                n.id
                for n in ast.walk(ast.parse(setup))
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            | {
                a.asname or a.name.split(".")[0]
                for n in ast.walk(ast.parse(setup))
                if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names
            }
        )
    except SyntaxError:
        pass
    prompt = (
        DEMO
        + "\n# Current data-science problem (Python 3).\n"
        + "\n".join(("# " + line for line in question.split("\n")))
        + "\n"
    )
    prompt += (
        "# Existing setup; these variables already exist when the solution is evaluated.\n"
        + setup
        + "\n"
    )
    prompt += "# " + LIBRARY_RULES.get(library, "Use the existing variables.") + "\n"
    prompt += "# Write only the missing solution. Do not repeat initialization, load_data(), sample data, or explanations.\n"
    if contract["output_names"]:
        prompt += "# Required output variables: " + ", ".join(contract["output_names"]) + ".\n"
    prompt += "# SOLUTION START\n" + prefix
    return dict(
        task_id=task_id,
        prompt=prompt,
        completion_prefix=prefix,
        official_prefix=official_prefix,
        contract=contract,
        adapter_version=VERSION,
        prompt_sha256=digest(prompt),
        source_prompt_sha256=digest(public.get("prompt", public.get("question", ""))),
    )


def clean_completion(raw, row):
    stops = (
        "<|endoftext|>",
        "<|endofmask|>",
        "<|eot_id|>",
        "<|mdm_mask|>",
        "</s>",
        "# END SOLUTION",
        "# SOLUTION END",
        "</code>",
        "\nQUESTION:",
        "\n# Current problem:",
    )
    found = [raw.find(stop) for stop in stops if raw.find(stop) >= 0]
    bare_ends = list(re.finditer("(?m)^[ \\t]*(?:END SOLUTION|SOLUTION END)[ \\t]*(?:\\n|$)", raw))
    found.extend((match.start() for match in bare_ends))
    body = raw[: min(found)] if found else raw
    fence = chr(96) * 3
    if body.lstrip().startswith(fence):
        body = body.lstrip().split("\n", 1)[1] if "\n" in body else ""
        body = body.split(fence, 1)[0]
    body = body.replace("\t", "    ").replace("<code>", "")
    marker_lines = {"SOLUTION START", "# SOLUTION START", "#SOLUTION START"}
    lines = body.split("\n")
    removed_markers = sum((line.strip() in marker_lines for line in lines))
    body = "\n".join((line for line in lines if line.strip() not in marker_lines))
    completion = row["completion_prefix"] + body
    program = row["official_prefix"] + completion
    removed = 0
    try:
        ast.parse(program)
    except (SyntaxError, ValueError):
        lines = program.split("\n")
        for end in range(len(lines) - 1, 0, -1):
            candidate = "\n".join(lines[:end]) + "\n"
            try:
                tree = ast.parse(candidate)
            except (SyntaxError, ValueError):
                continue
            entry = row["contract"].get("entry_point")
            if entry and (
                not any(
                    (
                        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry
                        for n in ast.walk(tree)
                    )
                )
            ):
                continue
            if not entry and (
                not any((not isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body))
            ):
                continue
            if candidate.startswith(row["official_prefix"]):
                removed = len(lines) - end
                program = candidate
                completion = program[len(row["official_prefix"]) :]
                break
    return (
        completion,
        program,
        dict(
            first_stop=min(found) if found else None,
            removed_trailing_lines=removed,
            removed_marker_lines=removed_markers,
        ),
    )


def static_contract(program, contract):
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError):
        return (0, 0, -10000)
    defined = set(dir(builtins)) | set(contract.get("provided_names", []))
    loaded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((alias.asname or alias.name.split(".")[0] for alias in node.names))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    required = set(contract.get("output_names", []))
    entry = contract.get("entry_point")
    if entry:
        required.add(entry)
    return (1, -len(required - defined), -len(loaded - defined))
