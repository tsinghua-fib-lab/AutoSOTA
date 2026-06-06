"""Generate the code reference pages and navigation."""

# modified from https://github.com/mkdocstrings/mkdocstrings/blob/main/scripts/gen_ref_nav.py

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'

module_name = "appl"
root = Path(__file__).parent.parent
target_dir = "docs"
src = root / "src" / module_name

for path in sorted(src.rglob("*.py")):
    module_path = module_name / path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path(target_dir, doc_path)

    parts = tuple(module_path.parts)

    if len(parts) > 1 and parts[1] == "cli":
        # exclude the cli module
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav_parts = [f"{mod_symbol} {part}" for part in parts]
    nav[tuple(nav_parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path.relative_to(root))

with mkdocs_gen_files.open(Path(target_dir, "SUMMARY.md"), "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
