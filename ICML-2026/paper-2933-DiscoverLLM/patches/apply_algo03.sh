#!/bin/bash
# ALGO-03: Intent Tree Depth and Breadth Constraint
set -e
cd /repo

# Backup
cp /repo/discoverllm/core/prompts/hierarchize_criteria.yaml /repo/patches/hierarchize_criteria.yaml.bak

python3 << "PYEOF"
PROMPT_FILE = "/repo/discoverllm/core/prompts/hierarchize_criteria.yaml"
with open(PROMPT_FILE) as f:
    content = f.read()

constraint_text = """
  ### 0. Depth and Breadth Constraints (CRITICAL)

  **Important:** To keep the intent tree focused and manageable:
  - **Maximum depth:** 3 levels (from root to deepest leaf). Do not exceed this.
  - **Maximum branching factor:** 4 children per node. If a parent would naturally have more than 4 children, keep only the 4 most specific/important ones.
  - **Minimum specificity gap:** Each child must be meaningfully more specific than its parent. Avoid adding children that are essentially restatements.

  These constraints ensure the tree remains explorable within the limited dialogue turns.
"""

insert_marker = "### 1. Parent-Child Relationships"
if insert_marker in content:
    content = content.replace(insert_marker, constraint_text + "\n" + insert_marker)
    with open(PROMPT_FILE, "w") as f:
        f.write(content)
    print("ALGO-03 applied: depth/breadth constraints added to hierarchize prompt")
else:
    print("ERROR: insertion point not found")
    import sys; sys.exit(1)
PYEOF
