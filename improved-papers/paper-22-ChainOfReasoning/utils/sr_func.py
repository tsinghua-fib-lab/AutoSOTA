# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from utils.lean4_import import _import_all_mathlib, _import_header

_import_all_mathlib = [i.strip() for i in _import_all_mathlib.split("\n") if i.strip()]
_import_header = [i.strip() for i in _import_header.split("\n") if i.strip()]

def execute_completions_math(verifier, completions):
    lean4_blocks_list = []
    valid_indices = []
    all_blocks = []
    
    for i, completion in enumerate(completions):
        lean4_blocks = re.findall(r'```lean4(.*?)```', completion, re.DOTALL)
        if lean4_blocks:
            generated_blocks = lean4_blocks[-1].strip()
            pattern = r'^(import\s+.+|open\s+.+|set_option\s+.+)$'
            imports_opens_options = re.findall(pattern, generated_blocks, re.MULTILINE)
            remaining_content = re.sub(pattern, '', generated_blocks, flags=re.MULTILINE).strip()

            import_statements = set()
            open_modules = set()
            set_options = set()

            for line in _import_all_mathlib + _import_header + imports_opens_options:
                line = line.strip()
                if line.startswith("import"):
                    import_statements.add(line)
                elif line.startswith("open"):
                    open_modules.update(line.replace("open", "").split())
                elif line.startswith("set_option"):
                    set_options.add(line)

            dedup_import = list(import_statements)
            if set_options:
                dedup_import.extend(sorted(set_options))
            if open_modules:
                dedup_import.append(f"open {' '.join(sorted(open_modules))}")

            dedup_import = "\n".join(dedup_import)

            lean4_blocks_list.append(dedup_import + "\n" + remaining_content)
            valid_indices.append(i)
            all_blocks.append(lean4_blocks)
        else:
            all_blocks.append([])
    
    successes = [False] * len(completions)
    outputs_data = [""] * len(completions)
    
    if lean4_blocks_list:
        request_id_list = verifier.submit_all_request(lean4_blocks_list)
        outputs_list = verifier.get_all_request_outputs(request_id_list)
        
        for idx, output in zip(valid_indices, outputs_list):
            if output['complete']:
                successes[idx] = True
                try:
                    outputs_data[idx] = output['infos'][0]['data']
                except:
                    outputs_data[idx] = ""
                    
    return successes, all_blocks, outputs_data

def execute_completions_prove(verifier, completions, formal_statement, header):
    lean4_blocks_list = []
    valid_indices = []
    all_blocks = []
    
    for i, completion in enumerate(completions):
        lean4_blocks = re.findall(r'```lean4(.*?)```', completion, re.DOTALL)
        if lean4_blocks and formal_statement in lean4_blocks[-1]:
            generated_blocks = lean4_blocks[-1].strip()
            pattern = r'^(import\s+.+|open\s+.+|set_option\s+.+)$'
            imports_opens_options = re.findall(pattern, header + generated_blocks, re.MULTILINE)
            remaining_content = re.sub(pattern, '', header + generated_blocks, flags=re.MULTILINE).strip()

            import_statements = set()
            open_modules = set()
            set_options = set()

            for line in _import_all_mathlib + _import_header + imports_opens_options:
                line = line.strip()
                if line.startswith("import"):
                    import_statements.add(line)
                elif line.startswith("open"):
                    open_modules.update(line.replace("open", "").split())
                elif line.startswith("set_option"):
                    set_options.add(line)

            dedup_import = list(import_statements)
            if set_options:
                dedup_import.extend(sorted(set_options))
            if open_modules:
                dedup_import.append(f"open {' '.join(sorted(open_modules))}")

            dedup_import = "\n".join(dedup_import)

            lean4_blocks_list.append(dedup_import + "\n" + remaining_content)
            valid_indices.append(i)
            all_blocks.append(lean4_blocks)
        else:
            all_blocks.append([])
    
    successes = [False] * len(completions)
    if lean4_blocks_list:
        request_id_list = verifier.submit_all_request(lean4_blocks_list)
        outputs_list = verifier.get_all_request_outputs(request_id_list)
        
        for idx, output in zip(valid_indices, outputs_list):
            if output['complete']:
                successes[idx] = True
    
    return successes, all_blocks
