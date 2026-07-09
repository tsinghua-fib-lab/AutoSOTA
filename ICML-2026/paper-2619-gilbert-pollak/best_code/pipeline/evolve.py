import argparse
from datetime import datetime
import subprocess
import shutil
import sys
from pathlib import Path
import re
import os
import time


def run(cmd, cwd=None, check=True):
  print("Running:", " ".join(cmd))
  res = subprocess.run(cmd, cwd=cwd)
  if check and res.returncode != 0:
    raise RuntimeError(f"Command failed: {' '.join(cmd)}")
  return res.returncode


def find_gpp():
  for name in ("g++", "clang++"):
    path = shutil.which(name)
    if path:
      return path
  return None


def modify_plot(src: Path, dst: Path, iter_k: int):
  text = src.read_text(encoding='utf-8')
  # include all formulas from F0 up to F{iter_k}
  includes = list(range(0, iter_k + 1))

  # build new include block (keep other includes intact)
  # replace the first occurrence of #include "formulas/F0" and following const m line
  # Create new includes string
  inc_lines = "\n".join([f'#include "formulas/F{k}"' for k in includes]) + "\n"

  # Replace the block that starts with an include to formulas/F and ends with const int m
  text = re.sub(r'(#include \"formulas/F\d+\"[\s\S]*?)const int m = M\d+;',
          lambda m: inc_lines + f"const int m = M{iter_k};",
          text,
          count=1)

  dst.write_text(text, encoding='utf-8')
  print(f"Wrote modified plot to {dst}")


def compile_plot(src: Path, out: Path):
  gpp = find_gpp()
  if gpp is None:
    raise RuntimeError('No C++ compiler (g++/clang++) found in PATH')
  # Use the same flags as the Makefile's plot target
  flags = [
    '-O3', '-Wall', '-Wextra', '-Wno-unused-parameter', '-Wno-unused-variable',
    '-std=c++23', '-pthread', '-ffast-math'
  ]
  cmd = [gpp] + flags + [str(src), '-o', str(out)]
  run(cmd)


def locate_binsearch(workdir: Path):
  # prefer workspace root file named binsearch or binsearch.exe
  candidates = [workdir / 'binsearch', workdir / 'binsearch.exe']
  for p in candidates:
    if p.exists():
      return p
  # fallback to PATH lookup
  bs = shutil.which('binsearch')
  if bs:
    return Path(bs)
  return None

added_tree_type = set()
rho_history = ['0.8000']
force_splus = False

def get_tree_type(splus_file: Path) -> str:
  global force_splus
  with open(splus_file, 'r', encoding='utf-8') as f:
    content = f.readline().strip() # like ('A', 'D', 'Q', 'R')
  assert content.startswith('(') and content.endswith(')'), "Invalid s_plus format"
  A, B, C, D = content[2], content[7], content[12], content[17]
  assert all(c.isalpha() for c in (A, B, C, D)), "Invalid characters in s_plus"
  treetype1 = f'({A}{B})-({C}{D})'
  treetype2 = f'({B}{C})-({D}{A})'
  if treetype1 not in added_tree_type:
    added_tree_type.add(treetype1)
    force_splus = True # next time must use splus to explore other type
    return treetype1
  else:
    force_splus = False
    added_tree_type.add(treetype2)
    return treetype2

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iterations', type=int, default=3)
  parser.add_argument('--start', type=int, default=1)
  parser.add_argument('--workdir', type=str, default='.')
  parser.add_argument('--plot-src', type=str, default='plot.cpp')
  parser.add_argument('--log', type=str, default=None, help='log file path (default log_{MM-DD-HH-MM}.md)')
  args = parser.parse_args()

  # prepare log file path
  if args.log is None:
    now = datetime.now().strftime('%m-%d-%H-%M')
    log_path = Path(f'log_{now}.md')
  else:
    log_path = Path(args.log)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  # if log file is new, write a markdown header
  if not log_path.exists():
    try:
      with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write('# Evolve run log\n\n')
        lf.write(f'Generated: {datetime.now().isoformat()}\n\n')
    except Exception as e:
      print('Failed to create log header:', e)

  workdir = Path(args.workdir).resolve()
  plot_src = (workdir / args.plot_src).resolve()
  if not plot_src.exists():
    print('plot.cpp not found at', plot_src)
    sys.exit(1)

  bottleneck_file = workdir / 'bottleneck.txt'
  splus_file = workdir / 'tmp' / 's_plus'
  improve_type = 'regular-point'

  for k in range(args.start, args.start + args.iterations):
    print('\n===== ITERATION', k, improve_type, '=====')
    dir_name = workdir / f'evolve_resp_{k}'
    dir_name.mkdir(parents=True, exist_ok=True)

    # 1) call llm.py
    llm_cmd = [sys.executable, 'llm.py', f'--{improve_type}', '--dir', str(dir_name)]
    if bottleneck_file.exists():
      llm_cmd += ['--bottleneck', str(bottleneck_file)]
    if improve_type == 'splus':
      try:
        run([workdir / 'split_rho', str(bottleneck_file)], cwd=workdir)
      except Exception as e:
        print('split_rho failed:', e)
        break
      if splus_file.exists():
        llm_cmd += ['--treetype', get_tree_type(splus_file)]
    agent_time = None
    try:
      llm_start = time.perf_counter()
      run(llm_cmd, cwd=workdir)
      agent_time = time.perf_counter() - llm_start
    except Exception as e:
      print('llm.py failed:', e)
      break

    # 2) extract (use --file to point to the response.md in the iteration directory)
    try:
      resp_file = dir_name / 'response.md'
      run([sys.executable, 'extract.py', f'--{improve_type}', '--file', str(resp_file)], cwd=workdir)
    except Exception as e:
      print('extract.py failed:', e)
      break

    # 3) calc.py -> expect it writes formulas/F{k}
    try:
      run([sys.executable, 'calc.py', f'--{improve_type}'], cwd=workdir)
    except Exception as e:
      print('calc.py failed:', e)
      break

    # 4) copy & modify plot.cpp
    plot_gen = workdir / f'plot_gen_{k}.cpp'
    try:
      modify_plot(plot_src, plot_gen, k)
    except Exception as e:
      print('modify_plot failed:', e)
      break
    
    # 5) compile
    out_exec = workdir / f'plot'
    if os.name == 'nt':
      out_exec = out_exec.with_suffix('.exe')
    try:
      compile_plot(plot_gen, out_exec)
    except Exception as e:
      print('compile failed:', e)
      break

    # 6) run binsearch
    bs = locate_binsearch(workdir)
    try:
      # run binsearch and get bottleneck.txt
      sym_start = time.perf_counter()
      subprocess.run([str(bs)], cwd=workdir)
      symbolic_time = time.perf_counter() - sym_start
      print('Wrote bottleneck to', bottleneck_file)
    except Exception as e:
      print('binsearch run failed:', e)
      break
      
    # 7) report rho
    try:
      with open('rho.txt', 'r', encoding='utf-8') as f:
        rho_value = f.read().strip()
        print(f'Iteration {k} completed. Current rho: {rho_value}')
        rho_history.append(rho_value)
    except Exception:
      print('rho.txt not found or unreadable')

    # read response.md (LLM final answer) if present
    resp_txt = ''
    resp_file = dir_name / 'response.md'
    if resp_file.exists():
      try:
        resp_txt = resp_file.read_text(encoding='utf-8')
      except Exception:
        resp_txt = ''

    # read bottleneck file content
    btxt = ''
    if bottleneck_file.exists():
      try:
        btxt = bottleneck_file.read_text(encoding='utf-8')
      except Exception:
        btxt = ''

    # read token usage if available
    total_tokens = None
    usage_file = dir_name / 'usage.json'
    if usage_file.exists():
      try:
        import json
        usage_payload = json.loads(usage_file.read_text(encoding='utf-8'))
        total_tokens = usage_payload.get('total_tokens')
        if agent_time is None:
          agent_time = usage_payload.get('duration_sec')
      except Exception:
        total_tokens = None
    # write log entry in Markdown
    try:
      treetype = None
      if improve_type == 'splus':
        for i, t in enumerate(llm_cmd[:-1]):
          if t == '--treetype' and i+1 < len(llm_cmd):
            treetype = llm_cmd[i+1]
        if treetype is None and splus_file.exists():
          try:
            treetype = get_tree_type(splus_file)
          except Exception:
            treetype = None

      with open(log_path, 'a', encoding='utf-8') as lf:
        lf.write(f'## Iteration {k} — {improve_type}\n\n')
        lf.write(f'- **Timestamp:** {datetime.now().isoformat()}\n')
        if treetype:
          lf.write(f'- **Tree type:** `{treetype}`\n')
        lf.write(f'- **Total tokens:** `{total_tokens if total_tokens is not None else ""}`\n')
        lf.write(f'- **AgentTime:** `{agent_time:.3f}s`\n' if agent_time is not None else '- **AgentTime:** ``\n')
        lf.write(f'- **SymbolicTime:** `{symbolic_time:.3f}s`\n' if 'symbolic_time' in locals() else '- **SymbolicTime:** ``\n')
        lf.write(f'- **Rho:** `{rho_value if "rho_value" in locals() else ""}`\n')
        lf.write('\n')
        lf.write('**Bottleneck**:\n\n')
        lf.write('```text\n')
        lf.write(btxt if btxt else '')
        lf.write('\n```\n\n')
        lf.write('**LLM final answer**:\n\n')
        lf.write('```markdown\n')
        lf.write(resp_txt if resp_txt else '')
        lf.write('\n```\n\n')
        lf.write('---\n\n')
    except Exception as e:
      print('Failed to write log:', e)

    print('Rho history so far:', ' -> '.join(rho_history))
    
    # 8) change improve_type if no improvement
    if float(rho_history[-1]) <= float(rho_history[-2]):
      if improve_type == 'regular-point':
        improve_type = 'splus'
        print('Switching improvement type to splus due to no improvement.')
      elif not force_splus: 
        improve_type = 'regular-point'
        print('Switching improvement type to regular-point due to no improvement.')
      else:
        print('Keep improving with splus due to forced requirement.')        
    else:
      added_tree_type.clear()  # reset tree types if improvement happened
    # small delay to avoid tight loop
    time.sleep(0.5)


if __name__ == '__main__':
  main()



