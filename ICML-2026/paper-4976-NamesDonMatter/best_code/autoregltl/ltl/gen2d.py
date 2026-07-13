import torch
import curses
from time import time
from collections import defaultdict
import pickle

import spot
from autoregltl.ltl.parser import ParseError, ltl_formula
from typing import Optional


def do_add(s, x):
  l = len(s)
  s.add(x)
  return len(s) != l


# Function to draw the heatmap using curses
def draw_heatmap(
        stdscr,
        tree_size_min = 1,
        tree_size_max = 50,
        ap_count = 10,
        time_limit:  Optional[float] = None,
        sample_limit: Optional[int] = None,
        output: str = 'output.pkl',
    ):
    time_limit = time_limit or float('inf')
    sample_limit = sample_limit or float('inf')
    # Initialize curses color pairs for heatmap
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        if i < 232:
            curses.init_pair(i + 1, i, -1)
        else:
            fg = 231 if 233 <= i+1 < 246 else 0
            curses.init_pair(i + 1, fg, i)
    stdscr.clear()

    start_time = time()

    matrix = torch.zeros(11, 51)
    max_value = 100
    last_update = 0
    key = "none"

    rows, cols = stdscr.getmaxyx()
    table_rows = min(rows - 5, matrix.size(0))
    table_cols = int(cols / 5) - 1
    shift_i = 1
    shift_j = 1

    last_generated = 0
    generated = 0
    last_accepted = 0
    accepted = 0

    def refresh():
        stdscr.erase()
        for i in range(table_rows):
            stdscr.addstr(i + 1, 0, "%4d " % (i+shift_i), curses.color_pair(197))
        for j in range(table_cols):
            stdscr.addstr(0, 5 + j*5, " %3d " % (j+shift_j), curses.color_pair(197))
        # Loop through the matrix and print each element
        for i in range(table_rows):
            for j in range(table_cols):
                newi = i + shift_i
                newj = j + shift_j
                if not (0 <= newi < matrix.size(0) and 0 <= newj < matrix.size(1)):
                    continue
                value = matrix[i+shift_i, j+shift_j].item()
                color = curses.color_pair(233 + round((value / max_value) * (255 - 233)))
                
                # Display each matrix element as a colored block
                stdscr.addstr(i + 1, 5 + j*5, " %3d " % value, color)

        stdscr.addstr(table_rows + 2, 1, f"Tree size: ({tree_size_min}, {tree_size_max})     AP Count: {ap_count}")
        
        nonlocal generated, last_generated, accepted, last_accepted
        stdscr.addstr(table_rows + 3, 1, f"Generated: {generated} + {last_generated}")
        stdscr.addstr(table_rows + 4, 1, f"Accepted: {accepted} + {last_accepted}")
        last_generated = 0
        last_accepted = 0
        stdscr.addstr(table_rows + 5, 1, f"Elapsed Time: {time() - start_time:.2f} s")

        stdscr.refresh()
        nonlocal last_update
        last_update = time()

    refresh()

    def get_input():
        try:
            return str(stdscr.getkey())
        except Exception:
            return None


    aps = []
    def get_formula_gen():
        nonlocal aps
        aps = list(map(chr, range(97, 97 + ap_count)))
        return spot.randltl(
            aps,
            seed=42,
            tree_size=(tree_size_min, tree_size_max),
            ltl_priorities='false=0,true=1,not=1,F=0,G=0,X=1,equiv=0,implies=0,xor=0,R=0,U=1,W=0,M=0,and=1,or=0',
            simplify=0,
        )
    formula_generator = get_formula_gen()

    formulas = defaultdict(set)

    exit_reason = None
    while True:
        formula_spot = next(formula_generator)
        formula_str = formula_spot.to_str()
        formula_str = ltl_formula(formula_str, 'spot').to_str('network-polish')
        formula_aps = [i for i in aps if i in formula_str]
        a = len(formula_aps)
        b = len(formula_str)
        if a < matrix.size(0) and b < matrix.size(1) and matrix[a, b] < max_value:
            if a < len(aps):
                # Make sure the aps are sorted
                # We don't want a formula like (e & f), it must be (a & b)
                lookup = {k: aps[i] for i, k in enumerate(formula_aps)}
                formula_str = "".join([lookup.get(c, c) for c in formula_str])
            if do_add(formulas[(a,b)], formula_str):
                matrix[a, b] += 1
                last_accepted += 1
                accepted += 1
                if accepted >= sample_limit:
                    exit_reason = "sample limit"
                    break
        last_generated += 1
        generated += 1

        current_time = time()
        if current_time - last_update > 0.016:
            refresh()
        if current_time - start_time > time_limit:
            exit_reason = "time limit"
            break

        if (key := get_input()) is not None:
            if key == "KEY_LEFT":
                shift_j -= 1
            elif key == "KEY_RIGHT":
                shift_j += 1

            elif key == "q":
                exit_reason = "user quit"
                break

            elif key == "j":
                tree_size_min = max(0, tree_size_min - 1)
                formula_generator = get_formula_gen()
            elif key == "u":
                tree_size_min = tree_size_min + 1
                formula_generator = get_formula_gen()

            elif key == "k":
                tree_size_max = max(0, tree_size_max - 1)
                formula_generator = get_formula_gen()
            elif key == "i":
                tree_size_max = tree_size_max + 1
                formula_generator = get_formula_gen()

            elif key == "l":
                ap_count = max(0, ap_count - 1)
                formula_generator = get_formula_gen()
            elif key == "o":
                ap_count = ap_count + 1
                formula_generator = get_formula_gen()

    curses.endwin()
    print(f"Exit reason: {exit_reason}")
    print(f"Generated: {generated}")
    print(f"Accepted: {accepted}")
    print(f"Elapsed Time: {time() - start_time:.2f} s")
    print(f"Saving to {output}...")

    formulas = {k: list(v) for k, v in formulas.items()}

    with open(output, 'wb') as f:
        pickle.dump(formulas, f)
    
    print("Done.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate heatmap for the given parameters.')

    # Positional or optional arguments
    parser.add_argument('--tree_size_min', type=int, default=1, 
                        help='Minimum tree size (default: 1)')
    parser.add_argument('--tree_size_max', type=int, default=50, 
                        help='Maximum tree size (default: 50)')
    parser.add_argument('--aps', type=int, default=10, 
                        help='Atomic proposition count (default: 10)')
    parser.add_argument('--time_limit', type=float, default=None, 
                        help='Optional time limit in seconds (default: None)')
    parser.add_argument('--sample_limit', type=int, default=None, 
                        help='Optional sample limit (default: None)')
    parser.add_argument('-o', '--output', type=str, default='output.pkl', 
                        help='Output file path (default: output.pkl)')

    # Parse the arguments
    args = parser.parse_args()

    curses.wrapper(
        draw_heatmap,
        tree_size_min=args.tree_size_min,
        tree_size_max=args.tree_size_max,
        ap_count=args.aps,
        time_limit=args.time_limit,
        sample_limit=args.sample_limit,
        output=args.output,
    )

