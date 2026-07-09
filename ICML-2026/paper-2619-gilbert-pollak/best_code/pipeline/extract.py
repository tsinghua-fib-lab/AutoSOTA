import os, argparse, sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--splus', action='store_true', help='add a new s_plus, cond will be stored at tmp/s_cond')
    parser.add_argument('--regular-point', action='store_true', help='add a new regular point, cond will be stored at tmp/x_cond')
    parser.add_argument('--file', type=str, help='the file to extract from, like response-*/response.md')

    args = parser.parse_args()
    if not args.splus and not args.regular_point:
        print('At least one of --splus and --regular-point should be set.')
        print('Run `python extract.py --help` to see the usage.')
        exit(0)
    elif args.splus and args.regular_point:
        print('Only one of --splus and --regular-point should be set.')
        print('Run `python extract.py --help` to see the usage.')
        exit(0)

    output_path = 'tmp/s_cond' if args.splus else 'tmp/x_cond'

    with open(args.file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract the (unique) fenced code block delimited by triple backticks
    code_lines = []
    in_code = False
    is_cpp = False
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.strip().startswith('```'):
            # toggle code block state: opening or closing fence
            if not in_code:
                in_code = True
                if 'cpp' in stripped:
                    is_cpp = True
                else:
                    is_cpp = False
                continue
            else:
                in_code = False
        if in_code and is_cpp:
            code_lines.append(stripped)

    if not code_lines:
        # no code block found -- write empty file
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write('')
    else:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(code_lines))
            out_f.write('\n')

