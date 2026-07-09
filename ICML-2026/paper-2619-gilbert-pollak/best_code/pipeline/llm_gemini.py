import subprocess, time, json
import os, argparse
from datetime import datetime
from google import genai
from google.genai import types

def run_python(code: str, timeout: int = 300):
    start = time.time()
    try:
        proc = subprocess.run(
            ['python', '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            timeout=timeout
        )
        duration = time.time() - start
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "return code": proc.returncode,
            "duration": duration
        }
    except subprocess.TimeoutExpired as e:
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1, "duration": time.time() - start}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "returncode": -2, "duration": time.time() - start}

def run_mathematica(code: str, timeout: int = 300):
    start = time.time()
    try:
        proc = subprocess.run(
            ['wolframscript', '-code', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            timeout=timeout
        )
        duration = time.time() - start
        def clip(s: str, max_len = 1000):
            if len(s) > max_len:
                s = s[:max_len] + f' ...({len(s) - max_len} bytes emitted)'
            return s
        return {
            "stdout": clip(proc.stdout),
            "stderr": proc.stderr,
            "return code": proc.returncode,
            "duration": duration
        }
    except subprocess.TimeoutExpired as e:
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1, "duration": time.time() - start}
    except Exception as e:
        return {"stdout": "", "stderr": f"Execution error: {e}", "returncode": -2, "duration": time.time() - start}

# If use Gemini-3-pro

API_URL = "API_URL"
API_KEY = "API_KEY"
MODEL_NAME = "gemini-3-pro-preview"

def parse_pair(line: str):
    parts = line.split()
    if len(parts) < 2:
        raise RuntimeError(f'unexpected line format: {line!r}')
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        raise RuntimeError(f'cannot parse floats from line: {line!r}')
    return lo, hi

def regular_point_bottleneck(bottleneck_file: str) -> str:
    # Read file and return formatted a_2, a_3, a_4 bounds extracted from specific lines.
    # Expected format (example):
    # line1: counts
    # line2..6: minn/maxn for variables b,c,d,s,e
    # We need: line5 -> a_2, line3 -> a_3, line6 -> a_4
    with open(bottleneck_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 6:
        raise RuntimeError(f'bottleneck file {bottleneck_file} has insufficient lines')

    lo2, hi2 = parse_pair(lines[4])  # line 5
    lo3, hi3 = parse_pair(lines[2])  # line 3
    lo4, hi4 = parse_pair(lines[5])  # line 6

    s2 = f"a_2 ∈ [{lo2:.3f}, {hi2:.3f}]"
    s3 = f"a_3 ∈ [{lo3:.3f}, {hi3:.3f}]"
    s4 = f"a_4 ∈ [{lo4:.3f}, {hi4:.3f}]"
    
    bottleneck = ", ".join([s2, s3, s4])
    print(f'Extracted bottleneck condition: {bottleneck}')
    return bottleneck

def splus_bottleneck(bottleneck_file: str) -> str:
    with open(bottleneck_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 6:
        raise RuntimeError(f'bottleneck file {bottleneck_file} has insufficient lines')

    lo2, hi2 = parse_pair(lines[1])  # line 2
    lo3, hi3 = parse_pair(lines[2])  # line 3
    lo4, hi4 = parse_pair(lines[3])  # line 4
    lo5, hi5 = parse_pair(lines[4])  # line 5
    lo6, hi6 = parse_pair(lines[5])  # line 6

    # Format to 3 decimal places as requested
    s2 = f"b ∈ [{lo2:.3f}, {hi2:.3f}]"
    s3 = f"c ∈ [{lo3:.3f}, {hi3:.3f}]"
    s4 = f"d ∈ [{lo4:.3f}, {hi4:.3f}]"
    s5 = f"s ∈ [{lo5:.3f}, {hi5:.3f}]"
    s6 = f"e ∈ [{lo6:.3f}, {hi6:.3f}]"
    
    bottleneck = ", ".join([s2, s3, s4, s5, s6])
    print(f'Extracted bottleneck condition: {bottleneck}')
    return bottleneck

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--splus', action='store_true', help='find a new s_plus, read prompt from splus_prompt.txt')
    parser.add_argument('--regular-point', action='store_true', help='find a new regular point, read prompt from regular_point_prompt.txt')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--dir', type=str, help='output directory', default=None)
    parser.add_argument('--bottleneck', type=str, help='the bottleneck condition file to add to the prompt', default=None)
    parser.add_argument('--treetype', type=str, help='the Steiner tree type to focus on', default="(AD)-(QR)")
    args = parser.parse_args()

    if not args.splus and not args.regular_point:
        print('At least one of --splus and --regular-point should be set.')
        print('Run `python llm.py --help` to see the usage.')
        exit(0)
    elif args.splus and args.regular_point:
        print('Only one of --splus and --regular-point should be set.')
        print('Run `python llm.py --help` to see the usage.')
        exit(0)

    if args.splus:
        prompt_file_name = 'splus_prompt'
        if args.bottleneck is None:
            bottleneck = "b ∈ [0.99, 1.01], c ∈ [1.00, 1.01], d ∈ [2.48, 2.50], s ∈ [1.70, 1.72], e ∈ [0.00, 0.01]"
        else:
            bottleneck = splus_bottleneck(args.bottleneck)
        treetype = args.treetype
    elif args.regular_point:
        prompt_file_name = 'regular_point_prompt'
        if args.bottleneck is None:
            bottleneck = "a_2 ∈ [1.70, 1.72], a_3 ∈ [1.00, 1.01], a_4 ∈ [0.00, 0.01]"
        else:
            bottleneck = regular_point_bottleneck(args.bottleneck)
        treetype = None

    now = datetime.now().strftime('%m-%d-%H-%M')
    dir = f'response-{now}' if args.dir is None else args.dir

    def _trans(s: str):
        s = s.replace('{Bottleneck}', bottleneck)
        if treetype is not None:
            s = s.replace('{A}', treetype[1])\
                .replace('{B}', treetype[2])\
                .replace('{C}', treetype[6])\
                .replace('{D}', treetype[7])
        return s

    if not args.audit:
        os.makedirs(dir, exist_ok=True)
        prompt_content = _trans(open(f'{prompt_file_name}.txt', 'r', encoding='utf-8').read())
    else:
        if not os.path.exists(f'{dir}/response.md'):
            print(f'{dir}/response.md not exist!')
            exit(0)
        prompt_content = _trans(open(f'{prompt_file_name}_audit.txt', 'r', encoding='utf-8').read())
        prompt_content = prompt_content.replace('{Original_Problem_Prompt}', _trans(open(f'{prompt_file_name}.txt', 'r', encoding='utf-8').read()))
        prompt_content = prompt_content.replace('{Candidate_Answer}', open(f'{dir}/response.md', 'r', encoding='utf-8').read())

        dir = f'{dir}/audit'
        os.makedirs(dir)

    open(f'{dir}/prompt.md', 'w', encoding='utf-8').write(prompt_content)


    tools_decl = [
        {
            "name": "run_mathematica",
            "description": """\
    You can use this function to run a Wolfram Language (Mathematica) code and get the result (stdout/stderr).
    For example, given `a`, if you want to calculate the maximum `y` such that there exists a `x` satisfying `x>0 && x^2+a*x*y+y^2=1`, you can run the code `Maximize[{y, x^2+a*x*y+y^2==1 && x>0}, {x, y}]`.
    If you don't know how to use a function FUNC in Wolfram Language, you can run `?FUNC` to see the usage.
    The result will be given to you in JSON format. There are four properties: stdout, stderr, return code, duration. In the above example, the result is {"stdout": "{Piecewise[{{1, a >= 0}, {2*Sqrt[-(-4 + a^2)^(-1)], Inequality[-2, Less, a, Less, 0]}}, Infinity], {x -> Piecewise[{{-(a*Sqrt[-(-4 + a^2)^(-1)]), Inequality[-2, Less, a, Less, 0]}}, Indeterminate], y -> Piecewise[{{1, a >= 0}, {2*Sqrt[-(-4 + a^2)^(-1)], Inequality[-2, Less, a, Less, 0]}}, Infinity]}}\n", "stderr": "", "return code": 0, "duration": 1.5105085372924805}.
    If you can use python to do the calculation yourself, you don't need to call this function. You may call this function multiple times to solve a complicated problem. In every single calling, only the result of the last line of the command will be returned.
    Note that all calls are independent; each call does not inherit anything from previous calls. Therefore, variables that need to be reused must be redefined in each call.
    """,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Wolfram Language code you want to run. For example, `Maximize[{y, x^2+a*x*y+y^2==1 && x>0}, {x, y}]`."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "run_python",
            "description": "You can use this function to run any python code and get the result (stdout/stderr).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code you want to run."
                    }
                }
            }
        }
    ]

    function_map = {
        "run_mathematica": run_mathematica,
        "run_python": run_python,
    }

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(base_url=API_URL)
    )
    tools = types.Tool(function_declarations=tools_decl)
    config = types.GenerateContentConfig(
        tools=[tools],
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_level="high" if MODEL_NAME == 'gemini-3-pro-preview' else None,
        ),
        system_instruction="You are a helpful assistant specializing in mathematics and geometry.",
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=prompt_content)])
    ]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

    id = 0
    answer = ""
    while True:
        new_response = None
        content = response.candidates[0].content
        id += 1
        with open(f'{dir}/response_{id}.json', 'w', encoding='utf-8') as f:
            f.write(response.model_dump_json(indent=2))
        contents.append(content)
        function_response_parts = []
        for part in content.parts:
            if part.text:
                answer += part.text
                print(part.text, end='')
            elif part.function_call:
                call = part.function_call
                name = call.name
                code = call.args['code']
                print(f"\n\nFunction call: {name}")
                print(code)
                func = function_map[name]
                result = func(code)
                print("\nFunction call result:", result)
                function_response_parts.append(types.Part.from_function_response(
                    name=name,
                    response={"result": json.dumps(result)},
                ))
        if function_response_parts:
            with open(f'{dir}/function_response_{id}.json', 'w', encoding='utf-8') as f:
                f.write(types.Content(role="user", parts=function_response_parts).model_dump_json(indent=2))
            contents.append(types.Content(
                role="user",
                parts=function_response_parts,
            ))
            new_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        if not new_response:
            break
        response = new_response

    with open(f'{dir}/audit.md' if args.audit else f'{dir}/response.md', 'w', encoding='utf-8') as f:
        f.write(answer)
