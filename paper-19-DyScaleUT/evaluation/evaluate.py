import os
import json
import time
import tempfile
from os import system
from tqdm import tqdm

UT_EXEC_FORMAT = """docker run -v $(pwd):/data kaka0605/exec_unit_test:24.12.30 \
    --input_path /data/output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl \
    --output_path /data/{docker_write}/{sol_num}_sol_{ut_num}_ut_result.jsonl \
    --mp_num {mp_num} \
    --chunk_size 1000 \
    --recover 0
"""

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(filename, dataset):
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")
    with open(filename, 'w', encoding='UTF-8') as fp:
        for data in tqdm(dataset):
            fp.write(json.dumps(data, ensure_ascii=False) + '\n')

def save_sol_and_ut_comb(benchmark, sol_model, ut_model, sol_num, ut_num):
    print('========== START PREPROCESSING ==========')
    output = []

    if benchmark != 'livecodebench':
        sol_dataset = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')
    else:
        sol_dataset = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_100_func.jsonl')
    
    ut_dataset = load_jsonl(f'data/result/{benchmark}/ut_{ut_model}_100.jsonl')

    for i in tqdm(range(len(sol_dataset))):
        for sol_id in range(sol_num):
            for ut_id in range(len(ut_dataset[i]['unit_tests'])):
                if ut_id == ut_num:
                    break
                if benchmark == 'livecodebench':
                    code = sol_dataset[i]['solutions'][sol_id] + '\n\n' + ut_dataset[i]['unit_tests'][ut_id]
                else:
                    code = sol_dataset[i]['solutions'][sol_id]['solution'] + '\n\n' + ut_dataset[i]['unit_tests'][ut_id]
                output.append({
                    'task_id': sol_dataset[i]['task_id'],
                    'sol_id': sol_id,
                    'ut_id': ut_id,
                    'code': code
                })
    
    save_jsonl(f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl', output)

def exec_ut(benchmark, sol_model, ut_model, sol_num, ut_num, docker_dir, mp_num):
    print('========== START EXECUTE UNIT TEST ==========')
    system(UT_EXEC_FORMAT.format_map({
        'benchmark': benchmark,
        'docker_write': docker_dir,
        'sol_model': sol_model,
        'ut_model': ut_model,
        'sol_num': sol_num,
        'ut_num': ut_num,
        'mp_num': mp_num
    }))
    system(f'mv {docker_dir}/{sol_num}_sol_{ut_num}_ut_result.jsonl output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details')
    system(f'rm output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl')

def select_sol(benchmark, sol_model, ut_model, sol_num, ut_num):
    print('========== START SELECT SOLUTION ==========')
    dataset = load_jsonl(f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{sol_num}_sol_{ut_num}_ut_result.jsonl')
    dataset = sorted(dataset, key=lambda x: (int(x["task_id"].split('/')[1]), x["sol_id"], x["ut_id"]))
    
    if benchmark == 'humaneval':
        current_task = 'HumanEval/0'
    elif benchmark == 'mbpp':
        current_task = 'Mbpp/2'
    solution_dict = {i: 0 for i in range(sol_num)}
    chosen_solution = []
    for data in tqdm(dataset):
        if data['task_id'] == current_task:
            if data['result'] == 'pass':
                solution_dict[data['sol_id']] += 1
        else:
            # calculate
            sorted_solution_dict = sorted(solution_dict.items(), key=lambda item: item[1], reverse=True)
            chosen_solution.append({
                'task_id': current_task,
                'chosen_solution': sorted_solution_dict[0][0]
            })

            # initialize
            current_task = data['task_id']
            solution_dict = {i: 0 for i in range(sol_num)}

    # the last task
    sorted_solution_dict = sorted(solution_dict.items(), key=lambda item: item[1], reverse=True)
    chosen_solution.append({
        'task_id': current_task,
        'chosen_solution': sorted_solution_dict[0][0]
    })

    sol_dataset = load_jsonl(f'data/{benchmark}/sol_{sol_model}_200.jsonl')
    output = []
    for i in range(len(chosen_solution)):
        output.append({
            'task_id': chosen_solution[i]['task_id'],
            'solution': sol_dataset[i]['solutions'][chosen_solution[i]['chosen_solution']]
        })

    save_name = f"select_in_{sol_num}_sol_by_{ut_num}_ut.jsonl"
    save_jsonl(f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{save_name}", output)
    return save_name

def main(benchmark, sol_model, ut_model, sol_num, ut_num, mp_num):
    system(f'mkdir output/{benchmark}/{sol_model}_sol_{ut_model}_ut')
    system(f'mkdir output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details')

    # Get a temporary folder to store intermediate results during Docker runtime
    current_directory = os.getcwd()
    temp_dir = tempfile.TemporaryDirectory(dir=current_directory, prefix='docker_write_')
    docker_dir = temp_dir.name.split('/')[-1]
    os.system(f'chmod 777 -R {docker_dir}')

    st = time.time()
    # get solution_unit_test data
    save_sol_and_ut_comb(benchmark, sol_model, ut_model, sol_num, ut_num)

    # execute unit test
    exec_ut(benchmark, sol_model, ut_model, sol_num, ut_num, docker_dir, mp_num)

    temp_dir.cleanup()
    print(time.time() - st)

if __name__ == '__main__':
    import argparse

    # parse parameter
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument('--benchmark', type=str, help='evaluate benchmark')
    parser.add_argument('--sol_model', type=str, help='the model that generate solutions')
    parser.add_argument('--ut_model', type=str, help='the model that generate unit test')
    parser.add_argument('--sol_num', type=int, help='the number of generated solutions')
    parser.add_argument('--ut_num', type=int, help='the number of generated unit test')
    parser.add_argument('--mp_num', type=int, help='the number of process used for code execution')
    args = parser.parse_args()

    main(args.benchmark, args.sol_model, args.ut_model, args.sol_num, args.ut_num, args.mp_num)
