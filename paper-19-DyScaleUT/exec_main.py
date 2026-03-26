import time
import json
import ctypes
import signal
import argparse
import unittest
import contextlib
import multiprocessing
from multiprocessing import Value, Array
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from io import StringIO
from operator import itemgetter


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def read_jsonline_in_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for i, line in enumerate(f):
            chunk.append(json.loads(line))
            if (i + 1) % chunk_size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def execute_unittest(
    code: str,
    time_limits,
    is_pass: Value,
    total_num: Value,
    pass_num: Value,
    fail_num: Value,
    error_num: Value,
    save_detail: bool,
    shared_array: Array
):
    # create a cache to catch the output
    output = StringIO()

    # execute the code and catch the output
    locals_dict = {}
    try:
        with time_limit(time_limits):
            exec(code, {'unittest': unittest, 'output': output, 'locals_dict': locals_dict})

        # obtain result
        if save_detail:
            shared_array.value = output.getvalue().encode('utf-8')
        result = locals_dict.get('result')
        is_pass.value = result.wasSuccessful()
        total_num.value = result.testsRun
        pass_num.value = result.testsRun - len(result.failures) - len(result.errors)
        fail_num.value = len(result.failures)
        error_num.value = len(result.errors)
    except TimeoutException as e:
        is_pass.value = False
        # text.shared_str = b'time out'
    except Exception as e:
        is_pass.value = False
        # text.shared_str = str(e)


def handle_execute(
    task_id,
    solution_id,
    test_case_id,
    code: str,
    time_limits,
    save_detail: bool,
):
    # initialize shared memory objects
    is_pass = Value('b', False)
    total_num = Value('i', -1)
    pass_num = Value('i', 0)
    fail_num = Value('i', 0)
    error_num = Value('i', 0)
    shared_array = multiprocessing.Array(ctypes.c_char, 2000)  # restrict the max length to 2000

    # execute
    p = multiprocessing.Process(
        target=execute_unittest,
        args=(
            code,
            time_limits,
            # return values
            is_pass,
            total_num,
            pass_num,
            fail_num,
            error_num,
            save_detail,
            shared_array,
        )
    )
    p.start()
    p.join(2)  # wait for 2 sec

    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    # obtain result from the queue
    if save_detail:
        try:
            details_text = shared_array.value.decode('utf-8')
        except:
            details_text = ""
    else:
        details_text = ""

    details = {
        'total_num': total_num.value,
        'pass_num': pass_num.value,
        'fail_num': fail_num.value,
        'error_num': error_num.value,
        'text': details_text
    }

    return task_id, solution_id, test_case_id, is_pass.value, details


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for unit test evaluation.")

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--mp_num', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--recover', type=int, default=0)
    parser.add_argument('--details', action='store_true', help='add detail result')
    parser.add_argument('--time_limit', type=int, default=1)
    
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_option()
    print(opt)

    # n_workers = max(1, multiprocessing.cpu_count() // 2)
    n_workers = opt.mp_num
    TIME_LIMITS = opt.time_limit
    data_loaded_num = 0
    for chunk in read_jsonline_in_chunks(opt.input_path, chunk_size=opt.chunk_size):
        data_loaded_num += len(chunk)
        if opt.recover >= data_loaded_num:
            continue
        # use multi-process to handle each chunk
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for data in tqdm(chunk):
                args = (
                    data['task_id'],
                    data['sol_id'],
                    data['ut_id'],
                    data['code'],
                    TIME_LIMITS,
                    opt.details
                )
                futures.append(executor.submit(handle_execute, *args))

            raw_results = []
            for future in tqdm(as_completed(futures)):
                task_id, solution_id, test_case_id, is_pass, details = future.result()
                raw_results.append({
                    'task_id': task_id,
                    'sol_id': solution_id,
                    'ut_id': test_case_id,
                    'result': 'pass' if is_pass else 'fail',
                    'details': details
                })
        
        # save result
        sorted_results = sorted(raw_results, key=itemgetter('task_id', 'sol_id', 'ut_id'))
        with open(opt.output_path, 'a+', encoding='UTF-8') as fp:
            for result in sorted_results:
                try:
                    fp.write(json.dumps(result, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(str(e))
                    continue
