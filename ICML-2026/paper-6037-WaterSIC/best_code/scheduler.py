import GPUtil
import time
import torch
from multiprocessing import Process


def run(task, gpu):
    f, args = task

    def wrapper():
        torch.cuda.set_device(gpu)
        f(*args)

    p = Process(target=wrapper)
    p.start()
    return p


def run_tasks(tasks, gpu_list=None):
    if gpu_list is None:
        gpu_list = GPUtil.getAvailable(order='first',
                                       maxLoad=0.05,
                                       maxMemory=0.05,
                                       limit=1000)
    print("Detected free GPUs:", gpu_list)
    gpu_status = {i: None for i in gpu_list}
    next_task = 0
    finished_tasks = 0
    while finished_tasks < len(tasks):
        for gpu_id in gpu_status.keys():
            process = gpu_status[gpu_id]
            if process is not None and not process.is_alive():
                print(f"GPU {gpu_id} is free now")
                gpu_status[gpu_id] = None
                finished_tasks += 1
            if gpu_status[gpu_id] is None and next_task < len(tasks):
                print(f"Allocating task {next_task} to GPU {gpu_id}")
                gpu_status[gpu_id] = run(tasks[next_task], gpu_id)
                next_task += 1
        time.sleep(0.1)
