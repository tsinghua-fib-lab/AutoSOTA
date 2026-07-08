import psutil
import torch
import os


def print_cpu_memory():
    mem = psutil.virtual_memory()
    total = str(round(mem.total / 1024 ** 3))
    used = str(round(mem.used / 1024 ** 3))
    use_per = str(round(mem.percent))
    free = str(round(mem.free / 1024 ** 3))
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_in_bytes = memory_info.rss
    # print("CPU memory size of all:" + total + "GB")
    # print("CPU memory used:" + used + "GB(" + use_per + "%)")
    print(f"CPU memory used:{memory_usage_in_bytes / 1024 ** 3}GB")
    print("CPU memory available :" + free + "GB")


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"GPU Allocated Memory: {allocated_memory:.2f} GB")
    print(f"GPU max_memory_allocated {max_allocated / (1024 ** 3)} GB")


def show_gpu_and_cpu_memory():
    print_gpu_memory()
    print_cpu_memory()
