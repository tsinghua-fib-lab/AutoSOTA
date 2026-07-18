import time
import utils.general_utils as utils
import torch


class Timer:
    def __init__(self, args, file=None):
        self.timers = {}
        self.args = args
        if args.enable_timer:
            # Enable time measure evaluated on python side.
            self.file = open(
                args.log_folder + "/python_time.log",
                "w",
            )
        else:
            self.file = None

    def start(self, key):
        """ 开始对指定的 key（代码块）进行计时 """
        if not utils.check_enable_python_timer():
            return
        """Start timer for the given key"""
        if key not in self.timers:
            self.timers[key] = {"start_time": None, "cnt": 0, "all_time": []}

        torch.cuda.synchronize()

        self.timers[key]["start_time"] = time.time()

    def stop(self, key, print_elapsed=False):
        if not utils.check_enable_python_timer():
            return

        """Stop the timer for the given key, and report the elapsed time"""
        if key not in self.timers or self.timers[key]["start_time"] is None:
            raise ValueError(f"Timer with key '{key}' is not running.")

        torch.cuda.synchronize()

        cur_time = time.time()
        duration = cur_time - self.timers[key]["start_time"]
        self.timers[key]["cnt"] += 1
        self.timers[key]["all_time"].append(duration)
        self.timers[key]["start_time"] = None
        if print_elapsed:
            print(f"Time for '{key}': {duration:.6f} seconds")
        return duration

    def printTimers(
        self, iteration, mode="this_iteration"
    ):  # this_iteration, average, sum
        """Get the elapsed time for the given key without stopping the timer"""
        if not utils.check_enable_python_timer():
            return

        for x in range(self.args.bsz):
            if (iteration + x) % self.args.log_interval == 1:
                iteration += x
                break

        for key in self.timers:
            if mode == "this_iteration":
                # print(f"iter {iteration}, TimeFor '{key}': {self.timers[key]['all_time'][-1]*1000:.6f} ms")
                self.file.write(
                    f"iter {iteration}, TimeFor '{key}': {self.timers[key]['all_time'][-1]*1000:.6f} ms\n"
                )
            elif mode == "average":
                average_time = (
                    sum(self.timers[key]["all_time"]) / self.timers[key]["cnt"]
                )
                # print(f"iter {iteration}, AverageTimeFor '{key}': {average_time*1000:.6f} ms")
                self.file.write(
                    f"iter {iteration}, AverageTimeFor '{key}': {average_time*1000:.6f} ms\n"
                )
            elif mode == "sum":
                sum_time = sum(self.timers[key]["all_time"])
                self.file.write(
                    f"iter {iteration}, TimeFor '{key}': {sum_time*1000:.6f} ms\n"
                )
        self.file.write("\n")
        self.file.flush()

    def clear(self):
        self.timers = {}


class End2endTimer:
    def __init__(self, args, file=None):
        self.total_time = 0
        self.last_time_point = None
        self.args = args

    def start(self):
        torch.cuda.synchronize()
        self.last_time_point = time.time()

    def stop(self):
        torch.cuda.synchronize()
        new_time_point = time.time()
        duration = new_time_point - self.last_time_point
        self.total_time += duration
        self.last_time_point = None

    def print_time(self, log_file, n_iterations):
        if self.last_time_point is not None:
            self.stop()
        log_file.write(
            "end2end total_time: {:.3f} s, iterations: {}, throughput {:.2f} it/s\n".format(
                self.total_time, n_iterations, n_iterations / self.total_time
            )
        )
