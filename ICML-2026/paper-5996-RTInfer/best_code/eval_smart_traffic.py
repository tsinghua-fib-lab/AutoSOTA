"""Evaluation script: Smart Traffic DMR and Relative Accuracy (paper metric).

Reproduces the rubric metric for paper #5996:
  - Application: Smart Traffic (YOLOv8-L 1080p + YOLOv8n KITTI)
  - Device: Jetson Xavier NX effective memory 6144 MiB
  - Deadline scaling factor: k=1.0
  - Policies: rtinfer (paper method), pantheon (strongest baseline)
  - Metrics: DMR (Deadline Miss Rate), relative accuracy
"""
import sys
sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/rebuttal_experiments")
from common import *
from modern_workloads import build_smart_traffic_case, scaled_deadlines

title, models, atlas, tasks, duration_ms = build_smart_traffic_case()
tasks_k1 = scaled_deadlines(tasks, 1.0)

results = run_policies(models, atlas, tasks_k1,
                       policies=("rtinfer", "pantheon"),
                       memory_mib=6144.0, duration_ms=1000, bandwidth_gbps=24.0)

for result in results:
    relative_accs = []
    for job in result.schedule_events:
        if job.task is None or job.variant is None:
            relative_accs.append(0.0)
            continue
        model = models.get(job.task.model_name)
        if model is None:
            relative_accs.append(0.0)
        else:
            relative_raw = job.variant.accuracy / model.full_accuracy
            relative_accs.append(0.0 if job.missed else relative_raw)
    dw_rel_acc = sum(relative_accs) / len(relative_accs) if relative_accs else 0.0
    dmr_pct = result.deadline_miss_rate * 100.0
    rel_acc_pct = dw_rel_acc * 100.0
    print("%-12s DMR=%.2f%%  relative_accuracy=%.2f%%  raw_acc=%.4f  jobs=%d missed=%d" %
          (result.policy, dmr_pct, rel_acc_pct, result.average_accuracy,
           result.total_jobs, result.missed_jobs))
