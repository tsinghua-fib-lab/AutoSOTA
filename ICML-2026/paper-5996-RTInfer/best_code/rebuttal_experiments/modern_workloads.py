from __future__ import annotations

from common import (
    JETSON_EFFECTIVE_MEMORY_MIB,
    RebuttalModelSpec,
    make_model,
    make_variants,
    print_result_table,
    run_policies,
)
from rtinfer.model import TaskSpec
from typing import Dict, List, Tuple


KAPPAS = (0.8, 1.0, 1.2)
POLICIES = ("rms-p", "dms-p", "pantheon", "rtinfer", "rtinfer-wo-alc", "rtinfer-wo-ms", "rtinfer-wo-dlp")
CaseBundle = Tuple[str, Dict, Dict, List[TaskSpec], int]


def periodic_task(
    model_name: str,
    deadline_ms: int,
    period_ms: int,
    duration_ms: int,
    shape: tuple[int, ...],
    start_ms: int = 0,
) -> TaskSpec:
    return TaskSpec(
        model_name=model_name,
        deadline_us=deadline_ms * 1000,
        period_us=period_ms * 1000,
        start_us=start_ms * 1000,
        end_us=duration_ms * 1000,
        shape=(1, *shape),
    )


def scaled_deadlines(tasks: list[TaskSpec], kappa: float) -> list[TaskSpec]:
    return [
        TaskSpec(
            model_name=task.model_name,
            deadline_us=max(1, int(task.deadline_us * kappa)),
            period_us=task.period_us,
            start_us=task.start_us,
            end_us=task.end_us,
            shape=task.shape,
            priority=task.priority,
        )
        for task in tasks
    ]


def build_smart_traffic_case() -> CaseBundle:
    yolo_l = make_model(
        RebuttalModelSpec(
            name="yolov8l_1080p",
            input_shape=(3, 1080, 1920),
            num_exits=4,
            full_latency_ms=145.0,
            full_memory_mib=1700.0,
            full_accuracy=0.924,
            earliest_accuracy=0.412,
            num_blocks=12,
        )
    )
    yolo_n = make_model(
        RebuttalModelSpec(
            name="yolov8n_kitti",
            input_shape=(3, 640, 640),
            num_exits=4,
            full_latency_ms=68.0,
            full_memory_mib=700.0,
            full_accuracy=0.896,
            earliest_accuracy=0.520,
            num_blocks=8,
        )
    )
    models = {yolo_l.name: yolo_l, yolo_n.name: yolo_n}
    atlas = {name: make_variants(model) for name, model in models.items()}
    tasks = [
        # High-frequency camera streams: RMS-P prioritizes these because the period is shortest.
        periodic_task("yolov8l_1080p", deadline_ms=320, period_ms=180, duration_ms=1000, shape=(3, 1080, 1920), start_ms=0),
        periodic_task("yolov8l_1080p", deadline_ms=320, period_ms=180, duration_ms=1000, shape=(3, 1080, 1920), start_ms=35),
        periodic_task("yolov8l_1080p", deadline_ms=340, period_ms=220, duration_ms=1000, shape=(3, 1080, 1920), start_ms=70),
        # Alert detector: DMS-P prioritizes it because the relative deadline is tighter.
        periodic_task("yolov8n_kitti", deadline_ms=180, period_ms=320, duration_ms=1000, shape=(3, 640, 640), start_ms=95),
        periodic_task("yolov8n_kitti", deadline_ms=210, period_ms=360, duration_ms=1000, shape=(3, 640, 640), start_ms=140),
    ]
    return "modern CNN smart traffic: YOLOv8-L 1080p + YOLOv8n", models, atlas, tasks, 1000


def smart_traffic_yolo() -> None:
    title, models, atlas, tasks, duration_ms = build_smart_traffic_case()
    for kappa in KAPPAS:
        results = run_policies(
            models,
            atlas,
            scaled_deadlines(tasks, kappa),
            policies=POLICIES,
            memory_mib=JETSON_EFFECTIVE_MEMORY_MIB,
            duration_ms=duration_ms,
            bandwidth_gbps=24.0,
        )
        print_result_table(f"{title} / kappa={kappa:.1f}", results)
    print("memory_note,weights_shared_conceptually,activation_memory_dominates,smart_traffic_naive_full_unpruned=6500MiB,jetson_effective_mib=6144")


def build_uav_vit_case() -> CaseBundle:
    vit_l = make_model(
        RebuttalModelSpec(
            name="vit_l_1024",
            input_shape=(3, 1024, 1024),
            num_exits=4,
            full_latency_ms=230.0,
            full_memory_mib=1850.0,
            full_accuracy=0.965,
            earliest_accuracy=0.550,
            num_blocks=12,
        )
    )
    mobilevit_s = make_model(
        RebuttalModelSpec(
            name="mobilevit_s_scene15",
            input_shape=(3, 512, 512),
            num_exits=4,
            full_latency_ms=96.0,
            full_memory_mib=760.0,
            full_accuracy=0.951,
            earliest_accuracy=0.610,
            num_blocks=8,
        )
    )
    models = {vit_l.name: vit_l, mobilevit_s.name: mobilevit_s}
    atlas = {name: make_variants(model) for name, model in models.items()}
    tasks = [
        # Large scene context has frequent refreshes but a looser control deadline.
        periodic_task("vit_l_1024", deadline_ms=500, period_ms=240, duration_ms=1200, shape=(3, 1024, 1024), start_ms=0),
        periodic_task("vit_l_1024", deadline_ms=500, period_ms=260, duration_ms=1200, shape=(3, 1024, 1024), start_ms=55),
        periodic_task("vit_l_1024", deadline_ms=540, period_ms=300, duration_ms=1200, shape=(3, 1024, 1024), start_ms=110),
        # Lightweight emergency scene classifier has a longer period but tighter deadline.
        periodic_task("mobilevit_s_scene15", deadline_ms=220, period_ms=360, duration_ms=1200, shape=(3, 512, 512), start_ms=145),
        periodic_task("mobilevit_s_scene15", deadline_ms=240, period_ms=420, duration_ms=1200, shape=(3, 512, 512), start_ms=210),
    ]
    return "ViT UAV scene recognition: ViT-L + MobileViT-S", models, atlas, tasks, 1200


def uav_vit() -> None:
    title, models, atlas, tasks, duration_ms = build_uav_vit_case()
    for kappa in KAPPAS:
        results = run_policies(
            models,
            atlas,
            scaled_deadlines(tasks, kappa),
            policies=POLICIES,
            memory_mib=JETSON_EFFECTIVE_MEMORY_MIB,
            duration_ms=duration_ms,
            bandwidth_gbps=24.0,
        )
        print_result_table(f"{title} / kappa={kappa:.1f}", results)


def build_mixed_modern_case() -> CaseBundle:
    gpt2_edge = make_model(
        RebuttalModelSpec(
            name="gpt2_edge_kv_bound",
            input_shape=(1024,),
            num_exits=3,
            full_latency_ms=180.0,
            full_memory_mib=1350.0,
            full_accuracy=0.900,
            earliest_accuracy=0.680,
            num_blocks=6,
        )
    )
    yolo_l = make_model(
        RebuttalModelSpec("yolov8l_1080p", (3, 1080, 1920), 4, 145.0, 1700.0, 0.924, 0.412, 12)
    )
    vit_l = make_model(
        RebuttalModelSpec("vit_l_1024", (3, 1024, 1024), 4, 230.0, 1850.0, 0.965, 0.550, 12)
    )
    models = {model.name: model for model in (gpt2_edge, yolo_l, vit_l)}
    atlas = {name: make_variants(model) for name, model in models.items()}
    tasks = [
        # Navigation perception: high-frequency, moderate deadline.
        periodic_task("yolov8l_1080p", 320, 190, 1000, (3, 1080, 1920), start_ms=0),
        periodic_task("yolov8l_1080p", 340, 230, 1000, (3, 1080, 1920), start_ms=45),
        # Transformer scene summary: slower period but tighter response requirement.
        periodic_task("vit_l_1024", 270, 360, 1000, (3, 1024, 1024), start_ms=80),
        # Command generation has the tightest deadline among long-period tasks.
        periodic_task("gpt2_edge_kv_bound", 220, 420, 1000, (1024,), start_ms=120),
        periodic_task("gpt2_edge_kv_bound", 240, 500, 1000, (1024,), start_ms=260),
    ]
    return "mixed modern deployment: YOLO + ViT + edge GPT-2", models, atlas, tasks, 1000


def mixed_modern_deployment() -> None:
    title, models, atlas, tasks, duration_ms = build_mixed_modern_case()
    for kappa in KAPPAS:
        results = run_policies(
            models,
            atlas,
            scaled_deadlines(tasks, kappa),
            policies=POLICIES,
            memory_mib=JETSON_EFFECTIVE_MEMORY_MIB,
            duration_ms=duration_ms,
            bandwidth_gbps=24.0,
        )
        print_result_table(f"{title} / kappa={kappa:.1f}", results)


def main() -> None:
    smart_traffic_yolo()
    uav_vit()
    mixed_modern_deployment()


if __name__ == "__main__":
    main()
