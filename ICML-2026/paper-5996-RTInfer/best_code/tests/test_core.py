from pathlib import Path
import os
import unittest

from rtinfer.atlas import AtlasConfig, build_variant_atlas
from rtinfer.delta_graph import DeltaGraph, Residency
from rtinfer.layout import BufferBlock, MemoryLayoutScheduler
from rtinfer.model import BlockProfile, ExitProfile, ModelProfile, TaskSpec
from rtinfer.pantheon_io import load_repository
from rtinfer.scheduler import OnlineScheduler


PANTHEON = Path(os.environ.get("PANTHEON_ROOT", "../Pantheon"))
PROFILE_ROOT = Path(os.environ.get("PROFILE_ROOT", "../Pantheon_Datasets_Models/3_Exported_JIT_Models"))


def synthetic_model(name: str = "tiny_detector") -> ModelProfile:
    blocks = tuple(BlockProfile(block_id=i, latency_us=10_000, memory_mib=64.0) for i in range(4))
    exits = (
        ExitProfile(exit_id=0, previous_block_id=1, latency_us=2_000, accuracy=0.72),
        ExitProfile(exit_id=1, previous_block_id=3, latency_us=2_000, accuracy=0.91),
    )
    return ModelProfile(name=name, dims=(1, 3, 224, 224), blocks=blocks, exits=exits)


class CoreTest(unittest.TestCase):
    def test_variant_atlas_contains_pareto_choices(self):
        models = {"tiny_detector": synthetic_model()}
        atlas = build_variant_atlas(models, DeltaGraph(), AtlasConfig(accuracy_cap=0.25))
        self.assertTrue(atlas["tiny_detector"])
        self.assertTrue(all(variant.latency_us > 0 for variant in atlas["tiny_detector"]))
        self.assertTrue(all(variant.memory_mib > 0 for variant in atlas["tiny_detector"]))

    def test_scheduler_runs_rtinfer_on_synthetic_workload(self):
        model = synthetic_model()
        models = {model.name: model}
        atlas = build_variant_atlas(models, DeltaGraph(), AtlasConfig(accuracy_cap=0.25))
        tasks = [
            TaskSpec(model_name=model.name, deadline_us=90_000, period_us=100_000, start_us=0, end_us=300_000, shape=model.dims),
            TaskSpec(model_name=model.name, deadline_us=90_000, period_us=100_000, start_us=25_000, end_us=300_000, shape=model.dims),
        ]
        result = OnlineScheduler(models, atlas, memory_budget_mib=256.0, delta_graph=DeltaGraph(), policy="rtinfer").run(tasks, 300_000)
        self.assertEqual(result.total_jobs, 6)
        self.assertLessEqual(result.deadline_miss_rate, 1.0)
        self.assertTrue(all(job.variant is not None for job in result.schedule_events))

    def test_delta_graph_residency_loads_missing_chunks_once(self):
        model = synthetic_model()
        delta = DeltaGraph(page_mib=1.0)
        chunks = delta.chunks_for_variant(model, pruning=0.0, exit_index=1)
        residency = Residency(memory_budget_bytes=10**9)
        first = residency.touch(chunks)
        second = residency.touch(chunks)
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_layout_rejects_overlapping_memory_pressure(self):
        scheduler = MemoryLayoutScheduler(memory_budget_mib=10.0)
        buffers = [
            BufferBlock(job_id=0, block_id=0, start_us=0, end_us=10, size_mib=6.0),
            BufferBlock(job_id=1, block_id=0, start_us=5, end_us=15, size_mib=6.0),
        ]
        self.assertIsNone(scheduler.place(buffers))

    def test_layout_accepts_non_overlapping_lifetimes(self):
        scheduler = MemoryLayoutScheduler(memory_budget_mib=10.0)
        buffers = [
            BufferBlock(job_id=0, block_id=0, start_us=0, end_us=10, size_mib=6.0),
            BufferBlock(job_id=1, block_id=0, start_us=10, end_us=15, size_mib=6.0),
        ]
        self.assertIsNotNone(scheduler.place(buffers))

    @unittest.skipUnless(PANTHEON.exists() and PROFILE_ROOT.exists(), "Pantheon artifacts are not available")
    def test_repository_fallback_loads_all_profile_models_when_artifacts_exist(self):
        models = load_repository(PANTHEON, PROFILE_ROOT)
        self.assertIn("age_classification", models)
        self.assertIn("object_detection", models)
        self.assertIn("scene_recognition", models)


if __name__ == "__main__":
    unittest.main()
