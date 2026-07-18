from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_EPOCH_SEED_BASE = 42


@dataclass(frozen=True)
class CameraBatchSchedule:
    iteration: int
    linear_batch_idx: int
    epoch: int
    within_epoch_idx: int
    num_cameras: int
    num_batches: int
    epoch_camera_offset: int
    batch_start_cam: int
    batch_indices: List[int]


def _circular_slice(schedule: Sequence[int], start: int, length: int) -> List[int]:
    num_items = len(schedule)
    if num_items == 0:
        return []
    return [int(schedule[(start + offset) % num_items]) for offset in range(length)]


def get_camera_batch_schedule(
    training_schedule: Sequence[int],
    iteration: int,
    batch_size: int,
    epoch_seed_base: int = DEFAULT_EPOCH_SEED_BASE,
    schedule_ordering: str = "trajectory",
) -> CameraBatchSchedule:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if len(training_schedule) == 0:
        raise ValueError("training_schedule must not be empty")

    num_cameras = len(training_schedule)
    num_batches = -(-num_cameras // batch_size)
    linear_batch_idx = max(0, (iteration - 1) // batch_size)
    epoch = linear_batch_idx // num_batches
    within_epoch_idx = linear_batch_idx % num_batches

    schedule_ordering = str(schedule_ordering).lower()
    rng = np.random.RandomState(seed=epoch_seed_base + epoch)

    if schedule_ordering == "shuffle":
        epoch_camera_offset = 0
        epoch_schedule = rng.permutation(np.asarray(training_schedule, dtype=np.int64)).tolist()
        batch_start_cam = within_epoch_idx * batch_size
        batch_indices = _circular_slice(epoch_schedule, batch_start_cam, batch_size)
    else:
        if epoch == 0:
            epoch_camera_offset = 0
        else:
            epoch_camera_offset = int(rng.randint(0, num_cameras))
        batch_start_cam = (within_epoch_idx * batch_size + epoch_camera_offset) % num_cameras
        batch_indices = _circular_slice(training_schedule, batch_start_cam, batch_size)

    return CameraBatchSchedule(
        iteration=iteration,
        linear_batch_idx=linear_batch_idx,
        epoch=epoch,
        within_epoch_idx=within_epoch_idx,
        num_cameras=num_cameras,
        num_batches=num_batches,
        epoch_camera_offset=epoch_camera_offset,
        batch_start_cam=batch_start_cam,
        batch_indices=batch_indices,
    )


def get_current_and_next_camera_batches(
    training_schedule: Sequence[int],
    iteration: int,
    batch_size: int,
    epoch_seed_base: int = DEFAULT_EPOCH_SEED_BASE,
    schedule_ordering: str = "trajectory",
) -> Tuple[CameraBatchSchedule, CameraBatchSchedule]:
    current_batch = get_camera_batch_schedule(
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=batch_size,
        epoch_seed_base=epoch_seed_base,
        schedule_ordering=schedule_ordering,
    )
    next_batch = get_camera_batch_schedule(
        training_schedule=training_schedule,
        iteration=iteration + batch_size,
        batch_size=batch_size,
        epoch_seed_base=epoch_seed_base,
        schedule_ordering=schedule_ordering,
    )
    return current_batch, next_batch
