import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class GPUResidentAdam:
    """Resident-block GPU Adam state keyed by block_id.

    The paper-mode GPU resident optimizer keeps Adam state only for the
    current resident set R_t. When a block is evicted from R_t, its
    optimizer state is dropped; if the block is readmitted later, its state
    is cold-restarted from zeros.
    """

    COMPONENT_SPECS = (
        ('xyz', '_xyz', 3, 0),
        ('opacity', '_opacity', 1, 1),
        ('scaling', '_scaling', 3, 2),
        ('rotation', '_rotation', 4, 3),
        ('features_dc', '_features_dc', 3, 4),
        ('features_rest', '_features_rest', 45, 5),
    )

    def __init__(self, batch_size: int = 1, block_size: int = 4096, device: str = 'cuda'):
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.state_mode = 'resident_blocks'

        self._resident_blocks = set()
        self._resident_block_states: Dict[int, Dict[str, Any]] = {}
        self._resident_streaks: Dict[int, int] = {}
        self._completed_streak_total = 0
        self._completed_streak_count = 0
        self._stats = {
            'state_mode': 'resident_blocks',
            'resident_target_blocks': 0,
            'resident_state_blocks': 0,
            'resident_state_bytes': 0,
            'cold_restarts': 0,
            'state_evictions': 0,
            'optimizer_rows_touched_total': 0,
            'cold_restarted_rows_touched_total': 0,
            'mean_resident_streak': 0.0,
        }

    def _update_state_stats(self) -> None:
        bytes_total = 0
        for state in self._resident_block_states.values():
            for name, _, _, _ in self.COMPONENT_SPECS:
                exp_avg = state['exp_avg'][name]
                exp_avg_sq = state['exp_avg_sq'][name]
                bytes_total += int(exp_avg.numel() * exp_avg.element_size())
                bytes_total += int(exp_avg_sq.numel() * exp_avg_sq.element_size())
        self._stats['resident_target_blocks'] = len(self._resident_blocks)
        self._stats['resident_state_blocks'] = len(self._resident_block_states)
        self._stats['resident_state_bytes'] = int(bytes_total)
        streak_total = self._completed_streak_total + sum(self._resident_streaks.values())
        streak_count = self._completed_streak_count + len(self._resident_streaks)
        self._stats['mean_resident_streak'] = float(streak_total) / max(1, streak_count)

    def _normalize_columns_lr(self, columns_lr) -> Dict[str, float]:
        if columns_lr is None:
            raise RuntimeError('[GPUResidentAdam] optimizer.columns_lr is required for gpu_resident backend')
        if torch.is_tensor(columns_lr):
            cols = columns_lr.detach()
            if cols.device.type != 'cpu':
                cols = cols.cpu()
            cols = cols.to(dtype=torch.float32)
        else:
            cols = torch.tensor(columns_lr, dtype=torch.float32)
        cols = cols.flatten().contiguous()
        if cols.numel() == 6:
            grouped = cols
        elif cols.numel() == 59:
            grouped = torch.tensor([
                float(cols[0].item()),
                float(cols[3].item()),
                float(cols[4].item()),
                float(cols[7].item()),
                float(cols[11].item()),
                float(cols[14].item()),
            ], dtype=torch.float32)
        else:
            raise RuntimeError(
                f'[GPUResidentAdam] Unexpected columns_lr width={cols.numel()}; expected 6 grouped or 59 expanded entries'
            )
        return {
            name: float(grouped[group_idx].item())
            for name, _, _, group_idx in self.COMPONENT_SPECS
        }

    def _ensure_block_state(self, block_id: int, row_count: int) -> Dict[str, Any]:
        state = self._resident_block_states.get(block_id)
        if state is not None:
            return state
        state = {
            'step': 0,
            'exp_avg': {},
            'exp_avg_sq': {},
        }
        for name, _, width, _ in self.COMPONENT_SPECS:
            state['exp_avg'][name] = torch.zeros((row_count, width), dtype=torch.float32, device=self.device)
            state['exp_avg_sq'][name] = torch.zeros((row_count, width), dtype=torch.float32, device=self.device)
        self._resident_block_states[block_id] = state
        self._stats['cold_restarts'] += 1
        self._update_state_stats()
        return state

    def set_resident_blocks(self, block_ids: List[int]) -> None:
        next_resident = {int(block_id) for block_id in block_ids}
        previous_resident = set(self._resident_blocks)
        evicted_from_residency = previous_resident - next_resident
        kept_resident = previous_resident & next_resident
        incoming_resident = next_resident - previous_resident
        for block_id in evicted_from_residency:
            streak = int(self._resident_streaks.pop(block_id, 0))
            if streak > 0:
                self._completed_streak_total += streak
                self._completed_streak_count += 1
        for block_id in kept_resident:
            self._resident_streaks[block_id] = int(self._resident_streaks.get(block_id, 0)) + 1
        for block_id in incoming_resident:
            self._resident_streaks[block_id] = 1

        evicted_blocks = [
            block_id for block_id in list(self._resident_block_states.keys())
            if block_id not in next_resident
        ]
        for block_id in evicted_blocks:
            self._resident_block_states.pop(block_id, None)
        self._stats['state_evictions'] += len(evicted_blocks)
        self._resident_blocks = next_resident
        self._update_state_stats()

    def step(
        self,
        iteration: int,
        gaussians,
        sparse_grad_local_ids: torch.Tensor,
        sparse_grad_components: Dict[str, torch.Tensor],
    ) -> Dict[str, int]:
        if sparse_grad_local_ids is None or sparse_grad_components is None:
            return {'updated_blocks': 0, 'touched_rows': 0, 'cold_rows': 0}
        if sparse_grad_local_ids.numel() == 0:
            return {'updated_blocks': 0, 'touched_rows': 0, 'cold_rows': 0}

        manager = getattr(gaussians, 'gpu_working_set_manager', None)
        if manager is None or manager.local_to_global_idx is None:
            raise RuntimeError('[GPUResidentAdam] gpu_working_set_manager with local_to_global_idx is required')

        local_ids_gpu = sparse_grad_local_ids
        if local_ids_gpu.device != self.device:
            local_ids_gpu = local_ids_gpu.to(self.device)
        local_ids_cpu = local_ids_gpu.detach()
        if local_ids_cpu.device.type != 'cpu':
            local_ids_cpu = local_ids_cpu.cpu()
        local_ids_cpu = local_ids_cpu.to(dtype=torch.long).contiguous()
        local_ids_np = local_ids_cpu.numpy()

        param_views = {
            name: getattr(gaussians, attr_name)
            for name, attr_name, _, _ in self.COMPONENT_SPECS
        }
        component_lrs = self._normalize_columns_lr(getattr(gaussians.optimizer, 'columns_lr', None))
        beta1, beta2 = gaussians.optimizer.param_groups[0]['betas']
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(gaussians.optimizer.param_groups[0]['eps'])

        touched_rows = int(local_ids_cpu.numel())
        self._stats['optimizer_rows_touched_total'] += touched_rows

        updated_blocks = 0
        cold_rows = 0

        with torch.no_grad():
            for block_id in manager.loaded_blocks:
                block_slice = manager.block_to_gpu_slice.get(block_id)
                if block_slice is None:
                    continue
                left = int(np.searchsorted(local_ids_np, block_slice.start, side='left'))
                right = int(np.searchsorted(local_ids_np, block_slice.stop, side='left'))
                if right <= left:
                    continue

                updated_blocks += 1
                count = right - left
                row_count = int(block_slice.stop - block_slice.start)
                block_local_rows = (local_ids_gpu[left:right] - int(block_slice.start)).to(dtype=torch.long)

                persistent_state = (not self._resident_blocks) or (int(block_id) in self._resident_blocks)
                if persistent_state:
                    was_cold_restart = int(block_id) not in self._resident_block_states
                    state = self._ensure_block_state(int(block_id), row_count)
                    if was_cold_restart:
                        cold_rows += int(count)
                        self._stats['cold_restarted_rows_touched_total'] += int(count)
                    state['step'] += 1
                    step = int(state['step'])
                else:
                    state = None
                    step = 1

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                denom_scale = math.sqrt(bias_correction2)

                for name, _, _, _ in self.COMPONENT_SPECS:
                    if name not in sparse_grad_components:
                        raise KeyError(f'[GPUResidentAdam] Missing sparse_grad_components[{name!r}] at iter={iteration}')
                    block_grads = sparse_grad_components[name][left:right]
                    if block_grads.device != self.device:
                        block_grads = block_grads.to(self.device)
                    block_grads = (block_grads / float(self.batch_size)).contiguous()

                    if persistent_state:
                        exp_avg = state['exp_avg'][name]
                        exp_avg_sq = state['exp_avg_sq'][name]
                        exp_avg_slice = exp_avg[block_local_rows]
                        exp_avg_sq_slice = exp_avg_sq[block_local_rows]
                    else:
                        exp_avg_slice = torch.zeros_like(block_grads)
                        exp_avg_sq_slice = torch.zeros_like(block_grads)

                    exp_avg_slice.mul_(beta1).add_(block_grads, alpha=1.0 - beta1)
                    exp_avg_sq_slice.mul_(beta2).addcmul_(block_grads, block_grads, value=1.0 - beta2)

                    if persistent_state:
                        exp_avg[block_local_rows] = exp_avg_slice
                        exp_avg_sq[block_local_rows] = exp_avg_sq_slice

                    denom = exp_avg_sq_slice.sqrt().div_(denom_scale).add_(eps)
                    update = (exp_avg_slice / denom) * (component_lrs[name] / bias_correction1)
                    param_block = param_views[name].data[block_slice]
                    param_block[block_local_rows] -= update

        self._update_state_stats()
        return {
            'updated_blocks': int(updated_blocks),
            'touched_rows': int(touched_rows),
            'cold_rows': int(cold_rows),
        }

    def get_stats(self) -> Dict[str, Any]:
        self._update_state_stats()
        return dict(self._stats)
