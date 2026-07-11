import pickle
import torch
import os
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from dyllm.config import Config
from dyllm.engine.sequence import Sequence
from dyllm.model_executor.models import LLaDAForDLM, DreamForDLM

from dyllm.model_executor.layers.sampler import LLaDASampler, DreamSampler

from dyllm.utils.context import set_context, get_context, reset_context
from dyllm.utils.weight_loader import load_model
from dyllm.utils.metadata import get_metadata


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.enforce_eager = True  # config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize distributed process group.
        # Prefer per-engine allocated port from config to avoid overlap across runs.
        port = config.dist_port if config.dist_port is not None else int(os.environ.get("DYLLM_DIST_PORT", "29500"))
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://localhost:{port}", world_size=self.world_size, rank=rank
        )

        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        if hf_config.model_type == "Dream":
            self.model = DreamForDLM(hf_config, config.threshold)
            self.sampler = DreamSampler("entropy")
        elif hf_config.model_type == "llada":
            self.model = LLaDAForDLM(hf_config, config.threshold)
            self.sampler = LLaDASampler("confidence")
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        load_model(self.model, config.model)
        if not self.enforce_eager:
            self.capture_cudagraph()
        # torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="dyllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="dyllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def prepare_full(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        idx_salient_row = []
        context_lens = []
        seq_ids = []
        metadata = get_metadata()
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq)
            positions.extend(list(range(seqlen)))
            seqlen_q = seqlen
            seqlen_k = seqlen
            context_lens.append(seqlen)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            idx_salient_row.extend(list(range(seqlen)))
            seq_ids.append(seq.seq_id)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_list = cu_seqlens_q
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_list = context_lens
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        metadata.running_seqs = seq_ids
        metadata.running_seqs_tensor = torch.tensor(seq_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_list,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            context_lens=context_lens,
            context_lens_cpu=context_lens_list,
            total_seqlen=cu_seqlens_q_list[-1],
        )
        return input_ids, positions

    def prepare_sparse(self, seqs: list[Sequence]):
        metadata = get_metadata()
        input_ids = []
        idx_updated_rows = []
        idx_salient_rows = []
        idx_salient_rows_k = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        cu_promptlens = [0]
        cu_updatedlens = [0]
        cu_salientlens = [0]
        positions = []
        positions_k = []
        seq_ids = []
        context_lens = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        for seq in seqs:
            seqlen = len(seq)
            seqlen_q = seq.max_new_tokens
            cu_promptlens.append(seq.num_prompt_tokens + cu_promptlens[-1])
            answer_start = seqlen - seq.max_new_tokens
            kept = [
                (p, t)
                for p, t in zip(seq.last_token_pos, seq.last_tokens)
                if p >= answer_start
            ]
            kept_pos = [p for p, _ in kept]
            kept_count = len(kept_pos)
            if seq.processed_steps > 4 and seq.processed_steps % 4 != 0:
                input_ids.extend(seq[-seq.max_new_tokens :])
                positions.extend(list(range(seqlen - seq.max_new_tokens, seqlen)))
                idx_salient_rows.extend(
                    [i - seq.num_prompt_tokens + cu_seqlens_q[-1] for i in kept_pos]
                )
                idx_salient_rows_k.extend([i + cu_seqlens_k[-1] for i in kept_pos])
                seqlen_q = seq.max_new_tokens
            else:
                input_ids.extend(seq)
                positions.extend(list(range(seqlen)))
                idx_salient_rows.extend([i + cu_seqlens_q[-1] for i in kept_pos])
                idx_salient_rows_k = None
                seqlen_q = seqlen
            positions_k.extend(kept_pos)
            idx_updated_rows.extend([i + cu_seqlens_k[-1] for i in kept_pos])
            cu_updatedlens.append(kept_count + cu_updatedlens[-1])
            cu_salientlens.append(kept_count + cu_salientlens[-1])
            seq_ids.append(seq.seq_id)
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            context_lens.append(seqlen)

        input_ids_cpu = torch.tensor(input_ids, dtype=torch.int64, device="cpu").pin_memory()
        positions_cpu = torch.tensor(positions, dtype=torch.int64, device="cpu").pin_memory()
        positions_k_cpu = torch.tensor(positions_k, dtype=torch.int64, device="cpu").pin_memory()

        cu_seqlens_q_list = cu_seqlens_q
        cu_seqlens_k_list = cu_seqlens_k

        cu_seqlens_q_cpu = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu").pin_memory()
        cu_seqlens_k_cpu = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu").pin_memory()
        cu_promptlens_cpu = torch.tensor(cu_promptlens, dtype=torch.int32, device="cpu").pin_memory()
        cu_updatedlens_cpu = torch.tensor(cu_updatedlens, dtype=torch.int32, device="cpu").pin_memory()
        cu_salientlens_cpu = torch.tensor(cu_salientlens, dtype=torch.int32, device="cpu").pin_memory()

        idx_updated_rows_cpu = torch.tensor(idx_updated_rows, dtype=torch.int32, device="cpu").pin_memory()
        idx_salient_rows_cpu = torch.tensor(idx_salient_rows, dtype=torch.int64, device="cpu").pin_memory()

        idx_salient_rows_k_cpu = (
            torch.tensor(idx_salient_rows_k, dtype=torch.int64, device="cpu").pin_memory()
            if idx_salient_rows_k is not None
            else None
        )

        context_lens_list = context_lens
        context_lens_cpu = torch.tensor(context_lens, dtype=torch.int32, device="cpu").pin_memory()

        metadata.running_seqs = seq_ids
        seq_ids_cpu = torch.tensor(seq_ids, dtype=torch.long, device="cpu").pin_memory()

        input_ids = input_ids_cpu.to(device="cuda", non_blocking=True)
        positions = positions_cpu.to(device="cuda", non_blocking=True)
        positions_k = positions_k_cpu.to(device="cuda", non_blocking=True)
        cu_seqlens_q = cu_seqlens_q_cpu.to(device="cuda", non_blocking=True)
        cu_seqlens_k = cu_seqlens_k_cpu.to(device="cuda", non_blocking=True)
        cu_promptlens = cu_promptlens_cpu.to(device="cuda", non_blocking=True)
        cu_updatedlens = cu_updatedlens_cpu.to(device="cuda", non_blocking=True)
        cu_salientlens = cu_salientlens_cpu.to(device="cuda", non_blocking=True)
        idx_updated_rows = idx_updated_rows_cpu.to(device="cuda", non_blocking=True)
        idx_salient_rows = idx_salient_rows_cpu.to(device="cuda", non_blocking=True)
        idx_salient_rows_k = (
            idx_salient_rows_k_cpu.to(device="cuda", non_blocking=True) if idx_salient_rows_k_cpu is not None else None
        )
        context_lens = context_lens_cpu.to(device="cuda", non_blocking=True)
        metadata.running_seqs_tensor = seq_ids_cpu.to(device="cuda", non_blocking=True)

        set_context(
            False,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_q_cpu=cu_seqlens_q_list,
            cu_seqlens_k=cu_seqlens_k,
            cu_promptlens=cu_promptlens,
            cu_updatedlens=cu_updatedlens,
            cu_salientlens=cu_salientlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            idx_updated_row=idx_updated_rows,
            idx_salient_row=idx_salient_rows,
            idx_salient_row_k=idx_salient_rows_k,
            context_lens=context_lens,
            context_lens_cpu=context_lens_list,
            total_seqlen_k=cu_seqlens_k_list[-1],
            total_seqlen=cu_seqlens_q_list[-1],
            positions_k=positions_k,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        mask_id = self.config.mask_id
        rels, counts, temps, transfers = [], [], [], []
        top_ps, top_ks = [], []
        conf_thrs = []
        batch_offsets = []
        has_p = has_k = False
        has_conf_thr = False

        is_sparse_k = get_context().idx_salient_row_k is not None

        for seq in seqs:
            limit = min(len(seq.token_ids), seq.block_size * (seq.block_idx + 1) + seq.num_prompt_tokens)
            mask_pos = [i for i, t in enumerate(seq.token_ids[:limit]) if t == mask_id]

            current_offset = 0
            if is_sparse_k:
                current_offset = len(seq) - seq.max_new_tokens
                mask_pos = [p - current_offset for p in mask_pos if p >= current_offset]

            rels.extend(mask_pos)
            batch_offsets.append(current_offset)
            counts.append(len(mask_pos))
            (
                temps.extend([seq.temperature] * len(mask_pos))
                if seq.temperature is not None and seq.temperature > 1e-6
                else temps.extend([None] * len(mask_pos))
            )
            transfers.append(1 if seq.confidence_threshold is not None else seq.num_transfer_tokens)
            p_val = seq.top_p if seq.top_p is not None else None
            top_ps.extend([p_val] * len(mask_pos))
            k_val = seq.top_k if seq.top_k is not None else None
            top_ks.extend([k_val] * len(mask_pos))
            conf_thrs.append(seq.confidence_threshold if seq.confidence_threshold is not None else float("inf"))
            has_p |= seq.top_p is not None
            has_k |= seq.top_k is not None
            has_conf_thr |= seq.confidence_threshold is not None

        block_size = max([seq.block_size if seq.block_size is not None else seq.max_new_tokens for seq in seqs])

        cu_cpu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device="cpu")
        cu_cpu[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32, device="cpu"), dim=0)
        cu_cpu = cu_cpu.pin_memory()

        if all(t is None for t in temps):
            temps_cpu = None
        else:
            safe_temps = [t if t is not None else 0.0 for t in temps]
            temps_cpu = torch.tensor(safe_temps, dtype=torch.float32, device="cpu").pin_memory()

        transfers_cpu = torch.tensor(transfers, dtype=torch.int32, device="cpu").pin_memory()
        p_cpu = torch.tensor(top_ps, dtype=torch.float32, device="cpu").pin_memory() if has_p else None
        k_cpu = torch.tensor(top_ks, dtype=torch.int32, device="cpu").pin_memory() if has_k else None
        conf_thr_cpu = (
            torch.tensor(conf_thrs, dtype=torch.float32, device="cpu").pin_memory() if has_conf_thr else None
        )

        rels_cpu = torch.tensor(rels, dtype=torch.long, device="cpu").pin_memory()
        offsets_cpu = torch.tensor(batch_offsets, dtype=torch.long, device="cpu").pin_memory()

        cu_filtered = cu_cpu.to(device="cuda", non_blocking=True)
        temperatures = temps_cpu.to(device="cuda", non_blocking=True) if temps_cpu is not None else None
        num_transfer_tokens = transfers_cpu.to(device="cuda", non_blocking=True)
        top_p = p_cpu.to(device="cuda", non_blocking=True) if p_cpu is not None else None
        top_k = k_cpu.to(device="cuda", non_blocking=True) if k_cpu is not None else None
        confidence_thresholds = (
            conf_thr_cpu.to(device="cuda", non_blocking=True) if conf_thr_cpu is not None else None
        )

        rel = rels_cpu.to(device="cuda", non_blocking=True)
        batch_offsets_gpu = offsets_cpu.to(device="cuda", non_blocking=True)

        return (
            (temperatures, top_k, top_p, block_size, confidence_thresholds),
            (rel, batch_offsets_gpu, cu_filtered),
            num_transfer_tokens,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_full: bool):
        if is_full or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_full: bool) -> list[int]:
        input_ids, positions = self.prepare_full(seqs) if is_full else self.prepare_sparse(seqs)

        logits = self.run_model(input_ids, positions, is_full)
        if self.rank == 0:
            sampler_params, input_indices, num_transfer_tokens = self.prepare_sample(seqs)
            pos, token_ids, counts = self.sampler(
                input_logits=logits,
                ctx=get_context(),
                input_indices=input_indices,
                confidence_thresholds=sampler_params[4],
                temperatures=sampler_params[0],
                num_transfer=num_transfer_tokens,
                top_k=sampler_params[1],
                top_p=sampler_params[2],
                block_size=sampler_params[3],
            )
        else:
            pos, token_ids, counts = None, None, None
        reset_context()
        return pos, token_ids, counts
