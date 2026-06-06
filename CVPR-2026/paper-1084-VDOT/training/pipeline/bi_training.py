from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class BiTrainingPipeline:
    def __init__(
            self,
            denoising_step_list: List[int],
            scheduler: SchedulerInterface,
            generator: WanDiffusionWrapper,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
            # indices = torch.randint(
            #     low=num_denoising_steps - 1,
            #     high=num_denoising_steps,
            #     size=(1,),
            #     device=device
            # )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            exit_flags=None,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_denoising_steps = len(self.denoising_step_list)

        if exit_flags is None:
            exit_flags = self.generate_and_sync_list(1, device=noise.device)
        # exit_flags = exit_flags
        print('exit_flags', exit_flags, flush=True)
        noisy_image_or_video = noise
        # print('dataset:', noisy_image_or_video.shape, flush=True)
        # Step 3.1: Spatial denoising loop
        # print('time_steps:', self.denoising_step_list, flush=True)
        denoising_step_list = [1000.0000,  960.0000,  888.8889,  727.2728]
        # for index, current_timestep in enumerate(self.denoising_step_list):
        for index, current_timestep in enumerate(denoising_step_list):
            # print(f"{index} generation....")
            # device = torch.cuda.current_device()
            # allocated = torch.cuda.memory_allocated(device)
            # reserved = torch.cuda.memory_reserved(device)
            # print(f"显存占用: {allocated / 1024 ** 2:.2f} MB (已分配) / {reserved / 1024 ** 2:.2f} MB (已保留)")
            exit_flag = (index == exit_flags[0])
            timestep = torch.ones(
                [batch_size, num_frames],
                device=noisy_image_or_video.device,
                dtype=torch.int64) * current_timestep

            if not exit_flag:
                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep=timestep
                    )  # [B, F, C, H, W]

                    # next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    #     noise.shape[:2], dtype=torch.long, device=noise.device)
                    next_timestep = denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)
                    noisy_image_or_video = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, denoised_pred.shape[:2])
                    # print('shape:', denoised_pred.shape, noisy_image_or_video.shape, next_timestep.shape, noisy_image_or_video.dtype, flush=True)
            else:
                # torch.cuda.empty_cache()
                # for getting real output
                # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()


        return denoised_pred, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
