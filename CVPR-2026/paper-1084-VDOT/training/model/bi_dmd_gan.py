from pipeline import BiTrainingVacePipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel
import copy
import time 

class BiDMD_GAN(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        print('---'*30)
        print('concat_time_embeddings', self.concat_time_embeddings)
        print('---'*30)
        # self.num_class = args.num_class
        self.relativistic_discriminator = getattr(args, "relativistic_discriminator", False)

        self.fake_score.adding_cls_branch(
            atten_dim=5120, num_class=1, time_embed_dim=5120 if self.concat_time_embeddings else 0)
        self.fake_score.model.requires_grad_(True)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: BiTrainingVacePipeline = None

        self.min_step_percent = [500, 0.02, 0.02, 1000]
        self.max_step_percent = [500, 0.98, 0.5, 1000]
        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.critic_timestep_shift = getattr(args, "critic_timestep_shift", self.timestep_shift)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        self.gan_g_weight = getattr(args, "gan_g_weight", 1e-2)
        self.gan_d_weight = getattr(args, "gan_d_weight", 1e-2)
        self.r1_weight = getattr(args, "r1_weight", 0.0)
        self.r2_weight = getattr(args, "r2_weight", 0.0)
        self.r1_sigma = getattr(args, "r1_sigma", 0.05)
        self.r2_sigma = getattr(args, "r2_sigma", 0.05)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
    
    def _run_cls_pred_branch(self,
                             noisy_image_or_video: torch.Tensor,
                             conditional_dict: dict,
                             timestep: torch.Tensor,
                             vace_context) -> torch.Tensor:
        """
            Run the classifier prediction branch on the generated image or video.
            Input:
                - image_or_video: a tensor with shape [B, F, C, H, W].
            Output:
                - cls_pred: a tensor with shape [B, 1, 1, 1, 1] representing the feature map for classification.
        """
        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
            vace_context=vace_context,
        )

        return noisy_logit


    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        vace_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        pred_fake_v, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            vace_context=vace_context,
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                vace_context=vace_context,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_v_cond, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            vace_context=vace_context,
        )

        pred_v_uncond, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            vace_context=vace_context,
        )

        pred_v = pred_v_uncond + (pred_v_cond - pred_v_uncond) * self.real_guidance_scale

        pred_real_image = self.real_score._convert_flow_pred_to_x0(
            flow_pred=pred_v.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, pred_v.shape[:2])

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            grad = grad * (timestep.reshape(timestep.shape[0], timestep.shape[1], 1, 1, 1) / 1000.0)
        grad = torch.nan_to_num(grad)
        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }


    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        vace_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_step
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.max_step
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)
            
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                vace_context=vace_context,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict
    

    # @torch.no_grad()
    def _safe_log(self, x, eps=1e-12):
        return torch.log(x + eps)


    def pairwise_cost(self, x, y, p=2):
        # x:[N,D], y:[M,D], return C:[N,M]
        d = torch.cdist(x, y, p=2)         # Euclidean distance
        if p == 1:
            return d
        elif p == 2:
            return d ** 2
        else:
            return d ** p


    def sinkhorn_logdomain_cost(self, x, y, eps=0.05, iters=100, p=2):
        """
        Entropy-regularized OT approximation W_p^p(x,y); log-domain Sinkhorn, end-to-end differentiable.
        x: [N,D], y: [M,D]
        eps: entropy regularization; smaller means closer to true OT, harder to compute
        """
        N, M = x.size(0), y.size(0)
        a = torch.full((N,), 1.0 / N, dtype=x.dtype, device=x.device)
        b = torch.full((M,), 1.0 / M, dtype=y.dtype, device=y.device)

        C = self.pairwise_cost(x, y, p=p)          # [N,M]
        K_log = -C / eps                      # log K = -C/eps

        # log u, log v
        f = torch.zeros_like(a)               # [N]
        g = torch.zeros_like(b)               # [M]

        # log-sum-exp iteration
        loga = self._safe_log(a)
        logb = self._safe_log(b)
        for _ in range(iters):
            # f <- log(a) - logsumexp(K_log + g[None,:], dim=1)
            f = loga - torch.logsumexp(K_log + g[None, :], dim=1)
            # g <- log(b) - logsumexp(K_log^T + f[None,:], dim=1)
            g = logb - torch.logsumexp(K_log.T + f[None, :], dim=1)

        # log Π = f[:,None] + K_log + g[None,:]
        logP = f[:, None] + K_log + g[None, :]
        P = torch.exp(logP)                   # [N,M], differentiable
        cost = torch.sum(P * C)               # scalar, differentiable
        return cost


    def sinkhorn_divergence(self, fake_bchw, real_bchw, eps=0.05, iters=50, p=2, debias=True):
        """
        Sinkhorn divergence: OT_eps(x,y) - 1/2 OT_eps(x,x) - 1/2 OT_eps(y,y)
        input : 
            x: [B, C, H, W]
            y: [B, C, H, W]
        """

        Bp, C, H, W = fake_bchw.shape
        x_bnd = fake_bchw.flatten(2).transpose(1, 2)   # [B', N, D]
        y_bnd = real_bchw.flatten(2).transpose(1, 2)

        total = torch.zeros(1, device=fake_bchw.device, dtype=fake_bchw.dtype, requires_grad=True)
        for b in range(Bp):
            ab = self.sinkhorn_logdomain_cost(x_bnd[b], y_bnd[b], eps=eps, iters=iters, p=p)
            if not debias: 
                total = total + ab
            else:
                aa = self.sinkhorn_logdomain_cost(x_bnd[b], x_bnd[b], eps=eps, iters=iters, p=p).detach()
                bb = self.sinkhorn_logdomain_cost(y_bnd[b], y_bnd[b], eps=eps, iters=iters, p=p).detach()
                total = total + (ab - 0.5 * (aa + bb))
        
        cost = total

        return cost

    def ot_grad_against_fake(self, fake_latent, real_latent, eps=0.05, iters=100, p=2, debias=False):
        """
        Returns: ∇_{fake} SinkhornDivergence(fake, real)
        fake_latent: [B, T, C, H, W]
        real_latent: [B, T, C, H, W]
        """

        orig_shape = fake_latent.shape
        B, T, C, H, W = orig_shape

        fake_latent = fake_latent.requires_grad_(True)

        fake_bchw = fake_latent.view(B * T, C, H, W)
        real_bchw = real_latent.view(B * T, C, H, W)

        # compress h and w to 1/4 
        # -> [1, 21, 16, 60/4, 104/4]
        # make sure H%4==0 and W%4==0
        down_factor = 2 
        fake_ds = F.avg_pool2d(fake_bchw, kernel_size=down_factor, stride=down_factor)
        real_ds = F.avg_pool2d(real_bchw, kernel_size=down_factor, stride=down_factor)
        
        time0 = time.time()
        cost = self.sinkhorn_divergence(fake_ds, real_ds, eps=eps, iters=iters, p=p, debias=debias)
        time1 = time.time()
        print(f'sinkhorn cost {time1-time0}s. ot distance: {cost.detach().item()}.')
        grad_bchw = torch.autograd.grad(cost, fake_bchw, create_graph=False, retain_graph=False, allow_unused=False)[0]
        grad_same_shape = grad_bchw.view(orig_shape)

        return grad_same_shape.to(fake_latent.dtype), cost.to(fake_latent.dtype)

    def _compute_ot_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        vace_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        with torch.no_grad():
            # Step 1: Compute the fake score
            pred_fake_v, pred_fake_image_cond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                vace_context=vace_context,
            )

            if self.fake_guidance_scale != 0.0:
                _, pred_fake_image_uncond = self.fake_score(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    vace_context=vace_context,
                )
                pred_fake_image = pred_fake_image_cond + (
                    pred_fake_image_cond - pred_fake_image_uncond
                ) * self.fake_guidance_scale
            else:
                pred_fake_image = pred_fake_image_cond

            # Step 2: Compute the real score
            # We compute the conditional and unconditional prediction
            # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
            pred_v_cond, pred_real_image_cond = self.real_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                vace_context=vace_context,
            )

            pred_v_uncond, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                vace_context=vace_context,
            )

            pred_v = pred_v_uncond + (pred_v_cond - pred_v_uncond) * self.real_guidance_scale

            pred_real_image = self.real_score._convert_flow_pred_to_x0(
                flow_pred=pred_v.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, pred_v.shape[:2])

        ot_grad, ot_cost = self.ot_grad_against_fake(
            fake_latent=pred_fake_image_cond, 
            real_latent=pred_real_image,
            eps=0.1, 
            iters=60, 
            p=2
        )

        ot_grad = ot_grad / (ot_grad.flatten(start_dim=2).norm(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-8)
        ot_grad = torch.nan_to_num(ot_grad)
        return ot_grad


    def compute_ot_loss(self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        vace_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the OT loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_step
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.max_step
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
        grad = self._compute_ot_grad(
            noisy_image_or_video=noisy_latent,
            estimated_clean_image_or_video=original_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            vace_context=vace_context,
        )

        dmd_log_dict = {}

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            alpha = 0.1
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - alpha * grad.double()).detach(), reduction="sum")
        return dmd_loss, dmd_log_dict


    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = BiTrainingVacePipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
        )

    def _run_generator(self,
                       image_or_video_shape,
                       conditional_dict: dict,
                       initial_latent: torch.tensor = None,
                       exit_flag=None,
                       vace_context: torch.Tensor = None,
                       ):
        noise_shape = image_or_video_shape.copy()
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            exit_flag=exit_flag,
            vace_context=vace_context,
            **conditional_dict,
        )
        return pred_image_or_video, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        exit_flag: list,
        vace_context: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, exit_flags=exit_flag, vace_context=vace_context, **conditional_dict
        )
    
    def renorm_to_std_2_shape(self, x, target_len=21):
        cur_len = x.shape[1]
        if cur_len < target_len:
            # 取最后一帧并重复补齐
            last = x[:, -1:, :, :, :]                    # shape [1,1,16,60,104]
            pad = last.repeat(1, target_len - cur_len, 1, 1, 1)
            x = torch.cat([x, pad], dim=1)
        elif cur_len > target_len:
            # 截取前 21 帧
            x = x[:, :target_len, :, :, :]
        return x

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        exit_flag = None,
        vace_context: torch.Tensor = None,
        step_now: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if step_now % 2 == 0: 
            conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].to(dtype=self.dtype)
            unconditional_dict["prompt_embeds"] = unconditional_dict["prompt_embeds"].to(dtype=self.dtype)

            pred_image, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                exit_flag=exit_flag,
                vace_context=vace_context,
            )

            pred_image_ = pred_image

            dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
                image_or_video=pred_image,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
                vace_context=vace_context,
            )

            print("generator dmd loss:", dmd_loss.item())

            otg_loss, dmd_log_dict = self.compute_ot_loss(
                image_or_video=pred_image_,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
                vace_context=vace_context,
            )

            print("generator ot loss:", otg_loss.item())
            
            return 0.2 * dmd_loss + 0.8 * otg_loss, dmd_log_dict

        else: 
            # =============== gan part =======================
            pred_image, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                exit_flag=exit_flag,
                vace_context=vace_context,
            )

            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.critic_timestep_shift > 1:
                critic_timestep = self.critic_timestep_shift * \
                    (critic_timestep / 1000) / (1 + (self.critic_timestep_shift - 1) * (critic_timestep / 1000)) * 1000

            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            # critic_timestep = self.scheduler.timesteps[critic_timestep]

            critic_noise = torch.randn_like(pred_image)
            noisy_fake_latent = self.scheduler.add_noise(
                pred_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            real_image_or_video = clean_latent.clone()
            real_image_or_video = self.renorm_to_std_2_shape(real_image_or_video)

            critic_noise = torch.randn_like(real_image_or_video)
            noisy_real_latent = self.scheduler.add_noise(
                real_image_or_video.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            conditional_dict["prompt_embeds"] = torch.concatenate(
                (conditional_dict["prompt_embeds"], conditional_dict["prompt_embeds"]), dim=0)
            critic_timestep = torch.concatenate((critic_timestep, critic_timestep), dim=0)

            noisy_latent = torch.concatenate((noisy_fake_latent, noisy_real_latent), dim=0)
            print('input:', noisy_latent.shape, len(conditional_dict['prompt_embeds']), critic_timestep.shape)
            _, _, noisy_logit = self.fake_score(
                noisy_image_or_video=noisy_latent,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                classify_mode=True,
                concat_time_embeddings=self.concat_time_embeddings,
                vace_context=vace_context,
            )
            noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)
            print('output:', noisy_real_logit.shape, noisy_fake_logit.shape)

            if not self.relativistic_discriminator:
                gan_G_loss = F.softplus(-noisy_fake_logit.float()).mean() * self.gan_g_weight
            else:
                relative_fake_logit = noisy_fake_logit - noisy_real_logit
                gan_G_loss = F.softplus(-relative_fake_logit.float()).mean() * self.gan_g_weight

            print("generator gan loss:", gan_G_loss.item())

            dmd_log_dict = {}

            return gan_G_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        real_image_or_video: torch.Tensor,
        initial_latent: torch.Tensor = None,
        exit_flag=None,
        vace_context: torch.Tensor = None,
        step_now: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        if step_now % 2 == 0: 
            # Step 1: Run generator on backward simulated noisy input
            with torch.no_grad():
                generated_image, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                    exit_flag=exit_flag,
                    vace_context=vace_context,
                )

            # Step 2: Compute the fake prediction
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_step
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.max_step
            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.timestep_shift > 1:
                critic_timestep = self.timestep_shift * \
                    (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            # critic_timestep = self.scheduler.timesteps[critic_timestep]
            print('critic:', critic_timestep, min_timestep, max_timestep)

            critic_noise = torch.randn_like(generated_image)
            noisy_generated_image = self.scheduler.add_noise(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            _, pred_fake_image = self.fake_score(
                noisy_image_or_video=noisy_generated_image,
                conditional_dict=conditional_dict,
                timestep=critic_timestep,
                vace_context=vace_context,
            )

            # Step 3: Compute the denoising loss for the fake critic
            if self.args.denoising_loss_type == "flow":
                from utils.wan_wrapper import WanDiffusionWrapper
                flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1)
                )
                pred_fake_noise = None
            else:
                flow_pred = None
                pred_fake_noise = self.scheduler.convert_x0_to_noise(
                    x0=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1)
                ).unflatten(0, image_or_video_shape[:2])

            denoising_loss = self.denoising_loss_func(
                x=generated_image.flatten(0, 1),
                x_pred=pred_fake_image.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=pred_fake_noise,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred
            )

            print('critic denoising loss:', denoising_loss.item())
            # Step 5: Debugging Log
            critic_log_dict = {
                "denoising_loss": denoising_loss.item(),
                "critic_timestep": critic_timestep.detach()
            }

            gan_D_loss = 0
            r1_loss = 0
            r2_loss = 0

            return (denoising_loss, gan_D_loss, r1_loss, r2_loss), critic_log_dict

        else: 
            # =============== gan part =====================
            # Step 1: Run generator on backward simulated noisy input
            with torch.no_grad():
                generated_image, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                    exit_flag=exit_flag,
                    vace_context=vace_context,
                )

            # Step 2: Get timestep and add noise to generated/real latents
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True
            )

            if self.critic_timestep_shift > 1:
                critic_timestep = self.critic_timestep_shift * \
                    (critic_timestep / 1000) / (1 + (self.critic_timestep_shift - 1) * (critic_timestep / 1000)) * 1000

            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            #critic_timestep = self.scheduler.timesteps[critic_timestep]

            real_image_or_video = self.renorm_to_std_2_shape(real_image_or_video)

            critic_noise = torch.randn_like(generated_image)
            noisy_fake_latent = self.scheduler.add_noise(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            # Step 4: Compute the real GAN discriminator loss
            noisy_real_latent = self.scheduler.add_noise(
                real_image_or_video.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

            conditional_dict_cloned = copy.deepcopy(conditional_dict)
            conditional_dict_cloned["prompt_embeds"] = torch.concatenate(
                (conditional_dict_cloned["prompt_embeds"], conditional_dict_cloned["prompt_embeds"]), dim=0)

            concat_noisy_video = torch.concatenate((noisy_fake_latent, noisy_real_latent), dim=0)

            _, _, noisy_logit = self.fake_score(
                noisy_image_or_video=concat_noisy_video,
                conditional_dict=conditional_dict_cloned,
                timestep=torch.concatenate((critic_timestep, critic_timestep), dim=0),
                classify_mode=True,
                concat_time_embeddings=self.concat_time_embeddings,
                vace_context=vace_context,
            )
            noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)

            if not self.relativistic_discriminator:
                gan_D_loss = F.softplus(-noisy_real_logit.float()).mean() + F.softplus(noisy_fake_logit.float()).mean()
            else:
                relative_real_logit = noisy_real_logit - noisy_fake_logit
                gan_D_loss = F.softplus(-relative_real_logit.float()).mean()
            gan_D_loss = gan_D_loss * self.gan_d_weight

            # R1 regularization
            if self.r1_weight > 0.:
                # with torch.no_grad():
                noisy_real_latent_perturbed = noisy_real_latent.clone()
                epison_real = self.r1_sigma * torch.randn_like(noisy_real_latent_perturbed)
                noisy_real_latent_perturbed = noisy_real_latent_perturbed + epison_real
                noisy_real_logit_perturbed = self._run_cls_pred_branch(
                    noisy_image_or_video=noisy_real_latent_perturbed,
                    conditional_dict=conditional_dict,
                    timestep=critic_timestep,
                    vace_context=vace_context,
                )

                r1_grad = (noisy_real_logit_perturbed - noisy_real_logit)
                r1_loss = self.r1_weight * torch.mean((r1_grad)**2)
            else:
                r1_loss = torch.zeros_like(gan_D_loss)

            # R2 regularization
            if self.r2_weight > 0.:
                # with torch.no_grad():
                noisy_fake_latent_perturbed = noisy_fake_latent.clone()
                epison_generated = self.r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
                noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epison_generated
                noisy_fake_logit_perturbed = self._run_cls_pred_branch(
                    noisy_image_or_video=noisy_fake_latent_perturbed,
                    conditional_dict=conditional_dict,
                    timestep=critic_timestep,
                    vace_context=vace_context,
                )

                r2_grad = (noisy_fake_logit_perturbed - noisy_fake_logit)
                r2_loss = self.r2_weight * torch.mean((r2_grad)**2)
            else:
                r2_loss = torch.zeros_like(gan_D_loss)
            
            print('critic gan loss:', gan_D_loss.item())
            print('critic r1 loss:', r1_loss.item())
            print('critic r2 loss:', r2_loss.item())

            denoising_loss = 0

            critic_log_dict = {}

            critic_log_dict.update({
                "critic_timestep": critic_timestep.detach(),
                'noisy_real_logit': noisy_real_logit.detach(),
                'noisy_fake_logit': noisy_fake_logit.detach(),
            })

            return (denoising_loss, gan_D_loss, r1_loss, r2_loss), critic_log_dict


    def update_step(self, global_step: int):
        min_step_percent = C(self.min_step_percent, global_step)
        max_step_percent = C(self.max_step_percent, global_step)
        self.min_step = int(self.num_train_timestep * min_step_percent)
        self.max_step = int(self.num_train_timestep * max_step_percent)
        print('update timestep:', global_step, self.min_step, self.max_step)

def C(value, global_step) -> float:
    start_step, start_value, end_value, end_step = value
    if global_step < start_step:
        return start_value
    current_step = global_step
    value = start_value + (end_value - start_value) * max(
        min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
    )
    return value