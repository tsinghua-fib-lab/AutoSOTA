import torch
import utils.util as util
from models.deblurring.image_deblur import ImageDeblur
from tqdm import tqdm


class JointDeblur(ImageDeblur):
    def __init__(self, opt):
        super(JointDeblur, self).__init__(opt)

    def deblur(self, y):
        """Deblur image
        Args:
            y: Blur image
        """
        y = util.img2tensor(y).unsqueeze(0).cuda()

        self.prepare_DIPs()
        self.reset_optimizers()

        warmup_k = torch.load(self.opt["warmup_k_path"]).cuda()
        self.warmup(y, warmup_k)

        # Input vector of DIPs is sampled from N(z, I)

        print("Deblurring")
        reg_noise_std = self.opt["reg_noise_std"]
        for step in tqdm(range(self.opt["num_iters"])):
            dip_zx_rand = self.dip_zx + reg_noise_std * torch.randn_like(self.dip_zx).cuda()
            dip_zk_rand = self.dip_zk + reg_noise_std * torch.randn_like(self.dip_zk).cuda()

            self.x_optimizer.zero_grad()
            self.k_optimizer.zero_grad()

            self.x_scheduler.step()
            self.k_scheduler.step()

            x = self.x_dip(dip_zx_rand)
            k = self.k_dip(dip_zk_rand)

            fake_y = self.kernel_wizard.adaptKernel(x, k)

            if step < self.opt["num_iters"] // 2:
                total_loss = 6e-1 * self.perceptual_loss(fake_y, y)
                total_loss += 1 - self.ssim_loss(fake_y, y)
                total_loss += 5e-5 * torch.norm(k)
                total_loss += 2e-2 * self.laplace_penalty(x)
            else:
                total_loss = self.perceptual_loss(fake_y, y)
                total_loss += 5e-2 * self.laplace_penalty(x)
                total_loss += 5e-4 * torch.norm(k)

            total_loss.backward()

            self.x_optimizer.step()
            self.k_optimizer.step()

            # debugging
            # if step % 100 == 0:
            #     print(torch.norm(k))
            #     print(f"{self.k_optimizer.param_groups[0]['lr']:.3e}")

        return util.tensor2img(x.detach())
