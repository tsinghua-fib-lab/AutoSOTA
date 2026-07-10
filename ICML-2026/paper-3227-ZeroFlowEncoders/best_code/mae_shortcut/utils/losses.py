from utils.shuffle import patchify


def mae_loss(imgs, pred, mask, patch_size, normalize=True):
            """
            Patch-level MSE loss
            imgs: (B, C, H, W)
            pred: (B, N, patch_dim)
            mask: (B, n_patches)          1 = masked, 0 = visible
            """
            target = patchify(imgs, patch_size)      # (B, n_patches, patch_dim)
            if normalize:
                mean = target.mean(dim=-1, keepdim=True)
                var  = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**0.5

            loss = (pred - target) ** 2  # (B, N, patch_dim)
            loss = loss.mean(dim=-1)                 # per patch
            loss = (loss * mask).sum() / mask.sum()  # only masked patches
            return loss