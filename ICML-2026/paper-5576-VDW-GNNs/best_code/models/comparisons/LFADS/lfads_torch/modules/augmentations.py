import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from ..tuples import SessionBatch


def pad_mask(mask, data, value):
    """Adds padding to I/O masks for CD and SV in cases where
    reconstructed data is not the same shape as the input data.
    """
    t_forward = data.shape[1] - mask.shape[1]
    n_heldout = data.shape[2] - mask.shape[2]
    pad_shape = (0, n_heldout, 0, t_forward)
    return F.pad(mask, pad_shape, value=value)


class AugmentationStack:
    def __init__(self, transforms=[], batch_order=[], loss_order=[]):
        # Build lists of input and output transformations to apply
        self.batch_transforms = [transforms[i] for i in batch_order]
        self.loss_transforms = [transforms[i] for i in loss_order]
        # Check that the transformations have the correct functions defined
        assert all([hasattr(t, "process_batch") for t in self.batch_transforms])
        assert all([hasattr(t, "process_losses") for t in self.loss_transforms])

    def process_batch(self, batch):
        for transform in self.batch_transforms:
            batch = transform.process_batch(batch)
        return SessionBatch(*batch)

    def process_losses(self, losses, batch, log_fn, data_split):
        for transform in self.loss_transforms:
            losses = transform.process_losses(losses, batch, log_fn, data_split)
        return losses

    def reset(self):
        for transform in {*self.batch_transforms, *self.loss_transforms}:
            if hasattr(transform, "reset"):
                transform.reset()


class SpikeJitter:
    def __init__(self, width=2):
        self.width = width

    def process_batch(self, batch):
        encod_data, recon_data, *other_data = batch
        encod_data = self._jitter_tensor(encod_data)
        recon_data = self._jitter_tensor(recon_data)
        return encod_data, recon_data, *other_data

    def _jitter_tensor(self, data):
        max_spike_ct = int(data.max())
        # Get indices of all spikes
        all_indices = torch.cat(
            [(data >= count).nonzero() for count in range(1, max_spike_ct + 1)],
        )
        # Sample shifts to add to the spikes
        shifts = torch.randint(
            low=-self.width,
            high=self.width + 1,
            size=(len(all_indices),),
            device=data.device,
        )
        # Split up the indices to shift only the time index
        b_i, t_i, n_i = torch.unbind(all_indices, dim=1)
        t_i += shifts
        # Reflect on the boundaries so we don't lose spikes
        B, T, N = data.shape
        t_i = torch.abs(t_i)
        oob_ixs = torch.nonzero(t_i > T - 1)
        t_i[oob_ixs] = 2 * (T - 1) - t_i[oob_ixs]
        # Recreate all indices and count the unique ones
        all_indices = torch.stack([b_i, t_i, n_i], 1)
        unique_indices, counts = torch.unique(all_indices, return_counts=True, dim=0)
        # Fill in a tensor of jittered data
        b_i, t_i, n_i = torch.unbind(unique_indices, dim=1)
        jittered_data = torch.zeros(B, T, N, device=data.device)
        jittered_data[b_i, t_i, n_i] = counts.float()
        return jittered_data

    def process_losses(self, recon_loss, *args):
        return recon_loss


class TemporalShift:
    def __init__(self, std=3.0, max_shift=6):
        self.std = std
        self.max = max_shift

    def process_batch(self, batch):
        encod_data, recon_data, *other_data = batch
        encod_data = self._shift_tensor(encod_data)
        recon_data = self._shift_tensor(recon_data)
        return encod_data, recon_data, *other_data

    def _shift_tensor(self, data):
        batch_size, n_steps, n_channels = data.shape
        # Sample a shift for each channel and each sample
        mean = torch.zeros(batch_size, 1, n_channels)
        shifts = torch.normal(mean, self.std).to(data.device)
        # Clamp the shifts within the range
        shifts = torch.clamp(shifts.round().long(), -self.max, self.max)
        # Create indices and shift them based on the sampled values
        indices = torch.arange(n_steps, device=data.device)[None, :, None]
        indices = indices.repeat(batch_size, 1, n_channels)
        # Repeat end values on the edges
        shifted_indices = torch.clamp(indices - shifts, 0, n_steps - 1)
        # Apply the shifts to the data
        shifted_data = torch.gather(data, dim=1, index=shifted_indices)
        return shifted_data

    def process_losses(self, recon_loss, *args):
        return recon_loss


class CoordinatedDropout:
    def __init__(self, cd_rate, cd_pass_rate, ic_enc_seq_len):
        self.cd_rate = cd_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        self.cd_pass_dist = Bernoulli(cd_pass_rate)
        # Use FIFO for grad masks
        self.grad_masks = []

    def process_batch(self, batch):
        encod_data, *other_data = batch
        # Only use CD where we are inferring rates (none inferred for IC segment)
        unmaskable_data = encod_data[:, : self.ic_enc_seq_len, :]
        maskable_data = encod_data[:, self.ic_enc_seq_len :, :]
        # Sample a new CD mask at each training step
        device = encod_data.device
        cd_mask = self.cd_input_dist.sample(maskable_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(maskable_data.shape).to(device)
        # Save the gradient mask for `process_outputs`
        if self.cd_rate > 0:
            grad_mask = torch.logical_or(torch.logical_not(cd_mask), pass_mask).float()
        else:
            # If cd_rate == 0, turn off CD
            grad_mask = torch.ones_like(cd_mask)
        # Store the grad_mask for later
        self.grad_masks.append(grad_mask)
        # Mask and scale post-CD input so it has the same sum as the original data
        cd_masked_data = maskable_data * cd_mask / (1 - self.cd_rate)
        # Concatenate the data from the IC encoder segment if using
        cd_input = torch.cat([unmaskable_data, cd_masked_data], axis=1)

        return cd_input, *other_data

    def process_losses(self, recon_loss, *args):
        # First-in-first-out
        grad_mask = self.grad_masks.pop(0)
        # Expand mask, but don't block gradients
        grad_mask = pad_mask(grad_mask, recon_loss, 1.0)
        # Block gradients with respect to the masked outputs
        grad_loss = recon_loss * grad_mask
        nograd_loss = (recon_loss * (1 - grad_mask)).detach()
        cd_loss = grad_loss + nograd_loss

        return cd_loss

    def reset(self):
        self.grad_masks = []


class CoordinatedDropoutTF2:
    def __init__(self, cd_rate, cd_pass_rate, ic_enc_seq_len):
        self.cd_rate = cd_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        self.cd_pass_dist = Bernoulli(cd_pass_rate)
        # Use FIFO for grad masks
        self.grad_masks = []

    def process_batch(self, batch):
        encod_data, recon_data, *other_data = batch
        # Only use CD where we are inferring rates (none inferred for IC segment)
        unmaskable_data = encod_data[:, : self.ic_enc_seq_len, :]
        maskable_data = encod_data[:, self.ic_enc_seq_len :, :]
        # Sample a new CD mask at each training step
        device = encod_data.device
        cd_mask = self.cd_input_dist.sample(recon_data.shape).to(device)
        pass_mask = self.cd_pass_dist.sample(recon_data.shape).to(device)
        # Save the gradient mask for `process_outputs`
        if self.cd_rate > 0:
            grad_mask = torch.logical_or(torch.logical_not(cd_mask), pass_mask).float()
        else:
            # If cd_rate == 0, turn off CD
            grad_mask = torch.ones_like(cd_mask)
        # Store the grad_mask for later
        self.grad_masks.append(grad_mask)
        # Mask and scale post-CD input so it has the same sum as the original data
        _, esl, edd = encod_data.shape
        cd_masked_data = maskable_data * cd_mask[:, :esl, :edd] / (1 - self.cd_rate)
        # Concatenate the data from the IC encoder segment if using
        cd_input = torch.cat([unmaskable_data, cd_masked_data], axis=1)

        return cd_input, recon_data, *other_data

    def process_losses(self, recon_loss, *args):
        # First-in-first-out
        grad_mask = self.grad_masks.pop(0)
        # Block gradients with respect to the masked outputs
        grad_loss = recon_loss * grad_mask
        nograd_loss = (recon_loss * (1 - grad_mask)).detach()
        cd_loss = grad_loss + nograd_loss

        return cd_loss

    def reset(self):
        self.grad_masks = []


class SampleValidation:
    def __init__(self, sv_rate, ic_enc_seq_len, recon_reduce_mean):
        self.sv_rate = sv_rate
        self.ic_enc_seq_len = ic_enc_seq_len
        self.recon_reduce_mean = recon_reduce_mean

    def process_batch(self, batch):
        encod_data, *other_data = batch
        sv_mask = batch[2]
        # Only use SV where we are inferring rates (none inferred for IC segment)
        unmaskable_data = encod_data[:, : self.ic_enc_seq_len, :]
        maskable_data = encod_data[:, self.ic_enc_seq_len :, :]
        # Set heldout data to zero and scale up heldin data
        sv_masked_data = maskable_data * sv_mask / (1 - self.sv_rate)
        # Concatenate the data from the IC encoder segment if using
        sv_input = torch.cat([unmaskable_data, sv_masked_data], axis=1)

        return sv_input, *other_data

    def process_losses(self, recon_loss, batch, log_fn, data_split):
        sv_mask = batch[2]
        # Aggregate and log recon cost for samples heldout for SV
        if self.sv_rate == 0:
            # Skip the masking if SV is not being used
            recon_heldin = recon_loss
            recon_heldout_agg = torch.tensor(float("nan"))
        else:
            # Rescale so means are comparable
            heldin_mask = sv_mask / (1 - self.sv_rate)
            heldout_mask = (1 - sv_mask) / self.sv_rate
            # Apply the heldin mask - expand the mask as necessary
            heldin_mask = pad_mask(heldin_mask, recon_loss, value=1.0)
            recon_heldin = recon_loss * heldin_mask
            # Apply the heldout mask - only include points with encoder input
            _, t_enc, n_enc = heldout_mask.shape
            encod_recon_loss = recon_loss[:, :t_enc, :n_enc]
            recon_heldout_masked = encod_recon_loss * heldout_mask
            # Aggregate the heldout cost for logging
            if not self.recon_reduce_mean:
                recon_heldout_masked = torch.sum(recon_heldout_masked, dim=(1, 2))
            recon_heldout_agg = torch.mean(recon_heldout_masked)
        # Log the heldout reconstruction cost
        log_fn(f"{data_split}/recon/sv", recon_heldout_agg)

        return recon_heldin


class SelectiveBackpropThruTime:
    def __init__(self):
        # Use FIFO for NaN masks
        self.isnan_masks = []

    def process_batch(self, batch):
        encod_data, recon_data, *other_data = batch
        # Remember where NaNs exist in the recon data
        self.isnan_masks.append(torch.isnan(recon_data))
        # Replace missing encod data with zeros
        encod_data_interp = torch.nan_to_num(encod_data, nan=0)
        # Temporarily replace missing recon data with tens
        # (zero won't work for ZIG and ones are used in place of zeros)
        recon_data_interp = torch.nan_to_num(recon_data, nan=10)
        return encod_data_interp, recon_data_interp, *other_data

    def process_losses(self, recon_loss, *args):
        # First-in-first-out
        isnan_mask = self.isnan_masks.pop(0)
        # Convert missing losses into zeros and scale up so mean is unchanged
        frac_isnan = isnan_mask.sum() / isnan_mask.numel()
        recon_loss[isnan_mask] = 0
        return recon_loss / (1 - frac_isnan)

    def reset(self):
        self.isnan_masks = []
