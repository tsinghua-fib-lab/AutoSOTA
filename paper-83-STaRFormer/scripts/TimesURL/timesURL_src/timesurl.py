import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from timesURL_src.models import TSEncoder
from timesURL_src.models.losses import hierarchical_contrastive_loss
from timesURL_src.utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from timesURL_src.utils import inter_cubic_sp_torch
from timesURL_src.utils import convert_coeff
from timesURL_src.lib import get_unlabeled_pretrain_data
from tqdm.auto import tqdm


def tp_noneffect(func, x, **kwargs):
    tp = x[..., -1:]
    x = func(x[..., :-1], **kwargs)
    return torch.cat([x, tp], dim=-1)


def freq_mix(x, rate=0.5, dim=1):
    x_f = torch.fft.fft(x, dim=dim)

    m = torch.cuda.FloatTensor(x_f.shape).uniform_() < rate
    amp = abs(x_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal = x_f.real.masked_fill(m, 0)
    fimag = x_f.imag.masked_fill(m, 0)

    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2 = x[b_idx]
    x2_f = torch.fft.fft(x2, dim=dim)

    m = torch.bitwise_not(m)
    freal2 = x2_f.real.masked_fill(m, 0)
    fimag2 = x2_f.imag.masked_fill(m, 0)

    freal += freal2
    fimag += fimag2

    x_f = torch.complex(freal, fimag)

    x = torch.abs(torch.fft.ifft(x_f, dim=dim))
    return x


def freq_dropout(x, dropout_rate=0.5):
    x_aug = x.clone()
    x_aug_f = torch.fft.fft(x_aug)
    m = torch.cuda.FloatTensor(x_aug_f.shape).uniform_() < dropout_rate
    amp = torch.abs(x_aug_f)
    _, index = amp.sort(dim=1, descending=True)
    dominant_mask = index > 5
    m = torch.bitwise_and(m, dominant_mask)
    freal = x_aug_f.real.masked_fill(m, 0)
    fimag = x_aug_f.imag.masked_fill(m, 0)
    x_aug_f = torch.complex(freal, fimag)
    x_aug = torch.abs(torch.fft.ifft(x_aug_f, dim=1))
    return x_aug


class TimesURL:
    '''The TimesURL model'''

    def __init__(
            self,
            input_dims,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=0.001,
            batch_size=16,
            sgd=False,
            max_train_length=None,
            temporal_unit=0,
            after_iter_callback=None,
            after_epoch_callback=None,
            args=None
    ):
        ''' Initialize a TimesURL model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''

        super().__init__()
        self.device = device
        self.lr = lr
        self.sgd = sgd
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.args = args

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False, is_scheduler=True, temp=1.0):
        ''' Training the TimesURL model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        train_data, mask = train_data['x'], train_data['mask']
        print(train_data.shape, mask.shape)

        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters

        if self.lr <= 1e-5 and n_iters is not None:
            n_iters *= 1.2

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
                mask = np.concatenate(split_with_nan(mask, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data, mask = centerize_vary_length_series(train_data, mask)

        mask = mask[~np.isnan(train_data[..., :-1]).all(axis=2).all(axis=1)]
        train_data = train_data[~np.isnan(train_data[..., :-1]).all(axis=2).all(axis=1)]
        mask[np.isnan(mask)] = 0
        x, t = train_data[..., :-1], train_data[..., -1:]
        obj = get_unlabeled_pretrain_data(np.concatenate([x, mask, t], axis=-1), self.args)
        train_loader = obj['train_dataloader']

        if self.sgd:
            optimizer = torch.optim.SGD(self._net.parameters(), lr=self.lr, weight_decay=5e-4, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=5e-4)
        if is_scheduler:
            if n_iters is not None and n_epochs is None:
                max_epochs = n_iters // len(train_loader)
            else:
                max_epochs = n_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

        loss_log = []
        pbar = tqdm(total=n_epochs)
        print(f'training on {self.device}')
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in tqdm(train_loader, desc=f'Epoch [{self.n_epochs} / {n_epochs}]', leave=False):
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                value = batch['value'].to(self.device)
                time = batch['time'].to(self.device)
                mask = batch['mask'].to(self.device)
                mask_origin = batch['mask_origin'].to(self.device)

                optimizer.zero_grad()

                loss = torch.tensor([0.]).to(self.device)
                for seq in range(value.size(1)):
                    x, t, m, m_old = value[:, seq], time[:, seq], mask[:, seq], mask_origin[:, seq]
                    dim = x.size(-1)
                    x = torch.cat([x, t.unsqueeze(2)], dim=-1)

                    ts_l = x.size(1)
                    crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                    crop_left = np.random.randint(ts_l - crop_l + 1)
                    crop_right = crop_left + crop_l
                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                    x_left = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                    x_right = tp_noneffect(freq_mix, take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), rate=0.5)

                    mask1 = take_per_row(m[..., :dim], crop_offset + crop_eleft, crop_right - crop_eleft)
                    mask2 = take_per_row(m[..., :dim], crop_offset + crop_left, crop_eright - crop_left)

                    mask1_inter = take_per_row(m[..., dim:], crop_offset + crop_eleft, crop_right - crop_eleft)
                    mask2_inter = take_per_row(m[..., dim:], crop_offset + crop_left, crop_eright - crop_left)

                    mask1_origin = take_per_row(m_old, crop_offset + crop_eleft, crop_right - crop_eleft)
                    mask2_origin = take_per_row(m_old, crop_offset + crop_left, crop_eright - crop_left)

                    out1, left_recon = self._net({'data': x_left, 'mask': mask1, 'mask_inter': mask1_inter, 'mask_origin': mask1_origin})
                    out2, right_recon = self._net({'data': x_right, 'mask': mask2, 'mask_inter': mask2_inter, 'mask_origin': mask2_origin})

                    out1, left_recon = out1[:, -crop_l:], left_recon[:, -crop_l:]
                    out2, right_recon = out2[:, :crop_l], right_recon[:, :crop_l]

                    x_left, x_right = x_left[:, -crop_l:], x_right[:, :crop_l]

                    mask1, mask2 = mask1[:, -crop_l:], mask2[:, :crop_l]
                    mask1_inter, mask2_inter = mask1_inter[:, -crop_l:], mask2_inter[:, :crop_l]

                    loss += self.args.lmd * hierarchical_contrastive_loss(
                        out1,
                        out2,
                        temporal_unit=self.temporal_unit,
                        temp=temp
                    )

                    if torch.sum(mask1_inter) > 0:
                        loss += 1 * torch.sum(torch.pow((x_left[..., :-1] - left_recon) * mask1_inter, 2)) / (
                                torch.sum(mask1_inter) + 1e-10) / 2
                    if torch.sum(mask2_inter) > 0:
                        loss += 1 * torch.sum(torch.pow((x_right[..., :-1] - right_recon) * mask2_inter, 2)) / (
                                torch.sum(mask2_inter) + 1e-10) / 2

                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            cum_loss /= n_epoch_iters if n_epoch_iters else 1
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            # Update progress bar
            pbar.update(1)

            if is_scheduler:
                scheduler.step()

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if interrupted:
                break   
        # end
        pbar.close()

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0,
               batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert isinstance(data, dict) or data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape if not isinstance(data, dict) else data['x'].shape

        org_training = self.net.training
        self.net.eval()

        if isinstance(data, dict):
            data = np.concatenate((data['x'], data['mask']), axis=-1)
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
