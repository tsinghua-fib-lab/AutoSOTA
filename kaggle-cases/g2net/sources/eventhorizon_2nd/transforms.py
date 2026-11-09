''' Albumentations like interface
Thank you, Arai-san nanoda!
https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
'''
from __future__ import annotations
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.fft import fft, rfft, ifft
from torchaudio.functional import bandpass_biquad
import colorednoise as cn
import librosa
import scipy
import pywt


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray | torch.Tensor):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray | torch.Tensor):
        raise NotImplementedError

    def __repr__(self):
        attrs = [item for item in dir(self)]
        repr_text = f'{self.__class__.__name__}('
        for attr in attrs:
            if attr[:1] == '_':
                continue
            elif attr in ['apply']:
                continue
            else:
                repr_text += f'{attr}={getattr(self, attr)}, '
        else:
            repr_text += ')'
        return repr_text


class AudioTransformPerChannel(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
    
    def __call__(self, y: np.ndarray | torch.Tensor):
        ch = y.shape[0]
        if isinstance(y, np.ndarray):
            augmented = y.copy()
        else:
            augmented = y.clone()
        for i in range(ch):
            if self.always_apply:
                augmented[i] = self.apply(y[i])
            else:
                if np.random.rand() < self.p:
                    augmented[i] = self.apply(y[i])
        return augmented


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y
    
    def __repr__(self):
        repr_text = 'Compose([\n'
        for trns in self.transforms:
            repr_text += f'{trns.__repr__()},\n'
        else:
            repr_text +='])'
        return repr_text


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)

    def __repr__(self):
        repr_text = 'OneOf([\n'
        for trns in self.transforms:
            repr_text += f'{trns.__repr__()},\n'
        else:
            repr_text +='])'
        return repr_text


'''Audio data augmentations
'''
def add_noise_snr(signal: np.ndarray, noise_shape: np.ndarray, snr: int):
    a_signal = np.sqrt(signal ** 2).max()
    a_noise = a_signal / (10 ** (snr / 20))
    a_white = np.sqrt(noise_shape ** 2).max()
    return (signal + noise_shape * 1 / a_white * a_noise).astype(signal.dtype)


def add_noise_snr_torch(signal: torch.Tensor, noise_shape: torch.Tensor, snr: int):
    a_signal = torch.sqrt(signal ** 2).max()
    a_noise = a_signal / (10 ** (snr / 20))
    a_white = torch.sqrt(noise_shape ** 2).max()
    return (signal + noise_shape * 1 / a_white * a_noise)


def change_volume(signal: np.ndarray, db: int, mode: str = 'uniform'):
    if mode == "uniform":
        db_translated = 10 ** (db / 20)
    elif mode == "fade":
        lin = np.arange(len(signal))[::-1] / (len(signal) - 1)
        db_translated = 10 ** (db * lin / 20)
    elif mode == "cosine":
        cosine = np.cos(np.arange(len(signal)) / len(signal) * np.pi * 2)
        db_translated = 10 ** (db * cosine / 20)
    else:
        sine = np.sin(np.arange(len(signal)) / len(signal) * np.pi * 2)
        db_translated = 10 ** (db * sine / 20)
    return signal * db_translated


class Normalize(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 factors=[1.0, 1.0, 1.0]):
        super().__init__(always_apply, p)
        assert len(factors) == 3
        self.factors = factors

    def apply(self, y: np.ndarray | torch.Tensor):
        for i in range(3):
            y[i] = y[i] / self.factors[i]
        return y

    
class Normalize2(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 mode='max'):
        super().__init__(always_apply, p)
        assert mode in ['max', 'min']
        self.mode = mode

    def apply(self, y: np.ndarray | torch.Tensor):
        if self.mode == 'max':
            y = y / y.max()
        elif self.mode == 'mean':
            pos_mean = y[y > 0].mean()
            y = y / pos_mean
        return y


class MinMaxScaler(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray):
        for i in range(3):
            y[i] = y[i] / np.max(np.abs(y[i]))
        return y


class WhitenTorch(AudioTransformPerChannel):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5,
                 signal_len=4096):
        super().__init__(always_apply, p)
        self.hann = torch.hann_window(signal_len, periodic=True, dtype=torch.float64)

    def apply(self, y: torch.Tensor):
        spec = fft(y*self.hann)
        mag = torch.sqrt(torch.real(spec*torch.conj(spec))) 
        return torch.real(ifft(spec/mag)).float() * np.sqrt(len(y)/2)


class GaussianNoiseSNR(AudioTransformPerChannel):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        white_noise = np.random.randn(len(y))
        return add_noise_snr(y, white_noise, snr)


class GaussianNoiseSNRTorch(AudioTransformPerChannel):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: torch.Tensor):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        white_noise = torch.randn(len(y))
        return add_noise_snr_torch(y, white_noise, snr)


class PinkNoiseSNR(AudioTransformPerChannel):
    '''
    Pink noise: exponent = 1
    Brown noise: exponent = 2
    '''
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, exponent=1):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.exponent = exponent

    def apply(self, y: np.ndarray):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        pink_noise = cn.powerlaw_psd_gaussian(self.exponent, len(y))
        return add_noise_snr(y, pink_noise, snr)


class AddNoiseSNR2(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5, 
                 target_channel=[0, 1, 2],
                 random_channel=0,
                 noise_type='gaussian', 
                 min_snr=5.0, 
                 max_snr=20.0):
        super().__init__(always_apply, p)
        self.target_channel = np.array(target_channel)
        self.random_channel = random_channel
        self.min_snr = min_snr
        self.max_snr = max_snr
        assert noise_type in ['gaussian', 'pink', 'brown']
        self.noise_type = noise_type

    def apply(self, y: np.ndarray,):
        augmented = y.copy()
        if self.random_channel > 0:
            noise_chans = np.random.choice(self.target_channel, self.random_channel, replace=False)
        else:
            noise_chans = self.target_channel
        _, l = y.shape
        if self.noise_type == 'gaussian':
            noise_shape = np.random.randn(l)
        elif self.noise_type == 'pink':
            noise_shape = cn.powerlaw_psd_gaussian(1, l)
        elif self.noise_type == 'brown':
            noise_shape = cn.powerlaw_psd_gaussian(2, l)
        for i in noise_chans:
            snr = np.random.uniform(self.min_snr, self.max_snr)
            augmented[i] = add_noise_snr(y[i], noise_shape, snr)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5, 
                 max_steps=5, 
                 sr=32000):
        super().__init__(always_apply, p)
        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray):
        ch = y.shape[0]
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        for i in range(ch):
            if n_steps == 0:
                continue
            y[i] = librosa.effects.pitch_shift(y[i], sr=self.sr, n_steps=n_steps)
        return y


class VolumeControl(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5, 
                 db_limit=10, 
                 mode="uniform"):
        super().__init__(always_apply, p)
        assert mode in ["uniform", "fade", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"
        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray):
        ch = y.shape[0]
        db = np.random.uniform(-self.db_limit, self.db_limit)
        for i in range(ch):
            y[i] = change_volume(y[i], db, self.mode)
        return y


class BandPass(AudioTransformPerChannel):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 lower=16,
                 upper=512,
                 sr=2048,
                 order=8,
                 ):
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper
        self.sr = sr
        self.order = order
        self._b, self._a = scipy.signal.butter(
            self.order, (self.lower, self.upper), btype='bandpass', fs=self.sr)
        
    def apply(self, y: np.ndarray):
        return scipy.signal.filtfilt(self._b, self._a, y)


class BandPass2(AudioTransform):
    '''
    Channel-wise band pass filter
    '''
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 bands=[[12, 500], [12, 500], [12, 500]], 
                 sr=2048,
                 order=8,
                 ):
        super().__init__(always_apply, p)
        self.sr = sr
        self.order = order
        self.bands = bands
        self._filters = []
        for lower, upper in self.bands:
            b, a = scipy.signal.butter(
                self.order, (lower, upper), btype='bandpass', fs=self.sr)
            self._filters.append([b, a])
        
    def apply(self, y: np.ndarray):
        for ch, (b, a) in enumerate(self._filters):
            y[ch] = scipy.signal.filtfilt(b, a, y[ch])
        return y


class BandPassTorch(AudioTransformPerChannel):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 lower=16,
                 upper=512,
                 sr=2048
                 ):
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper
        self.sr = sr
        
    def apply(self, y: torch.Tensor):
        return bandpass_biquad(y, 
                               self.sr, 
                               (self.lower + self.upper) / 2,
                               (self.upper - self.lower) / (self.upper + self.lower))


class DWTDenoise(AudioTransformPerChannel):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 wavelet='haar',
                 mode='per',
                 level=1
                 ):
        super().__init__(always_apply, p)
        self.wavelet = wavelet
        self.mode = mode
        self.level = level

    def _maddest(self, s):
        return np.mean(np.absolute(s - np.mean(s)))
        
    def apply(self, y: np.ndarray):
        coef = pywt.wavedec(y, self.wavelet, self.mode)
        sigma = (1/0.6745) * self._maddest(coef[-self.level])
        uthresh = sigma * np.sqrt(2*np.log(len(y)))
        coef[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coef[1:])
        return pywt.waverec(coef, self.wavelet, mode=self.mode)


class DropChannel(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 channels=[0]):
        super().__init__(always_apply, p)
        if not isinstance(channels, np.ndarray):
            self.channels = np.array(channels)
        else:
            self.channels = channels

    def apply(self, y: np.ndarray):
        y[self.channels] = 0.0
        return y


class SwapChannel(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5, 
                 channels=[0, 1]):
        super().__init__(always_apply, p)
        assert len(channels) == 2
        self.channels = channels

    def apply(self, y: np.ndarray):
        augmented = y.copy()
        augmented[self.channels[0]] = y[self.channels[1]]
        augmented[self.channels[1]] = y[self.channels[0]]
        return augmented


class ToTensor(AudioTransform):

    def __init__(self, 
                 always_apply=True, 
                 p=0.5, 
                 dtype=torch.float32):
        super().__init__(always_apply, p)
        self.dtype = dtype

    def apply(self, y: np.ndarray):
        return torch.tensor(y, dtype=self.dtype)


class GetDiff(AudioTransform):

    def __init__(self, 
                 always_apply=True, 
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray):
        augmented = y.copy()
        augmented[0] = y[1] - y[0]
        augmented[1] = y[2] - y[1]
        augmented[2] = y[0] - y[2]
        return augmented


class GlobalTimeShift(AudioTransform):

    def __init__(self, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray):
        shift = random.randint(0, y.shape[-1])
        augmented = np.roll(y, shift, axis=-1)
        return augmented


class IndependentTimeShift(AudioTransformPerChannel):

    def __init__(self, 
                 always_apply=False, 
                 frame_limit=(-20, 20), 
                 p=0.5):
        super().__init__(always_apply, p)
        self.frame_limit = frame_limit

    def apply(self, y: np.ndarray):
        shift = random.randint(*self.frame_limit)
        augmented = np.roll(y, shift)
        if shift > 0:
            augmented[:shift] = 0.0
        elif shift < 0:
            augmented[shift:] = 0.0
        return augmented


class AlignPhase(AudioTransform):

    def __init__(self, 
                 always_apply=True, 
                 shift_limit=40, 
                 p=0.5):
        super().__init__(always_apply, p)
        self.search_range = slice(4096-shift_limit, 4096+shift_limit)
        self.shift_limit = shift_limit

    def apply(self, y: np.ndarray):
        shift1 = scipy.signal.correlate(
            y[0], y[1], method='fft')[self.search_range].argmax() - self.shift_limit
        shift2 = scipy.signal.correlate(
            y[0], y[2], method='fft')[self.search_range].argmax() - self.shift_limit
        y[1] = np.roll(y[1], shift1)
        y[2] = np.roll(y[2], shift2)
        return y


class FlipWave(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray):
        return y * -1


class FlipWavePerChannel(AudioTransformPerChannel):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray):
        return y * -1

    
class Scale(AudioTransform):

    def __init__(self, 
                 always_apply=True, 
                 p=0.5,
                 scale=10):
        super().__init__(always_apply, p)
        self.scale = scale

    def apply(self, y: np.ndarray):
        return y * self.scale


'''Spectrogram augmentations
'''
class BatchFrequencyMask(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_bins=12, freq_limit=(0, 72), fill='zero'):
        super().__init__(always_apply, p)
        self.max_bins = max_bins
        self.freq_limit = freq_limit
        self.fill = fill
        assert self.fill in ['zero', 'mean', 'noise']

    def apply(self, y: torch.Tensor, **params):
        bs, ch, f, t = y.shape
        mask_bins = random.randint(1, self.max_bins)
        mask_freq = random.randint(
            self.freq_limit[0],
            min(f-self.max_bins, self.freq_limit[1]-self.max_bins))
        augmented = y.clone()
        if self.fill == 'zero':
            fill_color = 0.0
        elif self.fill == 'mean':
            fill_color = y.mean()
        elif self.fill == 'noise':
            raise NotImplementedError('noise fill is not implemented yet')
        augmented[:, :, mask_freq:mask_freq+mask_bins, :] = fill_color
        return augmented


InBatchFrequencyMask = BatchFrequencyMask


class BatchTimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, y: torch.Tensor, **params):
        bs, ch, f, t = y.shape
        shift_length = random.randint(1, t)
        augmented = y.clone()
        augmented[:, :, :, :t-shift_length] = y[:, :, :, shift_length:]
        augmented[:, :, :, t-shift_length:] = y[:, :, :, :shift_length]
        return augmented

    
class BatchTimeMask(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_width=64, fill='zero'):
        super().__init__(always_apply, p)
        self.max_width = max_width
        self.fill = fill
        assert self.fill in ['zero', 'mean', 'noise']

    def apply(self, y: torch.Tensor):
        bs, ch, f, t = y.shape
        mask_width = random.randint(1, self.max_width)
        start_time = random.randint(0, t)
        augmented = y.clone()
        if self.fill == 'zero':
            fill_color = 0.0
        elif self.fill == 'mean':
            fill_color = y.mean()
        elif self.fill == 'noise':
            raise NotImplementedError('noise fill is not implemented yet')
        augmented[:, :, :, start_time:start_time+mask_width] = fill_color
        return augmented


class RandomResizedCrop(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5,
                 size=(256, 256),
                 scale=(0.8, 1.0)):
        super().__init__(always_apply, p)
        self.size = size
        self.scale = scale
    
    def apply(self, y: torch.Tensor):
        area = self.size[0] * self.size[1]
        scale = random.uniform(*self.scale)
        ratio = self.size[1] / self.size[0]
        width = round(np.sqrt(area * scale / ratio))
        height = round(width * ratio)
        crop_x = random.randint(0, self.size[0] - width)
        crop_y = random.randint(0, self.size[1] - height)
        cropped = y[:, :, crop_x:crop_x+width, crop_y:crop_y+height]
        return F.interpolate(cropped, size=self.size, mode='bicubic')


class TimeResize(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5,
                 scale=(0.8, 1.0)):
        super().__init__(always_apply, p)
        self.scale = scale
    
    def apply(self, y: torch.Tensor):
        _, _, h, w = y.shape
        scale = random.uniform(*self.scale)
        crop_w = round(w*scale)
        if scale < 1.0:
            crop_x = random.randint(0, w - crop_w)
            cropped = y[:, :, :, crop_x:crop_x+crop_w]
        else:
            pad_w = crop_w - w
            cropped = F.pad(y, (0, pad_w), 'constant', 0)
        return F.interpolate(cropped, size=(h, w), mode='bicubic')


class HorizontalFlip(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply, p)
    
    def apply(self, y: torch.Tensor):
        return torch.flip(y, dims=(3,))


class VerticalFlip(AudioTransform):
    def __init__(self, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply, p)
    
    def apply(self, y: torch.Tensor):
        return torch.flip(y, dims=(2,))


class TimeTrim(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5,
                 trim_range=(0.25, 1.0), 
                 mode='trim'):
        super().__init__(always_apply, p)
        self.trim_range = trim_range
        assert mode in ['trim', 'mask']
        self.mode = mode
    
    def apply(self, y: torch.Tensor):
        start = int(y.shape[-1] * self.trim_range[0])
        end = int(y.shape[-1] * self.trim_range[1])
        if self.mode == 'trim':
            return y[:, :, :, start:end]
        elif self.mode == 'mask':
            y[:, :, :, :start] = 0.0
            y[:, :, :, end:] = 0.0
            return y


class NormalizeImage(AudioTransform):
    def __init__(self, 
                 always_apply=True, 
                 p=0.5,
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225)):
        super().__init__(always_apply, p)
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()
    
    def apply(self, y: torch.Tensor):
        self.mean = self.mean.to(y.device)
        self.std = self.std.to(y.device)
        return (y - self.mean[None, :, None, None]) / self.std[None, :, None, None]
