import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from modules import *

from kuma_utils.torch.modules import AdaptiveConcatPool2d, GeM, AdaptiveGeM
from kuma_utils.torch.utils import freeze_module


class MultiSpectrogram(nn.Module):

    def __init__(self, configs: tuple):
        super().__init__()
        self.spectrograms = nn.ModuleList()
        self.is_wavegram = []
        for spec, spec_params in configs:
            spectrogram = spec(**spec_params)
            is_trainable = spectrogram.__class__.__name__ in \
                ['WaveNetSpectrogram', 'CNNSpectrogram']
            self.is_wavegram.append(is_trainable)
            if is_trainable:
                freeze_module(spectrogram)
            self.spectrograms.append(spectrogram)
    
    def forward(self, s):
        bs, ch, w = s.shape
        specs = []
        for spectrogram, is_trainable in zip(self.spectrograms, self.is_wavegram):
            if is_trainable: # Use CNN as spectrogram
                specs.append(spectrogram(s))
            else:
                with torch.cuda.amp.autocast(enabled=False): # MelSpectrogram causes NaN with AMP
                    spec = spectrogram(s.view(bs * ch, w)) # spec: (batch size * wave channel, freq, time)
                _, f, t = spec.shape
                spec = spec.view(bs, ch, f, t)
                specs.append(spec)
        return torch.cat(specs, dim=1)


class SpectroCNN(nn.Module):

    def __init__(self, 
                 model_name='efficientnet_b7', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={}, 
                 custom_preprocess='none',
                 custom_classifier='none',
                 custom_attention='none', 
                 spectrogram=None,
                 spec_params={},
                 augmentations=None,
                 augmentations_test=None, 
                 resize_img=None,
                 upsample='nearest', 
                 mixup='mixup',
                 norm_spec=False,
                 return_spec=False):
        
        super().__init__()
        if isinstance(spectrogram, nn.Module): # deprecated
            self.spectrogram = spectrogram
        else:
            self.spectrogram = spectrogram(**spec_params)
        self.is_cnnspec = self.spectrogram.__class__.__name__ in [
            'WaveNetSpectrogram', 'CNNSpectrogram', 'MultiSpectrogram', 'ResNetSpectrogram']

        self.cnn = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes,
                                     **timm_params)
        self.mixup_mode = 'input'
        if custom_classifier != 'none' or custom_attention != 'none':
            model_type = self.cnn.__class__.__name__
            try:
                feature_dim = self.cnn.get_classifier().in_features
                self.cnn.reset_classifier(0, '')
            except:
                raise ValueError(f'Unsupported model type: {model_type}')

            if custom_preprocess == 'positional-add':
                preprocess = PositionalEncoding(feature_dim=resize_img, mode='add')
            elif custom_preprocess == 'positional-cat':
                preprocess = PositionalEncoding(feature_dim=resize_img, mode='concat')
            else:
                preprocess = nn.Identity()

            if custom_attention == 'triplet':
                attention = TripletAttention()
            elif custom_attention == 'mixup':
                attention = ManifoldMixup()
                self.mixup_mode = 'manifold'
            else:
                attention = nn.Identity()
            
            if custom_classifier == 'avg':
                global_pool = nn.AdaptiveAvgPool2d((1, 1))
            elif custom_classifier == 'max':
                global_pool = nn.AdaptiveMaxPool2d((1, 1))
            elif custom_classifier == 'concat':
                global_pool = AdaptiveConcatPool2d()
                feature_dim = feature_dim * 2
            elif custom_classifier == 'gem':
                global_pool = GeM(p=3, eps=1e-4)
            elif custom_classifier == 'positional':
                global_pool = nn.Sequential(
                    AdaptiveGeM((1, None), p=3),
                    PositionalEncoding(),
                    nn.AdaptiveAvgPool2d((1, 1)))
            elif custom_classifier == 'mixup':
                global_pool = nn.Sequential(
                    GeM(p=3, eps=1e-4),
                    ManifoldMixup())
                self.mixup_mode = 'manifold'
            else:
                raise ValueError(f'Unsupported classifier type: {custom_classifier}')
            
            if model_type[-11:] == 'Transformer' or model_type == 'XCiT': # transformer models
                global_pool = nn.Identity()

            self.cnn = nn.Sequential(
                preprocess, 
                self.cnn, 
                attention, 
                global_pool, 
                Flatten(),
                nn.Linear(feature_dim, 512), 
                nn.ReLU(inplace=True), 
                nn.Linear(512, num_classes)
            )
        self.norm_spec = norm_spec
        if self.norm_spec:
            self.norm = nn.BatchNorm2d(3)
        self.resize_img = resize_img
        if isinstance(self.resize_img, int):
            self.resize_img = (self.resize_img, self.resize_img)
        self.return_spec = return_spec
        self.augmentations = augmentations
        self.augmentations_test = augmentations_test
        self.upsample = upsample
        self.mixup = mixup
        assert self.mixup in ['mixup', 'cutmix']
    
    def feature_mode(self):
        self.cnn[-1] = nn.Identity()
        self.cnn[-2] = nn.Identity()

    def forward(self, s, lam=None, idx=None): # s: (batch size, wave channel, length of wave)
        bs, ch, w = s.shape
        if self.is_cnnspec: # Use CNN as spectrogram
            spec = self.spectrogram(s)
        else:
            s = s.view(bs * ch, w)
            with torch.cuda.amp.autocast(enabled=False): # MelSpectrogram causes NaN with AMP
                spec = self.spectrogram(s) # spec: (batch size * wave channel, freq, time)
            _, f, t = spec.shape
            spec = spec.view(bs, ch, f, t)
        if lam is not None: # in-batch mixup
            if self.mixup == 'mixup' and self.mixup_mode == 'input':
                spec, lam = mixup(spec, lam, idx)
            elif self.mixup == 'cutmix':
                spec, lam = cutmix(spec, lam, idx)
        
        if self.training and self.augmentations is not None:
            spec = self.augmentations(spec)
        if not self.training and self.augmentations_test is not None:
            spec = self.augmentations_test(spec)

        if self.resize_img is not None:
            spec = F.interpolate(spec, size=self.resize_img, mode=self.upsample)

        if self.norm_spec:
            spec = self.norm(spec)
        
        if self.mixup_mode == 'manifold':
            self.cnn[3][1].update(lam, idx)

        if self.return_spec and lam is not None:
            return self.cnn(spec), spec, lam
        elif self.return_spec:
            return self.cnn(spec), spec
        elif lam is not None:
            return self.cnn(spec), lam
        else:
            return self.cnn(spec)


class MultiInstanceSCNN(nn.Module):

    def __init__(self, 
                 model_name='efficientnet_b7', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={}, 
                 custom_preprocess='none',
                 custom_classifier='none',
                 custom_attention='none', 
                 spectrogram=None,
                 spec_params={},
                 augmentations=None,
                 augmentations_test=None, 
                 resize_img=None,
                 upsample='nearest',
                 n_patch=2,
                 return_spec=False):
        
        super().__init__()
        if isinstance(spectrogram, nn.Module): # deprecated
            self.spectrogram = spectrogram
        else:
            self.spectrogram = spectrogram(**spec_params)
        self.cnn = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes,
                                     **timm_params)
        model_type = self.cnn.__class__.__name__
        try:
            feature_dim = self.cnn.get_classifier().in_features
            self.cnn.reset_classifier(0, '')
        except:
            raise ValueError(f'Unsupported model type: {model_type}')

        if custom_preprocess == 'positional-add':
            preprocess = PositionalEncoding(feature_dim=resize_img, mode='add')
        elif custom_preprocess == 'positional-cat':
            preprocess = PositionalEncoding(feature_dim=resize_img, mode='concat')
        else:
            preprocess = nn.Identity()

        if custom_attention == 'triplet':
            attention = TripletAttention()
        else:
            attention = nn.Identity()
        
        if custom_classifier in ['avg', 'none']:
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            global_pool = GeM(p=3, eps=1e-4)
        elif custom_classifier == 'positional':
            global_pool = nn.Sequential(
                AdaptiveGeM((1, None), p=3),
                PositionalEncoding(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise ValueError(f'Unsupported classifier type: {custom_classifier}')
        
        if model_type[-11:] == 'Transformer' or model_type == 'XCiT': # transformer models
            global_pool = nn.Identity()

        self.cnn = nn.Sequential(
            preprocess, 
            self.cnn, 
            attention,
            global_pool, 
            Flatten())
        self.head = nn.Sequential(
            nn.Linear(feature_dim*n_patch, 512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes)
        )
        self.resize_img = resize_img
        if isinstance(self.resize_img, int):
            self.resize_img = (self.resize_img, self.resize_img)
        self.return_spec = return_spec
        self.augmentations = augmentations
        self.augmentations_test = augmentations_test
        self.upsample = upsample
        self.n_patch = n_patch

    @staticmethod
    def _make_patches(image: torch.Tensor, n_patch: int): # image: (bs, ch, w, h)
        _, _ , h, w = image.shape
        stride = (w - h) // (n_patch - 1)
        output = []
        for i in range(n_patch):
            output.append(image[:, :, :, stride*i:stride*i+h])
        return torch.stack(output, dim=1)

    def feature_mode(self):
        self.head[1] = nn.Identity()
        self.head[2] = nn.Identity()

    def forward(self, s):
        bs, ch, w = s.shape
        s = s.view(bs * ch, w)
        with torch.cuda.amp.autocast(enabled=False): # MelSpectrogram causes NaN with AMP
            spec = self.spectrogram(s) # spec: (batch size * wave channel, freq, time)
        _, f, t = spec.shape
        spec = spec.view(bs, ch, f, t)
        
        if self.training and self.augmentations is not None:
            spec = self.augmentations(spec)
        if not self.training and self.augmentations_test is not None:
            spec = self.augmentations_test(spec)

        if self.resize_img is not None:
            spec = F.interpolate(spec, size=self.resize_img, mode=self.upsample)

        if self.n_patch > 1:
            spec = self._make_patches(spec, self.n_patch) # spec: (bs, patch, ch, freq, time)
        else:
            spec = spec.unsqueeze(1)
        _, _, ch, f, t = spec.shape
        spec = spec.view(bs * self.n_patch, ch, f, t) # spec: (bs*patch, ch, freq, time)
        features = self.cnn(spec) # features: (bs*patch, feature)
        features = features.reshape(bs, -1)
        output = self.head(features)

        if self.return_spec:
            return output, spec
        else:
            return output
