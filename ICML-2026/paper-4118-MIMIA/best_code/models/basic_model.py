import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = 'concat'
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif  'cremad' in args.dataset :
            n_classes = 6
        elif  'AVE' in args.dataset :
            n_classes = 28
        elif args.dataset == 'balance':
            n_classes = 30
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual, modality_to_train='full'):

        if modality_to_train == 'audio':
            # 只训练audio时，在无梯度的上下文中运行visual_net
            # 这会阻止Opacus为visual_net保存任何输入拷贝
            with torch.no_grad():
                v = self.visual_net(visual)
            a = self.audio_net(audio)

        elif modality_to_train == 'visual':
            # 只训练visual时，在无梯度的上下文中运行audio_net
            with torch.no_grad():
                a = self.audio_net(audio)
            v = self.visual_net(visual)

        else:  # 'full' 模式或未指定模式
            # 正常执行，两个分支都计算梯度
            a = self.audio_net(audio)
            v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4).clone()  # <-- 在这里添加 .clone()

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out
