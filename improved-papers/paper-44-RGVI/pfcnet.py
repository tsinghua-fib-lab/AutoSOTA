import torch
import torch.nn as nn
import torch.nn.functional as F


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.LeakyReLU(0.2))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GatedConv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.conv_g = Conv(*conv_args)
        self.conv_x = ConvRelu(*conv_args)

    def forward(self, x):
        g = torch.sigmoid(self.conv_g(x))
        x = g * self.conv_x(x)
        return x


# encoding module
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(GatedConv(4, 32, 3, 1, 1), GatedConv(32, 64, 3, 1, 1), GatedConv(64, 128, 3, 1, 1))
        self.layer2 = nn.Sequential(GatedConv(128, 128, 4, 2, 1), GatedConv(128, 128, 3, 1, 3, 3), GatedConv(128, 128, 3, 1, 1))
        self.layer3 = nn.Sequential(GatedConv(128, 128, 4, 2, 1), GatedConv(128, 128, 3, 1, 3, 3), GatedConv(128, 128, 3, 1, 1))
        self.layer4 = nn.Sequential(GatedConv(128, 128, 4, 2, 1), GatedConv(128, 128, 3, 1, 3, 3), GatedConv(128, 128, 3, 1, 1))
        self.layer5 = nn.Sequential(GatedConv(128, 128, 4, 2, 1), GatedConv(128, 128, 3, 1, 3, 3), GatedConv(128, 128, 3, 1, 1))

    def forward(self, in_img):
        x = self.layer1(in_img)
        s1 = x
        x = self.layer2(x)
        s2 = x
        x = self.layer3(x)
        s4 = x
        x = self.layer4(x)
        s8 = x
        x = self.layer5(x)
        s16 = x
        return {'s1': s1, 's2': s2, 's4': s4, 's8': s8, 's16': s16}


# decoding module
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GatedConv(128, 128, 3, 1, 1)
        self.conv2 = GatedConv(128, 128, 3, 1, 1)
        self.conv3 = GatedConv(128, 128, 3, 1, 1)
        self.conv4 = GatedConv(128, 128, 3, 1, 1)
        self.conv5 = GatedConv(128, 128, 3, 1, 1)
        self.conv6 = GatedConv(128, 128, 3, 1, 1)
        self.conv7 = GatedConv(128, 128, 3, 1, 1)
        self.conv8 = GatedConv(128, 128, 3, 1, 1)
        self.conv9 = GatedConv(128, 128, 3, 1, 1)
        self.predictor = Conv(128, 3, 3, 1, 1)

    def forward(self, feats):
        x = self.conv1(feats['s16'])
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv3(feats['s8'] + x)
        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv5(feats['s4'] + x)
        x = self.conv6(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv7(feats['s2'] + x)
        x = self.conv8(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv9(feats['s1'] + x)
        pred_img = torch.tanh(self.predictor(x))
        return pred_img


# PFCNet model
class PFCNet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.load_state_dict(torch.load(model_path, map_location='cpu'))

    def forward(self, img, mask):
        norm_img = 2 * img - 1
        in_img = (1 - mask) * norm_img

        # query frame inference
        x = torch.cat([in_img, mask], dim=1)
        feats = self.encoder(x)
        pred_img = self.decoder(feats)
        out_img = ((in_img + mask * pred_img) + 1) / 2
        return out_img
