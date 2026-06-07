import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, deconv=False, is_3d=False, norm='instance', relu='leaky', **kwargs):
        super(BasicConv, self).__init__()
        self.use_norm = norm
        self.use_relu = relu

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channel, out_channel, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channel, out_channel, bias=False, **kwargs)
            if norm == 'batch':
                self.norm = nn.BatchNorm3d(out_channel)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm3d(out_channel)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channel, out_channel, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channel)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channel)

        if relu == 'leaky':
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == 'relu':
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        
        if self.use_norm:
            x = self.norm(x)
        if self.use_relu:
            x = self.relu(x)

        return x

class Conv2x(nn.Module):
    def __init__(self, in_channel, out_channel, deconv=False, is_3d=False, norm='instance', relu='leaky'):
        super(Conv2x, self).__init__()

        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = BasicConv(in_channel, out_channel, deconv, is_3d, norm, relu, kernel_size=kernel, stride=2, padding=1)
        self.conv2 = BasicConv(out_channel * 2, out_channel * 2, False, is_3d, norm, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = self.conv1(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv2(x)

        return x

class FeatureAtt(nn.Module):
    def __init__(self, cv_channel, feat_channel):
        super(FeatureAtt, self).__init__()
        self.feat_att = nn.Sequential(
            BasicConv(feat_channel, feat_channel // 2, kernel_size=1),
            nn.Conv2d(feat_channel // 2, cv_channel, 1)
        )

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv

        return cv

class HourGlass(nn.Module):
    def __init__(self, cfg):
        super(HourGlass, self).__init__()
        cv_channel = cfg.gwc_group

        self.corr_stem = BasicConv(cfg.gwc_group, cfg.gwc_group, is_3d=True, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            BasicConv(cv_channel, cv_channel * 2, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            BasicConv(cv_channel * 2, cv_channel * 4, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 6, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 6, cv_channel * 6, is_3d=True, kernel_size=3, padding=1)
        )

        self.conv3_up = BasicConv(cv_channel * 6, cv_channel * 4, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2_up = BasicConv(cv_channel * 4, cv_channel * 2, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1_up = BasicConv(cv_channel * 2, cv_channel, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        self.agg_0 = nn.Sequential(
            BasicConv(cv_channel * 8, cv_channel * 4, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.agg_1 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 2, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )

        self.feature_att_4 = FeatureAtt(cv_channel, cfg.feat_dim[0])
        self.feature_att_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])
        self.feature_att_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_32 = FeatureAtt(cv_channel * 6, cfg.feat_dim[3])
        self.feature_att_up_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_up_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])

    def forward(self, x, feat):
        x = self.corr_stem(x)
        x = self.feature_att_4(x, feat[0])

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, feat[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, feat[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, feat[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, feat[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, feat[1])

        conv = self.conv1_up(conv1)

        return conv

