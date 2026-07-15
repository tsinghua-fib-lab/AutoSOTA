import functools

from .. import arch_util
import torch
import torch.nn as nn
from ..backbones.resnet import ResidualBlock_noBN, ResnetBlock
from ..backbones.unet_parts import UnetSkipConnectionBlock


# The function F in the paper
class KernelExtractor(nn.Module):
    def __init__(self, opt):
        super(KernelExtractor, self).__init__()

        nf = opt["nf"]
        self.kernel_dim = opt["kernel_dim"]
        self.use_sharp = opt["KernelExtractor"]["use_sharp"]
        self.use_vae = opt["use_vae"]

        # Blur estimator
        norm_layer = arch_util.get_norm_layer(opt["KernelExtractor"]["norm"])
        n_blocks = opt["KernelExtractor"]["n_blocks"]
        padding_type = opt["KernelExtractor"]["padding_type"]
        use_dropout = opt["KernelExtractor"]["use_dropout"]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        input_nc = nf * 2 if self.use_sharp else nf
        output_nc = self.kernel_dim * 2 if self.use_vae else self.kernel_dim

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(nf),
            nn.ReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            inc = min(nf * mult, output_nc)
            ouc = min(nf * mult * 2, output_nc)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(nf * mult * 2),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    output_nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, sharp, blur):
        output = self.model(torch.cat((sharp, blur), dim=1))
        if self.use_vae:
            return output[:, : self.kernel_dim, :, :], output[:, self.kernel_dim :, :, :]

        return output, torch.zeros_like(output).cuda()


# The function G in the paper
class KernelAdapter(nn.Module):
    def __init__(self, opt):
        super(KernelAdapter, self).__init__()
        input_nc = opt["nf"]
        output_nc = opt["nf"]
        ngf = opt["nf"]
        norm_layer = arch_util.get_norm_layer(opt["Adapter"]["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x, k):
        """Standard forward"""
        return self.model(x, k)


class KernelWizard(nn.Module):
    def __init__(self, opt):
        super(KernelWizard, self).__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.1)
        front_RBs = opt["front_RBs"]
        back_RBs = opt["back_RBs"]
        num_image_channels = opt["input_nc"]
        nf = opt["nf"]

        # Features extraction
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []

        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)

        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())

        self.feature_extractor = nn.Sequential(*feature_extractor)

        # Kernel extractor
        self.kernel_extractor = KernelExtractor(opt)

        # kernel adapter
        self.adapter = KernelAdapter(opt)

        # Reconstruction
        recon_trunk = []
        for i in range(back_RBs):
            recon_trunk.append(resBlock_noBN_f())

        # upsampling
        recon_trunk.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True))
        recon_trunk.append(nn.PixelShuffle(2))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
        recon_trunk.append(lrelu)
        recon_trunk.append(nn.Conv2d(64, num_image_channels, 3, 1, 1, bias=True))

        self.recon_trunk = nn.Sequential(*recon_trunk)

    def adaptKernel(self, x_sharp, kernel):
        B, C, H, W = x_sharp.shape
        base = x_sharp

        x_sharp = self.feature_extractor(x_sharp)

        out = self.adapter(x_sharp, kernel)
        out = self.recon_trunk(out)
        out += base

        return out

    def forward(self, x_sharp, x_blur):
        x_sharp = self.feature_extractor(x_sharp)
        x_blur = self.feature_extractor(x_blur)

        output = self.kernel_extractor(x_sharp, x_blur)
        return output
