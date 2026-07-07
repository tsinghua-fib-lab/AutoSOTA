import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F

def gauss_kernel(kernlen=5, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return torch.from_numpy(out_filter)

def make_gauss_var(size, nsig, c_i):
    kernel = gauss_kernel(size, nsig, c_i)
    var = torch.nn.Parameter(kernel, requires_grad=False)
    return var

class GaussConv(nn.Module):
    def __init__(self, size, nsig, channels, padding='same'):
        super(GaussConv, self).__init__()
        self.padding = padding
        self.kernel = make_gauss_var(size, nsig, channels)

    def forward(self, img):
        c_i = img.shape[1]
        
        if self.padding == 'same':
            padding = self.kernel.shape[2] // 2
        else:
            padding = 0
        return F.conv2d(img, self.kernel, padding=padding, stride=1, groups=c_i)

# Sample usage
if __name__ == "__main__":
    img = torch.randn(1, 3, 10, 10)  # Random image with shape (batch_size, channels, height, width)
    gauss_conv = GaussConv(5, 3, img.shape[1])
    output = gauss_conv(img)
    print("Output shape:", output.shape)