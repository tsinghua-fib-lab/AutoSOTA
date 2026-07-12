import torch, numpy as np, h5py, sys
sys.path.insert(0, '/repo')
from model_siren import siren_model
from utils import build_coordinate_train, MYTVLoss, normalize01
import torch.nn as nn
from numpy import fft

device = torch.device('cuda:0')

# Load SIREN model
class SirenModel(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0
        self.model_mag = siren_model(num_layers=8, input_dim=2, hidden_dim=256, out_dim=1, w0=w0)
        self.model_phi = siren_model(num_layers=8, input_dim=2, hidden_dim=256, out_dim=1, w0=w0)
    def forward(self, coords):
        return self.model_mag(coords).float(), self.model_phi(coords).float()

model = SirenModel().to(device)
ckpt = torch.load('/repo/checkpoints/model_epoch_2500.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
print('SIREN checkpoint loaded from epoch', ckpt['epoch'])
print('Checkpoint eval PSNR:', ckpt.get('eval_psnr', 'N/A'))

# Load sample data
with h5py.File('/repo/data/sample_0009.h5', 'r') as f:
    gt_img = f['img_full'][:]
    mask = f['mask'][:]
    mask_t = mask.transpose(1,2,0)
    csmp = f['csmp'][:]
    csmp_t = csmp.transpose(1,2,0)
    fft_data = f['forward_fft'][:]
    fft_t = fft_data.transpose(1,2,0)
    print('Data loaded. GT shape:', gt_img.shape)
    print('Attrs:', dict(f.attrs))

nRow, nCol = gt_img.shape
coords = torch.from_numpy(build_coordinate_train(L_RO=nRow, L_PE=nCol)).to(device).float()

# Quick forward pass
with torch.no_grad():
    mag, phi = model(coords.view(-1, 2))
    pre_intensity = torch.complex(mag.view(nRow, nCol, 1), phi.view(nRow, nCol, 1))

    def calc_psnr(p, t):
        pn = normalize01(np.abs(p))
        tn = normalize01(np.abs(t))
        mse = np.mean((pn - tn) ** 2)
        if mse == 0:
            return 100.0
        dr = np.max(tn)
        if dr == 0:
            return 0.0
        return 20.0 * np.log10(dr / np.sqrt(mse))

    # Zero-filled reference
    zf_img = np.sum(
        fft.fftshift(fft.ifft2(fft.fftshift(fft_t, axes=(0,1)), axes=(0,1)), axes=(0,1)) *
        np.conj(csmp_t), axis=2
    )
    norm = np.max(np.abs(zf_img))

    pred = pre_intensity.squeeze().cpu().numpy()
    roi = mask_t[:,:,0] if len(mask_t.shape) == 3 else mask_t
    psnr = calc_psnr(pred * roi, gt_img / norm * roi)
    print('Zero-shot PSNR (no adaptation, SIREN w/ IPOD): {:.2f} dB'.format(psnr))
    print('Checkpoint meta_loss:', ckpt.get('meta_loss', 'N/A'))
