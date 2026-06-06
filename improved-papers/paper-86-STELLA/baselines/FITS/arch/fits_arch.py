import torch
import torch.nn as nn


class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, **model_args):
        super(FITS, self).__init__()
        self.seq_len = model_args['seq_len']
        self.pred_len = model_args['pred_len']
        self.individual = model_args['individual']
        self.channels = model_args['enc_in']

        self.cut_freq=model_args['cut_freq'] # 720/24

        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len
        if self.individual:
            self.freq_upsampler_real = nn.ModuleList()
            self.freq_upsampler_imag = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler_real.append(nn.Linear(self.cut_freq, int(self.cut_freq*self.length_ratio)))
                self.freq_upsampler_imag.append(nn.Linear(self.cut_freq, int(self.cut_freq*self.length_ratio)))

        else:
            self.freq_upsampler_real = nn.Linear(self.cut_freq, int(self.cut_freq*self.length_ratio)) # complex layer for frequency upcampling]
            self.freq_upsampler_imag = nn.Linear(self.cut_freq, int(self.cut_freq*self.length_ratio)) # complex layer for frequency upcampling]
        # pred_len=seq_len+pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # pred_len=self.pred_len


    def forward(self, history_data, **kwargs):
        
        x = history_data[..., 0]
        L = x.size(1)

        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)


        # if self.training:
        low_specx = torch.fft.rfft(x, dim=1)

        low_specx = torch.view_as_real(low_specx[:,0:self.cut_freq,:])
        low_specx_real = low_specx[:,:,:,0]
        low_specx_imag = low_specx[:,:,:,1]
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.cut_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            # print(low_specx.permute(0,2,1).size())
            # print(low_specx.permute(0,2,1).size())
            low_specxy_real = self.freq_upsampler_real(low_specx_real.permute(0,2,1)).permute(0,2,1)-self.freq_upsampler_imag(low_specx_imag.permute(0,2,1)).permute(0,2,1)
            low_specxy_imag = self.freq_upsampler_real(low_specx_imag.permute(0,2,1)).permute(0,2,1)+self.freq_upsampler_imag(low_specx_real.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        # low_specxy_ = torch.complex(low_specxy_real, low_specxy_imag)
        low_specxy_R = torch.zeros([low_specxy_real.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_real.size(2)],dtype=low_specxy_real.dtype).to(low_specxy_real.device)
        low_specxy_R[:,0:low_specxy_real.size(1),:]=low_specxy_real

        low_specxy_I = torch.zeros([low_specxy_imag.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_imag.size(2)],dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)
        low_specxy_I[:,0:low_specxy_imag.size(1),:]=low_specxy_imag

        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        low_xy=torch.fft.irfft(low_specxy, dim=1)

        low_xy=low_xy * self.length_ratio # compemsate the length change

        xy=(low_xy) * torch.sqrt(x_var) +x_mean # (B, L1+L2, N)
        
        y = xy[:, L:, :].unsqueeze(-1)
        return y

