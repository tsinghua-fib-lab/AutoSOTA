from convLast_std import Net
model_name = 'convLast_std_tem_data4'

#######################################################
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch
import os
import time
import math
import torch.utils.data as data
import torchvision
from math import sqrt,log10
from pytorch_msssim import ssim

import torchvision
import torch.nn.functional as F

class TEM(data.Dataset):
    def __init__(self, data_path):
        super(TEM, self).__init__()
        self.lq_filenames = []
        self.hq_filenames = []
        for i in range(1000):
            name = str(i) + '.png'
            lq_path = os.path.join(data_path, 'noisy', name)
            hq_path = os.path.join(data_path, 'gt', name)
            self.lq_filenames.append(lq_path)
            self.hq_filenames.append(hq_path)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # 读取图像（会得到[C, H, W]格式，C可能为1、3或4）
        lq = torchvision.io.read_image(self.lq_filenames[idx]).float()
        hq = torchvision.io.read_image(self.hq_filenames[idx]).float()
        
        # 只取R通道（对于RGB是第0通道，对于RGBA也是第0通道）
        # 如果是单通道图像则保持不变
        if lq.shape[0] > 1:  # 通道数大于1时取R通道
            lq = lq[:1]  # 取第一个通道（R通道）
        if hq.shape[0] > 1:
            hq = hq[:1]  # 取第一个通道（R通道）
            
        return lq, hq

def train():
    # bring model to device
    name = model_name
    model = Net()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)
    model = nn.DataParallel(model).to(device)

    # training settings
    batch_size = 8
    save_path = './'
    milestones = [250, 400, 425, 450, 475]
    learning_rate = 2e-4
    gamma = 0.5
    epochs = 100
    log_every = 100
    save_every = 100
    pretrain = None



    # get data loader
    dataset = TEM('./tem_data4')
    # dataset=TEM('../TEM')
    train_dataloader = DataLoader(dataset=dataset, num_workers=16, batch_size=batch_size,
                                shuffle=True, pin_memory=True, drop_last=True)

    # get loss and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    start_epoch = 1

    # use the last saved model to auto-resume training
    for i in range(epochs, 0, -1):
        s = save_path + f'{name}_{i}.pth'
        if os.path.exists(s):
            start_epoch = i + 1
            print(f'loading {s}')
            ckpt = torch.load(s)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            break

    # use pretrained model
    if pretrain is not None:
        if os.path.exists(pretrain):
            print(f'loading {pretrain}')
            ckpt = torch.load(pretrain)
            del ckpt['model_state_dict']['module.last_conv.weight']
            del ckpt['model_state_dict']['module.last_conv.bias']
            model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # train
    timer_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('fp32', epoch, opt_lr))
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            if (iter + 1) % log_every == 0:
                cur_steps = (iter + 1) * batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)
                epoch_width = math.ceil(math.log10(epochs))
                cur_epoch = str(epoch).zfill(epoch_width)
                avg_loss = epoch_loss / (iter + 1)
                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps,
                                                                        avg_loss, duration))

        # save model
        if epoch % save_every == 0:
            torch.set_grad_enabled(False)
            model = model.eval()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path + f'{name}_{epoch}.pth')
            sys.stdout.flush()
            torch.set_grad_enabled(True)

        scheduler.step()

def save_images():
    unet = nn.DataParallel(Net()).cuda()
    weights = torch.load(model_name + "_100.pth")
    unet.load_state_dict(weights['model_state_dict'])
    unet.eval()
    torch.set_grad_enabled(False)

    for i in range(100):
        name = str(i) + '.png'
        os.makedirs(model_name + '_result', exist_ok=True)
        in_path = os.path.join('tem_test_data4/noisy', name)
        out_path = os.path.join(model_name + '_result', name)
        in_img = torchvision.io.read_image(in_path).cuda()
        
        # 处理输入图像通道，只保留R通道
        if in_img.shape[0] > 1:
            in_img = in_img[:1]  # 取R通道
            
        in_img = torch.unsqueeze(in_img, 0).float()
        out_img = unet(in_img)
        out_img = torch.clip_(out_img, 0, 255)
        out_img = torch.squeeze(out_img, 0).byte()
        torchvision.io.write_png(out_img.cpu(), out_path)
        print(i)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse = diff.pow(2).mean()
    psnr = -10 * log10(mse)
    return float(psnr)


def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)

def calc_iou(sr, hr, threshold=127.5):
    """
    计算IoU指标
    1. 先将输入图像二值化（大于阈值为1，小于等于为0）
    2. 计算交并比：IoU = 交集 / 并集
    """
    # 转换为浮点数进行处理
    sr = sr.float()
    hr = hr.float()
    
    # 二值化处理
    sr_binary = (sr > threshold).float()  # 大于阈值为1，否则为0
    hr_binary = (hr > threshold).float()  # 大于阈值为1，否则为0
    
    # 计算交集和并集
    intersection = (sr_binary * hr_binary).sum()
    union = sr_binary.sum() + hr_binary.sum() - intersection
    
    # 避免除以零
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def get_metrics():
    gt = 'tem_test_data4/gt'
    path = os.listdir(gt)
    psnr, ssim1, iou = 0., 0., 0.
    for name in path:
        in_path = os.path.join(gt, name)
        out_path = os.path.join(model_name + '_result', name)
        
        # 读取并处理为单通道（无论输入是几通道，都只取第一个通道）
        in_img = torchvision.io.read_image(in_path).float()
        out_img = torchvision.io.read_image(out_path).float()
        
        # 关键修复：无论通道数是多少（包括4通道），都只保留第一个通道
        if in_img.shape[0] > 1:
            in_img = in_img[:1]  # 取第一通道（解决4通道问题）
        if out_img.shape[0] > 1:
            out_img = out_img[:1]  # 确保输出也是单通道
            
        in_img = torch.unsqueeze(in_img, 0)
        out_img = torch.unsqueeze(out_img, 0)
        
        # 计算各指标
        psnr += calc_psnr(in_img, out_img)
        ssim1 += calc_ssim(in_img, out_img)
        iou += calc_iou(in_img, out_img)
    
    avg_psnr = psnr / len(path)
    avg_ssim = ssim1 / len(path)
    avg_iou = iou / len(path)
    print(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    train()
    save_images()
    get_metrics()