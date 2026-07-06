import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.Network_Res2Net_GRA_NCD import Network
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/20201214-Network_Res2Net_GRA_NCD/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['vis']:
    data_path = '/{}/'.format(_data_name)
    save_path = './res/{}/middle_vis/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res_list = model(image)
        for i in range(4):
            res = res_list[i]
            res = -1 * (torch.sigmoid(res)) + 1
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            misc.imsave(save_path+name.replace('.png', '_{}.png'.format(i)), res)
