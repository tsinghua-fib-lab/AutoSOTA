import torch
import numpy as np
import os, argparse, cv2
from lib.Network_Res2Net_GRA_NCD_FeatureViz import Network
from utils.dataloader import test_dataset


def heatmap(feat_viz, ori_img, save_path=None):
    feat_viz = torch.mean(feat_viz, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    feat_viz = (feat_viz - feat_viz.min()) / (feat_viz.max() - feat_viz.min() + 1e-8)

    ori_img = ori_img.data.cpu().numpy().squeeze()
    ori_img = ori_img.transpose((1, 2, 0))
    ori_img = ori_img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    ori_img = ori_img[:, :, ::-1]
    ori_img = np.uint8(255 * ori_img)
    feat_viz = np.uint8(255 * feat_viz)
    feat_viz = cv2.applyColorMap(feat_viz, cv2.COLORMAP_JET)
    feat_viz = cv2.resize(feat_viz, (320, 320))
    ori_img = cv2.resize(ori_img, (320, 320))
    feat_viz = cv2.addWeighted(ori_img, 0.5, feat_viz, 0.5, 0)

    cv2.imwrite(save_path, feat_viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./snapshot/20201214-Network_Res2Net_GRA_NCD/Net_epoch_best.pth')
    opt = parser.parse_args()

    for _data_name in ['CAMO', 'COD10K', 'CHAMELEON']:
        data_path = '/media/nercms/NERCMS/GepengJi/2020ACMMM/Dataset/COD_New_data/TestDataset/{}/'.format(_data_name)
        save_path = './res/{}/Feature_Viz/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
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

            res5, res4, res3, res2, feat_viz = model(image)
            for i in range(0, 3):
                for j in range(0, 4):
                    for k in range(0, 2):
                        cur_feat_viz = feat_viz[i][j][k]
                        label = 'feat' if k == 0 else 'guid'
                        img_name = name.split('.')[0] + '_level{}_GRA{}_'.format(i+3, j+1) + label + '.png'
                        heatmap(feat_viz=cur_feat_viz, ori_img=image, save_path=save_path+img_name)
                        print('> Dataset: {}, Image: {}'.format(_data_name, save_path+img_name))