import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import cfg
from data import make_dataloader
from modeling import make_model
from utils.logger import setup_logger


# 自定义 Newdict 类，用于处理输入数据
class Newdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = [1, 2, 3, 4]

    # 定义 .to 方法，用于将数据转移到指定设备
    def to(self, device):
        for key in self.keys():
            self[key] = self[key].to(device)
        return self

    # 定义 size 方法，用于返回数据的尺寸
    def size(self, k):
        data = self['RGB']
        width, height = data.size(-1), data.size(-2)
        return width if k == -1 else height


# 定义 reshape_transform 函数，用于调整张量形状
def reshape_transform(tensor, height=16, width=8):
    """
    Reshape transformer-based model outputs for Grad-CAM visualization.

    Args:
        tensor: The input tensor to reshape.
        height: Target height after reshaping.
        width: Target width after reshaping.

    Returns:
        Reshaped tensor with channels as the first dimension.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


# 定义 Grad-CAM 可视化显示函数
def show_cam(index, imgpath, grayscale_cam, modality, cfg, n_iter):
    """
    Visualize Grad-CAM heatmaps on input images.

    Args:
        index: Index of the current image in the batch.
        imgpath: List of image paths.
        grayscale_cam: Grad-CAM heatmap.
        modality: Modality of the image (e.g., RGB, NI, TI).
        cfg: Configuration object.
        n_iter: Current iteration in the data loader.

    Saves the visualization to the output directory.
    """
    index = int(index)
    img_path = imgpath[index]
    print(f"Processing {index}: {img_path}")

    # Load image based on dataset and modality
    if cfg.DATASETS.NAMES == 'RGBNT201':
        img_path = f'../RGBNT201/test/{modality}/{img_path}'
    elif cfg.DATASETS.NAMES == 'RGBNT100':
        img_path = f'../RGBNT100/rgbir/query/{img_path}'

    grayscale_cam = grayscale_cam[index]
    if cfg.DATASETS.NAMES == 'RGBNT100':
        img = Image.open(img_path).convert('RGB')
        if modality == "RGB":
            cropped_image = img.crop((0, 0, 256, 128))
        elif modality == "NI":
            cropped_image = img.crop((256, 0, 512, 128))
        else:
            cropped_image = img.crop((512, 0, 768, 128))
        rgb_image = np.float32(cropped_image) / 255
    else:
        img = cv2.imread(img_path, 1)
        rgb_image = cv2.resize(img, (128, 256))
        rgb_image = np.float32(rgb_image) / 255

    # Generate and save visualization
    visualization = show_cam_on_image(rgb_image, grayscale_cam)
    output_dir = f'../gradcam_vis/{cfg.DATASETS.NAMES}/{modality}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{n_iter * cfg.TEST.IMS_PER_BATCH + index}.jpg'
    cv2.imwrite(save_path, visualization)


# 主函数入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDReID Testing")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("opts", help="Modify config options via command line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # 加载配置文件和选项
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 设置输出目录和日志
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("MDReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            logger.info(f"\n{cf.read()}")
    logger.info(f"Running with config:\n{cfg}")

    # 配置 CUDA 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda:"+cfg.MODEL.DEVICE_ID

    # 加载数据集
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 加载模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param("your_model.pth")
    model.eval()
    model.to(device)

    # 定义目标层
    target_layers = [model.BACKBONE.base]

    # 遍历验证集数据
    for n_iter, (img, pid, camids, camids_batch, viewids, imgpath) in enumerate(val_loader):
        img = Newdict({'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device),
                       'cam_label': camids.to(device)})

        # 应用 Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        grayscale_cam = cam(input_tensor=img, targets=None, eigen_smooth=False, aug_smooth=False)

        # 保存每张图片的可视化结果
        modality = "TI"  # 可选 RGB, NI, TI
        for i in range(cfg.TEST.IMS_PER_BATCH):
            show_cam(i, imgpath, grayscale_cam, modality, cfg, n_iter)

        # 仅处理一次迭代，以便查看一个batch的可视化结果, 可以注释掉这个条件
        if n_iter == 0:
            break
