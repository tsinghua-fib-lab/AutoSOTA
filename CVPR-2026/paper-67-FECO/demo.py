import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

from lib.core.config import cfg, update_config
from lib.models.model import FECO
from lib.utils.contact_utils import get_contact_thres
from lib.utils.func_utils import get_bbox_body
from lib.utils.vis_utils import ContactRenderer, draw_landmarks_on_image
from lib.utils.preprocessing import augmentation_contact


parser = argparse.ArgumentParser(description='Demo FECO')
parser.add_argument('--backbone', type=str, default='vit-h-14', choices=['vit-h-14', 'vit-l-16', 'vit-b-16', 'vit-s-16', 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'], help='backbone model')
parser.add_argument('--checkpoint', type=str, default='', help='model path for demo')
parser.add_argument('--input_path', type=str, default='asset/example_images', help='image path for demo')
args = parser.parse_args()


# Set device as CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize directories
experiment_dir = 'experiments_demo'


# Load config
update_config(backbone_type=args.backbone, exp_dir=experiment_dir)


# Initialize renderer
contact_renderer = ContactRenderer()


# Load demo images
input_dir = args.input_path
images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Initialize YOLO human detector
yolo_model_path = os.path.join('data', 'base_data', 'pretrained_models', 'yolo', 'yolov8s-pose.pt')
detector = YOLO(yolo_model_path)


############# Model #############
model = FECO().to(device)
model.eval()
############# Model #############


# Load model checkpoint if provided
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])


############################### Demo Loop ###############################
for i, frame_name in tqdm(enumerate(images), total=len(images)):
    print(f"Processing: {frame_name}")

    # Load and convert image
    frame_path = os.path.join(input_dir, frame_name)
    frame = cv2.imread(frame_path)
    orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_name_base = os.path.splitext(frame_name)[0]

    # Human detection
    detection_result = detector(frame_path)
    kpts = detection_result[0].keypoints.data[0] # first person detected
    kpts_np = kpts.detach().cpu().numpy()
    keypoints_2d = kpts_np[:, :2]
    keypoints_2d_valid = kpts_np[:, 2] > 0.5

    # Visualize human pose
    annotated_image, _ = draw_landmarks_on_image(orig_img.copy(), keypoints_2d)

    # image size (H, W)
    img_h, img_w, _ = orig_img.shape

    # left/right foot bboxes. YOLOv8 ankle indices: left=15, right=16
    bbox_foot_l, bbox_foot_r = get_bbox_body(
        keypoints_2d,
        keypoints_2d_valid,
        r_ankle_idx=16,
        l_ankle_idx=15,
        foot_frac=0.4,
        image_size=(img_h, img_w),
    ) # both in [x_min, y_min, x_max, y_max]

    # Image preprocessing
    crop_img, img2bb_trans, bb2img_trans, rot, do_flip, color_scale, do_extreme_crop, extreme_crop_lvl = augmentation_contact(orig_img.copy(), bbox_foot_r, 'test', enforce_flip=False)

    # Convert to model input format
    if 'resnet' in cfg.MODEL.backbone_type:
        from torchvision import transforms
        img_tensor = transforms.ToTensor()(crop_img.astype(np.float32) / 255.0)
    elif 'vit' in cfg.MODEL.backbone_type:
        from torchvision.transforms import Normalize
        normalize = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
        img_tensor = crop_img.transpose(2, 0, 1) / 255.0
        img_tensor = normalize(torch.from_numpy(img_tensor)).float()
    else:
        raise NotImplementedError(f"Unsupported backbone: {args.backbone}")

    ############# Run model #############
    with torch.no_grad():
        outputs = model({'input': {'image': img_tensor[None].to(device)}}, mode="test")
    ############# Run model #############

    # Save result
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/detection', exist_ok=True)
    os.makedirs('outputs/crop_img', exist_ok=True)
    os.makedirs('outputs/contact', exist_ok=True)

    cv2.imwrite(f'outputs/detection/{frame_name_base}.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'outputs/crop_img/{frame_name_base}.png', crop_img[..., ::-1])

    eval_thres = get_contact_thres(args.backbone)
    contact_mask = (outputs['contact_out'][0] > eval_thres).detach().cpu().numpy()
    contact_rendered = contact_renderer.render_contact(crop_img[..., ::-1], contact_mask)
    cv2.imwrite(f'outputs/contact/{frame_name_base}.png', contact_rendered)
############################### Demo Loop ###############################