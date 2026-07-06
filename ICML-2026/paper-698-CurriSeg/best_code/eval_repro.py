"""Evaluation: computes all metrics (M, Fbeta, Ephi, Salpha) on COD10K.
Supports H-flip TTA and sigmoid temperature scaling."""
import sys, os, argparse
sys.path.insert(0, "/repo")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from lib.Network import Network
from py_sod_metrics import Emeasure, Smeasure, WeightedFmeasure
import glob

CKPT_PATH = "/models/pretrained/Curriseg/model.pth"
TEST_DATA_PATH = "/datasets/TestDataset"
DATASET = "COD10K"

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=384)
parser.add_argument("--tta", action="store_true", help="Enable H-flip TTA")
parser.add_argument("--temperature", type=float, default=1.0, help="Sigmoid temperature")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

TESTSIZE = args.testsize
BATCH_SIZE = args.batch_size
TEMPERATURE = args.temperature

class TestDatasetBatched(torch.utils.data.Dataset):
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted(glob.glob(image_root + "/*.jpg") + glob.glob(image_root + "/*.png"))
        self.gts = sorted(glob.glob(gt_root + "/*.png") + glob.glob(gt_root + "/*.tif"))
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        orig_size = img.size
        img_tensor = self.transform(img)
        gt = Image.open(self.gts[idx]).convert("L")
        gt_tensor = self.gt_transform(gt)
        name = os.path.basename(self.images[idx])
        if name.endswith(".jpg"):
            name = name[:-4] + ".png"
        return img_tensor, gt_tensor, name, orig_size

model = Network(channels=192).cuda()
ckpt = torch.load(CKPT_PATH, map_location="cuda")
model.load_state_dict(ckpt, strict=True)
model.eval()

data_path = TEST_DATA_PATH + "/" + DATASET + "/"
dataset = TestDatasetBatched(data_path + "Imgs/", data_path + "GT/", TESTSIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

em_calc = Emeasure()
sm_calc = Smeasure()
wfm_calc = WeightedFmeasure()
mae_sum = 0.0
count = 0

for images, gts, names, orig_sizes in loader:
    images = images.cuda()
    B = images.size(0)
    
    with torch.no_grad():
        if args.tta:
            logits_orig = model(images)[4]
            images_hf = torch.flip(images, dims=[3])
            logits_hf = torch.flip(model(images_hf)[4], dims=[3])
            avg_logits = (logits_orig + logits_hf) / 2.0
            final_preds = torch.sigmoid(avg_logits / TEMPERATURE)
        else:
            final_preds = torch.sigmoid(model(images)[4] / TEMPERATURE)
    
    for i in range(B):
        pred = final_preds[i, 0].cpu().numpy()
        gt_val = gts[i, 0].cpu().numpy()
        orig_w = int(orig_sizes[1][i].item())
        orig_h = int(orig_sizes[0][i].item())
        
        pred_resized = cv2.resize(pred, (orig_w, orig_h))
        gt_resized = cv2.resize(gt_val, (orig_w, orig_h))
        
        p_min, p_max = pred_resized.min(), pred_resized.max()
        pred_norm = (pred_resized - p_min) / (p_max - p_min + 1e-8)
        mae_sum += np.abs(pred_norm - gt_resized).mean()
        count += 1
        
        gt_nn = cv2.resize(gt_val, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        gt_bool = gt_nn > 0.5
        em_calc.step(pred=pred_norm.astype(np.float64), gt=gt_bool, normalize=False)
        sm_calc.step(pred=pred_norm.astype(np.float64), gt=gt_bool, normalize=False)
        wfm_calc.step(pred=pred_norm.astype(np.float64), gt=gt_bool, normalize=False)

M = float(mae_sum / count)
em_res = em_calc.get_results()
Fbeta = float(wfm_calc.get_results()["wfm"])
Ephi = float(em_res["em"]["adp"])
Salpha = float(sm_calc.get_results()["sm"])

print(f"M: {M:.6f}")
print(f"Fbeta: {Fbeta:.6f}")
print(f"Ephi: {Ephi:.6f}")
print(f"Salpha: {Salpha:.6f}")
print(f"Config: testsize={TESTSIZE} tta={args.tta} temperature={TEMPERATURE}")
