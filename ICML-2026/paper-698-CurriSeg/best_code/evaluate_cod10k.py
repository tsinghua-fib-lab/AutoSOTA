import sys, os
sys.path.insert(0, "/repo")
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lib.Network import Network
from utils.data_val import test_dataset

# Config
CKPT_PATH = "/models/pretrained/Curriseg/model.pth"
TEST_DATA_PATH = "/datasets/TestDataset"
DATASET = "COD10K"
TESTSIZE = 384  # Match checkpoint training size
OUTPUT_DIR = "/repo/res/curriseg/" + DATASET + "/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading model...")
model = Network(channels=192).cuda()
ckpt = torch.load(CKPT_PATH, map_location="cuda")
model.load_state_dict(ckpt, strict=True)
model.eval()
print("Model loaded, params: %.1f M" % (sum(p.numel() for p in model.parameters()) / 1e6))

# Test dataset
data_path = TEST_DATA_PATH + "/" + DATASET + "/"
image_root = data_path + "Imgs/"
gt_root = data_path + "GT/"

print("Testing on", DATASET)
print("  Images:", image_root)
print("  GT:", gt_root)

test_loader = test_dataset(image_root, gt_root, TESTSIZE)
print("  Total images:", test_loader.size)

mae_sum = 0.0
count = 0

for i in range(test_loader.size):
    image, gt, name, _ = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt_norm = gt / (gt.max() + 1e-8)
    
    image = image.cuda()
    with torch.no_grad():
        result = model(image)
    
    res = F.interpolate(result[4], size=gt.shape, mode="bilinear", align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
    # Save prediction
    save_name = name if name.endswith(".png") else os.path.splitext(name)[0] + ".png"
    cv2.imwrite(OUTPUT_DIR + save_name, res * 255)
    
    # Accumulate MAE
    mae_sum += np.sum(np.abs(res - gt_norm)) / (gt.shape[0] * gt.shape[1])
    count += 1
    
    if (i + 1) % 500 == 0:
        print("  %d/%d done" % (i + 1, test_loader.size))

mae = mae_sum / max(count, 1)
print("\n[Done] MAE: %.6f on %s (%d images)" % (mae, DATASET, count))
print("Predictions saved to:", OUTPUT_DIR)
