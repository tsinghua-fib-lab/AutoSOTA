import sys, os
sys.path.insert(0, "/repo")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from lib.Network import Network
import glob

CKPT_PATH = "/models/pretrained/Curriseg/model.pth"
TEST_DATA_PATH = "/datasets/TestDataset"
DATASET = "COD10K"
TESTSIZE = 384
BATCH_SIZE = 32
OUTPUT_DIR = "/repo/res/curriseg/" + DATASET + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom batched test dataset
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
        print("  Images:", len(self.images), "GTs:", len(self.gts))
    
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

# Load model
print("Loading model...")
model = Network(channels=192).cuda()
ckpt = torch.load(CKPT_PATH, map_location="cuda")
model.load_state_dict(ckpt, strict=True)
model.eval()
print("Model ready.")

# Create dataloader
data_path = TEST_DATA_PATH + "/" + DATASET + "/"
dataset = TestDatasetBatched(data_path + "Imgs/", data_path + "GT/", TESTSIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Run inference
print("Starting inference...")
mae_sum = 0.0
count = 0

for batch_idx, (images, gts, names, orig_sizes) in enumerate(loader):
    images = images.cuda()
    with torch.no_grad():
        preds = model(images)
    
    final_preds = preds[4]  # main output
    final_preds = torch.sigmoid(final_preds)
    
    for i in range(images.size(0)):
        pred = final_preds[i, 0].cpu().numpy()
        gt = gts[i, 0].cpu().numpy()
        orig_w, orig_h = orig_sizes[1][i].item(), orig_sizes[0][i].item()
        
        # Resize to original GT size
        pred_resized = cv2.resize(pred, (orig_w, orig_h))
        gt_resized = cv2.resize(gt, (orig_w, orig_h))
        
        # Normalize
        pred_norm = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
        
        # Save
        name = names[i]
        cv2.imwrite(OUTPUT_DIR + name, pred_norm * 255)
        
        # MAE
        mae_sum += np.abs(pred_norm - gt_resized).mean()
        count += 1
    
    if (batch_idx + 1) % 10 == 0:
        print("  Batch %d/%d, %d images done" % (batch_idx + 1, len(loader), count))

mae = mae_sum / max(count, 1)
print("\n[Done] MAE: %.6f on %s (%d images)" % (mae, DATASET, count))
print("Output dir:", OUTPUT_DIR)
