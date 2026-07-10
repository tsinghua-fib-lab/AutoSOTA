import os, pickle, numpy as np
from PIL import Image

# Source: kagglehub PNG files
src_base = "/root/.cache/kagglehub/datasets/swaroopkml/cifar10-pngs-in-folders/versions/1/cifar10/cifar10"

# Destination: torchvision format
dst_dir = "/datasets/cifar-10-batches-py"
os.makedirs(dst_dir, exist_ok=True)

# Get class names from train dir
train_dir = os.path.join(src_base, "train")
class_names = sorted(os.listdir(train_dir))
print(f"Classes: {class_names}")

# Labels from class order (alphabetical)
# Actually, CIFAR-10 standard order is: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# The PNG files might use different ordering
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_split(split_dir, num_per_batch=10000):
    """Load all PNGs from a directory structure, return data and labels."""
    all_data = []
    all_labels = []
    labels = []
    
    # Map folder names to class indices
    # The folder names might be actual class names or numbered
    class_dirs = sorted(os.listdir(split_dir))
    print(f"Class dirs in {split_dir}: {class_dirs}")
    
    for cls_idx, cls_dir in enumerate(class_dirs):
        cls_path = os.path.join(split_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        files = sorted(os.listdir(cls_path))
        print(f"  {cls_dir}: {len(files)} images")
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(cls_path, fname)
                img = Image.open(fpath)
                img = img.resize((32, 32))
                arr = np.array(img)  # [H, W, C]
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                elif arr.shape[-1] == 4:
                    arr = arr[:, :, :3]
                all_data.append(arr)
                all_labels.append(cls_idx)
    
    data = np.array(all_data)  # [N, 32, 32, 3]
    labels = np.array(all_labels)
    
    # Reshape to CIFAR-10 format: [N, 3072] with R channel first
    data_flat = data.reshape(data.shape[0], -1)
    
    return data_flat, labels

# Load train data
train_data, train_labels = load_split(train_dir)
print(f"Train: {train_data.shape}, labels: {train_labels.shape}")

# Split into batches (5 batches of 10000 each, or as available)
batch_size = 10000
num_batches = max(1, len(train_data) // batch_size)
if len(train_data) % batch_size != 0:
    num_batches += 1

for i in range(min(num_batches, 5)):
    start = i * batch_size
    end = min(start + batch_size, len(train_data))
    batch_data = train_data[start:end]
    batch_labels = train_labels[start:end]
    
    batch_dict = {
        'batch_label': f'training batch {i+1} of {num_batches}',
        'labels': batch_labels.tolist(),
        'data': batch_data,
        'filenames': [f'batch_{i+1}_{j}.png' for j in range(len(batch_data))],
    }
    
    fname = os.path.join(dst_dir, f'data_batch_{i+1}')
    with open(fname, 'wb') as f:
        pickle.dump(batch_dict, f)
    print(f"Wrote {fname}: {len(batch_data)} samples")

# Load test data
test_dir = os.path.join(src_base, "test")
test_data, test_labels = load_split(test_dir)
test_dict = {
    'batch_label': 'testing batch 1 of 1',
    'labels': test_labels.tolist(),
    'data': test_data,
    'filenames': [f'test_{j}.png' for j in range(len(test_data))],
}
with open(os.path.join(dst_dir, 'test_batch'), 'wb') as f:
    pickle.dump(test_dict, f)
print(f"Wrote test_batch: {len(test_data)} samples")

# Write batches.meta
meta = {
    'label_names': CIFAR10_CLASSES,
    'num_cases_per_batch': batch_size,
    'num_vis': 3072,
}
with open(os.path.join(dst_dir, 'batches.meta'), 'wb') as f:
    pickle.dump(meta, f)
print("Done! All files written.")
