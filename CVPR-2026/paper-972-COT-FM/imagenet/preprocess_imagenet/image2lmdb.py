import lmdb
import os
from tqdm import tqdm
import pickle

def create_lmdb_from_imagenet(imagenet_root, lmdb_path, map_size=549755813888):
    """
    Convert ImageNet dataset to LMDB format
    
    Args:
        imagenet_root: ImageNet root directory (should contain class folders)
        lmdb_path: Output LMDB database path
        map_size: LMDB max size (default 512GB)
    """
    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    # Collect all image paths and labels
    image_data = []
    
    # Get sorted class names for consistent labeling
    class_names = sorted([d for d in os.listdir(imagenet_root) 
                         if os.path.isdir(os.path.join(imagenet_root, d))])
    
    # Process each class folder
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(imagenet_root, class_name)
        print(f"Processing class {class_idx}: {class_name}")
        
        # Add all images in this class
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, img_name)
                image_data.append((img_path, class_idx))
    
    print(f"Found {len(image_data)} images, starting conversion...")
    
    # Write to LMDB
    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(tqdm(image_data, desc="Converting")):
            try:
                # Read image binary data
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # Create data entry
                data_entry = {
                    'image': img_data,
                    'label': label
                }
                
                # Serialize data
                serialized_data = pickle.dumps(data_entry)
                
                # Use simple index as key
                key = f'{idx}'.encode()
                txn.put(key, serialized_data)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        # Save metadata
        txn.put('num_samples'.encode(), str(len(image_data)).encode())
        txn.put('num_classes'.encode(), str(len(class_names)).encode())
        txn.put('class_names'.encode(), pickle.dumps(class_names))
    
    env.close()
    print(f"LMDB database created: {lmdb_path}")
    print(f"Total images: {len(image_data)}")
    print(f"Total classes: {len(class_names)}")

if __name__ == "__main__":
    imagenet_root = "/path/to/imagenet/train"  # Change to your ImageNet path
    lmdb_path = "/path/to/imagenet_train.lmdb"  # Change to your output path
    
    create_lmdb_from_imagenet(imagenet_root, lmdb_path)