"""Generate edge maps from GT masks for training."""
import os
import cv2
import numpy as np
import argparse

def generate_edge(gt_path, edge_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print("Warning: cannot read", gt_path)
        return False
    # Canny edge detection
    edges = cv2.Canny(gt, 50, 150)
    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(edge_path, edges)
    return True

def main(gt_dir, edge_dir, num_workers=8):
    os.makedirs(edge_dir, exist_ok=True)
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith((".png", ".jpg"))])
    print("Found %d GT files" % len(gt_files))
    
    from concurrent.futures import ThreadPoolExecutor
    def process(f):
        gt_path = os.path.join(gt_dir, f)
        edge_name = os.path.splitext(f)[0] + ".png"
        edge_path = os.path.join(edge_dir, edge_name)
        if os.path.exists(edge_path):
            return (f, True, 0)
        ok = generate_edge(gt_path, edge_path)
        return (f, ok, 0 if ok else 1)
    
    # Process sequentially for simplicity
    for i, f in enumerate(gt_files):
        gt_path = os.path.join(gt_dir, f)
        edge_name = os.path.splitext(f)[0] + ".png"
        edge_path = os.path.join(edge_dir, edge_name)
        if os.path.exists(edge_path):
            if (i+1) % 500 == 0:
                print("  %d/%d (skipped existing)" % (i+1, len(gt_files)))
            continue
        ok = generate_edge(gt_path, edge_path)
        if (i+1) % 500 == 0:
            print("  %d/%d" % (i+1, len(gt_files)))
    
    generated = len([f for f in os.listdir(edge_dir) if f.endswith(".png")])
    print("Done! %d edge maps in %s" % (generated, edge_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--edge_dir", required=True)
    args = parser.parse_args()
    main(args.gt_dir, args.edge_dir)
