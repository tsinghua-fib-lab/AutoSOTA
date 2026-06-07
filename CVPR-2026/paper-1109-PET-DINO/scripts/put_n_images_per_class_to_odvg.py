import json
import random
import shutil
import os
from collections import defaultdict
import argparse


def extract_images_per_class_to_odvg(json_path, image_prefix, images_per_class, output_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from category ID to image IDs, and record instance count per image
    category_to_images = defaultdict(lambda: defaultdict(int))
    
    # Iterate over all annotations to build category-to-image mapping and count instances
    for ann in coco_data['annotations']:
        category_to_images[ann['category_id']][ann['image_id']] += 1

    # Create ODVG format annotation list
    with open(output_path, 'w') as output_f:
    
        # Process images for each category
        for category in coco_data['categories']:
            category_id = category['id']
            category_name = category['name']
            
            # Get all image IDs and their instance counts for this category
            image_counts = category_to_images[category_id]
            
            # Sort image IDs by instance count
            sorted_images = sorted(image_counts.items(), key=lambda x: x[1], reverse=True)

            # Remove non-existent images, encountered in odinw35-plantdoc dataset
            for image_id, instance_count in sorted_images:
                image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
                if 'file_name' not in image_info:   # lvis-v1
                    image_info['file_name'] = image_info['coco_url'].replace('http://images.cocodataset.org/', '')
                src_path = os.path.join(image_prefix, image_info['file_name'])
                if not os.path.exists(src_path):
                    print(f'missing {src_path} for category {category_name}')
                    sorted_images.remove((image_id, instance_count))
            
            # If image count is less than specified, use all
            num_to_select = min(images_per_class, len(sorted_images))
            
            # Select top N images with most instances 
            # selected_images = sorted_images[:num_to_select]

            # Randomly select N images instead of selecting those with most instances
            selected_images = random.sample(sorted_images, num_to_select) 
            
            # Copy selected images to corresponding category directory
            for image_id, instance_count in selected_images:
                # Find the corresponding image filename
                image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
                if 'file_name' not in image_info:   # lvis-v1
                    image_info['file_name'] = image_info['coco_url'].replace('http://images.cocodataset.org/', '')
                src_path = os.path.join(image_prefix, image_info['file_name'])

                # Get all annotations for this image
                image_annotations = [
                    ann for ann in coco_data['annotations'] 
                    if ann['image_id'] == image_id and ann['category_id'] == category_id
                ]

                instances = []
                for ann in image_annotations:
                    x, y, w, h = ann['bbox']    # [x, y, width, height]
                    bbox = [x, y, x + w, y + h]
                    instance = {
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'label': ann['category_id'],
                        'category': category_name,
                    }
                    instances.append(instance)      
                
                odvg_data = {
                    'filename': image_info['file_name'],    # src_path 
                    'height': image_info['height'],
                    'width': image_info['width'],
                    'detection': {'instances': instances}
                }

                # Dump each entry to file one by one
                json.dump(odvg_data, output_f)
                output_f.write('\n')
                
                print(f"{src_path} (instance count: {instance_count})")
                
            print(f"Processed category {category_name}: extracted {num_to_select} images")
        
        print(f"Processed {len(coco_data['categories'])} categories in total")
    
    output_f.close()

    print(f"Saved to {output_path}")


"""
# coco
python3 ./scripts/put_n_images_per_class_to_odvg.py --json_path data/coco/annotations/instances_train2017.json \
    --image_prefix data/coco/train2017 --output_path ./data_odvg_16/cocoTrain_odvg_16.json --number 16
# lvis
python3 ./scripts/put_n_images_per_class_to_odvg.py --json_path data/lvis/annotations/lvis_v1_train.json \
    --image_prefix data/lvis --output_path ./data_odvg_16/lvisTrain_odvg_16.json --number 16
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser('extract images per class to odvg.', add_help=True)
    parser.add_argument('--json_path', type=str, required=True, help='input json file name')
    parser.add_argument(
        '--image_prefix', '-o', type=str, required=True, help='output json file name')
    parser.add_argument(
        '--number', '-n', type=int, required=True, help='output json file name')
    parser.add_argument(
        '--output_path',
        required=True,
        type=str,
    )
    args = parser.parse_args()

    extract_images_per_class_to_odvg(args.json_path, args.image_prefix, args.number, args.output_path)
