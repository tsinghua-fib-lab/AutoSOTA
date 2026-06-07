
import json
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional
from pycocotools import mask as mask_utils

from tfd_utils.random_access import TFRecordRandomAccess

class PackedVRTEvalDataset:
    """Loader for packed VRT evaluation dataset"""
    
    def __init__(self, tfrecord_path: str):
        """
        Initialize the packed dataset loader
        
        Args:
            tfrecord_path: Path to the packed TFRecord file
        """
        self.tfrecord_path = tfrecord_path
        self.reader = TFRecordRandomAccess(tfrecord_path)
        self.keys = self.reader.get_keys()
        
    def get_sample(self, key: str) -> Optional[Dict]:
        """
        Get a sample by key
        
        Args:
            key: The sample key
            
        Returns:
            Dictionary containing all sample data
        """
        if key not in self.keys:
            return None
            
        try:
            tf_feat = self.reader[key].features.feature
            
            # 1. Image
            img_bytes = tf_feat['image'].bytes_list.value[0]
            image_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            # 2. Objects Info
            objects_info_json = tf_feat['objects_info_json'].bytes_list.value[0].decode('utf-8')
            objects_info_raw = json.loads(objects_info_json)
            
            objects_info = {}
            for obj_id, info in objects_info_raw.items():
                mask = mask_utils.decode(info['mask'])
                objects_info[int(obj_id)] = {
                    'mask': mask,
                    'original_id': info.get('original_id', ''),
                    'caption': info.get('caption', '')
                }
                
            # 3. Other fields
            question = tf_feat['question'].bytes_list.value[0].decode('utf-8')
            human_labeled_r_objs = list(tf_feat['human_labeled_r_objs'].int64_list.value)
            human_labeled_a_objs = list(tf_feat['human_labeled_a_objs'].int64_list.value)
            human_confidence = tf_feat['human_confidence'].int64_list.value[0]
            
            class_ids = [v.decode('utf-8') for v in tf_feat['class_ids'].bytes_list.value]
            
            human_reasoning = tf_feat['human_reasoning'].bytes_list.value[0].decode('utf-8') if 'human_reasoning' in tf_feat else ''
            human_answer_caption = tf_feat['human_answer_caption'].bytes_list.value[0].decode('utf-8') if 'human_answer_caption' in tf_feat else ''
            
            return {
                'key': key,
                'image': image_pil,
                'question': question,
                'objects_info': objects_info,
                'human_labeled_r_objs': human_labeled_r_objs,
                'human_labeled_a_objs': human_labeled_a_objs,
                'human_confidence': human_confidence,
                'class_ids': class_ids,
                'human_reasoning': human_reasoning,
                'human_answer_caption': human_answer_caption,
            }
            
        except Exception as e:
            print(f"Error loading data for key {key}: {e}")
            return None

    def get_evaluation_samples(self) -> List[Dict]:
        """Get all samples in the dataset"""
        samples = []
        for key in self.keys:
            sample = self.get_sample(key)
            if sample:
                samples.append(sample)
        return samples
        
    def get_all_keys(self) -> List[str]:
        """Get all available keys"""
        return self.keys

if __name__ == "__main__":
    # Test the dataloader
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', default='packed/vrt_eval.tfrecord')
    args = parser.parse_args()
    
    dataset = PackedVRTEvalDataset(args.tfrecord_path)
    print(f"Loaded dataset with {len(dataset.get_all_keys())} samples")
    
    samples = dataset.get_evaluation_samples()
    print(f"Retrieved {len(samples)} samples")
    
    if len(samples) > 0:
        sample = samples[0]
        print("Sample 0 keys:", sample.keys())
        print("Question:", sample['question'])
        print("Class IDs:", sample['class_ids'])
        print("Num objects:", len(sample['objects_info']))
