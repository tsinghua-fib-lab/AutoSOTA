from core.eval.eval_mask import db_eval_iou, db_eval_boundary
from motion_seg_inference import (
    MotionSegmentationInference,
    refine_sam,
    DEFAULT_SAM2_CONFIG_PATH,
    DEFAULT_SAM2_CHECKPOINT_PATH,
)
import numpy as np
import torch
import re
from PIL import Image
from tqdm import tqdm
import os
import json
from pathlib import Path
import pandas as pd
import time
from sam2 import build_sam

def _natural_key(s: str):
    # Natural sort key, e.g. frame2 < frame10.
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


class DAVISEvaluator:
    """Evaluate motion segmentation on DAVIS dataset"""
    
    def __init__(
        self,
        model_path,
        output_base_dir,
        pi3_model_path=None,
        raft_model_path=None,
        sam2_config_path=None,
        sam2_checkpoint_path=None,
    ):
        """
        Args:
            model_path: Path to trained model
            output_base_dir: Base directory to save all results
        """
        self.inference = MotionSegmentationInference(
            model_path=model_path,
            pi3_model_path=pi3_model_path,
            raft_model_path=raft_model_path,
        )
        self.output_base_dir = output_base_dir
        self.sam2_config_path = sam2_config_path
        self.sam2_checkpoint_path = sam2_checkpoint_path
        os.makedirs(output_base_dir, exist_ok=True)
        
        self.all_sequence_metrics = []
    
    def evaluate_davis(self, image_root, annotation_root, sequence_length=16, 
                       use_sam_refine=True, davis='2016', max_frames=None):
        """
        Evaluate on DAVIS dataset
        
        Args:
            image_root: Path to JPEGImages/480p
            annotation_root: Path to Annotations/480p
            sequence_length: Frames to process at once
            use_sam_refine: Whether to use SAM2 refinement
            max_frames: Max frames per sequence (None for all)
        
        Returns:
            dataset_metrics: Dictionary with overall dataset metrics
        """
        
        # Find all sequences from image directory
        if davis == '2016':
            sequences = self._find_davis_sequences_16(image_root)
            print(f"Found {len(sequences)} sequences in DAVIS-2016 dataset")
        elif davis == '2017':
            sequences = self._find_davis_sequences_17(image_root)
            print(f"Found {len(sequences)} sequences in DAVIS-2017 dataset")
        elif davis == 'davis-all':
            sequences = self._find_davis_sequences(image_root)
            print(f"Found {len(sequences)} sequences in DAVIS dataset")
        elif davis == '2016-M':
            sequences = self._find_davis_2016_sequences(image_root)
            print(f"Found {len(sequences)} sequences in DAVIS-2016M dataset")
        elif davis == '2017-M':
            sequences = self._find_davis_2017_sequences(image_root)
            print(f"Found {len(sequences)} sequences in DAVIS2017-M dataset")
        elif davis == 'fbms':
            sequences = self._find_fbms_sequences(image_root)
            print(f"Found {len(sequences)} sequences in FBMS dataset")
        elif davis == 'segtrack':
            sequences = self._find_segtrack_sequences(image_root)
            print(f"Found {len(sequences)} sequences in segtrack dataset")
        
        for seq_name in tqdm(sequences, desc="Evaluating sequences"):
            # if seq_name == 'libby':
            #     continue
            print(f"\nProcessing sequence: {seq_name}")
            
            frames_dir = os.path.join(image_root, seq_name)
            gt_dir = os.path.join(annotation_root, seq_name)
            
            if not os.path.exists(frames_dir):
                print(f"  Warning: frames directory not found, skipping")
                continue
            
            if not os.path.exists(gt_dir):
                print(f"  Warning: GT directory not found, skipping")
                continue
            
            # Evaluate single sequence
            seq_output_dir = os.path.join(self.output_base_dir, seq_name)
            os.makedirs(seq_output_dir, exist_ok=True)
            
            if davis == 'fbms':
                seq_metrics = self._evaluate_single_sequence_fbms(
                    seq_name, frames_dir, gt_dir, seq_output_dir,
                    sequence_length, use_sam_refine, max_frames,
                    rgb_from_gt=True
                )
            elif davis in ['2016', '2017', 'davis-all', '2016-M', '2017-M', 'segtrack']:
                seq_metrics = self._evaluate_single_sequence(
                    seq_name, frames_dir, gt_dir, seq_output_dir,
                    sequence_length, use_sam_refine, max_frames
                )
            else:
                raise ValueError(f"Unknown DAVIS variant: {davis}")
            
            if seq_metrics is not None:
                self.all_sequence_metrics.append(seq_metrics)
        
        # Compute dataset-level metrics
        dataset_metrics = self._compute_dataset_metrics()
        
        # Save results
        self._save_results(dataset_metrics)
        
        return dataset_metrics
    
    def _find_davis_sequences_16(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        val_set = [
            'blackswan', 'bmx-trees', 'breakdance', 'camel',
            'car-roundabout', 'car-shadow', 'cows', 'dance-twirl',
            'dog', 'drift-chicane', 'drift-straight', 'goat',
            'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
            'paragliding-launch', 'parkour', 'scooter-black', 'soapbox'
        ]

        # code/SegAnyMo/data/DAVIS/ImageSets/2016/val.txt
        for seq_name in os.listdir(image_root):
            if seq_name not in val_set:
                continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    
    def _find_davis_sequences_17(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        val_set=['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
                 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog',
                 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat',
                 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf',
                 'lab-coat', 'libby', 'loading', 'mbike-trick', 'motocross-jump',
                 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
                 'shooting', 'soapbox']

        # code/SegAnyMo/data/DAVIS/ImageSets/2016/val.txt
        for seq_name in os.listdir(image_root):
            if seq_name not in val_set:
                continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    def _find_davis_sequences(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        for seq_name in os.listdir(image_root):
            # if seq_name not in val_set:
            #     continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    
    def _find_davis_2016_sequences(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        val_set=['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',
                 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
                 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby',
                 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black',
                 'soapbox']
        for seq_name in os.listdir(image_root):
            if seq_name not in val_set:
                continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    
    def _find_davis_2017_sequences(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        val_set=['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
                 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog',
                 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat',
                 'gold-fish', 'horsejump-high', 'india', 'judo', 'kite-surf',
                 'lab-coat', 'libby', 'loading', 'mbike-trick', 'motocross-jump',
                 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
                 'shooting', 'soapbox']
        # val_set = ['dogs-jump', 'gold-fish']
        
        for seq_name in os.listdir(image_root):
            if seq_name not in val_set:
                continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    
    def _find_fbms_sequences(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        
        for seq_name in os.listdir(image_root):
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
    
    def _find_segtrack_sequences(self, image_root):
        """Find all sequences in DAVIS dataset"""
        sequences = []
        val_set=['birdfall', 'drift']
        
        for seq_name in os.listdir(image_root):
            # if seq_name not in val_set:
            #     continue
            seq_path = os.path.join(image_root, seq_name)
            if os.path.isdir(seq_path):
                sequences.append(seq_name)
        
        return sorted(sequences)
   
    def _evaluate_single_sequence(self, seq_name, frames_dir, gt_dir, 
                          output_dir, sequence_length, 
                          use_sam_refine, max_frames, stride=4):
        """
        Evaluate a single video sequence with multi-group frame sampling
        
        Splits all frames into multiple groups with configurable stride, enabling 
        denser predictions while avoiding small motion intervals from consecutive frames.
        
        Args:
            stride: Stride for frame grouping (default: 4). Will split frames into
                   'stride' number of groups, each taking every stride-th frame.
        
        Returns:
            seq_metrics: Dictionary with sequence-level metrics
        """
        time1 = time.time()
        # Load GT masks first to get original size
        gt_masks = self._load_davis_gt_masks(gt_dir, None)
        
        if gt_masks is None or len(gt_masks) == 0:
            print(f"  Error: No GT masks found")
            return None
        
        gt_height, gt_width = gt_masks[0].shape
        print(f"  GT mask size: {gt_width}x{gt_height}")
        
        # Load all frame files (without downsampling)
        all_frame_files = sorted([f for f in os.listdir(frames_dir)
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if max_frames:
            all_frame_files = all_frame_files[:max_frames]
        
        if len(all_frame_files) == 0:
            print(f"  Error: No frames found")
            return None
        
        # Get original frame sizes
        original_frame_sizes = []
        for frame_file in all_frame_files:
            img = Image.open(os.path.join(frames_dir, frame_file))
            original_frame_sizes.append(img.size)  # (width, height)
        
        # Split into groups with configurable stride
        frame_groups = [all_frame_files[i::stride] for i in range(stride)]
        
        print(f"  Total frames: {len(all_frame_files)}")
        print(f"  Stride: {stride}, Frame group sizes: {[len(g) for g in frame_groups]}")
        
        all_motion_masks = [None] * len(all_frame_files)
        all_video_frames = [None] * len(all_frame_files)
        
        # Build predictor
        model_config = self.sam2_config_path or os.environ.get("SAM2_CONFIG_PATH", DEFAULT_SAM2_CONFIG_PATH)
        checkpoint = self.sam2_checkpoint_path or os.environ.get("SAM2_CHECKPOINT_PATH", DEFAULT_SAM2_CHECKPOINT_PATH)
    
        predictor = build_sam.build_sam2_video_predictor(
            model_config, checkpoint, device='cuda'
        )
        # Process each group independently for inference
        for group_idx, frame_group in enumerate(frame_groups):
            print(f"  Processing group {group_idx + 1}/{stride} ({len(frame_group)} frames)...")
            
            # Load frames for this group
            video_frames = []
            original_indices = []
            
            for frame_idx, frame_file in enumerate(frame_group):
                img = Image.open(os.path.join(frames_dir, frame_file)).convert('RGB')
                # Resize frame to match GT mask size
                img_resized = img.resize((gt_width, gt_height), Image.BILINEAR)
                video_frames.append(img_resized)
                # Store original index in the full sequence
                original_idx = frame_idx * stride + group_idx
                original_indices.append(original_idx)
                all_video_frames[original_idx] = img_resized
            
            # Process in chunks for inference
            group_motion_masks = [None] * len(video_frames)
            for chunk_start in tqdm(range(0, len(video_frames), sequence_length),
                                desc=f"  Group {group_idx + 1}", leave=False):
                chunk_end = min(chunk_start + sequence_length, len(video_frames))
                chunk_frames = video_frames[chunk_start:chunk_end]
                
                # Predict motion mask
                motion_masks = self.inference.predict_motion_mask(chunk_frames)

                # TTA: Flip augmentation
                chunk_frames_flipped = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in chunk_frames]
                motion_masks_flipped = self.inference.predict_motion_mask(chunk_frames_flipped)
                # Flip masks back (horizontal flip)
                motion_masks_flipped = motion_masks_flipped[:, :, ::-1]
                # Average with original predictions
                motion_masks = (motion_masks + motion_masks_flipped) / 2.0

                # Store predictions temporarily
                for local_idx, motion_mask in enumerate(motion_masks):
                    group_motion_masks[chunk_start + local_idx] = motion_mask
            
            # Store in global array using original indices
            for local_idx, motion_mask in enumerate(group_motion_masks):
                if motion_mask is not None:
                    all_motion_masks[original_indices[local_idx]] = motion_mask
        
        # Filter out None values
        all_video_frames = [f for f in all_video_frames if f is not None]
        all_motion_masks = [m for m in all_motion_masks if m is not None]
        all_motion_masks = np.array(all_motion_masks)

        if len(all_motion_masks) == 0:
            print(f"  Error: No predictions generated")
            return None
        
        print(f"  Generated predictions for {len(all_motion_masks)} frames")
        
        # Now do SAM2 refinement on all frames together
        if use_sam_refine:
            print(f"  Applying SAM2 refinement on full sequence...")
            all_motion_masks = self._refine_with_sam(all_video_frames, all_motion_masks, strategy=False, predictor=predictor)
        time2 = time.time()
        print(f"  Time for inference and refinement: {time2 - time1:.2f} seconds")
        
        # Evaluate all frames
        frame_metrics = self._evaluate_chunk(all_motion_masks, gt_masks, 0)
        
        # Save predictions with original frame sizes
        self._save_davis_predictions(
            all_motion_masks, 
            output_dir, 
            seq_name, 
            original_frame_sizes,
            frames_dir=frames_dir,
            all_frame_files=all_frame_files,
            save_overlay=True
        )
        
        # Compute sequence metrics
        seq_metrics = {
            'sequence_name': seq_name,
            'num_frames': len(frame_metrics),
            'frame_metrics': frame_metrics,
            'mean_iou': np.mean([m['iou'] for m in frame_metrics]),
            'JR': np.mean([m['JR'] for m in frame_metrics]),
            'std_iou': np.std([m['iou'] for m in frame_metrics]),
            'mean_f_score': np.mean([m['f_score'] for m in frame_metrics]),
            'std_f_score': np.std([m['f_score'] for m in frame_metrics]),
            'J & F': np.mean([(m['iou'] + m['f_score']) / 2.0 for m in frame_metrics])
        }
        
        print(f"  Sequence metrics - IoU: {seq_metrics['mean_iou']:.4f}, "
            f"  JR: {seq_metrics['JR']:.4f}, "
            f"  F-Score: {seq_metrics['mean_f_score']:.4f}, "
            f"  J & F: {seq_metrics['J & F']:.4f}")
        
        return seq_metrics
    

    def _evaluate_single_sequence_fbms(
        self, seq_name, frames_dir, gt_dir,
        output_dir, sequence_length,
        use_sam_refine, max_frames,
        rgb_from_gt: bool = False,                 # Use only RGB frames with matching GT masks
        rgb_exts=(".jpg", ".jpeg", ".png")         # Candidate RGB file extensions to search
    ):
        """
        rgb_from_gt=True: only use RGB frames that have matching GT masks.
        rgb_from_gt=False: use all RGB frames found under frames_dir.
        """

        # Load GT file names and build a lookup map first.
        gt_files = sorted(
            [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=_natural_key
        )
        if len(gt_files) == 0:
            print(f"  Error: No GT masks found")
            return None

        gt_mask_dict = {}
        for gt_file in gt_files:
            gt_path = os.path.join(gt_dir, gt_file)
            gt_mask = np.array(Image.open(gt_path).convert('L'))
            mask_name = os.path.splitext(gt_file)[0]
            gt_mask_dict[mask_name] = gt_mask

        gt_height, gt_width = next(iter(gt_mask_dict.values())).shape
        print(f"  GT mask size: {gt_width}x{gt_height}")

        # Select RGB frames either from GT names or from all frames in frames_dir.
        if rgb_from_gt:
            # Match RGB files in frames_dir using GT mask file stems.
            selected_frame_files = []
            missing_rgb = []
            for stem in sorted(gt_mask_dict.keys(), key=_natural_key):
                found = None
                for ext in rgb_exts:
                    cand = stem + ext
                    if os.path.exists(os.path.join(frames_dir, cand)):
                        found = cand
                        break
                    # Also try uppercase extensions to handle mixed file naming.
                    cand2 = stem + ext.upper()
                    if os.path.exists(os.path.join(frames_dir, cand2)):
                        found = cand2
                        break
                if found is None:
                    missing_rgb.append(stem)
                else:
                    selected_frame_files.append(found)

            all_frame_files = selected_frame_files
            if len(missing_rgb) > 0:
                print(f"  Warning: {len(missing_rgb)} GT masks have no matched RGB. e.g. {missing_rgb[:5]}")
        else:
            # Use all RGB frames under frames_dir.
            all_frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.lower().endswith(rgb_exts)],
                key=_natural_key
            )

        if max_frames:
            all_frame_files = all_frame_files[:max_frames]

        if len(all_frame_files) == 0:
            print(f"  Error: No frames found")
            return None

        print(f"  Total RGB frames used: {len(all_frame_files)}, GT masks: {len(gt_mask_dict)}")
        
        frame_groups = [all_frame_files]
        print(f"  Frame group sizes: {[len(g) for g in frame_groups]}")

        original_frame_sizes = []
        for frame_file in all_frame_files:
            img = Image.open(os.path.join(frames_dir, frame_file))
            original_frame_sizes.append(img.size)  # (width, height)

        # Store all predictions with frame names as keys
        all_motion_masks_dict = {}  # {frame_name: motion_mask}
        all_video_frames_dict = {}  # {frame_name: frame_image}

        # Build predictor
        model_config = self.sam2_config_path or os.environ.get("SAM2_CONFIG_PATH", DEFAULT_SAM2_CONFIG_PATH)
        checkpoint = self.sam2_checkpoint_path or os.environ.get("SAM2_CHECKPOINT_PATH", DEFAULT_SAM2_CHECKPOINT_PATH)
    
        predictor = build_sam.build_sam2_video_predictor(
            model_config, checkpoint, device='cuda'
        )
        
        # Process each group independently for inference
        for group_idx, frame_group in enumerate(frame_groups):
            print(f"  Processing group {group_idx + 1}/1 ({len(frame_group)} frames)...")
            
            # Load frames for this group
            video_frames = []
            frame_names = []
            
            for frame_file in frame_group:
                img = Image.open(os.path.join(frames_dir, frame_file)).convert('RGB')
                # Resize frame to match GT mask size
                img_resized = img.resize((gt_width, gt_height), Image.BILINEAR)
                video_frames.append(img_resized)
                
                frame_name = os.path.splitext(frame_file)[0]
                frame_names.append(frame_name)
                all_video_frames_dict[frame_name] = img_resized
            
            # Process in chunks for inference
            group_motion_masks = [None] * len(video_frames)
            for chunk_start in tqdm(range(0, len(video_frames), sequence_length),
                                desc=f"  Group {group_idx + 1}", leave=False):
                chunk_end = min(chunk_start + sequence_length, len(video_frames))
                chunk_frames = video_frames[chunk_start:chunk_end]
                
                # Predict motion mask
                motion_masks = self.inference.predict_motion_mask(chunk_frames)
                
                # Store predictions temporarily
                for local_idx, motion_mask in enumerate(motion_masks):
                    group_motion_masks[chunk_start + local_idx] = motion_mask
            
            # Store in global dict using frame names
            for local_idx, motion_mask in enumerate(group_motion_masks):
                if motion_mask is not None:
                    all_motion_masks_dict[frame_names[local_idx]] = motion_mask
        
        if len(all_motion_masks_dict) == 0:
            print(f"  Error: No predictions generated")
            return None
        
        print(f"  Generated predictions for {len(all_motion_masks_dict)} frames")
        
        # Prepare lists for SAM2 refinement (all frames)
        all_video_frames_list = []
        all_motion_masks_list = []
        frame_names_order = []
        
        for frame_name in sorted(all_motion_masks_dict.keys()):
            all_video_frames_list.append(all_video_frames_dict[frame_name])
            all_motion_masks_list.append(all_motion_masks_dict[frame_name])
            frame_names_order.append(frame_name)
        
        all_motion_masks_array = np.array(all_motion_masks_list)
        
        # SAM2 refinement on ALL frames
        if use_sam_refine:
            print(f"  Applying SAM2 refinement on all {len(all_motion_masks_array)} frames...")
            all_motion_masks_array = self._refine_with_sam(
                all_video_frames_list, all_motion_masks_array, predictor=predictor
            )
        
        # Evaluate only frames with matching GT masks
        frame_metrics = []
        gt_matched_indices = []
        
        for idx, frame_name in enumerate(frame_names_order):
            if frame_name in gt_mask_dict:
                motion_mask = all_motion_masks_array[idx]
                if motion_mask.ndim == 3:
                    motion_mask = motion_mask[0]
                
                gt_mask = gt_mask_dict[frame_name]
                
                # Evaluate
                motion_masks_chunk = np.array([motion_mask])
                gt_masks_chunk = np.array([gt_mask])
                metric = self._evaluate_chunk(motion_masks_chunk, gt_masks_chunk, idx)
                frame_metrics.append(metric[0])
                gt_matched_indices.append(idx)
        
        if len(frame_metrics) == 0:
            print(f"  Error: No frames matched with GT masks")
            return None
        
        print(f"  Evaluated {len(frame_metrics)} frames (matched with GT)")
        
        # Save all predictions
        self._save_davis_predictions(
            all_motion_masks_array,
            output_dir,
            seq_name,
            original_frame_sizes,
            frames_dir=frames_dir,      # Original frame directory
            all_frame_files=all_frame_files,  # Full frame filename list
            save_overlay=True           # Save overlay visualizations
        )
        seq_metrics = {
            'sequence_name': seq_name,
            'num_frames': len(frame_metrics),
            'frame_metrics': frame_metrics,
            'JR': np.mean([m['JR'] for m in frame_metrics]),
            'mean_iou': np.mean([m['iou'] for m in frame_metrics]),
            'std_iou': np.std([m['iou'] for m in frame_metrics]),
            'mean_f_score': np.mean([m['f_score'] for m in frame_metrics]),
            'std_f_score': np.std([m['f_score'] for m in frame_metrics]),
            'J & F': np.mean([(m['iou'] + m['f_score']) / 2.0 for m in frame_metrics])
        }
        
        print(f"  Sequence metrics - IoU: {seq_metrics['mean_iou']:.4f}, "
            f"F-Score: {seq_metrics['mean_f_score']:.4f}, "
            f"J & F: {seq_metrics['J & F']:.4f}")
        
        return seq_metrics

    def _load_davis_gt_masks(self, gt_dir, num_frames=None):
        """Load ground truth masks from DAVIS annotation directory"""
        mask_files = sorted([f for f in os.listdir(gt_dir)
                            if f.endswith(('.png', '.jpg'))])
        
        if num_frames:
            mask_files = mask_files[:num_frames]
        
        gt_masks = []
        for mask_file in mask_files:
            try:
                mask = np.array(Image.open(
                    os.path.join(gt_dir, mask_file)).convert('L'))
                
                # DAVIS uses 0 for background, 255 for foreground
                # Convert to binary: 0 and 1
                mask = (mask > 0).astype(np.uint8)
                gt_masks.append(mask)
            except Exception as e:
                print(f"    Warning: Failed to load {mask_file}: {e}")
                continue
        
        return np.array(gt_masks) if gt_masks else None
    
    def _refine_with_sam(self, chunk_frames, motion_masks, strategy=True, predictor=None):
        """Apply SAM2 refinement to masks"""
        assert predictor is not None, "predictor must be provided (build it once outside and pass it in)."

        # motion_masks: numpy [T,H,W]
        T, mask_h, mask_w = motion_masks.shape

        # ---- frames -> tensor [T,3,H,W] on CPU (only used for SAM preprocessing / temporary files) ----
        frame_arrays = []
        for pil_img in chunk_frames:
            pil_img_resized = pil_img.resize((mask_w, mask_h), Image.BILINEAR)
            frame_array = np.array(pil_img_resized).astype(np.float32) / 255.0
            frame_arrays.append(frame_array)

        frame_arrays = np.stack(frame_arrays, axis=0)          # [T,H,W,3]
        frame_tensors = torch.from_numpy(frame_arrays).float() # CPU
        frame_tensors = frame_tensors.permute(0, 3, 1, 2)      # [T,3,H,W]

        # refine_sam writes CUDA tensors internally, so keep the output buffer on CUDA.
        refined_masks = torch.zeros((T, mask_h, mask_w), dtype=torch.float32, device="cuda")

        # Keep mask_list as numpy; preprocess_mask in refine_sam handles conversion.
        mask_list = motion_masks

        refine_sam(
            frame_tensors=frame_tensors,
            mask_list=mask_list,
            p_masks_sam=refined_masks,
            offset=0,
            predictor=predictor
        )

        return refined_masks.detach().cpu().numpy()
    
    def _evaluate_chunk(self, pred_masks, gt_masks, start_idx):
        """Evaluate a chunk of frames"""
        chunk_metrics = []
        
        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            frame_idx = start_idx + i
            
            # Resize pred_mask to match gt_mask size
            if pred_mask.shape != gt_mask.shape:
                pred_mask = self._resize_mask(pred_mask, gt_mask.shape)
            
            pred_binary = (pred_mask > 0.1).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)
            # import pdb;pdb.set_trace()
            # Compute IoU
            iou = db_eval_iou(gt_binary, pred_binary)
            
            # Compute F-Score
            f_score = db_eval_boundary(gt_binary, pred_binary, bound_th=0.008)
            # f_score = 0.0
            
            chunk_metrics.append({
                'frame_idx': frame_idx,
                'iou': iou,
                'f_score': f_score,
                'J & F': (iou + f_score) / 2.0
            })
        ious = np.array([x['iou'] for x in chunk_metrics])
        JR = (ious > 0.5).sum() / len(ious)
        # import pdb;pdb.set_trace()
        for x in chunk_metrics:
            x['JR'] = JR
        return chunk_metrics
    
    def _resize_mask(self, mask, target_shape):
        """Resize mask to target shape using PIL"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img_resized = mask_img.resize(
            (target_shape[1], target_shape[0]), 
            Image.BILINEAR
        )
        return np.array(mask_img_resized) / 255.0
    


    def _save_davis_predictions(self, all_motion_masks, output_dir, seq_name, 
                            original_frame_sizes, frames_dir=None, 
                            all_frame_files=None, save_overlay=True):
        """
        Save predictions as PNG masks (DAVIS format) at original frame sizes
        Optionally save overlay visualization on original images
        
        Args:
            all_motion_masks: numpy array of shape (N, H, W) with values in [0, 1]
            output_dir: output directory
            seq_name: sequence name
            original_frame_sizes: list of (width, height) tuples for each frame
            frames_dir: directory containing original frames (for overlay visualization)
            all_frame_files: list of frame filenames
            save_overlay: whether to save overlay visualization
        """
        # Save masks
        pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        
        # Save overlays
        if save_overlay and frames_dir is not None and all_frame_files is not None:
            overlay_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(overlay_dir, exist_ok=True)
        
        for i, (mask, original_size) in enumerate(zip(all_motion_masks, original_frame_sizes)):
            # Convert to 0-255 range
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Create PIL Image
            mask_img = Image.fromarray(mask_uint8, mode='L')
            
            # Resize to original frame size if different
            original_width, original_height = original_size
            if mask_img.size != original_size:
                mask_img = mask_img.resize(original_size, Image.NEAREST)
            
            # Save mask as PNG
            mask_img.save(os.path.join(pred_dir, f'{i:05d}.png'))
            
            # Save overlay visualization
            if save_overlay and frames_dir is not None and all_frame_files is not None:
                if i < len(all_frame_files):
                    # Load original frame
                    frame_path = os.path.join(frames_dir, all_frame_files[i])
                    original_frame = Image.open(frame_path).convert('RGB')
                    
                    # Create overlay
                    overlay = self._create_mask_overlay(original_frame, mask_img)
                    
                    # Save overlay
                    overlay.save(os.path.join(overlay_dir, f'{i:05d}.png'))
        
        print(f"  Saved {len(all_motion_masks)} masks to {pred_dir}")
        if save_overlay:
            print(f"  Saved {len(all_motion_masks)} overlays to {overlay_dir}")


    def _create_mask_overlay(self, image, mask, alpha=0.5, color=(255, 0, 0)):
        """
        Create visualization overlay of mask on image
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L mode, grayscale)
            alpha: transparency of overlay (0-1)
            color: RGB tuple for mask color
        
        Returns:
            PIL Image with overlay
        """
        # Ensure same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
        
        # Convert to numpy arrays
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create colored mask
        colored_mask = np.zeros_like(img_array)
        colored_mask[mask_array > 127] = color  # threshold at 127
        
        # Blend image and mask
        mask_binary = (mask_array > 127).astype(np.float32)
        overlay_array = img_array.copy()
        
        for c in range(3):
            overlay_array[:, :, c] = (
                img_array[:, :, c] * (1 - alpha * mask_binary) +
                colored_mask[:, :, c] * alpha * mask_binary
            ).astype(np.uint8)
        
        # Add contour for better visualization
        # overlay_with_contour = self._add_mask_contour(overlay_array, mask_array)
        
        # return Image.fromarray(overlay_with_contour)
        return Image.fromarray(overlay_array)


    def _add_mask_contour(self, image_array, mask_array, contour_color=(255, 255, 0), 
                        thickness=2):
        """
        Add contour around mask boundary
        
        Args:
            image_array: numpy array (H, W, 3)
            mask_array: numpy array (H, W)
            contour_color: RGB tuple for contour
            thickness: contour thickness in pixels
        
        Returns:
            numpy array with contour added
        """
        import cv2
        
        result = image_array.copy()
        
        # Find contours
        binary_mask = (mask_array > 127).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(result, contours, -1, contour_color, thickness)
        
        return result


    def _compute_dataset_metrics(self):
        """Compute aggregate metrics across all sequences"""
        all_ious = []
        all_JRs = []
        all_f_scores = []
        
        for seq_metrics in self.all_sequence_metrics:
            if seq_metrics is None:
                continue
            
            ious = [m['iou'] for m in seq_metrics['frame_metrics']]
            JRs = [m['JR'] for m in seq_metrics['frame_metrics']]
            f_scores = [m['f_score'] for m in seq_metrics['frame_metrics']]
            
            all_ious.extend(ious)
            all_JRs.extend(JRs)
            all_f_scores.extend(f_scores)
        
        dataset_metrics = {
            'num_sequences': len(self.all_sequence_metrics),
            'total_frames': len(all_ious),
            'JR': np.mean(all_JRs),
            'mean_iou': np.mean(all_ious),
            'std_iou': np.std(all_ious),
            'min_iou': np.min(all_ious),
            'max_iou': np.max(all_ious),
            'mean_f_score': np.mean(all_f_scores),
            'std_f_score': np.std(all_f_scores),
            'min_f_score': np.min(all_f_scores),
            'max_f_score': np.max(all_f_scores),
            'sequence_metrics': self.all_sequence_metrics
        }
        
        return dataset_metrics
    
    def _save_results(self, dataset_metrics):
        """Save evaluation results to files"""
        
        # Save JSON summary
        summary_file = os.path.join(self.output_base_dir, 'dataset_metrics.json')
        
        # Convert numpy types for JSON serialization
        summary_dict = {
            'num_sequences': int(dataset_metrics['num_sequences']),
            'total_frames': int(dataset_metrics['total_frames']),
            'mean_iou': float(dataset_metrics['mean_iou']),
            'JR': float(dataset_metrics['JR']),
            'std_iou': float(dataset_metrics['std_iou']),
            'min_iou': float(dataset_metrics['min_iou']),
            'max_iou': float(dataset_metrics['max_iou']),
            'mean_f_score': float(dataset_metrics['mean_f_score']),
            'std_f_score': float(dataset_metrics['std_f_score']),
            'min_f_score': float(dataset_metrics['min_f_score']),
            'max_f_score': float(dataset_metrics['max_f_score']),
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        # Save per-sequence CSV
        seq_data = []
        for seq_metrics in dataset_metrics['sequence_metrics']:
            seq_data.append({
                'Sequence': seq_metrics['sequence_name'],
                'Frames': seq_metrics['num_frames'],
                'Mean_IoU': seq_metrics['mean_iou'],
                'JR': seq_metrics['JR'],
                'Std_IoU': seq_metrics['std_iou'],
                'Mean_F_Score': seq_metrics['mean_f_score'],
                'Std_F_Score': seq_metrics['std_f_score'],
            })
        
        df = pd.DataFrame(seq_data)
        csv_file = os.path.join(self.output_base_dir, 'sequence_metrics.csv')
        df.to_csv(csv_file, index=False)
        
        # Print summary
        self._print_summary(dataset_metrics)
    
    def _print_summary(self, dataset_metrics):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("DAVIS DATASET EVALUATION SUMMARY")
        print("="*60)
        print(f"Sequences: {dataset_metrics['num_sequences']}")
        print(f"Total Frames: {dataset_metrics['total_frames']}")
        print(f"\nIoU (Jaccard Index):")
        print(f"  Mean:  {dataset_metrics['mean_iou']:.4f}")
        print(f"  JR:  {dataset_metrics['JR']:.4f}")
        print(f"  Std:   {dataset_metrics['std_iou']:.4f}")
        print(f"  Range: [{dataset_metrics['min_iou']:.4f}, "
              f"{dataset_metrics['max_iou']:.4f}]")
        print(f"\nF-Score (Boundary):")
        print(f"  Mean:  {dataset_metrics['mean_f_score']:.4f}")
        print(f"  Std:   {dataset_metrics['std_f_score']:.4f}")
        print(f"  Range: [{dataset_metrics['min_f_score']:.4f}, "
              f"{dataset_metrics['max_f_score']:.4f}]")
        print("="*60 + "\n")


# Legacy local example block removed to avoid hard-coded paths in the open-source version.

import argparse
def main():
    parser = argparse.ArgumentParser(description="Evaluate motion segmentation model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pth file")
    parser.add_argument("--pi3_model_path", type=str, default=None,
                        help="Path to pi3 .safetensors checkpoint. If omitted, use PI3_MODEL_PATH env var.")
    parser.add_argument("--raft_model_path", type=str, default=None,
                        help="Path to RAFT checkpoint. If omitted, use RAFT_MODEL_PATH env var.")
    parser.add_argument("--sam2_config_path", type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help="Path to SAM2 config yaml. If omitted, use SAM2_CONFIG_PATH env var.")
    parser.add_argument("--sam2_checkpoint_path", type=str, default=None,
                        help="Path to SAM2 checkpoint. If omitted, use SAM2_CHECKPOINT_PATH env var.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--image_root", type=str, required=True, help="Path to image directory")
    parser.add_argument("--annotation_root", type=str, required=True, help="Path to annotation directory")
    parser.add_argument("--sequence_length", type=int, default=32, help="Sequence length for evaluation")
    parser.add_argument("--use_sam_refine", type=lambda x: x.lower() in ["true", "1", "yes"], default=False,
                        help="Whether to apply SAM refinement")
    parser.add_argument("--davis", type=str, default="2017-M",
                        help="Dataset name, e.g., '2017-M', '2016', 'fbms', 'segtrack'")

    args = parser.parse_args()

    print("\n========== Evaluation Configuration ==========")
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("==============================================\n")

    # Initialize evaluator
    evaluator = DAVISEvaluator(
        model_path=args.model_path,
        output_base_dir=args.output_dir,
        pi3_model_path=args.pi3_model_path,
        raft_model_path=args.raft_model_path,
        sam2_config_path=args.sam2_config_path,
        sam2_checkpoint_path=args.sam2_checkpoint_path,
    )

    # Run evaluation
    dataset_metrics = evaluator.evaluate_davis(
        image_root=args.image_root,
        annotation_root=args.annotation_root,
        sequence_length=args.sequence_length,
        use_sam_refine=args.use_sam_refine,
        davis=args.davis
    )

    print("\n========== Evaluation Results ==========")
    print(f"Mean IoU (J):       {dataset_metrics.get('mean_iou', 0):.4f}")
    print(f"Mean F-Score (F):   {dataset_metrics.get('mean_f_score', 0):.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
