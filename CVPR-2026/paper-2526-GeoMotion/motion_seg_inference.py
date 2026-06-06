import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from glob import glob
import tempfile
import shutil
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from code.SegAnyMo import inference
from tqdm import tqdm
from pi3.utils.geometry import process_video_with_improved_sliding_window
from sam2 import build_sam

DEFAULT_RAFT_MODEL_PATH = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DEFAULT_SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT_PATH = "./sam2-main/checkpoints/sam2.1_hiera_large.pt"


def initialize_raft_model(device, raft_model_path=None):
    """Initialize the RAFT optical-flow model."""
    global RAFT_MODEL, RAFT_TRANSFORMS
    if raft_model_path is None:
        raft_model_path = os.environ.get("RAFT_MODEL_PATH", DEFAULT_RAFT_MODEL_PATH)
    
    # Check for torchvision and optical flow models
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        print("Torchvision optical flow models not found (requires torchvision >= 0.13).")
        print("Will fallback to OpenCV optical flow.")
        return False
        
    try:
        # If a specific path is provided, load from it
        if raft_model_path and os.path.exists(raft_model_path):
            print(f"Loading RAFT model from: {raft_model_path}")
            RAFT_MODEL = raft_large(weights=None, progress=False)
            state_dict = torch.load(raft_model_path, map_location=device)
            RAFT_MODEL.load_state_dict(state_dict)
            # When loading custom weights, transforms are often not included, so we create a basic one
            RAFT_TRANSFORMS = lambda img1, img2: (img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)
        else:
            # Otherwise, load the default pre-trained model from torchvision
            print("Loading RAFT model with default torchvision weights...")
            weights = Raft_Large_Weights.DEFAULT
            RAFT_MODEL = raft_large(weights=weights, progress=False)
            RAFT_TRANSFORMS = weights.transforms()
        
        RAFT_MODEL = RAFT_MODEL.to(device).eval()
        print("RAFT model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize RAFT model: {e}")
        print("Will fallback to OpenCV optical flow.")
        RAFT_MODEL = None # Ensure model is None on failure
        return False

def compute_optical_flow_raft(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    使用RAFT计算光流 (输入为 [0,1] 范围的Tensor)
    """
    global RAFT_MODEL, RAFT_TRANSFORMS
    import torchvision.transforms.functional as F
    
    # This function assumes RAFT_MODEL is not None
    original_H, original_W = frame1.shape[1], frame1.shape[2]
    
    # 确保输入是batch格式 (1, 3, H, W)
    if frame1.dim() == 3: frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3: frame2 = frame2.unsqueeze(0)
    
    # RAFT预处理：调整尺寸确保能被8整除
    H, W = frame1.shape[2], frame1.shape[3]
    new_H = ((H + 7) // 8) * 8
    new_W = ((W + 7) // 8) * 8
    
    img1_resized = F.resize(frame1, size=[new_H, new_W], antialias=False)
    img2_resized = F.resize(frame2, size=[new_H, new_W], antialias=False)
    
    # 应用模型特定的变换
    img1_processed, img2_processed = RAFT_TRANSFORMS(img1_resized, img2_resized)
    
    # RAFT推理
    with torch.no_grad():
        flow_predictions = RAFT_MODEL(img1_processed, img2_processed)
        flow_pred = flow_predictions[-1] # 取最后一个（最精细的）预测
    
    # 调整回原始尺寸
    if flow_pred.shape[2] != original_H or flow_pred.shape[3] != original_W:
        # 使用bilinear插值调整光流场大小
        # flow_pred = F.interpolate(flow_pred, size=(original_H, original_W), mode='bilinear', align_corners=False)
        flow_pred = F.resize(flow_pred, size=[original_H, original_W], antialias=False)
                
        # 相应地缩放光流值
        scale_W = original_W / new_W
        scale_H = original_H / new_H
        flow_pred[:, 0] *= scale_W
        flow_pred[:, 1] *= scale_H

    # 返回 (2, H, W)
    return flow_pred.squeeze(0)

def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    将ImageNet标准化的tensor恢复到[0,1]范围
    Args:
        tensor: [..., C, H, W] 标准化后的tensor
    Returns:
        denormalized tensor in [0,1] range
    """
    # ImageNet标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    
    # 反标准化：x = x_norm * std + mean
    denormalized = tensor * std + mean
    
    # 确保在[0,1]范围内
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    return denormalized

def compute_optical_flow_magnitude_batch(video_tensor: torch.Tensor, device, 
                                        bidirectional: bool = True,
                                        fusion_method: str = 'max',
                                        return_flow: bool = False):
    """
    为整个视频批次计算光流（支持返回向量）
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        bidirectional: 是否使用双向光流
        fusion_method: 双向光流融合方法
        return_flow: 是否返回光流向量（NEW!）
    
    Returns:
        如果 return_flow=False:
            flow_magnitudes: [B, N, H, W]
        如果 return_flow=True:
            (flow_vectors, flow_magnitudes)
            flow_vectors: [B, N, 2, H, W] 或 [B, N, H, W, 2]
            flow_magnitudes: [B, N, H, W]
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized, using zero flow")
        B, N, _, H, W = video_tensor.shape
        flow_magnitudes = torch.zeros(B, N, H, W, device=device)
        if return_flow:
            flow_vectors = torch.zeros(B, N, 2, H, W, device=device)
            return flow_vectors, flow_magnitudes
        return flow_magnitudes
    
    B, N, C, H, W = video_tensor.shape
    flow_magnitudes = torch.zeros(B, N, H, W, device=device)
    flow_vectors = torch.zeros(B, N, 2, H, W, device=device) if return_flow else None
    
    # 反标准化到[0,1]范围
    video_denorm = denormalize_imagenet(video_tensor)
    
    if not bidirectional:
        for b in range(B):
            for t in range(1, N):
                frame1 = video_denorm[b, t-1]
                frame2 = video_denorm[b, t]
                
                flow = compute_optical_flow_raft(frame1, frame2)  # [2, H, W]
                flow_magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
                
                flow_magnitudes[b, t] = flow_magnitude
                if return_flow:
                    flow_vectors[b, t] = flow  # [2, H, W]
    else:
        # 双向光流
        for b in range(B):
            forward_flows = []
            backward_flows = []
            forward_flow_vecs = [] if return_flow else None
            backward_flow_vecs = [] if return_flow else None
            
            for t in range(N-1):
                frame1 = video_denorm[b, t]
                frame2 = video_denorm[b, t+1]
                
                # 前向光流
                forward_flow = compute_optical_flow_raft(frame1, frame2)
                forward_magnitude = torch.sqrt(forward_flow[0]**2 + forward_flow[1]**2)
                forward_flows.append(forward_magnitude)
                if return_flow:
                    forward_flow_vecs.append(forward_flow)
                
                # 后向光流
                backward_flow = compute_optical_flow_raft(frame2, frame1)
                backward_magnitude = torch.sqrt(backward_flow[0]**2 + backward_flow[1]**2)
                backward_flows.append(backward_magnitude)
                if return_flow:
                    backward_flow_vecs.append(backward_flow)
            
            # 为每一帧分配光流
            for t in range(N):
                flow_values = []
                flow_vecs = [] if return_flow else None
                
                if t > 0:
                    flow_values.append(backward_flows[t-1])
                    if return_flow:
                        flow_vecs.append(backward_flow_vecs[t-1])
                
                if t < N-1:
                    flow_values.append(forward_flows[t])
                    if return_flow:
                        flow_vecs.append(forward_flow_vecs[t])
                
                if len(flow_values) > 0:
                    flow_magnitudes[b, t] = fuse_flow_magnitudes(flow_values, fusion_method)
                    if return_flow:
                        # 融合向量：用 magnitude 做加权平均
                        if fusion_method == 'max':
                            max_idx = torch.argmax(torch.stack([v.max() for v in flow_values]))
                            flow_vectors[b, t] = flow_vecs[max_idx]
                        else:  # mean
                            flow_vectors[b, t] = torch.stack(flow_vecs).mean(dim=0)
    
    if return_flow:
        return flow_vectors, flow_magnitudes
    return flow_magnitudes


def fuse_flow_magnitudes(flow_values: list, method: str = 'max') -> torch.Tensor:
    """
    融合多个光流幅度值
    Args:
        flow_values: 光流幅度值列表
        method: 融合方法
    Returns:
        融合后的光流幅度
    """
    if len(flow_values) == 1:
        return flow_values[0]
    
    if method == 'max':
        # 取最大值
        return torch.max(torch.stack(flow_values), dim=0)[0]
    elif method == 'mean':
        # 取平均值
        return torch.mean(torch.stack(flow_values), dim=0)
    elif method == 'sum':
        # 求和
        return torch.sum(torch.stack(flow_values), dim=0)
    elif method == 'weighted_mean':
        # 加权平均（给予时间上更近的帧更高权重）
        weights = torch.tensor([1.0] * len(flow_values), device=flow_values[0].device)
        weights = weights / weights.sum()
        weighted_flows = torch.stack([w * f for w, f in zip(weights, flow_values)])
        return torch.sum(weighted_flows, dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

class MotionSegmentationInference:
    """
    Inference class for motion segmentation using trained VGGT model
    """
    def __init__(self, model_path, pi3_model_path=None, raft_model_path=None, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model (you might need to adjust these parameters)
        from pi3.models.pi3_conf_flow_cam_lowfeature_wo_mean import create_pi3_motion_segmentation_model
        if pi3_model_path is None:
            pi3_model_path = os.environ.get("PI3_MODEL_PATH", None)
        self.model = create_pi3_motion_segmentation_model(
            pi3_model_path=pi3_model_path,
        ).to(device)
        initialize_raft_model(device=device, raft_model_path=raft_model_path)
        
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully. Best IoU: {checkpoint.get('best_iou', 'N/A')}")
        
        # Setup image preprocessing
        self.img_size = 518
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_video(self, video_frames):
        """
        Preprocess video frames for VGGT input
        
        Args:
            video_frames: List of PIL images or numpy arrays
            
        Returns:
            torch.Tensor: Preprocessed video tensor [1, S, 3, H, W]
        """
        processed_frames = []
        
        for frame in video_frames:
            # Convert to PIL if numpy
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # Resize to target size
            frame = frame.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            # Convert to tensor and normalize
            frame_tensor = transforms.ToTensor()(frame)  # [3, H, W] in [0, 1]
            frame_tensor = self.normalize(frame_tensor)  # Normalize for VGGT
            
            processed_frames.append(frame_tensor)
        
        # Stack and add batch dimension
        video_tensor = torch.stack(processed_frames, dim=0)  # [S, 3, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, S, 3, H, W]
        
        return video_tensor
    
    def predict_motion_mask(self, video_frames, return_confidence=False):
        """
        Predict motion segmentation mask for video frames
        
        Args:
            video_frames: List of PIL images or numpy arrays
            return_confidence: Whether to return additional confidence maps
            
        Returns:
            numpy array: Motion masks [S, H, W] in range [0, 1]
            dict (optional): Additional outputs if return_confidence=True
        """
        # from pi3.models.pi3_conf import run_pi3_attention_analysis
        # Preprocess input
        video_tensor = self.preprocess_video(video_frames).to(self.device)

        # Forward pass
        with torch.no_grad():
            flow_magnitudes = compute_optical_flow_magnitude_batch(
                video_tensor, self.device, 
                bidirectional=True,
                fusion_method='max'
            )  # [B, N, H, W]

            # predictions = self.model(video_tensor)
            predictions = self.model(video_tensor, flow_magnitudes)
            motion_mask = predictions['motion_mask']  # [1, S, H, W]
            motion_mask = motion_mask.squeeze(0).cpu().numpy()  # [S, H, W]

        return motion_mask
    
    def load_video_from_directory(self, video_dir, max_frames=None):
        """
        Load video frames from directory
        
        Args:
            video_dir: Directory containing video frames
            max_frames: Maximum number of frames to load
            
        Returns:
            List of PIL images
        """
        # Get image paths
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(os.path.join(video_dir, ext)))
        
        # img_paths = sorted(img_paths)[::10]
        img_paths = sorted(img_paths)
        
        if max_frames:
            img_paths = img_paths[:max_frames]
        
        # Load images
        images = []
        for path in tqdm(img_paths, desc="Loading frames"):
            img = Image.open(path).convert('RGB')
            images.append(img)
        
        return images
    
def visualize_motion_segmentation(images, motion_masks, save_dir, sequence_name="video", 
                                threshold=0.5, overlay_alpha=0.6):
    """
    Create comprehensive visualizations of motion segmentation results
    
    Args:
        images: List of PIL images (original size)
        motion_masks: numpy array [S, H, W] of motion masks
        save_dir: Directory to save visualizations
        sequence_name: Name for the sequence
        threshold: Binary threshold for mask visualization
        overlay_alpha: Alpha for overlay visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    
    # Create individual frame visualizations
    frame_dir = os.path.join(save_dir, f"{sequence_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    save_frame_visualizations(images, motion_masks, frame_dir, threshold, overlay_alpha)

    # Create summary grid visualization
    create_summary_grid(images, motion_masks, save_dir, sequence_name, threshold)

    print(f"Visualizations saved to {save_dir}")


def save_frame_visualizations(images, motion_masks, frame_dir, threshold=0.5, overlay_alpha=0.6):
    """Save overlay visualization for each frame."""
    for frame_idx, (img, mask) in enumerate(zip(images, motion_masks)):
        orig_w, orig_h = img.size
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        img_array = np.array(img)
        overlay = img_array.astype(np.float32).copy()
        motion_pixels = mask_resized > threshold
        overlay[motion_pixels] = (
            overlay[motion_pixels] * (1.0 - overlay_alpha)
            + np.array([255, 0, 0], dtype=np.float32) * overlay_alpha
        )
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_idx:05d}.png"), overlay_bgr)


def create_summary_grid(images, motion_masks, save_dir, sequence_name, threshold=0.5):
    """
    Create a grid summary of the entire sequence
    """
    num_frames = len(images)
    grid_size = min(8, num_frames)  # Show up to 8 frames
    
    if grid_size == 0:
        return
    
    # Sample frames evenly
    frame_indices = np.linspace(0, num_frames-1, grid_size, dtype=int)
    
    fig, axes = plt.subplots(3, grid_size, figsize=(2*grid_size, 6))
    if grid_size == 1:
        axes = axes.reshape(3, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        img = images[frame_idx]
        mask = motion_masks[frame_idx]
        
        # Resize mask to match image
        orig_w, orig_h = img.size
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Motion mask
        axes[1, i].imshow(mask_resized, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title('Motion Mask')
        axes[1, i].axis('off')
        
        # Overlay
        img_array = np.array(img)
        overlay = img_array.copy()
        motion_pixels = mask_resized > threshold
        overlay[motion_pixels] = overlay[motion_pixels] * 0.4 + np.array([255, 0, 0]) * 0.6
        
        axes[2, i].imshow(overlay.astype(np.uint8))
        axes[2, i].set_title('Overlay')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Motion Segmentation Summary - {sequence_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{sequence_name}_summary.png'), 
               dpi=200, bbox_inches='tight')
    plt.close()


def run_inference_on_video_directory(args, inference, video_dir, output_dir, 
                                   max_frames=None, sequence_length=16, use_sam_refine=True):
    """
    Run inference on a directory of video frames
    
    Args:
        model_path: Path to trained model
        video_dir: Directory containing video frames
        output_dir: Directory to save results
        max_frames: Maximum frames to process
        sequence_length: Number of frames to process at once
        use_sam_refine: Whether to use SAM2 for refinement
    """
    # Initialize inference
    # inference = MotionSegmentationInference(model_path)
    
    # Load video frames
    print(f"Loading video from {video_dir}")
    video_frames = inference.load_video_from_directory(video_dir, max_frames)
    
    if len(video_frames) == 0:
        print("No frames found in directory")
        return
    
    print(f"Processing {len(video_frames)} frames")
    
    # Process video in chunks if it's too long
    all_motion_masks = []
    all_motion_masks_before = []
    all_motion_masks_after = []

    model_config = args.sam2_config_path or os.environ.get("SAM2_CONFIG_PATH", DEFAULT_SAM2_CONFIG_PATH)
    checkpoint = args.sam2_checkpoint_path or os.environ.get("SAM2_CHECKPOINT_PATH", DEFAULT_SAM2_CHECKPOINT_PATH)

    predictor = build_sam.build_sam2_video_predictor(
        model_config, checkpoint, device='cuda'
    )
    
    
    for start_idx in tqdm(range(0, len(video_frames), sequence_length), desc="Processing chunks"):
        end_idx = min(start_idx + sequence_length, len(video_frames))
        chunk_frames = video_frames[start_idx:end_idx]
        
        # ---------- Predict ----------
        motion_masks = inference.predict_motion_mask(chunk_frames)
        
        # ⭐ 保存 refine 前结果（一定要 copy）
        motion_masks_before = motion_masks.copy()
        
        # ---------- SAM refine ----------
        if use_sam_refine:
            print(f"Refining masks with SAM2 for frames {start_idx}-{end_idx}")
            
            mask_h, mask_w = motion_masks.shape[1], motion_masks.shape[2]
            
            frame_arrays = []
            for pil_img in chunk_frames:
                pil_img_resized = pil_img.resize((mask_w, mask_h), Image.BILINEAR)
                frame_array = np.array(pil_img_resized).astype(np.float32) / 255.0
                frame_arrays.append(frame_array)
            
            frame_arrays = np.stack(frame_arrays, axis=0)
            frame_tensors = torch.from_numpy(frame_arrays).float().permute(0, 3, 1, 2)
            
            refined_masks = torch.zeros(
                motion_masks.shape[0],
                motion_masks.shape[1],
                motion_masks.shape[2],
                dtype=torch.float32
            )
            
            mask_list = torch.from_numpy(motion_masks).float()
            refine_sam(frame_tensors, mask_list, refined_masks, offset=0, predictor=predictor)
            
            motion_masks = refined_masks.cpu().numpy()
        
        # ---------- Collect ----------
        all_motion_masks_before.append(motion_masks_before)
        all_motion_masks_after.append(motion_masks)

    # Concatenate all chunks
    all_motion_masks_before = np.concatenate(all_motion_masks_before, axis=0)
    all_motion_masks_after  = np.concatenate(all_motion_masks_after, axis=0)
    # all_motion_masks = np.concatenate(all_motion_masks, axis=0)
    
    # Save results
    sequence_name = os.path.basename(video_dir)
    # save_motion_masks(all_motion_masks, video_frames, output_dir, sequence_name)
    
    # Create visualizations
    visualize_motion_segmentation(
        video_frames,
        all_motion_masks_before,
        output_dir,
        sequence_name + "_before_sam"
    )

    visualize_motion_segmentation(
        video_frames,
        all_motion_masks_after,
        output_dir,
        sequence_name + "_after_sam"
    )
    

    return all_motion_masks_after

def preprocess_mask(mask_tensor, threshold=0.8):
    # 1. 二值化
    # binary_mask = (mask_tensor > threshold).float()
    binary_mask = (mask_tensor > threshold)
    
    # 2. (可选) 连通域去噪: 如果你想更狠一点，可以把太小的点去掉
    # 注意: 这需要在 CPU numpy 上做，如果不想切设备，只做二值化也足够有效
    # mask_np = binary_mask.cpu().numpy()
    # ... cv2.connectedComponents ...
    
    return binary_mask

def refine_sam(frame_tensors, mask_list, p_masks_sam, offset=0, predictor=None):
  """Refine the final motion masks with SAM2.

  Args:
    frame_tensors: video frames
    mask_list: list of final motion masks
    p_masks_sam: returned SAM2-refined masks
    offset: video frame offset
  """
  # make tmp dir for SAMv2
  tmp_dir = tempfile.mkdtemp()
  for i in range(frame_tensors.shape[0]):
    frame = frame_tensors[i].permute(1, 2, 0).cpu().numpy()
    img_path = os.path.join(tmp_dir, f'{i:05}.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, np.uint8(frame * 255))
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    inference_state = predictor.init_state(tmp_dir)
    shutil.rmtree(tmp_dir)
    ann_obj_id = 1
    # Process even frames
    predictor.reset_state(inference_state)

    for idx, mask in enumerate(mask_list):
      if idx % 2 == 1:
        # import pdb;pdb.set_trace()
        clean_mask = preprocess_mask(mask)
        if clean_mask.sum() > 0:
            _, _, _ = predictor.add_new_mask(
                inference_state,
                frame_idx=idx,
                obj_id=ann_obj_id,
                mask=clean_mask,
            )
    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state, start_frame_idx=0):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).float()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
    for out_frame_idx in range(mask_list.shape[0]):
      if out_frame_idx % 2 == 0:
        p_masks_sam[offset + out_frame_idx] = video_segments[out_frame_idx][
            ann_obj_id
        ][0]

    # Process odd frames
    predictor.reset_state(inference_state)
    for idx, mask in enumerate(mask_list):
      if idx % 2 == 0:
        clean_mask = preprocess_mask(mask)
        
        if clean_mask.sum() > 0:
            _, _, _ = predictor.add_new_mask(
                inference_state,
                frame_idx=idx,
                obj_id=ann_obj_id,
                mask=clean_mask,
            )
    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state, start_frame_idx=0):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).float()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
    for out_frame_idx in range(mask_list.shape[0]):
      if out_frame_idx % 2 == 1:
        p_masks_sam[offset + out_frame_idx] = video_segments[out_frame_idx][
            ann_obj_id
        ][0]

def main():
    parser = argparse.ArgumentParser(
        description='Motion Segmentation Inference (directory input only)'
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pi3_model_path', type=str, default=None,
                        help='Path to pi3 .safetensors checkpoint. If omitted, uses PI3_MODEL_PATH env var.')
    parser.add_argument('--raft_model_path', type=str, default=None,
                        help='Path to RAFT checkpoint. If omitted, uses RAFT_MODEL_PATH env var.')
    parser.add_argument('--sam2_config_path', type=str, default=None,
                        help='Path to SAM2 config yaml. If omitted, uses SAM2_CONFIG_PATH env var.')
    parser.add_argument('--sam2_checkpoint_path', type=str, default=None,
                        help='Path to SAM2 checkpoint. If omitted, uses SAM2_CHECKPOINT_PATH env var.')

    # Single-sequence mode (one RGB frame directory).
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to a single rgb frame directory')

    # Multi-sequence mode (e.g., MOSE-0000 style folders).
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset root dir containing */rgb folders')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory to save results')

    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')

    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Number of frames to process at once')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for visualization')

    args = parser.parse_args()
    if args.sam2_config_path:
        os.environ["SAM2_CONFIG_PATH"] = args.sam2_config_path
    if args.sam2_checkpoint_path:
        os.environ["SAM2_CHECKPOINT_PATH"] = args.sam2_checkpoint_path

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Multi-sequence mode (equivalent to the original bash loop).
    # ============================================================
    inference = MotionSegmentationInference(
        model_path=args.model_path,
        pi3_model_path=args.pi3_model_path,
        raft_model_path=args.raft_model_path,
    )
    if args.dataset_dir is not None:
        print(f"Processing dataset: {args.dataset_dir}")

        rgb_dirs = sorted(
            # glob(os.path.join(args.dataset_dir, '*', 'rgb'))
            glob(os.path.join(args.dataset_dir, '*'))
        )
        # import pdb; pdb.set_trace()

        if len(rgb_dirs) == 0:
            raise RuntimeError(f"No rgb folders found in {args.dataset_dir}")

        for rgb_dir in rgb_dirs:
            if not os.listdir(rgb_dir):
                print(f"[Warning] Empty folder: {rgb_dir}, skipping.")
                continue

            # seq_name = os.path.basename(os.path.dirname(rgb_dir))
            seq_name = os.path.basename(os.path.normpath(rgb_dir))
            out_dir = os.path.join(args.output_dir, seq_name)
            # import pdb; pdb.set_trace()
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n=== Processing sequence: {seq_name} ===")
            print(f"Input : {rgb_dir}")
            print(f"Output: {out_dir}")

            motion_masks = run_inference_on_video_directory(
                args,
                inference=inference,
                video_dir=rgb_dir,
                output_dir=out_dir,
                max_frames=args.max_frames,
                sequence_length=args.sequence_length,
            )

            if motion_masks is not None:
                motion_ratio = np.mean(motion_masks > args.threshold)
                print(f"[{seq_name}] Motion pixel ratio: {motion_ratio:.2%}")

        print("Dataset inference completed.")
        return

    # ============================================================
    # Single-sequence mode (useful for debugging).
    # ============================================================
    if args.input_dir is not None:
        if not os.path.isdir(args.input_dir):
            raise ValueError(f"Invalid input_dir: {args.input_dir}")

        seq_name = os.path.basename(os.path.dirname(args.input_dir))
        out_dir = os.path.join(args.output_dir, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing single sequence: {seq_name}")
        print(f"Input : {args.input_dir}")
        print(f"Output: {out_dir}")

        motion_masks = run_inference_on_video_directory(
            args,
            inference=inference,
            video_dir=args.input_dir,
            output_dir=out_dir,
            max_frames=args.max_frames,
            sequence_length=args.sequence_length,
        )

        if motion_masks is not None:
            motion_ratio = np.mean(motion_masks > args.threshold)
            print(f"Motion pixel ratio: {motion_ratio:.2%}")

        print("Inference completed.")
        return

    # ============================================================
    # Argument fallback
    # ============================================================
    raise ValueError("You must specify either --dataset_dir or --input_dir")


if __name__ == "__main__":
    main()
