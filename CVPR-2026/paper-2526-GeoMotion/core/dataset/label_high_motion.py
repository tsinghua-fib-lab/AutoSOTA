import os
import json
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import argparse


class VideoSequenceDataset(Dataset):
    """将视频序列封装为 Dataset，方便多 GPU 处理"""
    
    def __init__(self, video_roots):
        self.video_roots = video_roots
    
    def __len__(self):
        return len(self.video_roots)
    
    def __getitem__(self, idx):
        return idx, self.video_roots[idx]


class OpticalFlowLabelerGPU:
    """GPU 加速的光流计算和标注"""
    
    def __init__(self, flow_threshold=2.0, device='cuda'):
        self.flow_threshold = flow_threshold
        self.device = device
        
        # 使用 RAFT 或其他 GPU 光流模型（这里先用 CPU cv2，后面可以替换）
        # 如果要用 GPU 光流，可以集成 RAFT 或 torch 实现的光流
    
    def load_mask_npz(self, mask_path):
        """加载 .npz 格式的 mask"""
        data = np.load(mask_path)
        if 'mask' in data:
            mask = data['mask']
        elif 'arr_0' in data:
            mask = data['arr_0']
        else:
            mask = data[list(data.keys())[0]]
        return mask
    
    def compute_optical_flow_batch(self, frames1, frames2, masks):
        """
        批量计算光流（可以在这里集成 GPU 光流算法）
        
        Args:
            frames1: [B, H, W, 3] numpy array
            frames2: [B, H, W, 3] numpy array
            masks: [B, H, W] numpy array
        
        Returns:
            avg_flows: [B] 平均光流
        """
        batch_size = len(frames1)
        avg_flows = []
        max_flows = []
        
        for i in range(batch_size):
            frame1 = frames1[i]
            frame2 = frames2[i]
            mask = masks[i]
            
            # 计算光流（使用 cv2，可以替换为 GPU 实现）
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            
            # 只计算运动区域
            motion_region = mask > 0.5
            if motion_region.sum() > 0:
                avg_flow = magnitude[motion_region].mean()
                max_flow = magnitude[motion_region].max()
            else:
                avg_flow = 0.0
                max_flow = 0.0
            
            avg_flows.append(avg_flow)
            max_flows.append(max_flow)
        
        return avg_flows, max_flows
    
    def analyze_sequence(self, video_root, max_frames=16):
        """
        分析单个视频序列（采样处理）
        
        Args:
            video_root: 视频根目录
            max_frames: 最多采样的帧数
        """
        img_dir = video_root
        mask_dir = os.path.join(video_root, "masks")
        
        # 获取图片文件
        img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(sorted(Path(img_dir).glob(ext)))
        
        if len(img_paths) < 2:
            return None
        
        # 找到对应的 mask 文件
        valid_pairs = []
        for img_path in img_paths:
            img_name = img_path.stem
            mask_path = os.path.join(mask_dir, f"{img_name}.npz")
            if os.path.exists(mask_path):
                valid_pairs.append((str(img_path), mask_path))
        
        if len(valid_pairs) < 2:
            return None
        
        # 采样：如果帧数超过 max_frames，均匀采样
        total_frames = len(valid_pairs)
        if total_frames > max_frames:
            # 均匀采样 max_frames 帧
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            valid_pairs = [valid_pairs[i] for i in indices]
        
        # 批量读取并处理
        frames1_batch = []
        frames2_batch = []
        masks_batch = []
        frame_flows = []
        
        for i in range(len(valid_pairs) - 1):
            img_path1, mask_path1 = valid_pairs[i]
            img_path2, _ = valid_pairs[i + 1]
            
            frame1 = np.array(Image.open(img_path1).convert('RGB'))
            frame2 = np.array(Image.open(img_path2).convert('RGB'))
            mask = self.load_mask_npz(mask_path1)
            
            if len(mask.shape) == 3:
                mask = mask[0]
            
            frames1_batch.append(frame1)
            frames2_batch.append(frame2)
            masks_batch.append(mask)
        
        # 批量计算光流
        avg_flows, max_flows = self.compute_optical_flow_batch(
            frames1_batch, frames2_batch, masks_batch
        )
        
        for i, (avg_flow, max_flow) in enumerate(zip(avg_flows, max_flows)):
            motion_region = masks_batch[i] > 0.5
            frame_flows.append({
                'frame_idx': i,
                'avg_flow': float(avg_flow),
                'max_flow': float(max_flow),
                'motion_pixels': int(motion_region.sum())
            })
        
        # 计算统计信息
        if len(avg_flows) > 0:
            avg_flow = np.mean(avg_flows)
            max_flow = np.max(avg_flows)
            min_flow = np.min(avg_flows)
            std_flow = np.std(avg_flows)
            
            is_high_motion = avg_flow >= self.flow_threshold
            
            return {
                'sequence_name': os.path.basename(video_root),
                'num_frames': total_frames,  # 原始总帧数
                'sampled_frames': len(valid_pairs),  # 实际采样帧数
                'avg_flow': float(avg_flow),
                'max_flow': float(max_flow),
                'min_flow': float(min_flow),
                'std_flow': float(std_flow),
                'is_high_motion': is_high_motion,
                'frame_flows': frame_flows
            }
        
        return None


def worker_process(rank, world_size, video_roots, flow_threshold, max_frames, output_dir):
    """每个 GPU 上的工作进程"""
    
    # 设置设备
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # 数据分片：每个 GPU 处理一部分数据
    total_seqs = len(video_roots)
    seqs_per_gpu = (total_seqs + world_size - 1) // world_size
    start_idx = rank * seqs_per_gpu
    end_idx = min(start_idx + seqs_per_gpu, total_seqs)
    
    local_video_roots = video_roots[start_idx:end_idx]
    
    print(f"GPU {rank}: 处理 {len(local_video_roots)} 个序列 (索引 {start_idx}-{end_idx}), 每序列采样 {max_frames} 帧")
    
    # 创建标注器
    labeler = OpticalFlowLabelerGPU(flow_threshold=flow_threshold, device=device)
    
    # 处理数据
    results = {}
    progress_bar = tqdm(
        local_video_roots,
        desc=f"GPU {rank}",
        position=rank,
        leave=True
    )
    
    for video_root in progress_bar:
        try:
            analysis = labeler.analyze_sequence(video_root, max_frames=max_frames)
            if analysis is not None:
                # 确保所有值都是 Python 原生类型，避免 numpy 类型
                analysis['is_high_motion'] = bool(analysis['is_high_motion'])
                analysis['num_frames'] = int(analysis['num_frames'])
                analysis['sampled_frames'] = int(analysis['sampled_frames'])
                results[analysis['sequence_name']] = analysis
        except Exception as e:
            print(f"GPU {rank}: 处理 {video_root} 时出错: {e}")
            continue
    
    # 保存本地结果
    output_file = os.path.join(output_dir, f'results_gpu_{rank}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"GPU {rank}: 完成，处理了 {len(results)} 个序列，结果保存到 {output_file}")


def merge_results(output_dir, output_file, flow_threshold):
    """合并所有 GPU 的结果"""
    all_results = {}
    
    # 读取所有 GPU 的结果
    result_files = sorted(Path(output_dir).glob('results_gpu_*.json'))
    
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            gpu_results = json.load(f)
            all_results.update(gpu_results)
    
    # 计算统计信息
    high_motion_count = sum(1 for info in all_results.values() if info['is_high_motion'])
    
    summary = {
        'total_sequences': len(all_results),
        'high_motion_sequences': high_motion_count,
        'low_motion_sequences': len(all_results) - high_motion_count,
        'high_motion_ratio': high_motion_count / len(all_results) if len(all_results) > 0 else 0,
        'flow_threshold': flow_threshold
    }
    
    # 保存合并结果
    output_data = {
        'summary': summary,
        'sequences': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"合并完成！")
    print(f"总序列数: {summary['total_sequences']}")
    print(f"高运动序列: {summary['high_motion_sequences']} ({summary['high_motion_ratio']*100:.2f}%)")
    print(f"低运动序列: {summary['low_motion_sequences']}")
    print(f"最终结果已保存到: {output_file}")
    print(f"{'='*60}")
    
    # 清理临时文件
    for result_file in result_files:
        os.remove(result_file)
    print("已清理临时文件")


def scan_dataset_multigpu(data_dir, output_file, flow_threshold=2.0, max_frames=16, num_gpus=None):
    """
    多 GPU 并行扫描数据集
    
    Args:
        data_dir: 数据集根目录或 txt 文件
        output_file: 输出 JSON 文件路径
        flow_threshold: 光流阈值
        max_frames: 每个序列最多采样的帧数
        num_gpus: 使用的 GPU 数量，None 表示使用所有可用 GPU
    """
    
    # 获取所有视频序列
    video_roots = []
    
    if os.path.isfile(data_dir) and data_dir.endswith(".txt"):
        print(f"从 txt 文件读取序列列表: {data_dir}")
        with open(data_dir, "r") as f:
            video_roots = [line.strip() for line in f if line.strip()]
    else:
        print(f"扫描目录: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            if "masks" in dirs:
                video_roots.append(root)
    
    print(f"找到 {len(video_roots)} 个视频序列")
    print(f"每个序列将采样最多 {max_frames} 帧进行分析")
    
    # 确定使用的 GPU 数量
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count())
    
    if num_gpus == 0:
        raise RuntimeError("没有可用的 GPU！")
    
    print(f"使用 {num_gpus} 个 GPU 进行并行处理")
    
    # 创建临时输出目录
    output_dir = os.path.dirname(output_file) or '.'
    temp_dir = os.path.join(output_dir, 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 启动多进程
    mp.spawn(
        worker_process,
        args=(num_gpus, video_roots, flow_threshold, max_frames, temp_dir),
        nprocs=num_gpus,
        join=True
    )
    
    # 合并结果
    merge_results(temp_dir, output_file, flow_threshold)
    
    # 删除临时目录
    os.rmdir(temp_dir)


def export_high_motion_list(json_file, output_txt):
    """从标签文件导出高运动序列列表"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    high_motion_sequences = []
    for seq_name, info in data['sequences'].items():
        if info['is_high_motion']:
            high_motion_sequences.append(seq_name)
    
    with open(output_txt, 'w') as f:
        for seq in sorted(high_motion_sequences):
            f.write(f"{seq}\n")
    
    print(f"已导出 {len(high_motion_sequences)} 个高运动序列到: {output_txt}")


def visualize_statistics(json_file):
    """可视化统计信息"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = data['sequences']
    avg_flows = [info['avg_flow'] for info in sequences.values()]
    
    print(f"\n{'='*60}")
    print("光流统计信息:")
    print(f"{'='*60}")
    print(f"平均光流范围: {min(avg_flows):.2f} - {max(avg_flows):.2f}")
    print(f"平均值: {np.mean(avg_flows):.2f}")
    print(f"中位数: {np.median(avg_flows):.2f}")
    print(f"标准差: {np.std(avg_flows):.2f}")
    
    # 光流分布
    bins = [0, 1, 2, 3, 5, 10, float('inf')]
    labels = ['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
    
    print(f"\n光流分布:")
    for i in range(len(bins)-1):
        count = sum(1 for f in avg_flows if bins[i] <= f < bins[i+1])
        print(f"  {labels[i]:6s}: {count:4d} ({count/len(avg_flows)*100:5.1f}%)")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='多 GPU 并行为视频序列标注高运动标签')
    parser.add_argument('--data_dir', type=str, default='/data1/got_train_video_roots_with_masks.txt', 
                        help='数据集根目录或包含序列路径的 txt 文件')
    parser.add_argument('--output', type=str, default='motion_labels.json',
                        help='输出标签文件路径 (默认: motion_labels.json)')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='光流阈值 (默认: 2.0)')
    parser.add_argument('--max-frames', type=int, default=16,
                        help='每个序列采样的最大帧数 (默认: 16)')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='使用的 GPU 数量 (默认: 使用所有可用 GPU)')
    parser.add_argument('--export-list', type=str, default=None,
                        help='导出高运动序列列表到 txt 文件')
    parser.add_argument('--visualize', action='store_true',
                        help='显示统计信息')
    
    args = parser.parse_args()
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 环境")
    
    print(f"检测到 {torch.cuda.device_count()} 个 GPU")
    
    # 多 GPU 并行生成标签
    scan_dataset_multigpu(
        args.data_dir,
        args.output,
        args.threshold,
        args.max_frames,
        args.num_gpus
    )
    
    # 可视化统计
    if args.visualize:
        visualize_statistics(args.output)
    
    # 导出高运动列表
    if args.export_list:
        export_high_motion_list(args.output, args.export_list)


if __name__ == "__main__":
    main()
# ============================================================================
# 使用示例
# ============================================================================
"""
# 1. 基本使用 - 扫描目录
python label_high_motion.py /path/to/dataset --output motion_labels.json --threshold 2.0

    parser.add_argument('--data_dir', type=str, default='/data1/got_train_video_roots_with_masks.txt', 
                        help='数据集根目录或包含序列路径的 txt 文件')

# 2. 从 txt 文件读取序列列表
python label_high_motion.py /path/to/sequences.txt --output motion_labels.json

# 3. 同时导出高运动序列列表
python label_high_motion.py /path/to/dataset \
    --output motion_labels.json \
    --export-list high_motion_sequences.txt \
    --visualize

# 4. 使用不同的阈值
python label_high_motion.py /path/to/dataset --threshold 3.0

# ============================================================================
# 输出 JSON 格式示例
# ============================================================================
{
  "summary": {
    "total_sequences": 1000,
    "high_motion_sequences": 350,
    "low_motion_sequences": 650,
    "high_motion_ratio": 0.35,
    "flow_threshold": 2.0
  },
  "sequences": {
    "video_001": {
      "sequence_name": "video_001",
      "num_frames": 16,
      "avg_flow": 3.45,
      "max_flow": 5.67,
      "min_flow": 1.23,
      "std_flow": 0.89,
      "is_high_motion": true,
      "frame_flows": [
        {"frame_idx": 0, "avg_flow": 3.2, "max_flow": 5.1, "motion_pixels": 12450},
        {"frame_idx": 1, "avg_flow": 3.7, "max_flow": 5.8, "motion_pixels": 13200}
      ]
    }
  }
}

# ============================================================================
# 在 Dataset 中使用标签
# ============================================================================
import json

class MotionSegmentationDatasetWithLabels(Dataset):
    def __init__(self, data_dir, split, label_file="motion_labels.json", **kwargs):
        super().__init__(data_dir, split, **kwargs)
        
        # 加载标签
        with open(label_file, 'r') as f:
            self.motion_labels = json.load(f)['sequences']
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if sample is None:
            return None
        
        seq_name = sample['sequence_name']
        
        # 从预计算的标签中获取信息
        if seq_name in self.motion_labels:
            label_info = self.motion_labels[seq_name]
            sample['is_high_motion'] = label_info['is_high_motion']
            sample['avg_flow_magnitude'] = label_info['avg_flow']
        else:
            # 如果标签文件中没有，默认为低运动
            sample['is_high_motion'] = False
            sample['avg_flow_magnitude'] = 0.0
        
        return sample
"""