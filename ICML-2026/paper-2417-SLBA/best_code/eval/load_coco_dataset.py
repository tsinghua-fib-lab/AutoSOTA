#!/usr/bin/env python3
"""
加载sayakpaul/coco-30-val-2014数据集的train split中的前5000条caption
保存为coco-30-val-2014_prompt.json文件
"""

import json
from datasets import load_dataset
from tqdm import tqdm

def load_coco_captions(dataset_name='sayakpaul/coco-30-val-2014', start_idx=0, end_idx=4999, output_path='coco-30-val-2014_prompt.json'):
    """
    从Hugging Face加载COCO数据集的caption并保存为JSON格式
    
    Args:
        dataset_name: Hugging Face数据集名称
        start_idx: 开始索引
        end_idx: 结束索引（包含）
        output_path: 输出JSON文件路径
    """
    try:
        print(f"正在加载数据集: {dataset_name}")
        
        # 加载数据集
        dataset_info = load_dataset(dataset_name, split=None)
        available_splits = list(dataset_info.keys())
        print(f"可用的数据集分割: {available_splits}")
        
        # 使用train分割
        if 'train' in available_splits:
            split = 'train'
        else:
            split = available_splits[0]
            
        print(f"使用数据集分割: {split}")
        dataset = dataset_info[split]
        
        print(f"数据集总大小: {len(dataset)}")
        print(f"提取范围: {start_idx} - {end_idx}")
        
        # 检查索引范围
        actual_end_idx = min(end_idx, len(dataset) - 1)
        if actual_end_idx != end_idx:
            print(f"注意: 调整结束索引为 {actual_end_idx}（数据集大小限制）")
        
        # 提取caption
        prompts = []
        
        # 先查看一个样本的结构
        sample = dataset[0]
        print(f"数据样本字段: {list(sample.keys())}")
        
        for i in tqdm(range(start_idx, actual_end_idx + 1), desc='提取caption'):
            item = dataset[i]
            
            # 提取caption字段
            caption = None
            if 'caption' in item:
                caption = item['caption']
                if isinstance(caption, list):
                    caption = caption[0]  # 取第一个caption
            elif 'text' in item:
                caption = item['text']
            elif 'prompt' in item:
                caption = item['prompt']
            else:
                # 查找包含文本的字段
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:  # 假设caption长度大于10
                        caption = value
                        break
            
            if caption:
                prompts.append([i - start_idx, caption])  # 重新编号从0开始
            else:
                print(f"警告: 索引 {i} 处未找到有效的caption")
        
        print(f"成功提取 {len(prompts)} 条caption")
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        print(f"Caption已保存到: {output_path}")
        
        # 显示前几个样本
        print("\n前5个样本:")
        for i, (idx, caption) in enumerate(prompts[:5]):
            print(f"{idx}: {caption}")
            
        return prompts
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        raise

def main():
    """主函数"""
    print("开始加载COCO数据集...")
    
    # 加载前5000条caption（索引0-4999）
    prompts = load_coco_captions(
        dataset_name='sayakpaul/coco-30-val-2014',
        start_idx=0,
        end_idx=4999,
        output_path='eval/coco-30-val-2014_prompt.json'
    )
    
    print(f"\n任务完成！共保存了 {len(prompts)} 条caption")

if __name__ == '__main__':
    main()
