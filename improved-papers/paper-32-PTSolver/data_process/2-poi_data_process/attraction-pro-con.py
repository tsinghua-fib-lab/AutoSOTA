#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List, Set
import openai  # 引入 openai 库
from tqdm import tqdm  # 引入 tqdm 库

# API配置保持不变
api_config = {
    'glm-4-flashx': {'api_key': os.getenv('CHATGLM_API_KEY', '940831e246c06af9b10cdfe5cf7ce375.A6HuDJFmLm6Skuxa'),
                     'url': 'https://open.bigmodel.cn/api/paas/v4/', 'model': 'glm-4-flashx'},
    'deepseek': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-2dd9ac932e8b4951968e78fbe9b6ddfa'),
                 'url': 'https://api.deepseek.com/v1/', 'model': 'deepseek-chat'},
    'glm-4-flash': {'api_key': os.getenv('CHATGLM_API_KEY', '940831e246c06af9b10cdfe5cf7ce375.A6HuDJFmLm6Skuxa'),
                    'url': 'https://open.bigmodel.cn/api/paas/v4/', 'model': 'glm-4-flash'},
    'glm-4-plus': {'api_key': os.getenv('CHATGLM_API_KEY', 'd6646b7655abf0af4f28f0881ec8a173.Oju5XTevor9EQr6O'),
                   'url': 'https://open.bigmodel.cn/api/paas/v4/', 'model': 'glm-4-plus'},
    'yi-lightning': {'api_key': os.getenv('01_API_KEY', '67cf2ba4980b47338b70aabb0133c90a'),
                     'url': 'https://api.lingyiwanwu.com/v1/', 'model': 'yi-lightning'},
    'yi-large': {'api_key': os.getenv('01_API_KEY', '67cf2ba4980b47338b70aabb0133c90a'),
                 'url': 'https://api.lingyiwanwu.com/v1/', 'model': 'yi-large'},
    'gpt-4o': {'api_key': os.getenv('Gpt_API_KEY', 'sk-F6Zl9geYkxv8MX5P02A0A6Ff037c4e22Ac0712179b358c16'),
               'url': 'https://api3.apifans.com/v1/', 'model': 'gpt-4o'},
    'gpt-4o-mini': {'api_key': os.getenv('Gpt_API_KEY', 'sk-F6Zl9geYkxv8MX5P02A0A6Ff037c4e22Ac0712179b358c16'),
                    'url': 'https://api3.apifans.com/v1/', 'model': 'gpt-4o-mini'},
    'sli-deepseek3': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-akmniudwimlmimvvtwxcycmpebcbvofcxiacsvkyfxlqkrkb'),
                     'url': 'https://api.siliconflow.cn/v1/', 'model': 'deepseek-ai/DeepSeek-V3'},
    'sli-Qwen7B': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-akmniudwimlmimvvtwxcycmpebcbvofcxiacsvkyfxlqkrkb'),
                   'url': 'https://api.siliconflow.cn/v1/', 'model': 'Qwen/Qwen2.5-7B-Instruct'},
    'sli-llama8B': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-akmniudwimlmimvvtwxcycmpebcbvofcxiacsvkyfxlqkrkb'),
                    'url': 'https://api.siliconflow.cn/v1/', 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct'},
    'llama405': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-akmniudwimlmimvvtwxcycmpebcbvofcxiacsvkyfxlqkrkb'),
                 'url': 'https://api.siliconflow.cn/v1/', 'model': 'meta-llama/Meta-Lllama-3.1-405B-Instruct'},
    'llama70': {'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-akmniudwimlmimvvtwxcycmpebcbvofcxiacsvkyfxlqkrkb'),
                'url': 'https://api.siliconflow.cn/v1/', 'model': 'nvidia/Llama-3.1-Nemotron-70B-Instruct'},
    'gemini-exp-1206': {'api_key': os.getenv('GEMINI_API_KEY', 'AIzaSyCKRiX_93o7_N66jm_hbpAna5BCTScSacA'),
                        'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-exp-1206:generateContent',
                        'model': 'gemini-exp-1206'},
    'ali-deepseek3': {
            'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-24b3211b550c49aa8a051e7b99344ec0'),
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'deepseek-v3'},
    'ali-qwen-max': {
            'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-09f1d6152ed34744b59d517bf6793b60'),
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'qwen-max'},
    'ali-qwen-plus': {
            'api_key': os.getenv('DEEPSEEK_API_KEY', 'sk-09f1d6152ed34744b59d517bf6793b60'),
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'qwen-plus'},
}


class ProcessingError(Exception):
    """自定义异常类，用于跟踪处理失败的原因"""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class ErrorTracker:
    """
    跟踪和记录处理失败的记录
    """

    def __init__(self, output_dir: str):
        self.failed_records_file = os.path.join(output_dir, "failed_records.json")
        self.failed_records: Dict[str, Dict] = {}
        self.load_failed_records()

    def load_failed_records(self):
        """加载之前记录的失败记录"""
        if os.path.exists(self.failed_records_file):
            try:
                with open(self.failed_records_file, 'r', encoding='utf-8') as f:
                    self.failed_records = json.load(f)
                print(f"Loaded {len(self.failed_records)} failed records")
            except Exception as e:
                print(f"Error loading failed records file: {e}")
                self.failed_records = {}

    def save_failed_records(self):
        """保存失败记录到文件"""
        try:
            with open(self.failed_records_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving failed records: {e}")

    def add_failed_record(self, gmap_id: str, error_message: str, attempts: int):
        """添加一条失败记录"""
        self.failed_records[gmap_id] = {
            'error_message': error_message,
            'attempts': attempts,
            'last_attempt': datetime.datetime.now().isoformat()
        }
        self.save_failed_records()


class CheckpointManager:
    """
    管理处理进度的检查点，支持断点续传功能
    """

    def __init__(self, output_dir: str):
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        self.processed_ids: Set[str] = set()
        self.load_checkpoint()

    def load_checkpoint(self):
        """加载已处理的记录ID"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self.processed_ids = set(checkpoint_data.get('processed_ids', []))
                print(f"Loaded checkpoint with {len(self.processed_ids)} processed records")
            except Exception as e:
                print(f"Error loading checkpoint file: {e}")
                self.processed_ids = set()

    def save_checkpoint(self):
        """保存当前处理进度"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_ids': list(self.processed_ids),
                    'last_update': datetime.datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def mark_processed(self, gmap_id: str):
        """标记一条记录为已处理"""
        self.processed_ids.add(gmap_id)
        if len(self.processed_ids) % 10 == 0:
            self.save_checkpoint()

    def is_processed(self, gmap_id: str) -> bool:
        """检查记录是否已处理"""
        return gmap_id in self.processed_ids


def retry_llm_call(prompt: str, model_name: str, max_retries: int = 5, retry_delay: int = 30) -> str:
    """
    使用重试机制调用LLM API（基于 openai 库）
    Args:
        prompt: 提示词文本
        model_name: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试等待时间（秒）
    Returns:
        API响应文本
    Raises:
        ProcessingError: 当所有重试都失败时抛出，包含尝试次数信息
    """
    attempts = 0
    last_error = None
    model_config = api_config.get(model_name)
    if not model_config:
        raise ValueError(f"Model {model_name} not found in api_config.")

    openai.api_key = model_config['api_key']
    openai.api_base = model_config['url']

    while attempts < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model_config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            attempts += 1
            last_error = str(e)
            if attempts < max_retries:
                print(f"Attempt {attempts} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                break
    raise ProcessingError(f"All {attempts} attempts failed. Last error: {last_error}", attempts)


def process_attraction(record: Dict, meta_dict: Dict, model_name: str, max_reviews: int,
                       checkpoint_manager: CheckpointManager, error_tracker: ErrorTracker) -> Optional[Dict]:
    """
    处理单个景点的评论，包含重试机制和错误跟踪
    """
    gmap_id = record.get("gmap_id")
    if checkpoint_manager.is_processed(gmap_id):
        print(f"Skipping already processed attraction: {gmap_id}")
        return None
    reviews = record.get("review_attraction", [])[:max_reviews]
    reviews_text = "\n\n".join([rev.get("text", "") for rev in reviews])
    meta = meta_dict.get(gmap_id, {})
    attraction_name = meta.get("name", "Unknown Attraction")
    prompt = f"""
You are a travel review expert. Please read the following user reviews for the attraction "{attraction_name}" .
Based on the reviews, please summarize the advantages and disadvantages of the attraction by considering the following aspects:
1. Environment (landscapes, cleanliness, etc.);
2. Facilities and Services (guided tours, rest areas, restrooms, staff service, etc.);
3. Accessibility and Convenience (location, parking, public transportation, etc.);
4. Activities and Experience (interactive features, historical/cultural aspects, entertainment, etc.);
5. Value for Money (ticket prices, overall experience, etc.).
Please output your summary using the following format with special markers:
【PRO】: [Your summary of advantages]
【CON】: [Your summary of disadvantages]
Only output exactly as above (with the markers) and nothing else.
Here are some user reviews:
{reviews_text}
    """.strip()
    try:
        # 使用重试机制调用LLM
        response = retry_llm_call(prompt, model_name)
        pattern = r"【PRO】:\s*(.*?)\s*【CON】:\s*(.*)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            pro_text = match.group(1).strip()
            con_text = match.group(2).strip()
        else:
            pro_text = response
            con_text = ""
        checkpoint_manager.mark_processed(gmap_id)
        return {"gmap_id": gmap_id, "pro": pro_text, "con": con_text}
    except ProcessingError as e:
        print(f"Failed to process attraction {gmap_id} after {e.attempts} attempts")
        error_tracker.add_failed_record(gmap_id, str(e), e.attempts)
        return {"gmap_id": gmap_id, "pro": "", "con": "", "error": str(e)}


def process_records_with_retry(records: List[Dict], meta_dict: Dict, model_name: str, max_reviews: int,
                               output_dir: str, num_threads: int):
    """
    使用重试机制处理所有记录，并显示进度条
    """
    checkpoint_manager = CheckpointManager(output_dir)
    error_tracker = ErrorTracker(output_dir)
    output_file = os.path.join(output_dir, "summary.jsonl")

    # 加载已有结果
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    existing_results[result['gmap_id']] = result
                except Exception as e:
                    print(f"Error reading existing result: {e}")

    print(f"Processing {len(records)} attractions using {num_threads} threads...")

    # 使用 tqdm 显示进度条
    with ThreadPoolExecutor(max_workers=num_threads) as executor, \
            tqdm(total=len(records), desc="Processing Attractions", unit="attraction") as pbar:
        futures = []
        for record in records:
            gmap_id = record.get("gmap_id")
            if checkpoint_manager.is_processed(gmap_id):
                if gmap_id in existing_results:
                    pbar.update(1)  # 如果已处理，直接更新进度条
                    continue
            futures.append(
                executor.submit(process_attraction, record, meta_dict, model_name,
                                max_reviews, checkpoint_manager, error_tracker)
            )

        # 处理新的结果
        for future in futures:
            result = future.result()
            pbar.update(1)  # 每完成一个任务，更新进度条
            if result:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 最后保存进度
    checkpoint_manager.save_checkpoint()

    # 输出失败记录统计
    if error_tracker.failed_records:
        print("\nProcessing completed with errors:")
        print(f"Failed to process {len(error_tracker.failed_records)} attractions:")
        for gmap_id, info in error_tracker.failed_records.items():
            print(f"- {gmap_id}: {info['error_message']} (Attempts: {info['attempts']})")
    else:
        print("\nProcessing completed successfully with no errors.")
    print(f"Results saved in: {output_file}")
    print(f"Failed records details saved in: {error_tracker.failed_records_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize pros and cons from attraction reviews using a large language model")
    parser.add_argument("--model_name", type=str, default="ali-qwen-max", help="The model name to use")
    parser.add_argument("--max_reviews", type=int, default=10, help="Maximum number of reviews to use per attraction")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing")
    parser.add_argument("--reviews_file", type=str, default="10_selected_reviews-error.json",
                        help="Path to the reviews jsonl file")
    parser.add_argument("--meta_file", type=str, default="meta-city-all-attraction-19976.jsonl",
                        help="Path to the attraction meta information jsonl file")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Maximum number of retries for failed API calls")
    parser.add_argument("--retry_delay", type=int, default=30,
                        help="Delay in seconds between retry attempts")
    args = parser.parse_args()

    # 加载meta信息
    meta_dict = {}
    with open(args.meta_file, "r", encoding="utf-8") as f_meta:
        for line in f_meta:
            try:
                meta = json.loads(line)
                meta_dict[meta.get("gmap_id")] = meta
            except Exception as e:
                print(f"Error reading meta information: {e}")

    # 加载评论记录
    records = []
    with open(args.reviews_file, "r", encoding="utf-8") as f_reviews:
        for line in f_reviews:
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                print(f"Error reading review record: {e}")

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join("output-02-08-error", f"{args.model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 使用新的处理函数
    process_records_with_retry(
        records, meta_dict, args.model_name, args.max_reviews, output_dir, args.num_threads
    )


if __name__ == "__main__":
    main()