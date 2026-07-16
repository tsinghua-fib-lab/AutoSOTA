
"""
Main run script
Run model evaluation and save results
"""

import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

# Import configuration
from config import DATASET_CONFIG, MODEL_CONFIG

# Import datasets
from data import (
    CIFAR100Dataset,
    CIFAR10Dataset,
    ImageNet1kDataset,
    AGNewsDataset,
    MMLUDataset,
    TruthfulQADataset,
    HaluEvalDataset,
    Flickr30kDataset,
    COCOCaptionDataset
)

# Import models
from models import (
    BLIP2ImageClassifier,
    Qwen3VLImageClassifier, 
    MetaCLIP2ImageClassifier,
    OpenAIVisionImageClassifier,
    CLIPImageClassifier,
    InternVLImageClassifier,
    SAILVLImageClassifier,
    Ministral3VLImageClassifier,
    LlamaTextClassifier,
    OpenAITextClassifier,
    Qwen3TextClassifier,
    MinistralTextClassifier,
    LlamaGenerator,
    OpenAIGenerator,
    Qwen3Generator,
    MinistralGenerator,
    BLIP2Captioner,
    Qwen3VLCaptioner,
    MetaCLIP2Captioner,
    OpenAICaptioner,
    Llama32VisionCaptioner,
    InternVLCaptioner,
    SAILVLCaptioner,
    Ministral3VLCaptioner,
    Pixtral12BCaptioner,
    GLM46VCaptioner,
    GLM46VImageClassifier,
    Pixtral12BImageClassifier,
    Idefics3ImageClassifier,
    Idefics3Captioner,
    Gemma3ImageClassifier,
    Step3VLImageClassifier,
    Step3VLCaptioner
)


class ModelEvaluator:
    """Model Evaluator"""
    
    def __init__(self, task_type: str, dataset_name: str, model_name: str, output_dir: str = "./outputs"):
        """
        Initialize the evaluator

        Args:
            task_type: Task type (image_classification, text_classification, llm_generation, vlm_tagging)
            dataset_name: Dataset name
            model_name: Model name
            output_dir: Output directory
        """
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset and model
        self.dataset = self._load_dataset()
        self.dataset.load_data() 
        self.model = self._load_model()
        
    def _load_dataset(self):
        """Load dataset"""
        print(f"\n{'='*50}")
        print(f"Loading dataset: {self.dataset_name}")
        print(f"{'='*50}")
        
        if self.task_type == "image_classification":
            if self.dataset_name == "CIFAR-100":
                return CIFAR100Dataset(split='test', num_samples=10000)
            elif self.dataset_name == "CIFAR-10":
                return CIFAR10Dataset(split='test', num_samples=10000)
            elif self.dataset_name == "ImageNet-1k":
                return ImageNet1kDataset(split='validation', num_samples=10000)

        
        elif self.task_type == "text_classification":
            if self.dataset_name == "AG_News":
                return AGNewsDataset(split='train', num_samples=10000)
            elif self.dataset_name == "MMLU":
                return MMLUDataset(split='test', num_samples=None)
        
        elif self.task_type == "llm_generation":    
            if self.dataset_name == "TruthfulQA":
                return TruthfulQADataset(split='validation', num_samples=817)
            elif self.dataset_name == "HaluEval":
                return HaluEvalDataset(subset='dialogue', split='data', num_samples=10000)
        
        elif self.task_type == "vlm_tagging":
            if self.dataset_name == "Flickr30k":
                return Flickr30kDataset(split='test', num_samples=10000)
            elif self.dataset_name == "COCO":
                return COCOCaptionDataset(split='val', num_samples=10000)
        
        raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_model(self):
        """Load model"""
        print(f"\n{'='*50}")
        print(f"Loading model: {self.model_name}")
        print(f"{'='*50}")
        
        config = MODEL_CONFIG[self.task_type][self.model_name]
        
        if self.task_type == "image_classification":
            if config['model_type'] == 'blip2':
                model = BLIP2ImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'qwen3_vl':
                model = Qwen3VLImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'metaclip2':  
                model = MetaCLIP2ImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'openai_vision':
                model = OpenAIVisionImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'clip':
                model = CLIPImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'internvl':
                model = InternVLImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'sailvl':
                model = SAILVLImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'ministral3_vl':
                model = Ministral3VLImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'pixtral':
                model = Pixtral12BImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'glm4v':
                model = GLM46VImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'idefics3':
                model = Idefics3ImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'gemma3':
                model = Gemma3ImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'step3vl':
                model = Step3VLImageClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
                
        elif self.task_type == "text_classification":
            if config['model_type'] == 'llama':
                model = LlamaTextClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'openai':
                model = OpenAITextClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'qwen3':
                model = Qwen3TextClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            elif config['model_type'] == 'ministral':
                model = MinistralTextClassifier(config, self.dataset.class_names)
                model.load_model()
                return model
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
        
        elif self.task_type == "llm_generation":
            if config['model_type'] == 'llama':
                model = LlamaGenerator(config)
                model.load_model()
                return model
            elif config['model_type'] == 'qwen3':
                model = Qwen3Generator(config)
                model.load_model()
                return model
            elif config['model_type'] == 'openai':
                model = OpenAIGenerator(config)
                model.load_model()
                return model
            elif config['model_type'] == 'ministral':
                model = MinistralGenerator(config)
                model.load_model()
                return model
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
        
        elif self.task_type == "vlm_tagging":
            if config['model_type'] == 'blip2':
                model = BLIP2Captioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'qwen3_vl': 
                model = Qwen3VLCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'metaclip2':     
                model = MetaCLIP2Captioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'openai':      
                model = OpenAICaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'llama-vision':
                model = Llama32VisionCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'internvl':
                model = InternVLCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'sailvl':
                model = SAILVLCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'ministral3_vl':
                model = Ministral3VLCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'pixtral':
                model = Pixtral12BCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'glm4v':
                model = GLM46VCaptioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'idefics3':
                model = Idefics3Captioner(config)
                model.load_model()
                return model
            elif config['model_type'] == 'step3vl':
                model = Step3VLCaptioner(config)
                model.load_model()
                return model
        
        raise ValueError(f"Unknown model: {self.model_name}")
    def evaluate(self):
        """Run evaluation"""
        print(f"\n{'='*50}")
        print(f"Starting evaluation")
        print(f"Task: {self.task_type}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model_name}")
        print(f"{'='*50}\n")
        
        results = []
        
        # VLM tasks use a different processing flow
        if self.task_type == "vlm_tagging":
            # Iterate over the dataset
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                item = self.dataset[idx]
                
                # Get image and ground truth captions (using standard format)
                image = item['input']
                true_captions = item['label']  # Keep as list
                
                # If not a list, convert to list
                if not isinstance(true_captions, list):
                    true_captions = [true_captions]
                
                # Take the first one for prediction
                true_caption_for_pred = true_captions[0] if true_captions else ""
                
                # Model prediction
                prediction = self.model.predict(
                    image=image,
                    true_caption=true_caption_for_pred,
                    index=idx
                )
                
                # Build result record (VLM specific format, keep all captions)
                result = {
                    'index': idx,
                    'true_captions': true_captions,  # Keep all captions (list)
                    'predicted_caption': prediction['predicted_caption'],
                    'metadata': item.get('metadata', {})
                }
                
                results.append(result)
        elif self.task_type == "llm_generation":
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                item = self.dataset[idx]
                
                prompt = item['input']
                reference = item['label']  # right_response or best_answer
                
                # Model generation
                prediction = self.model.predict(prompt)
                
                # Build result based on dataset type
                if self.dataset_name == "HaluEval":
                    result = {
                        'index': idx,
                        'prompt': prompt,
                        'generated_answer': prediction['generated_answer'],
                        'right_response': reference,
                        'hallucinated_response': item['metadata'].get('hallucinated_response', ''),
                        'knowledge': item['metadata'].get('knowledge', ''),
                        'dialogue_history': item['metadata'].get('dialogue_history', ''),
                        'raw_output': prediction.get('raw_output', {})
                    }
                else:  # TruthfulQA and other datasets
                    result = {
                        'index': idx,
                        'question': prompt,
                        'best_answer': reference,
                        'generated_answer': prediction['generated_answer'],
                        'metadata': item['metadata'],
                        'raw_output': prediction.get('raw_output', {})
                    }
                
                results.append(result)
        elif self.task_type == "text_classification":  # Special handling for text classification
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                item = self.dataset[idx]
                
                text = item['input']
                true_label = item['label']
                
                # Check if there are choices (multiple-choice datasets like MMLU)
                choices = item['metadata'].get('choices', None)
                
                # Call different method depending on whether choices exist
                if choices:
                    prediction = self.model.predict(text, choices=choices)
                else:
                    prediction = self.model.predict(text)
                
                # Build result record
                result = {
                    'index': idx,
                    'input_text': text[:200],  # Save first 200 characters
                    'true_label': true_label,
                    'true_label_name': item.get('label_name'),
                    'predicted_label': prediction['prediction'],
                    'predicted_label_name': prediction.get('prediction_name'),
                    'confidence': prediction['confidence'],
                    'correct': true_label == prediction['prediction'],  # Add correctness flag
                    'top5_predictions': prediction['top5_predictions'],
                    'top5_prediction_names': prediction.get('top5_prediction_names'),
                    'top5_confidences': prediction['top5_confidences'],
                    'metadata': item['metadata'],
                    'raw_output': prediction.get('raw_output', {})
                }
                
                # If multiple-choice, add actual answer content
                if choices:
                    result['choices'] = choices
                    result['true_answer'] = choices[true_label] if 0 <= true_label < len(choices) else None
                    result['predicted_answer'] = choices[prediction['prediction']] if 0 <= prediction['prediction'] < len(choices) else None
                
                results.append(result)
        
        elif self.task_type == "image_classification":  # Image classification
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                item = self.dataset[idx]
                
                # Model prediction
                prediction = self.model.predict(item['input'])

                # Build result record
                result = {
                    'index': idx,
                    'input_info': "Image data (not serializable)",
                    'true_label': item['label'],
                    'true_label_name': item.get('label_name'),
                    'predicted_label': prediction['prediction'],
                    'predicted_label_name': prediction.get('prediction_name'),
                    'confidence': prediction['confidence'],
                    'correct': item['label'] == prediction['prediction'],  # Add correctness flag
                    'top5_predictions': prediction['top5_predictions'],
                    'top5_prediction_names': prediction.get('top5_prediction_names'),
                    'top5_confidences': prediction['top5_confidences'],
                    'metadata': item['metadata'],
                    'raw_output': prediction.get('raw_output', {})
                }
                
                results.append(result)
        
        else:  # Other tasks (LLM scoring, etc.)
            for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
                item = self.dataset[idx]
                
                # Model prediction
                prediction = self.model.predict(item['input'])

                # Build result record
                result = {
                    'index': idx,
                    'true_label': item['label'],
                    'true_label_name': item.get('label_name'),
                    'predicted_label': prediction['prediction'],
                    'predicted_label_name': prediction.get('prediction_name'),
                    'confidence': prediction['confidence'],
                    'top5_predictions': prediction['top5_predictions'],
                    'top5_prediction_names': prediction.get('top5_prediction_names'),
                    'top5_confidences': prediction['top5_confidences'],
                    'metadata': item['metadata'],
                    'raw_output': prediction.get('raw_output', {})
                }

                results.append(result)
        # else:
        #     # Original flow for other tasks (classification, scoring)
        #     for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
        #         item = self.dataset[idx]
                
        #         # Model prediction
        #         prediction = self.model.predict(item['input'])
                
        #         # Build result record
        #         result = {
        #             'index': idx,
        #             'true_label': item['label'],
        #             'true_label_name': item.get('label_name'),
        #             'predicted_label': prediction['prediction'],
        #             'predicted_label_name': prediction.get('prediction_name'),
        #             'confidence': prediction['confidence'],
        #             'top5_predictions': prediction['top5_predictions'],
        #             'top5_prediction_names': prediction.get('top5_prediction_names'),
        #             'top5_confidences': prediction['top5_confidences'],
        #             'metadata': item['metadata'],
        #             'raw_output': prediction.get('raw_output', {})
        #         }
                
        #         # Add task-specific information based on task type
        #         if self.task_type == "image_classification":
        #             result['input_info'] = "Image data (not serializable)"
        #         elif self.task_type == "text_classification":
        #             result['input_text'] = item['input'][:200]  # Save first 200 characters
        #         elif self.task_type == "llm_generation":
        #             result['input_question'] = item['input']
                
        #         results.append(result)
        
        # Save results
        self._save_results(results)
        
        return results

    def _save_results(self, results):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create task-specific output directory
        task_dir = os.path.join(self.output_dir, self.task_type)
        os.makedirs(task_dir, exist_ok=True)
        
        # Filename
        filename = f"{self.dataset_name}_{self.model_name}_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(task_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
        
        # VLM tasks also generate CSV
        if self.task_type == "vlm_tagging":
            # Generate CSV (one sample per row)
            csv_data = []
            for r in results:
                # Merge multiple ground truth captions for display
                true_captions_str = " | ".join(r['true_captions'])
                
                csv_row = {
                    'index': r['index'],
                    'predicted_caption': r['predicted_caption'],
                    'true_caption_1': r['true_captions'][0] if len(r['true_captions']) > 0 else "",
                    'true_caption_2': r['true_captions'][1] if len(r['true_captions']) > 1 else "",
                    'true_caption_3': r['true_captions'][2] if len(r['true_captions']) > 2 else "",
                    'true_caption_4': r['true_captions'][3] if len(r['true_captions']) > 3 else "",
                    'true_caption_5': r['true_captions'][4] if len(r['true_captions']) > 4 else "",
                    'num_true_captions': len(r['true_captions']),
                    'filename': r['metadata'].get('filename', ''),
                    'img_id': r['metadata'].get('img_id', '')
                }
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(task_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV saved to: {csv_path}")
            
            # Print VLM statistics
            print(f"\n{'='*50}")
            print(f"VLM Captioning Results")
            print(f"{'='*50}")
            print(f"Total samples: {len(results)}")
            
            # Show first few examples
            print(f"\nSample predictions (first 3):")
            for result in results[:3]:
                print(f"\n[Index {result['index']}]")
                print(f"  Predicted: {result['predicted_caption'][:80]}...")
                print(f"  True (5 captions):")
                for i, cap in enumerate(result['true_captions'][:5], 1):
                    print(f"    {i}. {cap[:70]}...")
            
            print(f"\n{'='*50}")
            print("Note: Use separate scripts to calculate BLEU, CIDEr, etc.")
            print(f"{'='*50}")
            
            # Calculate file size
            file_size_kb = os.path.getsize(json_path) / 1024
            print(f"\nJSON file size: {file_size_kb:.2f} KB")
            print(f"CSV file size: {os.path.getsize(csv_path) / 1024:.2f} KB")
        elif self.task_type == "llm_generation":
            for r in results:
                print(f"\n[Index {r['index']}]")
                print(f"  Question: {r['question']}")
                print(f"  Best Answer: {r['best_answer']}")
                print(f"  Generated Answer: {r['generated_answer']}")
                print(f"  Metadata: {r['metadata']}")
                print(f"  Raw Output: {r['raw_output']}")
                print(f"{'='*50}")
        else:
            # Save CSV summary for other tasks
            df_results = []
            for r in results:
                df_row = {
                    'index': r['index'],
                    'true_label': r['true_label'],
                    'predicted_label': r['predicted_label'],
                    'confidence': r['confidence'],
                    'correct': r['true_label'] == r['predicted_label'] if isinstance(r['true_label'], type(r['predicted_label'])) else False
                }
                df_results.append(df_row)
            
            df = pd.DataFrame(df_results)
            csv_path = os.path.join(task_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Summary saved to: {csv_path}")
            
            # # Print simple statistics
            # if 'correct' in df.columns:
            #     accuracy = df['correct'].mean()
            #     print(f"\nAccuracy: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})")
            # # Print simple statistics
            if 'correct' in df.columns:
                accuracy = df['correct'].mean()
                print(f"\nAccuracy: {accuracy:.4f} ({df['correct'].sum()}/{len(df)})") 
                
                # Add the following code to print detailed sample output
                print(f"\n{'='*80}")
                print(f"Detailed Sample Results (first 5):")
                print(f"{'='*80}")
                for r in results[:5]:  # Print first 5 samples
                    print(f"\n[Sample {r['index']}]")
                    print(f"Question: {r.get('input_text', 'N/A')}")

                    # If there are choices (multiple-choice like MMLU)
                    if 'choices' in r:
                        print(f"Choices:")
                        for i, choice in enumerate(r['choices']):
                            marker = "✓" if i == r['true_label'] else " "
                            pred_marker = "→" if i == r['predicted_label'] else " "
                            print(f"  {marker}{pred_marker} {chr(65+i)}. {choice}")
                    
                    print(f"Correct answer: {r.get('true_answer', r.get('true_label_name', r['true_label']))}")
                    print(f"Predicted answer: {r.get('predicted_answer', r.get('predicted_label_name', r['predicted_label']))}")
                    print(f"Confidence: {r['confidence']:.4f}")
                    print(f"Result: {'Correct' if r.get('correct', False) else 'Wrong'}")
                    
                    # Print top-5 predictions
                    if 'top5_prediction_names' in r and r['top5_prediction_names']:
                        print(f"Top-5 predictions: {r['top5_prediction_names']}")
                        print(f"Top-5 confidences: {[f'{c:.3f}' for c in r['top5_confidences']]}")

                    # Print raw output (if available)
                    if 'raw_output' in r and r['raw_output']:
                        print(f"Raw output: {r['raw_output']}")
                    
                    print(f"{'-'*80}")
                
                # Print indices of all incorrect samples
                wrong_indices = [r['index'] for r in results if not r.get('correct', True)]
                print(f"\n{'='*80}")
                print(f"Incorrect sample indices ({len(wrong_indices)} total): {wrong_indices}")
                print(f"{'='*80}")
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Independence Study - Evaluation")
    parser.add_argument('--task', type=str, required=True,
                       choices=['image_classification', 'text_classification', 'llm_generation', 'vlm_tagging'],
                       help='Task type')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., CIFAR-100, AG_News, TruthfulQA, Flickr30k)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., blip2, llama3.1-8b, gpt-4o-mini)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = ModelEvaluator(
        task_type=args.task,
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()