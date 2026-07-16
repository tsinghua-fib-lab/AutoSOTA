
"""
Image classification models - unified single-generation approach
Both BLIP2 and Qwen3 use single generation + match scoring
"""

from typing import Dict, Any, List
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModel
)
from PIL import Image
import numpy as np
import re
import random
from .base_model import BaseModel
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import base64
from io import BytesIO


class BLIP2ImageClassifier(BaseModel):
    """BLIP2 for image classification - single generation version"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load BLIP2 model"""
        print(f"Loading BLIP2 model: {self.model_name}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Single generation strategy - identical to Qwen3
        """
        if self.model is None:
            self.load_model()
        
        # Method 1: First generate a free-form description
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, 
            torch.float16 if self.device == 'cuda' else torch.float32
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=30)
        
        generated_desc = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Method 2: Match categories using the description (no prompt, since BLIP2 doesn't follow instructions)
        class_scores = self._calculate_match_scores_from_description(generated_desc)
        
        # Format output
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5 (avoid double softmax)
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_description': generated_desc,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _calculate_match_scores_from_description(self, description: str) -> np.ndarray:
        """
        Calculate match scores based on the generated description
        Same logic as Qwen3
        """
        desc_lower = description.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            # Scoring rules (identical to Qwen3)
            if class_lower == desc_lower:
                class_scores[i] = 10.0
            elif class_lower in desc_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif desc_lower in class_lower:
                class_scores[i] = 3.0
            else:
                # Word-level matching
                class_words = set(class_lower.split())
                desc_words = set(desc_lower.split())
                common_words = class_words.intersection(desc_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        # Normalize (same as Qwen3)
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class Qwen3VLImageClassifier(BaseModel):
    """Qwen3-VL for image classification - single generation version"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load Qwen3-VL model"""
        print(f"Loading Qwen3-VL model: {self.model_name}")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"Model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - same as BLIP2"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        output = self._generate_with_qwen(messages, image, max_tokens=20, temperature=0.1)
        
        # Use the same matching algorithm as BLIP2
        class_scores = self._calculate_match_scores(output)
        
        # predicted_class = int(np.argmax(class_scores))
        # confidence = float(class_scores[predicted_class])
        
        # top5_indices, top5_probs = self.get_top_k_predictions(
        #     class_scores, 
        #     k=min(5, self.num_classes)
        # )
        
        # if predicted_class not in top5_indices:
        #     top5_indices[-1] = predicted_class
        #     top5_probs[-1] = confidence
        
        # return {
        #     'prediction': predicted_class,
        #     'prediction_name': self.class_names[predicted_class],
        #     'confidence': confidence,
        #     'top5_predictions': top5_indices,
        #     'top5_prediction_names': [self.class_names[i] for i in top5_indices],
        #     'top5_confidences': top5_probs,
        #     'raw_output': {
        #         'generated_text': output,
        #         'all_scores': class_scores.tolist(),
        #         'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
        #     }
        # }
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - same as BLIP2"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores
    
    def _generate_with_qwen(self, messages: list, image: Image.Image, 
                           max_tokens: int = 30, temperature: float = 0.1) -> str:
        """Qwen3 generation helper function"""
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        return output_text
class MetaCLIP2ImageClassifier(BaseModel):
    """MetaCLIP2 - Image-text contrastive approach (Zero-shot)"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        print(f"Loading MetaCLIP2 model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()
        
        # Image processing
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Text processing
        text_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.class_names]
        text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Feature extraction and similarity computation
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features @ text_features.T) * 100
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        return self._format_output(probs)
    
    def _format_output(self, class_scores: np.ndarray, text: str = "") -> Dict[str, Any]:
        """Format output - fixed version"""
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])
        
        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()
        
        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence
        
        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': text,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
class OpenAIVisionImageClassifier(BaseModel):
    """OpenAI Vision model - optimized version"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        
    def load_model(self):
        print(f"Using OpenAI model: {self.model_name}")
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.api_key is None:
            self.load_model()
        
        # Optimization: compress image
        image_base64 = self._image_to_base64(image)
        
        # Optimization: shorten prompt
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""Classify this image into ONE category: {class_list}
Answer ONLY the category name."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"  # Use low-resolution mode
                        }
                    }
                ]
            }
        ]
        
        # Detect model type
        model_lower = self.model_name.lower()
        is_reasoning = any(x in model_lower for x in ['o1', 'o3', 'gpt-5'])
        
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": 100 if is_reasoning else 20,  # Reduce tokens
        }
        
        if not is_reasoning:
            api_params['temperature'] = 0.1
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content
            output = content.strip() if content else ""
            
            print(f"GPT Output: '{output}'")
            
            class_scores = self._calculate_match_scores(output)
            return self._format_output(class_scores, output)
            
        except Exception as e:
            print(f"API Error: {e}")
            class_scores = np.ones(self.num_classes) / self.num_classes
            return self._format_output(class_scores, f"Error: {str(e)}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Compress image to reduce tokens"""
        buffered = BytesIO()
        
        # Convert mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Resize
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Reduce quality
        image.save(buffered, format="JPEG", quality=70)
        return base64.b64encode(buffered.getvalue()).decode()

    
    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """
        Calculate match scores - identical to BLIP2/Qwen3
        """
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            # Exact match
            if class_lower == output_lower:
                class_scores[i] = 10.0
            # Output contains class name
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            # Class name contains output
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            # Word-level matching
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = sum(len(word) * 0.1 for word in common_words if len(word) > 2)
                    class_scores[i] = score
        
        # Normalize (consistent with other models)
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores
    
    def _format_output(self, class_scores: np.ndarray, text: str = "") -> Dict[str, Any]:
        """Format output - fixed version"""
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])
        
        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()
        
        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence
        
        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': text,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
class InternVLImageClassifier(BaseModel):
    """InternVL3.5-8B for image classification - single generation version"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load InternVL model"""
        print(f"Loading InternVL model: {self.model_name}")
        
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"Model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - consistent with other models"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        output = self._generate_with_internvl(image, prompt, max_tokens=20)
        
        # Use the same matching algorithm as other models
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _generate_with_internvl(self, image: Image.Image, prompt: str, 
                                max_tokens: int = 30) -> str:
        """InternVL generation helper function"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # InternVL uses specific image preprocessing
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size=448):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        transform = build_transform(input_size=448)
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self.model.device)
        
        # Build conversation format
        question = f"<image>\n{prompt}"
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        
        try:
            # InternVL uses the chat method
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config
            )
            return response.strip()
        except Exception as e:
            print(f"InternVL generation error: {e}")
            return ""

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with other models"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class SAILVLImageClassifier(BaseModel):
    """SAIL-VL2-8B for image classification - single generation version"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load SAIL-VL2 model"""
        print(f"Loading SAIL-VL2 model: {self.model_name}")
        
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        
        # Load model first (this triggers the correct config download)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        # Then load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print(f"SAIL-VL2 model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - consistent with other models"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        output = self._generate_with_sailvl(image, prompt, max_tokens=20)
        
        # Use the same matching algorithm as other models
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _generate_with_sailvl(self, image: Image.Image, prompt: str, 
                               max_tokens: int = 30) -> str:
        """SAIL-VL2 generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device).to(torch.bfloat16)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        # Decode
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Remove chat template end marker
        response = response.split('<|im_end|>')[0].strip()

        # Try to remove the input prompt portion (if included in output)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with other models"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class CLIPImageClassifier(BaseModel):
    """CLIP (LAION) - Image-text contrastive approach (Zero-shot)"""
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load CLIP model"""
        print(f"Loading CLIP model: {self.model_name}")
        from transformers import CLIPProcessor, CLIPModel
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        CLIP zero-shot classification
        Same logic as MetaCLIP2
        """
        if self.model is None:
            self.load_model()
        
        # Image processing
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Text processing (use template to improve performance)
        text_prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.class_names]
        text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # Feature extraction and similarity computation
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (scaling factor 100)
            logits = (image_features @ text_features.T) * 100
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        return self._format_output(probs)
    
    def _format_output(self, class_scores: np.ndarray) -> Dict[str, Any]:
        """Format output - consistent with other models"""
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])
        
        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()
        
        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence
        
        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }


class Ministral3VLImageClassifier(BaseModel):
    """Ministral 3 8B Vision (mistralai/Ministral-3-8B-Instruct-2512) for image classification

    This is Mistral AI's vision-language model, different from the text-only Ministral-8B-Instruct-2410.
    Supports vision capabilities for image understanding and classification.

    https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
    """
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load Ministral 3 VL model"""
        print(f"Loading Ministral 3 VL model: {self.model_name}")
        
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        
        self.tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto"
        )
        self.model.eval()
        print(f"Ministral 3 VL model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - consistent with other models"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        output = self._generate_with_ministral3(image, prompt, max_tokens=20)
        
        # Use the same matching algorithm as other models
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _generate_with_ministral3(self, image: Image.Image, prompt: str, 
                                   max_tokens: int = 30) -> str:
        """Ministral 3 VL generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to base64 URL
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/png;base64,{img_base64}"

        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True
        )
        
        # Move to device
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=torch.bfloat16, 
                device=self.model.device
            )
        
        # Get image_sizes
        image_sizes = None
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            image_sizes = [inputs["pixel_values"].shape[-2:]]
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                image_sizes=image_sizes,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Decode - only take the generated portion
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
        ).strip()
        
        return response

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with other models"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class Pixtral12BImageClassifier(BaseModel):
    """Pixtral-12B (mistral-community/pixtral-12b) for image classification

    This is Mistral community's Pixtral model, using the Llava architecture.
    Supports vision capabilities for image understanding and classification.

    https://huggingface.co/mistral-community/pixtral-12b
    """
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Pixtral-12B model"""
        print(f"Loading Pixtral-12B model: {self.model_name}")
        
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print(f"Pixtral-12B model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - consistent with other models"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        output = self._generate_with_pixtral(image, prompt, max_tokens=20)
        
        # Use the same matching algorithm as other models
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _generate_with_pixtral(self, image: Image.Image, prompt: str, 
                                max_tokens: int = 30) -> str:
        """Pixtral-12B generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Build message format (using chat template)
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text_prompt = self.processor.apply_chat_template(chat)

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Decode
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Remove input prompt portion (if included in output)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with other models"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class GLM46VImageClassifier(BaseModel):
    """GLM-4.6V-Flash (zai-org/GLM-4.6V-Flash) for image classification

    This is the lightweight version (9B parameters) of the GLM-4.6V series by Zhipu AI.
    Supports visual understanding and native multimodal function calling capabilities.

    Acceleration optimizations (by priority):
    - SDPA: PyTorch built-in Scaled Dot Product Attention (stable, recommended)
    - Flash Attention 2: Faster but requires compatible CUDA environment
    - torch.compile: PyTorch 2.0+ compilation optimization

    https://huggingface.co/zai-org/GLM-4.6V-Flash
    """
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.use_flash_attn = config.get('use_flash_attn', False)  # Disabled by default (CUDA compatibility issues)
        self.use_sdpa = config.get('use_sdpa', True)  # Enabled by default (more stable)
        self.use_compile = config.get('use_compile', False)  # Disabled by default
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load GLM-4.6V-Flash model with acceleration optimizations"""
        print(f"Loading GLM-4.6V-Flash model: {self.model_name}")
        print(f"  - Flash Attention 2: {self.use_flash_attn}")
        print(f"  - SDPA: {self.use_sdpa}")
        print(f"  - torch.compile: {self.use_compile}")
        
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Model loading parameters
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        # Select attention implementation (priority: flash_attention_2 > sdpa > eager)
        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  → Using Flash Attention 2...")
        elif self.use_sdpa:
            model_kwargs["attn_implementation"] = "sdpa"
            print("  → Using SDPA (Scaled Dot Product Attention)...")
        else:
            print("  → Using default (eager) attention...")
        
        try:
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            # If the specified attention implementation fails, fall back to default
            print(f"  → Attention implementation failed: {e}")
            print("  → Falling back to default (eager) attention...")
            model_kwargs.pop("attn_implementation", None)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        
        self.model.eval()
        
        # Try torch.compile (PyTorch 2.0+)
        if self.use_compile:
            try:
                print("  → Applying torch.compile optimization...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  → torch.compile applied successfully")
            except Exception as e:
                print(f"  → torch.compile failed (will use eager mode): {e}")
        
        print(f"GLM-4.6V-Flash model loaded successfully!")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Single generation strategy - consistent with other models"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        output = self._generate_with_glm4v(image, prompt, max_tokens=20)
        
        # Use the same matching algorithm as other models
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 URL (JPEG is faster)"""
        from io import BytesIO
        import base64
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)  # JPEG is much faster than PNG
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_base64}"
    
    def _generate_with_glm4v(self, image: Image.Image, prompt: str, 
                             max_tokens: int = 30) -> str:
        """GLM-4.6V-Flash generation helper function (single sample)"""
        image_url = self._image_to_base64(image)
        
        # Build message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Remove token_type_ids (if present) - per official documentation
        inputs.pop("token_type_ids", None)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Decode - only take the generated portion
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return output_text
    
    def predict_batch(self, images: List[Image.Image], batch_size: int = 4) -> List[Dict[str, Any]]:
        """Batch prediction - improve GPU utilization"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        prompt = f"""This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and tell me which category it belongs to. Answer with ONLY the category name."""
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_outputs = self._generate_batch(batch_images, prompt, max_tokens=20)
            
            for output in batch_outputs:
                class_scores = self._calculate_match_scores(output)
                predicted_class = int(np.argmax(class_scores))
                confidence = float(class_scores[predicted_class])
                
                sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
                top5_indices = sorted_indices.tolist()
                top5_probs = class_scores[sorted_indices].tolist()
                
                if predicted_class not in top5_indices:
                    top5_indices[-1] = predicted_class
                    top5_probs[-1] = confidence
                
                results.append({
                    'prediction': predicted_class,
                    'prediction_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'top5_predictions': top5_indices,
                    'top5_prediction_names': [self.class_names[j] for j in top5_indices],
                    'top5_confidences': top5_probs,
                    'raw_output': {
                        'generated_text': output,
                        'all_scores': class_scores.tolist(),
                        'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
                    }
                })
        
        return results
    
    def _generate_batch(self, images: List[Image.Image], prompt: str, 
                        max_tokens: int = 30) -> List[str]:
        """Batch generation - true batch processing"""
        # Prepare all messages
        batch_messages = []
        for image in images:
            image_url = self._image_to_base64(image)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt}
                ]
            }]
            batch_messages.append(messages)
        
        # Batch apply chat template (with padding)
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True  # Key: enable padding
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        input_lengths = [inputs["input_ids"].shape[1]] * len(images)
        
        # Batch generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Batch decode
        outputs = []
        for i in range(len(images)):
            output_text = self.processor.decode(
                generated_ids[i][input_lengths[i]:],
                skip_special_tokens=True
            ).strip()
            outputs.append(output_text)
        
        return outputs

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with other models"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class Idefics3ImageClassifier(BaseModel):
    """Idefics3-8B-Llama3 (HuggingFaceM4/Idefics3-8B-Llama3) for image classification

    This is an open-source multimodal model released by Hugging Face, based on SigLIP + Llama 3.1 8B.
    Significantly improved over Idefics2 in OCR, document understanding, and visual reasoning.

    https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
    """
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Idefics3 model"""
        print(f"Loading Idefics3 model: {self.model_name}")
        
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print(f"Idefics3 model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Simplified prediction - directly output model-generated class name without computing confidence"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        # Improved prompt: emphasize that an exact class name must be selected from the list
        prompt = f"""This is an image classification task. The possible categories are:
{class_list}

Look at the image carefully and identify which EXACT category from the list above it belongs to.
You MUST choose one category name from the list. Answer with ONLY the exact category name, nothing else."""
        
        output = self._generate_with_idefics3(image, prompt, max_tokens=20)
        
        # Clean output
        import re
        output_clean = re.sub(r'[^\w\s\-]', '', output).strip().lower()
        
        # Find the best matching category
        best_match_idx = 0
        best_match_score = 0
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            # Exact match
            if class_lower == output_clean:
                best_match_idx = i
                best_match_score = 100
                break

            # Output contains class name
            if class_lower in output_clean:
                score = 50 + len(class_lower)
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i
            
            # Class name contains output
            elif output_clean in class_lower and len(output_clean) > 2:
                score = 30 + len(output_clean)
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i
            
            # Word matching
            else:
                class_words = set(class_lower.split())
                output_words = set(output_clean.split())
                common = class_words & output_words
                if common:
                    score = sum(len(w) for w in common if len(w) > 2)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = i
        
        return {
            'prediction': best_match_idx,
            'prediction_name': self.class_names[best_match_idx],
            'confidence': 1.0 if best_match_score > 0 else 0.0,
            'top5_predictions': [best_match_idx],
            'top5_prediction_names': [self.class_names[best_match_idx]],
            'top5_confidences': [1.0],
            'raw_output': {
                'generated_text': output
            }
        }
    
    def _generate_with_idefics3(self, image: Image.Image, prompt: str, 
                                 max_tokens: int = 30) -> str:
        """Idefics3 generation helper function - following official examples"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Build message format (following Idefics3 official documentation)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template (returns a string)
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs (following the official example approach)
        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Record input length for later extraction
        input_len = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Method 1: Only decode newly generated tokens (most reliable method)
        new_token_ids = generated_ids[0, input_len:]
        response = self.processor.decode(new_token_ids, skip_special_tokens=True).strip()
        
        # DEBUG: Print output of first 3 samples for debugging
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"\n[DEBUG Idefics3 #{self._debug_count}]")
            print(f"  input_len: {input_len}")
            print(f"  generated_ids shape: {generated_ids.shape}")
            print(f"  new_token_ids: {new_token_ids.tolist()}")
            print(f"  response (decoded): '{response}'")
            # Also decode full output for comparison
            full_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"  full_output length: {len(full_output)}")
            print(f"  full_output last 200 chars: '{full_output[-200:]}'")
            self._debug_count += 1
        
        # If response is empty, try fallback method
        if not response:
            full_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Look for "assistant" keyword
            lower_output = full_output.lower()
            if "assistant" in lower_output:
                idx = lower_output.rfind("assistant")
                response = full_output[idx + len("assistant"):].strip()
            elif prompt in full_output:
                response = full_output.split(prompt)[-1].strip()
        
        # Clean up potentially remaining special markers
        for marker in ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        return response.strip()

    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - improved version: supports generalized category matching"""
        import re
        
        # Clean output: remove punctuation, convert to lowercase
        output_clean = re.sub(r'[^\w\s]', '', output.lower()).strip()
        output_words = set(output_clean.split())
        
        class_scores = np.zeros(self.num_classes)
        
        # Generalized category mapping (common hypernyms -> possible ImageNet subclass keywords)
        generic_category_hints = {
            'bird': ['bird', 'cock', 'hen', 'owl', 'eagle', 'hawk', 'finch', 'sparrow', 
                     'warbler', 'robin', 'jay', 'magpie', 'chickadee', 'coucal', 'hummingbird',
                     'parrot', 'macaw', 'lorikeet', 'cockatoo', 'toucan', 'hornbill', 'kingfisher',
                     'bee eater', 'jacamar', 'drake', 'goose', 'swan', 'pelican', 'flamingo',
                     'crane', 'bustard', 'limpkin', 'coot', 'oystercatcher', 'kite', 'vulture',
                     'goldfinch', 'junco', 'brambling', 'indigo bunting', 'albatross'],
            'dog': ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'setter', 'pointer',
                    'collie', 'shepherd', 'bulldog', 'poodle', 'corgi', 'beagle', 'pug',
                    'mastiff', 'boxer', 'rottweiler', 'doberman', 'schnauzer', 'husky',
                    'malamute', 'samoyed', 'dalmatian', 'greyhound', 'whippet', 'basenji',
                    'chihuahua', 'papillon', 'maltese', 'shih-tzu', 'pekinese', 'blenheim',
                    'wolfhound', 'deerhound', 'saluki', 'afghan', 'redbone', 'vizsla'],
            'cat': ['cat', 'tabby', 'persian', 'siamese', 'tiger cat', 'egyptian cat', 'cougar',
                    'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah'],
            'fish': ['fish', 'tench', 'goldfish', 'shark', 'ray', 'sturgeon', 'gar', 'lionfish',
                     'puffer', 'barracouta', 'eel', 'coho', 'rock beauty', 'anemone fish'],
            'snake': ['snake', 'boa', 'python', 'cobra', 'mamba', 'viper', 'rattlesnake',
                      'garter snake', 'water snake', 'vine snake', 'night snake', 'sidewinder'],
            'insect': ['insect', 'ant', 'bee', 'fly', 'butterfly', 'moth', 'beetle', 'dragonfly',
                       'damselfly', 'grasshopper', 'cricket', 'cockroach', 'mantis', 'cicada',
                       'leafhopper', 'lacewing', 'walking stick', 'ladybug', 'weevil'],
            'car': ['car', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'sedan',
                    'sports car', 'racer', 'ambulance', 'beach wagon', 'pickup'],
            'flower': ['flower', 'daisy', 'rose', 'tulip', 'sunflower', 'orchid', 'lily',
                       'lotus', 'hibiscus', 'buttercup', 'columbine', 'poppy'],
            'fruit': ['fruit', 'apple', 'orange', 'banana', 'strawberry', 'lemon', 'fig',
                      'pineapple', 'pomegranate', 'custard apple', 'jackfruit', 'granny smith'],
            'vegetable': ['vegetable', 'cucumber', 'zucchini', 'artichoke', 'bell pepper',
                          'cardoon', 'mushroom', 'cauliflower', 'broccoli', 'cabbage', 'head cabbage'],
            'mountain': ['mountain', 'volcano', 'alp', 'cliff', 'promontory', 'valley'],
            'boat': ['boat', 'canoe', 'kayak', 'gondola', 'catamaran', 'trimaran', 'fireboat',
                     'speedboat', 'lifeboat', 'yawl', 'aircraft carrier', 'submarine'],
            'plane': ['plane', 'airplane', 'airliner', 'warplane', 'airship'],
            'furniture': ['furniture', 'chair', 'table', 'desk', 'bed', 'sofa', 'couch',
                          'bookcase', 'cabinet', 'wardrobe', 'throne', 'rocking chair'],
        }
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            class_words_set = set(class_lower.split())
            
            # 1. Exact match (highest priority)
            if class_lower == output_clean:
                class_scores[i] = 10.0
                continue
            
            # 2. Output contains complete class name
            if class_lower in output_clean:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
                continue
            
            # 3. Class name contains output (e.g., output "dog" matches "italian greyhound")
            if output_clean in class_lower:
                class_scores[i] = 4.0
                continue
            
            # 4. Generalized category matching (e.g., output "bird" matches all bird types)
            for generic, hints in generic_category_hints.items():
                if generic in output_words:
                    # Check if class name contains any hint words
                    for hint in hints:
                        if hint in class_lower or any(h in class_lower for h in hint.split()):
                            class_scores[i] = 2.0  # Generalized match gets a lower score
                            break
                    if class_scores[i] > 0:
                        break
            
            if class_scores[i] > 0:
                continue
            
            # 5. Word intersection matching
            common_words = class_words_set.intersection(output_words)
            if common_words:
                score = 0
                for word in common_words:
                    if len(word) > 2:
                        score += len(word) * 0.1
                class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            # When there is still no match, return uniform distribution
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores


class Gemma3ImageClassifier(BaseModel):
    """Google Gemma 3 4B (google/gemma-3-4b-it) for image classification

    Gemma 3 is a multimodal model released by Google, supporting text and image input with a 128K context window.
    Requires transformers >= 4.50.0

    Improvements (referencing Qwen3-VL and official documentation):
    1. Use do_pan_and_scan=True to handle high-resolution images
    2. Add system prompt to provide expert role
    3. Use matching algorithm consistent with Qwen3-VL

    https://huggingface.co/google/gemma-3-4b-it
    https://huggingface.co/docs/transformers/en/model_doc/gemma3
    """
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Gemma 3 model"""
        print(f"Loading Gemma 3 model: {self.model_name}")
        
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print(f"Gemma 3 model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Prediction - using the same strategy as Qwen3-VL"""
        if self.model is None:
            self.load_model()
        
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        # Use few-shot examples to guide the model to output precise category names
        prompt = f"""Classify this image into ONE of the ImageNet categories below.

IMPORTANT: You must output EXACTLY one category name from the list, not a general description.
For example:
- If you see a small black and white bird, output "chickadee" (not "bird")
- If you see a thin elegant dog, output "Italian greyhound" (not "dog")  
- If you see a mountain with smoke, output "volcano" (not "mountain")

Categories: {class_list}

Look at the image carefully. Output ONLY the exact category name from the list above:"""
        
        output = self._generate_with_gemma3(image, prompt, max_tokens=30)
        
        # Use the same matching algorithm as Qwen3-VL
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])

        # Manually compute Top5
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()

        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence

        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores - consistent with Qwen3-VL"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores
    
    def _generate_with_gemma3(self, image: Image.Image, prompt: str, 
                              max_tokens: int = 30) -> str:
        """Gemma 3 generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Build message format - use system prompt to set expert role
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert image classifier specializing in fine-grained visual recognition. You can identify specific species, breeds, and object types with high accuracy. Always output the most specific category name."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template and process inputs
        # Not using do_pan_and_scan, keeping it simple
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        # Record input length for later extraction
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Only decode newly generated tokens
        new_tokens = generated_ids[0][input_len:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        
        # DEBUG: Print output of first 3 samples for debugging
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"[DEBUG Gemma3 #{self._debug_count}]")
            print(f"  input_len: {input_len}")
            print(f"  generated_ids shape: {generated_ids.shape}")
            print(f"  response (decoded): '{response}'")
            self._debug_count += 1
        
        return response.strip()


class Step3VLImageClassifier(BaseModel):
    """Step3-VL-10B for image classification - StepFun's 10B vision-language model"""
    
    # key_mapping required by Step3-VL (per official documentation)
    KEY_MAPPING = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "vit_large_projector": "model.vit_large_projector",
    }
    
    def __init__(self, config: Dict[str, Any], class_names: list):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def load_model(self):
        """Load Step3-VL-10B model"""
        print(f"Loading Step3-VL-10B model: {self.model_name}")
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # Load processor (requires trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # Load model (requires key_mapping and trust_remote_code)
        # Note: Use dtype instead of torch_dtype per official documentation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype="auto",  # Recommended by official documentation
            key_mapping=self.KEY_MAPPING
        ).eval()
        
        print(f"Step3-VL-10B model loaded")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Image classification prediction"""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Build class list
        class_list = ", ".join([c.replace('_', ' ') for c in self.class_names])
        
        # Force English output, prevent model from defaulting to Chinese
        prompt = f"""[IMPORTANT: Reply in English only]

This is an image classification task. The image belongs to ONE of these categories:
{class_list}

Look at the image and identify which category it belongs to.
Output ONLY the exact category name from the list above, nothing else."""
        
        # Generate (increase max_tokens to ensure complete output)
        output = self._generate_with_step3vl(image, prompt, max_tokens=50)
        
        # Match classification
        class_scores = self._calculate_match_scores(output)
        
        predicted_class = int(np.argmax(class_scores))
        confidence = float(class_scores[predicted_class])
        
        # Top5 computation
        sorted_indices = np.argsort(class_scores)[-min(5, self.num_classes):][::-1]
        top5_indices = sorted_indices.tolist()
        top5_probs = class_scores[sorted_indices].tolist()
        
        if predicted_class not in top5_indices:
            top5_indices[-1] = predicted_class
            top5_probs[-1] = confidence
        
        return {
            'prediction': predicted_class,
            'prediction_name': self.class_names[predicted_class],
            'confidence': confidence,
            'top5_predictions': top5_indices,
            'top5_prediction_names': [self.class_names[i] for i in top5_indices],
            'top5_confidences': top5_probs,
            'raw_output': {
                'generated_text': output,
                'all_scores': class_scores.tolist(),
                'num_nonzero_scores': int(np.count_nonzero(class_scores > 0))
            }
        }
    
    def _calculate_match_scores(self, output: str) -> np.ndarray:
        """Calculate match scores"""
        output_lower = output.lower().strip()
        class_scores = np.zeros(self.num_classes)
        
        for i, class_name in enumerate(self.class_names):
            class_lower = class_name.lower().replace('_', ' ').strip()
            
            if class_lower == output_lower:
                class_scores[i] = 10.0
            elif class_lower in output_lower:
                class_scores[i] = 5.0 + len(class_lower) * 0.1
            elif output_lower in class_lower:
                class_scores[i] = 3.0
            else:
                class_words = set(class_lower.split())
                output_words = set(output_lower.split())
                common_words = class_words.intersection(output_words)
                
                if common_words:
                    score = 0
                    for word in common_words:
                        if len(word) > 2:
                            score += len(word) * 0.1
                    class_scores[i] = score
        
        if class_scores.sum() > 0:
            class_scores = class_scores / class_scores.sum()
        else:
            class_scores = np.ones(self.num_classes) / self.num_classes
        
        return class_scores
    
    def _generate_with_step3vl(self, image: Image.Image, prompt: str, 
                                max_tokens: int = 30) -> str:
        """Step3-VL generation helper function"""
        import io
        import base64
        
        # Convert PIL Image to base64 URL (Step3-VL documentation uses URL format)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_base64}"
        
        # Build message format - use system prompt to force English output and disable thinking
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Always respond directly in English without any reasoning or thinking process. Output only the final answer."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": prompt + " /no_think"}
                ]
            }
        ]
        
        # Apply chat template
        try:
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt",
                enable_thinking=False
            ).to(self.model.device)
        except TypeError:
            # If enable_thinking parameter is not supported
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Decode
        decoded = self.processor.decode(
            generated_ids[0, input_len:], 
            skip_special_tokens=True
        ).strip()
        
        # Step3-VL special handling: extract content after </think> tag (if present)
        import re
        if '</think>' in decoded:
            decoded = decoded.split('</think>')[-1].strip()
        
        # If output starts with Chinese characters, try to extract English content within quotes
        if decoded and ord(decoded[0]) > 127:
            quoted = re.findall(r'"([^"]+)"', decoded)
            if quoted:
                decoded = quoted[-1]
            else:
                # Try to extract English sentences
                english_parts = re.findall(r'[A-Z][a-z\s\-]+', decoded)
                if english_parts:
                    decoded = english_parts[-1]
        
        # Remove all Chinese characters and Chinese punctuation (clean up residuals)
        decoded = re.sub(r'[\u4e00-\u9fff，。！？、；：""''【】（）]', '', decoded)
        decoded = decoded.strip()
        
        # DEBUG: Print first 3 samples
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"[DEBUG Step3-VL #{self._debug_count}] output: '{decoded}'")
            self._debug_count += 1
        
        return decoded