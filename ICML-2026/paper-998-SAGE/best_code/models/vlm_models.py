"""
VLM Image Captioning Models
"""

from typing import Dict, Any, List
import torch
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    # Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoModel
)
from PIL import Image
from .base_model import BaseModel
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import base64
from io import BytesIO


class BLIP2Captioner(BaseModel):
    """BLIP2 for image captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load BLIP2 model"""
        print(f"Loading BLIP2 model: {self.model_name}")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        if "flan-t5-xxl" in self.model_name.lower():
            # XXL: recommend using device_map + bfloat16/8bit
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",          # multi-GPU / auto-split
            )
        else:
            # For smaller models like blip2-opt-2.7b
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model.to(self.device)
        # self.model = Blip2ForConditionalGeneration.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        # )
        # self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Generate caption
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, 
            torch.float16 if self.device == 'cuda' else torch.float32
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=3
            )
        
        predicted_caption = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }
class Qwen3VLCaptioner(BaseModel):
    """Qwen3-VL for image captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load Qwen3-VL model"""
        print(f"Loading Qwen3-VL model: {self.model_name}")
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"Model loaded")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (simple caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Generate caption
        predicted_caption = self._generate_with_qwen(messages, image, max_tokens=18)  
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }
    
    def _generate_with_qwen(self, messages: list, image: Image.Image,
                           max_tokens: int = 30) -> str:
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
                do_sample=False,
                # repetition_penalty=1.1          # Avoid repetition
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
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        return self._generate_with_qwen(messages, image, max_tokens=max_tokens)

    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(r'^(The image shows?|This image depicts?|In this image|I see)\s*', '', caption, flags=re.IGNORECASE)
        caption = caption.strip()
        
        # 2. If sentence is incomplete (ends with comma, "and", "with", etc.), truncate to last complete part
        # Check if ends with incomplete word
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                # Find the content before the last comma
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    # If there is no comma, truncate the last few words
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length (avoid excessive length)
        max_length = 120
        if len(caption) > max_length:
            # Truncate to the last complete word
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()

# class Qwen3VLCaptioner(BaseModel):
#     """Qwen3-VL for image caption generation"""
    
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
#     def load_model(self):
#         """Load Qwen3-VL model"""
#         print(f"Loading Qwen3-VL model: {self.model_name}")
        
#         self.model = Qwen3VLForConditionalGeneration.from_pretrained(
#             self.model_name,
#             dtype=torch.bfloat16,
#             device_map="auto"
#         )
#         self.processor = AutoProcessor.from_pretrained(self.model_name)
#         print(f"Model loaded")
    
#     def predict(self, 
#                 image: Image.Image, 
#                 true_caption: str = None,
#                 index: int = None) -> Dict[str, Any]:
#         """Generate image caption"""
#         if self.model is None:
#             self.load_model()
        
#         # Build prompt (simple caption generation task)
#         prompt = "Describe this image in one short sentence. Focus on the main subject and action."
        
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": prompt}
#             ]
#         }]
        
#         # Generate caption
#         predicted_caption = self._generate_with_qwen(messages, image, max_tokens=20)
        
        
#         return {
#             'index': index,
#             'true_caption': true_caption if true_caption else "",
#             'predicted_caption': predicted_caption
#         }
    
#     def _generate_with_qwen(self, messages: list, image: Image.Image, 
#                            max_tokens: int = 50) -> str:
#         """Qwen3 generation helper function"""
#         text = self.processor.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#         )
        
#         image_inputs, video_inputs = process_vision_info(messages)
        
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         )
#         inputs = inputs.to(self.device)
        
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 temperature=0.1,
#                 do_sample=False
#             )
        
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] 
#             for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
        
#         output_text = self.processor.batch_decode(
#             generated_ids_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0].strip()
        
#         return output_text


class MetaCLIP2Captioner(BaseModel):
    """MetaCLIP2 for image captioning (via image-text feature generation)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load MetaCLIP2 model"""
        print(f"Loading MetaCLIP2 model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """
        Generate image caption
        Note: MetaCLIP itself doesn't directly generate text, using simplified strategy here
        Select the best matching caption by similarity with predefined caption templates
        """
        if self.model is None:
            self.load_model()
        
        # Predefine some generic caption templates
        caption_templates = [
            "a photo of an object",
            "a picture showing a scene",
            "an image depicting something",
            "a photograph of a subject",
            "a view of something interesting"
        ]
        
        # Image processing
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
        # Text processing
        text_inputs = self.processor(text=caption_templates, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Compute similarity
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits = (image_features @ text_features.T) * 100
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Select the most similar caption
        best_idx = int(probs.argmax())
        predicted_caption = caption_templates[best_idx]
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }


class OpenAICaptioner(BaseModel):
    """OpenAI Vision for image captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        
    def load_model(self):
        """OpenAI API doesn't need explicit loading"""
        print(f"Using OpenAI model: {self.model_name}")
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.api_key is None:
            self.load_model()
        
        # Compress image
        image_base64 = self._image_to_base64(image)
        
        # Build prompt
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
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
            "max_completion_tokens": 100 if is_reasoning else 20,
        }
        
        if not is_reasoning:
            api_params['temperature'] = 0.1
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content
            predicted_caption = content.strip() if content else ""
            
            return {
                'index': index,
                'true_caption': true_caption if true_caption else "",
                'predicted_caption': predicted_caption
            }
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            return {
                'index': index,
                'true_caption': true_caption if true_caption else "",
                'predicted_caption': f"Error: {str(e)}"
            }
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Compress image to reduce tokens"""
        buffered = BytesIO()
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Reduce size
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Reduce quality
        image.save(buffered, format="JPEG", quality=70)
        return base64.b64encode(buffered.getvalue()).decode()

class Llama32VisionCaptioner(BaseModel):
    """Llama-3.2-11B-Vision-Instruct for image captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load Llama-3.2-Vision model"""
        print(f"Loading Llama-3.2-Vision model: {self.model_name}")
        
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"Model loaded")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Build messages according to official format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Just a placeholder, doesn't contain actual image data
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Generate caption
        predicted_caption = self._generate_with_llama_vision(messages, image, max_tokens=20)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }
    
    def _generate_with_llama_vision(self, messages: list, image: Image.Image,
                                    max_tokens: int = 30) -> str:
        """Llama-3.2-Vision generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Key fix: use correct parameters
        inputs = self.processor(
            images=image,  # Note: changed to images (plural)
            text=input_text,  # Use text parameter
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generation config (minimize parameters to avoid conflicts)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                # Don't pass any other parameters!
            )
        
        # Decode (full decode, then manually extract)
        full_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract generated part from full text
        # Llama-3.2-Vision output will include input prompt, need to manually separate
        prompt_text = self.processor.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        if full_text.startswith(prompt_text):
            output_text = full_text[len(prompt_text):].strip()
        else:
            output_text = full_text.strip()
        
        return output_text

    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        return self._generate_with_llama_vision(messages, image, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If the sentence is incomplete (ends with comma, "and", "with", etc.), truncate to the last complete part
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class InternVLCaptioner(BaseModel):
    """InternVL3.5-8B for image captioning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
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
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_internvl(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
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
        
        # Build dialogue format
        question = f"<image>\n{prompt}"
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
        }
        
        try:
            # InternVL uses chat method
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
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_internvl(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class SAILVLCaptioner(BaseModel):
    """SAIL-VL2-8B for image captioning
    
    BytedanceDouyinContent/SAIL-VL2-8B
    https://huggingface.co/BytedanceDouyinContent/SAIL-VL2-8B
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """Load SAIL-VL2 model"""
        print(f"Loading SAIL-VL2 model: {self.model_name}")
        
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print(f"SAIL-VL2 model loaded")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_sailvl(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
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
        
        # Process input
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
        
        # Try to remove input prompt part (if included in output)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_sailvl(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class Ministral3VLCaptioner(BaseModel):
    """Ministral 3 8B Vision (mistralai/Ministral-3-8B-Instruct-2512) for image captioning

    This is Mistral AI's vision-language model, different from text-only Ministral-8B-Instruct-2410.
    Supports visual capabilities, can understand and describe images.
    
    https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
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
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_ministral3(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }
    
    def _generate_with_ministral3(self, image: Image.Image, prompt: str,
                                   max_tokens: int = 30) -> str:
        """Ministral 3 VL generation helper function"""
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to base64 URL
        import base64
        from io import BytesIO
        
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
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_ministral3(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class Pixtral12BCaptioner(BaseModel):
    """Pixtral-12B (mistral-community/pixtral-12b) for image captioning

    This is the Pixtral model released by Mistral community, using Llava architecture.
    Supports visual capabilities, can understand and describe images.
    
    https://huggingface.co/mistral-community/pixtral-12b
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_pixtral(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
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
        
        # Process input
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
        
        # Remove input prompt part (if included in output)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_pixtral(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class GLM46VCaptioner(BaseModel):
    """GLM-4.6V-Flash (zai-org/GLM-4.6V-Flash) for image captioning

    This is the lightweight version (9B parameters) of GLM-4.6V series released by Zhipu AI.
    Supports visual understanding, native multimodal function calling, interleaved image-text generation.

    Acceleration options (by priority):
    - SDPA: PyTorch built-in Scaled Dot Product Attention (stable, recommended)
    - Flash Attention 2: Faster but requires compatible CUDA environment
    - torch.compile: PyTorch 2.0+ compilation optimization
    
    https://huggingface.co/zai-org/GLM-4.6V-Flash
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.use_flash_attn = config.get('use_flash_attn', False)  # Default off (CUDA compatibility issues)
        self.use_sdpa = config.get('use_sdpa', True)  # Default on SDPA (more stable)
        self.use_compile = config.get('use_compile', False)  # Default off
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load GLM-4.6V-Flash model with acceleration optimization"""
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
            # If specified attention implementation fails, fall back to default
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
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_glm4v(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
        }
    
    def _generate_with_glm4v(self, image: Image.Image, prompt: str,
                             max_tokens: int = 30) -> str:
        """GLM-4.6V-Flash generation helper function"""
        import base64
        from io import BytesIO
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to base64 data URL (official recommended format)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/png;base64,{img_base64}"
        
        # Build message format (following official documentation)
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
        
        # Remove token_type_ids (if exists) - official requirement
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
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_glm4v(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class Idefics3Captioner(BaseModel):
    """Idefics3-8B-Llama3 (HuggingFaceM4/Idefics3-8B-Llama3) for image captioning

    This is an open-source multimodal model released by Hugging Face, based on SigLIP + Llama 3.1 8B.
    Significantly improved over Idefics2 in OCR, document understanding and visual reasoning.
    
    https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Build prompt (caption generation task)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action."
        
        # Generate caption
        predicted_caption = self._generate_with_idefics3(image, prompt, max_tokens=18)
        
        # Post-processing: ensure it's a complete sentence
        predicted_caption = self._clean_caption(predicted_caption)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': predicted_caption
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
        
        # Process input (following the official example approach)
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
        
        # If response is empty, try fallback method
        if not response:
            full_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Find "assistant" keyword
            lower_output = full_output.lower()
            if "assistant" in lower_output:
                idx = lower_output.rfind("assistant")
                response = full_output[idx + len("assistant"):].strip()
            elif prompt in full_output:
                response = full_output.split(prompt)[-1].strip()
        
        # Clean up possible remaining special markers
        for marker in ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        return response.strip()

    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 10) -> str:
        """
        Generate with image: receive image and custom prompt, return generated text
        For scenarios requiring both image and text like scoring
        """
        if self.model is None:
            self.load_model()
        
        return self._generate_with_idefics3(image, prompt, max_tokens=max_tokens)
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and fix caption, ensure it's a complete sentence
        """
        import re
        
        # 1. Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # 2. If sentence is incomplete, truncate
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the']
        
        for ending in incomplete_endings:
            if caption.endswith(ending):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-2])
                break
        
        # 3. Ensure first letter is capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # 4. Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # 5. Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()


class Step3VLCaptioner(BaseModel):
    """Step3-VL-10B (stepfun-ai/Step3-VL-10B) for image captioning
    
    StepFun's 10B vision-language model with excellent performance on multimodal tasks.
    https://huggingface.co/stepfun-ai/Step3-VL-10B
    """
    
    # Key mapping required by Step3-VL (according to official docs)
    KEY_MAPPING = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "vit_large_projector": "model.vit_large_projector",
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype="auto",
            key_mapping=self.KEY_MAPPING
        ).eval()
        
        print(f"Step3-VL-10B model loaded")
    
    def predict(self, 
                image: Image.Image, 
                true_caption: str = None,
                index: int = None) -> Dict[str, Any]:
        """Generate image caption"""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Unified prompt: concise, focus on subject and action (consistent with other models)
        prompt = "Describe this image in one complete sentence under 15 words. Focus on the main subject and action only."
        
        # Generate caption
        # Thinking process about 100-150 tokens, English answer about 20-30 tokens
        # 200 tokens should be enough (shorter thinking after clearer prompt)
        output = self._generate_with_step3vl(image, prompt, max_tokens=200)
        
        # Clean output
        cleaned_caption = self._clean_step3vl_caption(output)
        
        return {
            'index': index,
            'true_caption': true_caption if true_caption else "",
            'predicted_caption': cleaned_caption
        }
    
    def _generate_with_step3vl(self, image: Image.Image, prompt: str,
                                max_tokens: int = 100) -> str:
        """Step3-VL generation helper function"""
        import io
        
        # Convert PIL Image to base64 URL
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/png;base64,{img_base64}"

        # Build message format - force direct English answer output
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a concise image captioner. Respond with ONLY the caption in English. No explanation, no thinking, just the caption."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_url},
                    {"type": "text", "text": prompt}
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
            # If enable_thinking parameter not supported
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
        
        # DEBUG: Print raw output for first 5 samples
        # Debug output disabled (printing doesn't affect speed, main time is model inference)
        # if not hasattr(self, '_vlm_debug_count'):
        #     self._vlm_debug_count = 0
        # if self._vlm_debug_count < 10:
        #     print(f"[DEBUG Step3-VL VLM #{self._vlm_debug_count}] raw output ({len(decoded)} chars): '{decoded[:200]}...'")
        #     self._vlm_debug_count += 1
        
        return decoded
    
    def generate_with_image(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """Generate with image: receive image and custom prompt, return generated text"""
        if self.model is None:
            self.load_model()
        
        # Force English
        english_prompt = f"[Reply in English only] {prompt}"
        return self._generate_with_step3vl(image, english_prompt, max_tokens=max_tokens)
    
    def _clean_step3vl_caption(self, caption: str) -> str:
        """Clean and fix Step3-VL caption output (reference Qwen implementation)"""
        import re
        
        # Step3-VL special handling: extract content after </think> tag (if exists)
        if '</think>' in caption:
            caption = caption.split('</think>')[-1].strip()
        
        # If output starts with Chinese (thinking process truncated, no </think>)
        if caption and ord(caption[0]) > 127:
            # First try to extract English content after "organize language:" pattern
            patterns = [
                r'所以组织语言：([A-Z][^。]+)',  # English after colon
                r'语言：([A-Z][^。]+)',
                r'：([A-Z][A-Za-z\s,\'\-\.\"\d]+[.!?])',  # English sentence after any colon
            ]
            for pattern in patterns:
                match = re.search(pattern, caption)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > 15:  # Ensure extracted content is long enough
                        caption = extracted
                        break
            else:
                # Find English content in quotes (prefer longest)
                quoted = re.findall(r'"([^"]+)"', caption)
                if quoted:
                    # Take longest quoted content (usually complete English sentence)
                    caption = max(quoted, key=len)
                else:
                    # Try to extract complete English sentence (support more chars: numbers, quotes, etc.)
                    english_sentences = re.findall(r'[A-Z][A-Za-z\s,\'\-\"\d]+[.!?]', caption)
                    if english_sentences:
                        caption = max(english_sentences, key=len)
                    else:
                        # Last attempt: extract fragments starting with uppercase letter
                        english_parts = re.findall(r'[A-Z][A-Za-z\s,\'\-\"\d]+', caption)
                        if english_parts:
                            caption = max(english_parts, key=len)
        
        # Remove all Chinese characters and punctuation (clean up residue)
        caption = re.sub(r'[\u4e00-\u9fff，。！？、；：""''【】（）]', '', caption)
        caption = caption.strip()
        
        # Remove extra quotes
        caption = caption.strip('"\'')
        
        # Remove redundant words at the beginning
        caption = re.sub(
            r'^(The image shows?|This image depicts?|In this image|I see|This is)\s*', 
            '', 
            caption, 
            flags=re.IGNORECASE
        )
        caption = caption.strip()
        
        # Check if ends with incomplete word (reference Qwen)
        incomplete_endings = [',', ' and', ' with', ' at', ' in', ' on', ' of', ' by', ' one', ' a', ' the', ' to', ' passing', ' holding']
        for ending in incomplete_endings:
            if caption.lower().rstrip('.').endswith(ending.strip()):
                if ',' in caption:
                    caption = caption.rsplit(',', 1)[0]
                else:
                    words = caption.split()
                    if len(words) > 3:
                        caption = ' '.join(words[:-1])
                break
        
        # Remove trailing numbers and extra characters
        caption = re.sub(r'\s*\d+\s*[.!?]?\s*$', '.', caption)
        caption = re.sub(r'\s+[.!?]', '.', caption)
        
        # If output too short or incomplete, return default description
        if len(caption) < 10 or caption.endswith(' in') or caption.endswith(' in.'):
            return "An image."
        
        # Check if Chinese characters still remain (German words like Vorsicht may be sign text in image, should not filter)
        if re.search(r'[\u4e00-\u9fff]', caption):
            return "An image."
        
        # Ensure first letter capitalized
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Ensure ends with period
        if caption and not caption.endswith(('.', '!', '?')):
            caption = caption + '.'
        
        # Limit maximum length
        max_length = 120
        if len(caption) > max_length:
            caption = caption[:max_length].rsplit(' ', 1)[0]
            if not caption.endswith(('.', '!', '?')):
                caption = caption + '.'
        
        return caption.strip()