from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class LLaVANeXT:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
            device_map='auto'
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id # 显式设置 pad_token_id=eos_token_id

    def generate(self, image, question, temp):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": question
                    },
                    {
                        "type": "image",
                    }
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(0)
        output = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp
        )
        if '7b' in self.version:
            answer = self.processor.decode(output[0], skip_special_tokens=True).split('[/INST] ')[-1].strip()
        elif '13b' in self.version:
            answer = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        return answer

    def generate_batch(self, image, question, args):
        all_answers = []
        batch_size = args.batch_size
        num = args.sampling_time
        for start_index in range(0, num, batch_size):
            end_idx = min(start_index + batch_size -1, num-1)
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            images = []
            prompts = []
            for i in range(end_idx - start_index + 1):
                images.append(image)
                conversation = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": question
                                            },
                                            {
                                                "type": "image"
                                            }
                                        ]
                                    }
                                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                prompts.append(prompt)
            inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True).to(0, torch.float16)
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=args.sampling_temp,
                    num_beams=1,
                    top_p=0.99,
                    top_k=10,
                    # pad_token_id=self.model.config.eos_token_id,  # 显式传入
                )
            # 批量解码
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            if '7b' in self.version:
                answers = [out.split('[/INST] ')[-1].strip() for out in decoded]
            elif '13b' in self.version:
                answers = [out.split('ASSISTANT: ')[-1].strip() for out in decoded]
            all_answers.extend(answers)

        return all_answers