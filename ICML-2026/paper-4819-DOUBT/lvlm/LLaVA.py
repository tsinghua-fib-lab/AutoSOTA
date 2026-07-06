import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class LLaVA:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

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
                                        "type": "image"
                                    }
                                ]
                            }
                        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
        )
        final_ans = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        return final_ans

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
                )
            # 批量解码
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [out.split("ASSISTANT: ")[-1].strip() for out in decoded]
            all_answers.extend(answers)

        return all_answers