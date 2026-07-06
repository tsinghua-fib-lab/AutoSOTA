import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")


class Qwen2VL:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="auto",
                    )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

        # self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image, question, temp):
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image
                                },
                                {
                                    "type": "text",
                                    "text": question
                                }
                            ]
                        }
                    ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print("text:", text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(0)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
            repetition_penalty=1.05,
            top_k=50,
            top_p=0.95,
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return answer[0]

    def generate_batch(self, image, question, args):
        all_answers = []
        batch_size = args.batch_size
        num = args.sampling_time
        for start_index in range(0, num, batch_size):
            end_idx = min(start_index + batch_size - 1, num - 1)
            messages_list = []
            for i in range(end_idx - start_index + 1):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ]
                messages_list.append(messages)
            # text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                     for messages in messages_list]
            image_inputs, video_inputs = process_vision_info(messages_list)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(0)
            # 批次生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=args.sampling_temp,
                    repetition_penalty=1.05,
                    top_k=10,
                    top_p=0.99,
                )
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            answer = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
            all_answers.extend(answer)
        return all_answers