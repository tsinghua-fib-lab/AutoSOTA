import os.path

from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:

    def __init__(self, version):
        self.version = version
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"
        # Use local model path if available
        local_path = f"/models/{self.version}"
        if os.path.exists(local_path):
            model_name = local_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=None, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, question, temp):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
            repetition_penalty=1.05,
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
