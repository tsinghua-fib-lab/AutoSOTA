from data import load_gsm8k
from utils import model_inference, initialize_text_to_text_model
from fire import Fire
import re
import os
from tqdm import tqdm
from peft import PeftModel

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
        print(text)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def main(model_name = "llama/llama-2-7b-hf"):
    _, _, test_set = load_gsm8k()
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, True, flash_attention=True
    )
    model = PeftModel.from_pretrained(model, "checkpoint_dir")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    all = 0
    correct = 0
    t = tqdm(test_set)
    for example in t:
        # print(example['x'])
        pred_text = model_inference(model, tokenizer, example['x'], model_type, max_target_length=512)
        gt = extract_num(example["y"])
        pred = extract_num(pred_text)
        correct += int(gt == pred)
        all += 1
        t.set_description(f"Accuracy: {correct / all * 100:02f}%")

    print("Acc:", correct / all)
    # append to gsm8k_results.txt (create if not exists)
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open("gsm8k_results.txt", "a") as f:
        f.write(f"{model_name} {correct / all}\n")

if __name__ == "__main__":
    Fire(main)



