import os
import json
import argparse

def extract_result(folder_path):
    summary = {"results": {}}

    for filename in os.listdir(folder_path):
        if filename.startswith("results"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "gsm8k_cot_zeroshot" in data["results"]:
                gsm8k_data = data["results"]["gsm8k_cot_zeroshot"]
                acc_value = gsm8k_data["exact_match,flexible-extract"]
                summary["results"][filename] = {"gsm8k_cot_zeroshot": acc_value}


    output_file = os.path.join(folder_path, "summary_results_gsm8k.json")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(summary, f_out, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, required=True)
    args = parser.parse_args()

    extract_result(args.directory_path)

if __name__ == '__main__':
    main()