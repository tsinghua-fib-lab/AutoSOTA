import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autointerp-path", type=str)
    args = parser.parse_args()

    full_explanation_dict = {}
    with open(args.autointerp_path, "rb") as f:
        tsaedict = json.load(f)
        for key in tsaedict['eval_result_unstructured'].keys():
            entry = tsaedict['eval_result_unstructured'][key]["explanation"]
            full_explanation_dict[int(key)] = entry

    autointerp_path = args.autointerp_path.split(".json")[0]+"_formatted.json"
    with open(autointerp_path, "w") as f:
        f.write(json.dumps(full_explanation_dict, indent=2, ensure_ascii=True))