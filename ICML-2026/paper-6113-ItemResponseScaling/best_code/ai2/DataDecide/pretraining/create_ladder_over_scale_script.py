import argparse
import json

def main(args):
    with open(args.scales, 'r') as scales_file:
        scales = scales_file.readlines()
    data_mixes = []
    with open(args.data_mixes, 'r') as data_mixes_file:
        for line in data_mixes_file:
            data_mixes.append(json.loads(line))
    
    for scale in scales:
        scale = scale.strip()
        for data_mix_info in data_mixes:
            for seed in args.seeds:
                data_mix = data_mix_info['name'].strip()
                s3 = data_mix_info['s3_only']
                if seed is not None:
                    seed_str = f"--seed {seed}"
                else:
                    seed_str = ""
                if args.name_suffix:
                    data_mix = f"{data_mix}-{args.name_suffix}"
                print(f"scripts/beaker/ladder-launch.sh 1 normal --model {scale} --data {data_mix} --length {args.length} --name {data_mix}" + (" --s3" if s3 else "") + f" {seed_str}" + " --save_overwrite")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a ladder of runs over the scales and data mixes provided')
    parser.add_argument('--scales', type=str, help='Path to the file containing the scales')
    parser.add_argument('--data-mixes', type=str, help='Path to the file containing the data mixes (jsonl)')
    parser.add_argument('--name-suffix', type=str, default=None, help='suffix (after data-mix) of the experiment')
    parser.add_argument('--length', type=str, help='Length of the experiment')
    parser.add_argument('--seeds', nargs='+', type=int, help='Seeds to run')
    args = parser.parse_args()
    if not args.seeds:
        args.seeds = [None]
    main(args)
