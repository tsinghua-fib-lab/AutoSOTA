import traceback
import torch
import pickle, os
import editdistance
import numpy as np
import time
from autoregltl import ted, dataset
from autoregltl.ltl.chars import CHARS
import random

from tqdm.auto import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
from autoregltl.ltl import trace_check

device = torch.device('cuda')

redgreen = mpl.colors.LinearSegmentedColormap(
    "redgreen",
    {
        'red': (
            (0.0, 1.0, 1.0),
            (0.15873*3, 1.0, 1.0),
            (0.174603*3, 0.96875, 0.96875),
            (1.0, 0.0, 0.0),
        ),
        'green': (
            (0.0, 0.0, 0.0),
            (0.15873*3, 0.9375, 0.9375),
            (0.174603*3, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        'blue': (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    }
)

@torch.no_grad
def eval2d(
        model_path,
        eval_ds,
        repeat_count=10,
        figsize=(6, 5),
        gen_args=None,
        output = "eval2da1.pkl",
    ):
    if gen_args is None:
        gen_args = dict(
            alpha=1.0,
            beam_size=3,
            gen_batch_size=128,
        )
    save_loc = os.path.join(model_path, output)
    model = ted.load_model(model_path, device)
    model.eval()

    with open(eval_ds, 'rb') as f:
        dsdict = pickle.load(f)
    
    min_aps = min([i[0] for i in dsdict.keys()])
    max_aps = max([i[0] for i in dsdict.keys()])
    min_length = min([i[1] for i in dsdict.keys()])
    max_length = max([i[1] for i in dsdict.keys()])
    
    datasets = {}
    all_pairs = []
    for ap in range(min_aps, max_aps+1):
        sizes = []
        datas = []
        for l in range(min_length, max_length+1):
            data = dsdict.get((ap, l), [])
            sizes.append(len(data))
            datas += data
            all_pairs += data
        test_dataset = dataset.SeqDataset(datas)
        datasets[ap] = (test_dataset, sizes)

    print("All pairs:", len(all_pairs))
    filedict = {}

    model.config.vocab.aps = CHARS[:max_aps]
    if (merged_embedder := getattr(model, "merged_embedder", None)):
        merged_embedder.prepare()

    correct_matrix = torch.zeros(max_aps + 1, max_length)
    count_matrix = torch.zeros(max_aps + 1, max_length)
    all_results = {}
    timing_info = {}
    total_time = 0.0
    total_samples = 0
    for apcount in tqdm(list(range(min_aps, max_aps+1))[::-1], desc="APs"):
        model.config.vocab.aps = CHARS[:apcount]
        model.merged_embedder.shrink_w()

        test_dataset, sizes = datasets[apcount]
        dataset_size = len(test_dataset)
        
        start_time = time.time()
        cum_preds = model.generate_predictions(
            test_dataset,
            max_length=22,
            gen_args=gen_args,
            leave_tqdm=False,
            prepare_embedder=False,  # generate_predictions should NOT re-prep embedder
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Track timing info
        avg_time_per_sample = elapsed_time / dataset_size if dataset_size > 0 else 0.0
        timing_info[apcount] = {
            'time': elapsed_time,
            'dataset_size': dataset_size,
            'avg_time_per_sample': avg_time_per_sample
        }
        total_time += elapsed_time
        total_samples += dataset_size
        
        # Print timing info for this AP count
        print(f"AP count {apcount}: {elapsed_time:.2f}s for {dataset_size} samples ({avg_time_per_sample:.4f}s/sample)")
        
        cum_results = trace_check.evaluate_ltl(cum_preds, timeout=30, leave_tqdm=False)
        for l, size in zip(range(min_length, max_length+1), sizes):
            results, cum_results = cum_results[:size], cum_results[size:]
            correct = 0
            for r in results:
                if r['result'] == 'semantically correct' or r['result'] == 'exact match':
                    correct += 1
            all_results[(apcount, l)] = results
            correct_matrix[apcount, l-1] += correct
            count_matrix[apcount, l-1] += len(results)
    
    # Print total timing info
    print(f"\nTotal time: {total_time:.2f}s for {total_samples} samples ({total_time/total_samples if total_samples > 0 else 0:.4f}s/sample)")
    
    filedict |= {
        "correct_matrix": correct_matrix,
        "count_matrix": count_matrix,
        "correct": correct_matrix.sum().item(),
        "count": count_matrix.sum().item(),
        "repeat_count": repeat_count,
        "eval_ds": eval_ds,
        "all_results": all_results,
        "timing_info": timing_info,
        "total_time": total_time,
        "total_samples": total_samples,
        "avg_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
    }
    print("Correct:", filedict["correct"])
    print("Count:", filedict["count"])
    print("Correct ratio:", filedict["correct"]/filedict["count"])

    # SAVE
    with open(save_loc, 'wb') as f:
        pickle.dump(filedict, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample_rate = count_matrix / 100.0
    eval_results = torch.where(count_matrix > 0, correct_matrix / count_matrix, 0.0)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    common_kwargs = dict(
        aspect='auto',
    )
    ax.imshow(eval_results, cmap=redgreen, vmin=0.0, vmax=1.0, **common_kwargs)
    # Plotting the modulus array as the 'value' part
    black = torch.zeros(max_aps+1, max_length, 4)
    black[:, :, -1] = 1.0 - sample_rate
    #black[:, :, -1] = torch.where(sample_rate > 0, 0.0, 1.0)
    ax.imshow(black, **common_kwargs)

    ax.set_ylabel("AP count")
    ax.set_xlabel("Formula length")
    xticks = [1, 10, 20, 30, 40, 50]
    ax.set_xticks([i -1 for i in xticks], xticks)
    # # 35 is not actually inclusive
    # ax.add_patch(mpl.patches.Rectangle((-0.5, -0.5), 35, 5+1, fill=False, edgecolor='white', lw=2))

    fig.colorbar(plt.cm.ScalarMappable(cmap=redgreen), ax=ax)
    plt.savefig(save_loc + ".png", bbox_inches="tight", dpi=192, pad_inches=0.02)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, nargs='+')
    parser.add_argument('--repeat-count', type=int, default=10)
    parser.add_argument('--figsize', type=str, default="(6,3)")
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--input', type=str, default="data-prop/eval2d-10ap.pkl")
    parser.add_argument('--output', type=str, default="eval2da1.pkl")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for generation')
    parser.add_argument('--beam-size', type=int, default=3, help='Beam size for generation')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

    for model_path in args.model_path:
        print("Evaluating:", model_path)
        try:
            eval2d(
                model_path,
                repeat_count=args.repeat_count,
                figsize=eval(args.figsize),
                eval_ds=args.input,
                output=args.output,
                gen_args=dict(
                    alpha=1.0,
                    beam_size=args.beam_size,
                    gen_batch_size=args.batch_size,
                ),
            )
        except Exception as e:
            print("Error:")
            traceback.print_exc()