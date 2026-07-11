import os

import torch


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.runtime_results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def add_runtime(self, run, epoch_time):
        assert 0 <= run < len(self.runtime_results)
        self.runtime_results[run].append(float(epoch_time))

    def get_runtime_summary(self):
        run_means = []
        run_stds = []
        run_epochs = []
        pooled = []

        for run_times in self.runtime_results:
            if len(run_times) == 0:
                continue
            t = torch.tensor(run_times, dtype=torch.float64)
            run_means.append(t.mean().item())
            run_stds.append(t.std(unbiased=False).item())
            run_epochs.append(int(t.numel()))
            pooled.extend(run_times)

        if len(run_means) == 0:
            return None

        run_means_t = torch.tensor(run_means, dtype=torch.float64)
        pooled_t = torch.tensor(pooled, dtype=torch.float64)
        return {
            "run_mean": run_means_t.mean().item(),
            "run_std": run_means_t.std(unbiased=False).item(),
            "run_means": run_means,
            "run_stds": run_stds,
            "run_epochs": run_epochs,
            "pooled_mean": pooled_t.mean().item(),
            "num_runs": len(run_means),
            "num_epochs": int(pooled_t.numel()),
        }

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {ind + 1}')
            print(f'Final Train: {result[ind, 0]:.2f}')
            print(f'Final Test: {result[ind, 2]:.2f}')
            self.test = result[ind, 2]

            if len(self.runtime_results[run]) > 0:
                runtimes = torch.tensor(self.runtime_results[run], dtype=torch.float64)
                print(f'Avg Epoch Time: {runtimes.mean():.4f}s')
                print(f'Std Epoch Time: {runtimes.std(unbiased=False):.4f}s')
                print(f'Epochs Timed: {int(runtimes.numel())}')

        else:
            # NOTE: runs may have different lengths under early stopping.
            # Do NOT do torch.tensor(self.results) (ragged list). Handle per-run.
            best_results = []

            for run_list in self.results:
                if len(run_list) == 0:
                    raise RuntimeError("Logger has an empty run (no epochs logged).")

                r = 100 * torch.tensor(run_list)  # shape: [T, 4], T can vary per run

                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()

                if mode == 'max_acc':
                    idx = r[:, 1].argmax().item()
                else:
                    idx = r[:, 3].argmin().item()

                train2 = r[idx, 0].item()
                test2 = r[idx, 2].item()

                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} +- {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} +- {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} +- {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} +- {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} +- {r.std():.2f}')

            self.test = r.mean()
            runtime_summary = self.get_runtime_summary()
            if runtime_summary is not None:
                print(
                    f"Avg Epoch Time (run mean): {runtime_summary['run_mean']:.4f}s "
                    f"+- {runtime_summary['run_std']:.4f}s"
                )
                print(
                    f"Avg Epoch Time (pooled): {runtime_summary['pooled_mean']:.4f}s "
                    f"over {runtime_summary['num_epochs']} epochs"
                )
                for idx, (run_mean, run_std) in enumerate(
                    zip(runtime_summary["run_means"], runtime_summary["run_stds"]),
                    start=1,
                ):
                    print(
                        f"Run {idx:02d} Avg Epoch Time: {run_mean:.4f}s "
                        f"+- {run_std:.4f}s"
                    )
            return best_result[:, 4]

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')


def save_model(args, model, optimizer, run):
    if not os.path.exists(f'models/{args.dataset}'):
        os.makedirs(f'models/{args.dataset}')
    if(args.model == 'MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{args.gnn}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)


def load_model(args, model, optimizer, run):
    if(args.model == 'MPNN'):
        model_path = f'models/{args.dataset}/{args.model}_{args.gnn}_{run}.pt'
    else:
        model_path = f'models/{args.dataset}/{args.model}_{run}.pt'
    device = next(model.parameters()).device
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def save_result(args, results, runtime_summary=None):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    if(args.model == 'MPNN'):
        filename = f'results/{args.dataset}/{args.model}_{args.gnn}.csv'
    else:
        filename = f'results/{args.dataset}/{args.model}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        if(args.model == 'MPNN'):
            runtime_suffix = ""
            if runtime_summary is not None:
                per_run_runtime = "; ".join(
                    f"run{idx + 1}:{mean:.4f}s+-{std:.4f}s"
                    for idx, (mean, std) in enumerate(
                        zip(runtime_summary["run_means"], runtime_summary["run_stds"])
                    )
                )
                runtime_suffix = (
                    f" epoch_time {runtime_summary['run_mean']:.4f}s "
                    f"$\\pm$ {runtime_summary['run_std']:.4f}s"
                    f" per_run_epoch_time [{per_run_runtime}]"
                )
            write_obj.write(
                f"{args.model} " + f"{args.lr} " + f"{args.hidden_channels} " + f"{args.local_layers} " + f"{args.dropout} " + f"{args.ln} " +
                f"{args.bn} " + f"{args.res} " +
                f"{results.mean():.2f} $\\pm$ {results.std():.2f}" + runtime_suffix + " \n")
        else:
            write_obj.write(
                f"{args.model} " + f"{args.lr} " +
                f"{results.mean():.2f} $\\pm$ {results.std():.2f} \n")
