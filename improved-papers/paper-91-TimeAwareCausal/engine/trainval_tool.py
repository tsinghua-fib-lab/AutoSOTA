import pdb
import time

import torch
from tqdm import tqdm
from engine.utils import AverageMeter, ProgressMeter, accuracy
import os
import torch.nn.functional as F


def train(
    args,
    epoch,
    algorithm,
    minibatches_loader,
    use_cuda=True,
    writer=None,
):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Cla Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(args.iterations, batch_time, losses, top1, prefix="Epoch: [{}]".format(epoch))

    algorithm.train()
    end = time.time()
    iterations = args.iterations

    for iteration in range(iterations):
        it = iteration + 1
        if use_cuda:
            minibatches = [[data.cuda() for data in batch_data] for batch_data in next(minibatches_loader)]
        else:
            minibatches = [batch_data for batch_data in next(minibatches_loader)]
        *loss, pred, targets = algorithm.update(minibatches, it, writer, epoch, iterations)

        acc1 = accuracy(pred, targets, topk=(1,))[0]

        top1.update(acc1[0], targets.size(0))
        losses.update(loss[0].item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0 and it != 0:
            progress.print(it)

    algorithm.update_scheduler()

    return losses, top1


def val(
    algorithm,
    dataloaders,
    use_cuda=True,
    progress_bar=False,
    writer=None,
):
    losses, top1s, accuracies = [], [], []

    for env_name, domain_idx, dataloader in dataloaders:
        loss, top1, acc = val_one_domain(
            domain_idx, algorithm, dataloader, use_cuda, progress_bar,
        )

        losses.append(loss.avg)
        top1s.append(top1.avg.item())
        accuracies.append(acc)

        if hasattr(algorithm, "prepare_for_next_test"):
            algorithm.prepare_for_next_test()

    return losses, top1s, accuracies


def val_one_domain(
    domain_idx,
    algorithm,
    dataloader,
    use_cuda=True,
    progress_bar=False,
):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    accuracies = []

    # Define iterator
    episodes = enumerate(dataloader)
    stop_iter = len(dataloader)
    if progress_bar:
        episodes = tqdm(episodes, unit='episodes',
                        total=len(dataloader), initial=0, leave=False)

    algorithm.eval()
    if hasattr(algorithm, "dynamic_encoder"):
        algorithm.dynamic_encoder.reset_hidden_bank()

    for iteration, episode in episodes:
        with torch.no_grad():
            # data, targets = episode  # [bs, 1, 28, 28] [64] [64] [64]
            if use_cuda:
                episode = [item.cuda() for item in episode]
            data, targets = episode[0], episode[1]

            targets = targets.view(-1)

            if len(episode) > 2:
                data = dict(inputs=data, additional_info=episode[2:])

            need_cache = True
            final_cache = True if iteration + 1 == stop_iter else False
            output = algorithm.predict(data, domain_idx, algorithm.training, need_cache=need_cache, final_cache=final_cache)

            loss = algorithm.criterion(output, targets)

            acc1 = accuracy(output, targets, (1,))[0]
            top1.update(acc1[0], targets.size(0))
            losses.update(loss.item(), targets.size(0))
            accuracies.append(acc1)

            if progress_bar:
                episodes.set_postfix(avg_acc=torch.tensor(accuracies).mean().item())

    return losses, top1, accuracies