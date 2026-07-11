from data_trace.data_trace import DataTrace
from model.parrot import utils
from model.models import ParrotModel, get_fraction_train_file
from model import device_manager
from utils.aligner import NormalAligner, ShiftAligner
from cache.hash import BrightKiteHashFunction, CitiHashFunction, ShiftHashFunction
from cache.cache import ParrotTrainingCache
from cache.evict import *
from typing import Callable
import os
import torch
import io
import tqdm
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("-f", "--model_fraction", type=str, default='1')
    parser.add_argument("-p", "--model_config_path", type=str, default='checkpoints/parrot/model_config.json')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--traces_root_dir", type=str, default='traces')
    parser.add_argument("--real_test", action='store_true')
    args = parser.parse_args()
    device_manager.set_device(args.device)

    traces_dir = os.path.join(args.traces_root_dir, args.dataset)
    if not os.path.exists(traces_dir):
        raise ValueError(f'Parrot: {traces_dir} not found')

    train_file_path = get_fraction_train_file(args.traces_root_dir, args.dataset, args.model_fraction)
    if args.real_test:
        eval_file_path = os.path.join(traces_dir, f'{args.dataset}_test.csv')
    else:
        eval_file_path = os.path.join(traces_dir, f'{args.dataset}_valid.csv')

    if args.dataset == 'brightkite':
        cache_line_size = 1
        capacity = 1000
        associativity = 10
        align_type = NormalAligner
        hash_type = BrightKiteHashFunction
    elif args.dataset == 'citi':
        cache_line_size = 1
        capacity = 1200
        associativity = 100
        align_type = NormalAligner
        hash_type = CitiHashFunction
    else:
        cache_line_size = 64
        capacity = 2097152
        associativity = 16
        align_type = ShiftAligner
        hash_type = ShiftHashFunction
#################################################################################################
    if not os.path.exists(args.model_config_path):
        raise ValueError(f'Parrot: {args.model_config_path} not found')
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)

    lr = model_config['lr']
    total_steps = model_config['total_steps']
    eval_freq = model_config['eval_freq']
    save_freq = model_config['save_freq']
    batch_size = model_config['batch_size']

    print(f'Parrot: lr[{lr}], total_steps[{total_steps}], eval_freq[{eval_freq}], save_freq[{save_freq}], batch_size[{batch_size}]')

    collection_multiplier = model_config['collection_multiplier']
    dagger_init = model_config['dagger_init']
    dagger_final = model_config['dagger_final']
    dagger_steps = model_config['dagger_steps']
    dagger_update_freq = model_config['dagger_update_freq']
    print(f'Parrot: Dagger collection_multiplier[{collection_multiplier}], dagger_init[{dagger_init}], dagger_final[{dagger_final}], dagger_steps[{dagger_steps}], dagger_update_freq[{dagger_update_freq}]')
    
    checkpoint_dir = os.path.join(args.checkpoints_root_dir, 'parrot', args.dataset, args.model_fraction)
    os.makedirs(checkpoint_dir, exist_ok=True)

    parrot_model = ParrotModel.from_config(args.model_config_path, None)
    optimizer = torch.optim.Adam(parrot_model._model.parameters(), lr=lr)
#################################################################################################
    evict_type = partial(CombineWeightsAlgorithm, 
                         candidate_algorithms=[PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis'), 
                                               PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_model)], 
                         weights=[1, 0], lazy_evictor_type=None)
    
    def get_model_prob(get_step_lambda):
        fraction = min(float(get_step_lambda()) / dagger_steps, 1.0)
        model_prob = dagger_init + fraction * (dagger_final - dagger_init)
        return model_prob

    def generate_snapshots(file_path: str, max_examples=None, model_prob_gen=lambda:0):
        if max_examples is None:
            max_examples = np.inf
        cache = ParrotTrainingCache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc=f'Collecting data on DataTrace [{file_path}]') as pbar:
                while not trace.done():
                    model_prob = model_prob_gen()
                    pbar.set_postfix({"Model Prob": model_prob})
                    cache.reset(model_prob)
                    data = []
                    total_cnt = 0
                    hit_cnt = 0
                    while len(data) < max_examples and not trace.done():
                        pc, address = trace.next()
                        snapshot, hit = cache.collect(pc, address)
                        if hit:
                            hit_cnt += 1
                        total_cnt += 1
                        data.append(snapshot)
                        pbar.update(1)
                    
                    if total_cnt == 0:
                        hit_rate = 0
                    else:
                        hit_rate = hit_cnt / total_cnt
                    yield data, hit_rate

    oracle_data, oracle_hit_rate = next(generate_snapshots(eval_file_path))
    print('oracle hit rate:', oracle_hit_rate, flush=True)

    step = 0
    get_step = lambda: step
    eval_hit_rate = 0
    with tqdm.tqdm(total=total_steps, desc='Training Process: ') as pbar:
        postfix_dict = {
            "train_hit_rate": 0,
            "eval_now_hit_rate": 0,
            "eval_oracle_hit_rate": oracle_hit_rate,
            "loss/total": 0,
        }
        
        while True:
            max_examples = (dagger_update_freq * collection_multiplier * batch_size)
            train_data_generator = generate_snapshots(train_file_path, max_examples, partial(get_model_prob, get_step_lambda=get_step))
            for train_data, train_hit_rate in train_data_generator:
                postfix_dict['train_hit_rate'] = train_hit_rate
                for batch_num, batch in enumerate(utils.as_batches([train_data], batch_size, model_config.get("sequence_length"))):
                    if step % eval_freq == 0 and step != 0:
                        eval_data, eval_hit_rate = next(generate_snapshots(eval_file_path, None, lambda:1))
                        postfix_dict['eval_now_hit_rate'] = eval_hit_rate
                    
                    if step % save_freq == 0 and step != 0:
                        save_path = os.path.join(checkpoint_dir, f"{step}_{eval_hit_rate}.ckpt")
                        with open(save_path, "wb") as save_file:
                            checkpoint_buffer = io.BytesIO()
                            torch.save(parrot_model._model.state_dict(), checkpoint_buffer)
                            print(f"Saving model checkpoint to: {save_path}", flush=True)
                            save_file.write(checkpoint_buffer.getvalue())
                    optimizer.zero_grad()
                    losses = parrot_model._model.loss(batch, model_config.get("sequence_length") // 2)
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    step += 1
                    postfix_dict["loss/total"] = total_loss.item()
                    pbar.set_postfix(postfix_dict)
                    # for loss_name, loss_value in losses.items():
                    #     pbar.set_postfix({f"loss/{loss_name}": loss_value})
                    if step == total_steps:
                        exit(0)
                    if batch_num == dagger_update_freq:
                        break