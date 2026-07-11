import copy
import json
import lightgbm as lgb
import argparse
import numpy as np
import pandas as pd
import os
import csv
from prettytable import PrettyTable
import tqdm
from cache.evict.algorithms import PredictAlgorithm, PredictAlgorithmFactory
from data_trace.data_trace import DataTrace
from model.models import LightGBMModel, get_fraction_train_file
from utils.aligner import ShiftAligner
from cache.hash import ShiftHashFunction
from cache.cache import BoostCache, LightGBMTrainingCache
from utils.aligner import ShiftAligner, NormalAligner
from cache.hash import ShiftHashFunction, BrightKiteHashFunction, CitiHashFunction
from model import device_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("-f", "--model_fraction", type=str, default='1')
    parser.add_argument("-p", "--model_config_path", type=str, default='checkpoints/lightgbm/model_config.json')
    parser.add_argument("-c", "--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("-t", "--traces_root_dir", type=str, default='traces')

    parser.add_argument("-i", "--iter_threshold", action='store_true')
    parser.add_argument("--real_test", action='store_true')
    args = parser.parse_args()
    device_manager.set_device(args.device)

    traces_dir = os.path.join(args.traces_root_dir, args.dataset)
    if not os.path.exists(traces_dir):
        raise ValueError(f'LightGBM: {traces_dir} not found')
    
    if not os.path.exists(args.model_config_path):
        raise ValueError(f'LightGBM: {args.model_config_path} not found')
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)
        deltanums = model_config['delta_nums']
        edcnums = model_config['edc_nums']
        training_config = model_config['training']

    print(f'LightGBM: Hyper Params Train Fraction[{args.model_fraction}], Delta[{deltanums}], EDC[{edcnums}]')

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
    
    train_file_path = get_fraction_train_file(args.traces_root_dir, args.dataset, args.model_fraction)
    valid_file_path = os.path.join(traces_dir, f'{args.dataset}_valid.csv')
    test_file_path = os.path.join(traces_dir, f'{args.dataset}_test.csv')
    
    def generate_feature_path(trace_path, label_path):
        features = []
        evict_type = PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleBin', associativity=associativity)
        cache = LightGBMTrainingCache(trace_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity, deltanums, edcnums)
        with DataTrace(trace_path) as trace:
            with tqdm.tqdm(desc="Collecting bin labels on DataTrace") as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    bin_label = cache.collect(pc, address)
                    features.append(bin_label)
                    pbar.update(1) 
        with open(label_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for item in features:
                writer.writerow(item)
        print(f"Generate Feature Label File: Written to {label_path}, Len[{len(features)}]")

    bin_label_dir = os.path.join(traces_dir, 'labels')
    os.makedirs(bin_label_dir, exist_ok=True)
    
    valid_bin_file_path = os.path.join(bin_label_dir, f'valid_{deltanums}_{edcnums}.csv')
    if not os.path.exists(valid_bin_file_path):
        generate_feature_path(valid_file_path, valid_bin_file_path)
    if not os.path.exists(valid_bin_file_path):
        raise ValueError(f'LightGBM: {valid_bin_file_path} not found, generate failed')
    
    test_bin_file_path = os.path.join(bin_label_dir, f'test_{deltanums}_{edcnums}.csv')
    if not os.path.exists(test_bin_file_path):
        generate_feature_path(test_file_path, test_bin_file_path)
    if not os.path.exists(test_bin_file_path):
        raise ValueError(f'LightGBM: {test_bin_file_path} not found, generate failed')

    if args.model_fraction == '1':
        train_bin_file_path = os.path.join(bin_label_dir, f'train_{deltanums}_{edcnums}.csv')
    else:
        train_bin_file_path = os.path.join(bin_label_dir, f'train_{args.model_fraction}_{deltanums}_{edcnums}.csv')
    if not os.path.exists(train_bin_file_path):
        generate_feature_path(train_file_path, train_bin_file_path)
    if not os.path.exists(train_bin_file_path):
        raise ValueError(f'LightGBM: {train_bin_file_path} not found, generate failed')

    print(f'LightGBM: Train Path[{train_file_path}], Label[{train_bin_file_path}]')
    print(f'LightGBM: Valid Path[{valid_file_path}], Label[{valid_bin_file_path}]')
    print(f'LightGBM: Test Path[{test_file_path}], Label[{test_bin_file_path}]')

    def load_dataset(bin_path):
        df = pd.read_csv(bin_path, header=None)
        label = df.iloc[:, -1].to_numpy()
        features = df.iloc[:, :-1].to_numpy().astype(np.float64)
        data = lgb.Dataset(features, label=label)
        return data
        
    train_data = load_dataset(train_bin_file_path)
    valid_data = load_dataset(valid_bin_file_path)
    test_data = load_dataset(test_bin_file_path)

    #####################################################
    bst = lgb.train(training_config, train_data, valid_sets=[valid_data], callbacks=[
        lgb.early_stopping(stopping_rounds=50),
    ])
    ypred = bst.predict(test_data.data)
    def accuracy_score(test_bins, predictions):
        success = 0
        for i in range(test_bins.size):
            if test_bins[i] == predictions[i]:
                success += 1
        return success
    
    bench_predicitons = None
    if args.iter_threshold:
        print('Light GBM: Trained finished, generate Threshold...')
        threshold_range = np.arange(0, 1, 0.01)
        best_accuracy = 0
        for threshold in threshold_range:
            predictions = (ypred > threshold).astype(int)
            accuracy = accuracy_score(test_data.label, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                bench_predicitons = predictions
    else:
        best_threshold = 0.5
        predictions = (ypred > best_threshold).astype(int)
        best_accuracy = accuracy_score(test_data.label, predictions)
        bench_predicitons = predictions

    checkpoint_dir = os.path.join(args.checkpoints_root_dir, 'lightgbm', args.dataset, args.model_fraction)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    this_ckpt_path = os.path.join(checkpoint_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
    bst.save_model(this_ckpt_path)
    
    with open(os.path.join(checkpoint_dir, 'threshold'), 'w') as f:
        f.write(str(best_threshold))
    print(f"LightGBM: Model Checkpoint write to [{checkpoint_dir}]")
    print(f"LightGBM: Best Threshold [{best_threshold}]")
    print(f"LightGBM: Best Accuracy [{best_accuracy/test_data.label.size}]")

    #########################################
    if args.real_test:
        gbm_gen = lambda : LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, best_threshold)
        oracle_predicitons = test_data.label

        bench_cache = BoostCache(copy.deepcopy(bench_predicitons.tolist()), test_file_path, align_type, PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), hash_type, cache_line_size, capacity, associativity)
        oracle_cache = BoostCache(copy.deepcopy(oracle_predicitons.tolist()), test_file_path, align_type, PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), hash_type, cache_line_size, capacity, associativity)

        with DataTrace(test_file_path) as trace:
            with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    bench_cache.access(pc, address)
                    oracle_cache.access(pc, address)
                    pbar.update(1)
        
        _, opt_miss, _, _ = oracle_cache.stat()
        table = PrettyTable() 
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate", "Competitive Ratio"]
        hit, miss, total, rate = oracle_cache.stat()
        table.add_row(['OPT', hit, miss, total, rate, f"{miss / opt_miss:.3f}"])
        hit, miss, total, rate = bench_cache.stat()
        table.add_row(['LightGBM', hit, miss, total, rate, f"{miss / opt_miss:.3f}"])
        print(table)