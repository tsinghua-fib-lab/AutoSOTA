from datetime import datetime
import numpy as np
import logging
import random, os
import numpy as np
import torch

def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)

def formatting_list(number_list, prefix=None):
    formatted_line = ""
    for i, number in enumerate(number_list):
        if prefix is None:
            formatted_line += f"{number:.2f}, "
        else:
            formatted_line += f"{prefix}{i}: {number:.2f},"
    return formatted_line

def calculate_separate_acc(all_attr, all_y, all_predict, avg_acc):

    majority_one_indices = np.where((all_attr == 1) & (all_y == 0))
    majority_two_indices = np.where((all_attr == 0) & (all_y == 1))
    minority_one_indices = np.where((all_attr == 1) & (all_y == 1))
    minority_two_indices = np.where((all_attr == 0) & (all_y == 0))

    majority_one_acc = (all_predict[majority_one_indices] == all_y[majority_one_indices]).sum() / majority_one_indices[0].shape[0]
    logging.info(f"Majority one (10): {majority_one_acc:.3f}, {(all_predict[majority_one_indices] == all_y[majority_one_indices]).sum()} / {majority_one_indices[0].shape[0]}")
    majority_two_acc = (all_predict[majority_two_indices] == all_y[majority_two_indices]).sum() / majority_two_indices[0].shape[0]
    logging.info(f"Majority two (01): {majority_two_acc:.3f}, {(all_predict[majority_two_indices] == all_y[majority_two_indices]).sum()} / {majority_two_indices[0].shape[0]}")
    all_majority = majority_one_indices[0].shape[0] + majority_two_indices[0].shape[0]
    all_majority_correct = (all_predict[majority_one_indices] == all_y[majority_one_indices]).sum() + (all_predict[majority_two_indices] == all_y[majority_two_indices]).sum()
    logging.info(f"Majority: {(all_majority_correct / all_majority):.3f}, {all_majority_correct} / {all_majority}")

    minority_one_acc = (all_predict[minority_one_indices] == all_y[minority_one_indices]).sum() / minority_one_indices[0].shape[0]
    logging.info(f"Minority one (11): {minority_one_acc:.3f}, {(all_predict[minority_one_indices] == all_y[minority_one_indices]).sum()} / {minority_one_indices[0].shape[0]}")
    minority_two_acc = (all_predict[minority_two_indices] == all_y[minority_two_indices]).sum() / minority_two_indices[0].shape[0]
    logging.info(f"Minority two (00): {minority_two_acc:.3f}, {(all_predict[minority_two_indices] == all_y[minority_two_indices]).sum()} / {minority_two_indices[0].shape[0]}")
    all_minority = minority_one_indices[0].shape[0] + minority_two_indices[0].shape[0]
    all_minority_correct = (all_predict[minority_one_indices] == all_y[minority_one_indices]).sum() + (all_predict[minority_two_indices] == all_y[minority_two_indices]).sum()
    logging.info(f"Minority: {(all_minority_correct / all_minority):.3f}, {all_minority_correct} / {all_minority}")

    EO = max(np.abs(majority_one_acc - minority_two_acc), np.abs(majority_two_acc - minority_one_acc))
    worst_acc = min([majority_one_acc, minority_one_acc, majority_two_acc, minority_two_acc])
    logging.info("\n")
    logging.info(f"EO: {EO:.3f}")
    logging.info("\n")
    logging.info(f"Overall acc: {avg_acc:.3f}")
    logging.info("\n")
    logging.info(f"Worst Acc: {worst_acc:.3f}")
    logging.info(f"Average Acc: {(majority_one_acc + minority_one_acc +majority_two_acc+minority_two_acc)/4:.3f}")
    

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True