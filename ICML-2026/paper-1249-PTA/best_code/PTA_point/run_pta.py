import os
import sys
import wandb

import torch
import torch.nn.functional as F
import operator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

def update_text_features(image_feature, probs, text_features, history_text, alpha=0.01):
    # probs: [1, C], image_feature: [1, D]
    C = text_features.size(0)
    T = 50

    w = probs.squeeze(0)  # [C]
    w_new = torch.zeros_like(w)
    mask = w >= 1e-1
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)  # [C, 1]
    history_text[mask] = (1 - w_new[mask]) * history_text[mask] + w_new[mask] * image_feature.squeeze(0)

    refined_text = alpha * text_features + (1 - alpha) * history_text
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, history_text

@torch.no_grad()
def run_pta(args, test_loader, lm3d_model, clip_weights, task_name):
    ''' NOTE Build cache in advance '''
    accuracies = []
    text_features = clip_weights.t()  # [C, D]
    history_text = torch.zeros_like(text_features).cuda()
    #Test-time adaptation
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()

        # pc_feats: (1, emb_dim)
        # clip_logits: (1, n_cls)
        # loss: a scalar
        # prob_map: (1, n_cls)
        # pred: a scalar, class index
        pc_feats, clip_logits, _, _, _ = get_logits(args, feature, lm3d_model, clip_weights)
        target = target.cuda()

        soft_logits = F.softmax(clip_logits, dim = -1)
        text_features, history_text = update_text_features(
            pc_feats, soft_logits, text_features, history_text
        )

        final_logits = clip_logits.clone()
        final_logits += 100. * pc_feats.half() @ text_features.half().T
            
        acc = cls_acc(final_logits, target)  
        accuracies.append(acc)

        if i % args.print_freq == 0:
            print("---- PTA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    print("---- ***Final*** PTA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
    with open('outputs/point_result.txt', 'a') as f:
        f.write("PTA's performance on {}: Top1- {:.2f}.\n".format(task_name, sum(accuracies)/len(accuracies))) 
    return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    # Set random seed
    set_random_seed(args.seed)

    clip_model, lm3d_model = load_models(args)

    # NOTE *** need to be implemented
    preprocess = None

    dataset_name = args.dataset
    cor_type_name = args.cor_type
    print(f"Processing {dataset_name}_{cor_type_name}.")
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    
    print(f'>>> classnames:', classnames)
    
    # `clip_weights` are text features of shape (emb_dim, n_cls)
    clip_weights = clip_classifier(args, classnames, template, clip_model)
    task_name = f"{dataset_name}_{cor_type_name}"
    acc = run_pta(args, test_loader, lm3d_model, clip_weights, task_name)

if __name__ == "__main__":
    main()