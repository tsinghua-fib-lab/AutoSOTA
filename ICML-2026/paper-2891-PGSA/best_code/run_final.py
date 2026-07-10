import sys, os, json, torch
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/main')
import models as models_module
from args import apply_psahs_defaults, build_parser
from data_loader import load_domain_pair
from psahs.paths import checkpoint_suffix, output_dir
from psahs import training_utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_eval(seed):
    parser = build_parser('test')
    args = parser.parse_args(['-d', 'dblp_acm', '--src_name', 'ACMv9', '--tgt_name', 'DBLPv8', '--method', 'DANN_rw', '--seeds', str(seed)])
    args = apply_psahs_defaults(args)
    
    utils.set_seed(seed)
    source, target = load_domain_pair(args, adjust_source=True)
    input_dim = source.num_node_features
    output_dim = source.num_classes
    
    model = models_module.GNN_adv(input_dim, output_dim, args).to(device)
    mlp_model = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    path = output_dir(args)
    ck = checkpoint_suffix(args)
    mlp_ckpt = os.path.join(path, 'best_mlp_model_seed%d%s.pt' % (seed, ck))
    mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    
    target.rw_stats_history = []
    best_valid, best_test = 0.0, 0.0
    
    for epoch in range(args.epochs):
        model.train()
        src_data = source.to(device)
        tgt_data = target.to(device)
        mlp_model.eval()
        with torch.no_grad():
            _, tgt_mlp_pred = mlp_model(tgt_data)
        _, mlp_pred_tgt = tgt_mlp_pred.max(dim=1)
        
        if args.reweight and epoch >= (args.start_epoch - 1):
            _, [pred_tgt, _] = model.forward(tgt_data, 1)
            _, pred_tgt = pred_tgt.max(dim=1)
            mask = pred_tgt == mlp_pred_tgt
            tgt_data.y_hat[mask] = mlp_pred_tgt[mask]
            if (epoch - 1) % args.rw_freq == 0:
                from psahs.data import datasets
                datasets.adjust_graph_structure_fast_target_Plabel(tgt_data, h_thresh=args.h_threshold)
                tgt_data = tgt_data.to(device)
        
        da_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)
        [_, _], [pred_src, pred_domain_src] = model.forward(src_data, da_alpha)
        [_, _], [_, pred_domain_tgt] = model.forward(tgt_data, da_alpha)
        mask_src = src_data.source_training_mask
        cls_loss = utils.ce_loss(pred_src[mask_src], src_data.y[mask_src])
        domain_loss = utils.bce_loss(pred_domain_src[src_data.source_mask], torch.zeros_like(pred_domain_src[src_data.source_mask]))
        domain_loss += utils.bce_loss(pred_domain_tgt[tgt_data.target_mask], torch.ones_like(pred_domain_tgt[tgt_data.target_mask]))
        loss = cls_loss + domain_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            [_, _], [pred, _] = model.forward(tgt_data, 1)
            tgt_test_acc, _ = utils.classification_scores(pred[tgt_data.target_testing_mask], tgt_data.y[tgt_data.target_testing_mask])
            tgt_val_acc, _ = utils.classification_scores(pred[tgt_data.target_validation_mask], tgt_data.y[tgt_data.target_validation_mask])
        if tgt_val_acc > best_valid:
            best_valid = tgt_val_acc
            best_test = tgt_test_acc
        if (epoch + 1) % 50 == 0:
            print('[seed=%d epoch=%d] test=%.4f val=%.4f' % (seed, epoch+1, tgt_test_acc, tgt_val_acc))
    
    return best_test

results = {}
for seed in [1, 2, 3, 4, 5]:
    acc = train_and_eval(seed)
    results[seed] = acc
    print('[Seed %d] Final best test: %.4f' % (seed, acc))

accs = list(results.values())
mean_acc = sum(accs) / len(accs)
std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs)) ** 0.5
print('[Summary] Mean: %.4f +/- %.4f' % (mean_acc, std_acc))
print(json.dumps(results, indent=2))
