import sys, os, json, torch, copy, numpy as np
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/main')
import models as models_module
from args import apply_psahs_defaults, build_parser
from psahs.paths import checkpoint_suffix, output_dir
from psahs import training_utils as utils
from psahs.data import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate data once with fixed seed
utils.set_seed(42)
parser = build_parser('test')
args = parser.parse_args(['-d', 'dblp_acm', '--src_name', 'ACMv9', '--tgt_name', 'DBLPv8', '--method', 'DANN_rw', '--seeds', '1'])
args = apply_psahs_defaults(args)

data_root = '/repo/dataset'
source_template = datasets.prepare_dblp_acm(data_root, 'ACMv9')
target_template = datasets.prepare_dblp_acm(data_root, 'DBLPv8')
datasets.adjust_graph_structure_fast_source(source_template, h_thresh=args.h_threshold)

input_dim = source_template.num_node_features
output_dim = source_template.num_classes
print('Data loaded: src=%d nodes, tgt=%d nodes, feats=%d, classes=%d' % (
    source_template.num_nodes, target_template.num_nodes, input_dim, output_dim))

# MLP training with same data
path = output_dir(args)
ck = checkpoint_suffix(args)
for seed in [1, 2, 3, 4, 5]:
    utils.set_seed(seed)
    mlp_model = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    scheduler, opt = utils.build_optimizer(args, mlp_model.parameters())
    best_valid = 0.0
    for epoch in range(300):  # MLP epochs
        mlp_model.train()
        src_data = source_template.clone().to(device)
        opt.zero_grad()
        _, logits = mlp_model(src_data)
        mask = src_data.source_training_mask
        loss = utils.ce_loss(logits[mask], src_data.y[mask])
        loss.backward()
        opt.step()
        scheduler.step()
        # Simple validation
        mlp_model.eval()
        with torch.no_grad():
            _, logits = mlp_model(src_data)
            tgt_data = target_template.clone().to(device)
            _, tgt_logits = mlp_model(tgt_data)
            tgt_val_acc, _ = utils.classification_scores(tgt_logits[tgt_data.target_validation_mask], tgt_data.y[tgt_data.target_validation_mask])
        if tgt_val_acc > best_valid:
            best_valid = tgt_val_acc
            torch.save(mlp_model.state_dict(), os.path.join(path, 'best_mlp_model_seed%d%s.pt' % (seed, ck)))
    print('MLP seed %d done, best val=%.4f' % (seed, best_valid))

# PSAHS training with same data but different model seeds
results = {}
for seed in [1, 2, 3, 4, 5]:
    utils.set_seed(seed)
    source = source_template.clone()
    target = target_template.clone()
    datasets.adjust_graph_structure_fast_source(source, h_thresh=args.h_threshold)
    
    model = models_module.GNN_adv(input_dim, output_dim, args).to(device)
    mlp_model = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
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
    
    results[seed] = best_test
    print('[Seed %d] Final best test: %.4f' % (seed, best_test))

accs = list(results.values())
mean_acc = sum(accs) / len(accs)
std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs)) ** 0.5
print('[Summary] Mean: %.4f +/- %.4f' % (mean_acc, std_acc))
