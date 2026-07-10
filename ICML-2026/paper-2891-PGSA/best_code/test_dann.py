import sys, os, json, torch
sys.path.insert(0, '/repo')
sys.path.insert(0, '/repo/main')
import models as models_module
from args import apply_psahs_defaults, build_parser
from data_loader import load_domain_pair
from psahs.paths import checkpoint_suffix, output_dir
from psahs import training_utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = build_parser('test')
args = parser.parse_args(['-d', 'dblp_acm', '--src_name', 'ACMv9', '--tgt_name', 'DBLPv8', '--method', 'DANN', '--seeds', '1', '--epochs', '300', '--conv_type', 'gcn', '--jk', 'cat'])
args = apply_psahs_defaults(args)
args.epochs = 300
args.conv_type = 'gcn'

utils.set_seed(1)
source, target = load_domain_pair(args, adjust_source=True)
input_dim = source.num_node_features
output_dim = source.num_classes
print('Dims: in=%d out=%d src_nodes=%d tgt_nodes=%d' % (input_dim, output_dim, source.num_nodes, target.num_nodes))

model = models_module.directed_GNN_adv(input_dim, output_dim, args).to(device)
mlp_model = models_module.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
path = output_dir(args)
ck = checkpoint_suffix(args)
mlp_ckpt = os.path.join(path, 'best_mlp_model_seed1%s.pt' % ck)
mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device))
scheduler, opt = utils.build_optimizer(args, model.parameters())

best_valid, best_test = 0.0, 0.0
for epoch in range(args.epochs):
    model.train()
    src_data = source.to(device)
    tgt_data = target.to(device)
    mlp_model.eval()
    da_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)
    [_, _], [pred_src, pred_domain_src] = model.forward(src_data, da_alpha)
    [_, _], [_, pred_domain_tgt] = model.forward(tgt_data, da_alpha)
    mask_src = src_data.source_training_mask
    label_src = src_data.y[mask_src]
    pred_src = pred_src[mask_src]
    pred_domain_src = pred_domain_src[src_data.source_mask]
    pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]
    cls_loss = utils.ce_loss(pred_src, label_src)
    domain_loss = utils.bce_loss(pred_domain_src, torch.zeros_like(pred_domain_src))
    domain_loss += utils.bce_loss(pred_domain_tgt, torch.ones_like(pred_domain_tgt))
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
        print('epoch %d: test=%.4f val=%.4f best_test=%.4f' % (epoch+1, tgt_test_acc, tgt_val_acc, best_test))
print('Final: val=%.4f test=%.4f' % (best_valid, best_test))
