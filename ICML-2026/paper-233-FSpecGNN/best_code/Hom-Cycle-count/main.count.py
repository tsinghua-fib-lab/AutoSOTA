import torch
from src.utils import *
from src.model import *
from src.eigen_model import *
from torch_geometric.loader import DataLoader

# --------------------------------- PREPROCESS --------------------------------- #

def collate_fn(ds):
    X, E, U, S, A, Y, M1, M2, Me = [], [], [], [], [], [], [], [], []
    for data in ds:
        X.append(data[0])
        E.append(data[1])
        U.append(data[2])
        S.append(data[3])
        A.append(data[4]) if data[4] is not None else A.append(torch.zeros(0))
        Y.append(data[5])
        M1.append(data[6])
        M2.append(data[7])
        Me.append(data[8])
        
    X, E, U, S, A, Y, M1, M2, Me = map(lambda x: torch.stack(x, dim=0), (X, E, U, S, A, Y, M1, M2, Me))
    
    return [X, E, U, S, A, Y, M1, M2, Me]

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument("--task", type=str, nargs=2, required=True, help="name&task")
parser.add_argument("--test", action="store_true", dest="test")

parser.add_argument("--seed", type=int, default=19260817, help="random seed")
parser.add_argument("--indir", type=str, default="data/count", help="dataset")
parser.add_argument("--outdir", type=str, default="result.count", help="output")
parser.add_argument("--device", type=int, default=None, help="CUDA device")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=2, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=150, help="embedding dimension")

parser.add_argument("--bs", type=int, default=512, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=1200, help="training epochs")

args = parser.parse_args()
print(f"""Run:
    model: {args.model}
    task: {args.task}
    seed: {args.seed}
""")

id = f"{args.model}-{args.task}-{args.max_dis}-{args.num_layer}x{args.dim_embed}-{args.bs}-{args.lr}-{args.seed}"

torch.manual_seed(args.seed)
if args.device is None: device = torch.device("cpu") 
else: device = torch.device(f"cuda:{args.device}") 

from src import dataset

pt_dir = 'data/count/processed/{}_{}.pt'.format(args.task[0], args.task[1])
if os.path.exists(pt_dir):
    train_ds, val_ds, test_ds = torch.load(pt_dir, weights_only=False)
else:
    train_ds = dataset.GraphCount(args.indir, 'train', *args.task, transform=eigen_graph)
    val_ds = dataset.GraphCount(args.indir, 'val', *args.task, transform=eigen_graph)
    test_ds = dataset.GraphCount(args.indir, 'test', *args.task, transform=eigen_graph)
    
    train_ds = collate_fn(train_ds)
    val_ds = collate_fn(val_ds)
    test_ds = collate_fn(test_ds)
    
    print(len(train_ds), len(val_ds), len(test_ds))
    torch.save([train_ds, val_ds, test_ds], pt_dir)

train_ds = torch.utils.data.TensorDataset(*train_ds)
val_ds = torch.utils.data.TensorDataset(*val_ds)
test_ds = torch.utils.data.TensorDataset(*test_ds)

dataloader = {
    'train': torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn),
    'val': torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn),
    'test': torch.utils.data.DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn),
}

# ----------------------------------- MODEL ---------------------------------- #

model = Eigen_GNN(args, args.task[1], odim=1, bn=False)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ----------------------------------- ITER ----------------------------------- #

def critn(pred, target, M1, M2, Me):

    # print(pred.shape, target.shape, Me.shape)
    # pred = pred.squeeze(-1)
    # assert pred.shape == target.shape == Me

    if len(pred.shape) == 3:
        pred = pred.squeeze(-1).view(target.shape)

    L1 = torch.abs(pred - target)
    
    normalize = {
        "cycle3,v": 3, "cycle3,e": 6,
        "cycle4,v": 4, "cycle4,e": 8,
        "cycle5,v": 5, "cycle5,e": 10,
        "cycle6,v": 6, "cycle6,e": 12,
        "chordal4,v": 4, "chordal4,e": 10,
        "chordal5,v": 5, "chordal5,e": 14,
    }
    task = f"{args.task[0]},{args.task[1]}"
    if task in normalize:
        L1 = L1 / normalize[task]

    if args.task[1] == 'v':
        L1 = L1 * M1
    if args.task[1] == 'e':
        L1 = L1 * Me

    return L1.flatten(start_dim=1).sum(dim=1).mean()
    # return pys.scatter(L1, batch["y_batch"]).mean()

def train(model, loader, critn, optim, args):
    model.train()
    losses = []
    for batch in loader:
        #batch = batch.to(device)
        X, E, U, S, A, Y, M1, M2, Me = [t.to(device) for t in batch]
        pred = model(args, X, E, U, S, A, M2)

        optim.zero_grad()
        loss = critn(pred, Y, M1, M2, Me)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return np.array(losses).mean()

def eval(model, loader, critn, args):
    model.eval()
    errors = []
    for batch in loader:
        X, E, U, S, A, Y, M1, M2, Me = [t.to(device) for t in batch]

        with torch.no_grad():
          pred = model(args, X, E, U, S, A, M2)
          err = critn(pred, Y, M1, M2, Me)
          errors.append(err.item())

    return np.array(errors).mean()

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.9,
                                                   patience=10,
                                                #    min_lr=1e-5,
                                                #    verbose=True
                                                   )

if args.test:
    # print(gnn.summary(model, next(iter(dataloader["train"])), max_depth=5))
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel(), p.std().item())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import code;
    exit(code.interact(local=dict(globals(), **dict(locals()))))

# ------------------------------------ RUN ----------------------------------- #

record = []

os.makedirs(output_dir:=args.outdir, exist_ok=True)
log=f"{output_dir}/{id}.txt"
# assert not os.path.exists(log:=f"{output_dir}/{id}.txt")

from tqdm import trange
for epoch in (pbar:=trange(args.epochs)):

    # for group in optim.param_groups:
    #     group['lr'] = args.lr * (1 + np.cos(np.pi * epoch / args.epochs))/2
    lr = optim.param_groups[0]['lr']
    loss = train(model, dataloader["train"], critn, optim, args)
    test = eval(model, dataloader["test"], critn, args)
    val = eval(model, dataloader["val"], critn, args)
    
    sched.step(val)

    record.append([lr, loss, val, test])
    pbar.set_postfix(lr=lr, loss=loss, val=val, test=test)

    np.savetxt(log, record, delimiter='\t')
