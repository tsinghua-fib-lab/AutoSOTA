import numpy as np
import torch


import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

from models_sudoku import BLOSudokuLearnA, OptNetSudokuLearnA, SingleOptLayerSudoku
from utils_sudoku import computeErr, create_logger, decode_onehot
import logger as logger
import wandb

def train_test_loop(args, experiment_dir, n):
    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    
    board_side_len = n**2
    
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    #################### PREPARE DATA ##################
    
    train_data_dir_path = f"sudoku/data/{n}"
    features = torch.load(os.path.join(train_data_dir_path, "features.pt"))
    labels = torch.load(os.path.join(train_data_dir_path, "labels.pt"))
    m = 15
    features = torch.tensor(features, dtype=torch.float32).to(device)[:]#[m:m+1]
    labels   = torch.tensor(labels, dtype=torch.float32).to(device)[:]#[m:m+1]
    print(features.shape)
    print(labels.shape)
    # print(f"15th puzzle: ")
    # print(decode_onehot(features[0]))
    
    # Create TensorDataset
    dataset = TensorDataset(features, labels)   
    
   # Fixed split indices
    num_samples = len(dataset)
    train_split = 0.9 #1
    train_size = int(num_samples * train_split)
    test_size = num_samples - train_size

    # Use first 80% for training, last 20% for testing
    train_indices = list(range(0, train_size))
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_indices = list(range(train_size, num_samples))
    test_dataset  = Subset(dataset, test_indices)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_dataset = train_dataset
    # test_loader = train_loader

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    ###############################################
    alpha = args.alpha
    dual_cutoff = args.dual_cutoff
    Qpenalty = 0.1
    model = SingleOptLayerSudoku(n, learnable_parts=['eq'], layer_type=method, Qpenalty=Qpenalty, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=args.slack_tol)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    
    directory = experiment_dir
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = '{}_n{}_lr{}_seed{}_{time_str}.csv'.format(method, n, learning_rate, seed, time_str=time_str)
    if os.path.exists(directory + filename):
        os.remove(directory + filename)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    start_epoch = 0

    if args.resume_epoch != None:
        print(f"RESUMING TRAINING from epoch {args.resume_epoch}")
        start_epoch = args.resume_epoch
        checkpoint = torch.load(os.path.join(directory, f"checkpoint_epoch{start_epoch}.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch+=1
        
    if start_epoch==0:
        mode = 'w'
    else:
        mode = 'a'

    with open(directory + filename, mode) as file:
        if start_epoch==0:
            file.write('epoch, train_loss, test_loss, forward_time, backward_time, train_error, test_error\n')
            file.flush()
        
        writer = SummaryWriter(log_dir=f"runs/sudoku_n{n}_{method}_bs{batch_size}_lr{learning_rate}_seed{seed}_{time_str}")
        logger.set_writer(writer, tag=f"BLO_{method}_{n}_{alpha}_{dual_cutoff}_{batch_size}_{learning_rate}_{seed}")
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        loss_fn = torch.nn.MSELoss()
        
        avg_train_loss = [] # avg training loss per epoch
        avg_test_loss = []
        avg_train_err = []
        avg_test_err = []
        
        for epoch in range(start_epoch, start_epoch+num_epochs):
            print(f"##### epoch {epoch}: ")
            train_loss_list, test_loss_list = [], []
            train_err, test_err = 0,0
            
            forward_time = 0
            backward_time = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                # i = 37
                if i%10==0:
                    print(f"\t\t train example: {i}/{len(train_loader)}")
                x = x.to(device)
                y = y.to(device)
                iter_start_time = time.time()
                
                start_time = time.time()
                pred = model(x)
                # print(f"pred abs max: {torch.max(torch.abs(pred))}, pred abs min:{torch.min(torch.abs(pred))}")
                # print(f"pred : {decode_onehot(pred[0])}")
                loss = loss_fn(pred, y)
                
                forward_time += time.time() - start_time
                iter_time = time.time() - iter_start_time

                start_time = time.time()
                loss.backward()
                backward_time += time.time() - start_time
                
                with torch.no_grad():
                    train_err += computeErr(pred)
                
                #####################################
                ######### compare grads with cvxpy:
                ######################################
                # init_learnable_vals = {
                #     "A": model.A.data.detach().clone(),
                #     "z0_a": model.z0_a.data.detach().clone(),
                # }
                # model_cvxpy = SingleOptLayerSudoku(n, learnable_parts=["eq"], layer_type="cvxpylayer", Qpenalty=Qpenalty, alpha=alpha, init_learnable_vals=init_learnable_vals)
                # model_cvxpy.train()
                # pred_cvxpy = model_cvxpy(x)
                # loss_cvxpy = loss_fn(pred_cvxpy, y)
                # loss_cvxpy.backward()
                
                # grad_A_cvxpy = model_cvxpy.A.grad.clone().detach().cpu().reshape(-1)#.numpy()
                # grad_z0_cvxpy = model_cvxpy.z0_a.grad.clone().detach().cpu().reshape(-1)#.numpy()
                
                # grad_A = model.A.grad.clone().detach().cpu().reshape(-1)#.numpy()
                # grad_z0 = model.z0_a.grad.clone().detach().cpu().reshape(-1)#.numpy()
                
                # with torch.no_grad():
                #     grad_diff_A = torch.norm(grad_A - grad_A_cvxpy, p=1).item()
                #     grad_cos_sim_A = torch.nn.functional.cosine_similarity(grad_A, grad_A_cvxpy, dim=0).item()
                
                #     ######## calculate cosine similarity between grads
                #     writer.add_scalar('grad_A/train_cos_sim', grad_cos_sim_A, (epoch)*len(train_loader)+i)
                #     ######## calculate l1 diff between grads
                #     writer.add_scalar('grad_A/train_l1_diff', grad_diff_A, (epoch)*len(train_loader)+i)
                    
                #     grad_diff_z0 = torch.norm(grad_z0 - grad_z0_cvxpy, p=1).item()
                #     grad_cos_sim_z0 = torch.nn.functional.cosine_similarity(grad_z0, grad_z0_cvxpy, dim=0).item()
                
                #     ######## calculate cosine similarity between grads
                #     writer.add_scalar('grad_z0/train_cos_sim', grad_cos_sim_z0, (epoch)*len(train_loader)+i)
                #     ######## calculate l1 diff between grads
                #     writer.add_scalar('grad_z0/train_l1_diff', grad_diff_z0, (epoch)*len(train_loader)+i)
                    
                #     rank_A = np.linalg.matrix_rank(model.A.data.clone().cpu().detach().numpy())
                #     cond_A = np.linalg.cond(model.A.data.clone().cpu().detach().numpy())
                #     print(f"{(epoch)*len(train_loader)+i} : grad cos sim A: {grad_cos_sim_A} || grad l1 diff A: {grad_diff_A} || \n grad cos sim z0: {grad_cos_sim_z0} || grad l1 diff z0: {grad_diff_z0}")
                #     print(f"rank A: {rank_A}, cond A: {cond_A}")
                #     writer.add_scalar('constraints/rank A', rank_A, (epoch)*len(train_loader)+i)
                #     writer.add_scalar('constraints/condition number A', cond_A, (epoch)*len(train_loader)+i)
                    
                optimizer.step()

                optimizer.zero_grad()

                train_loss_list.append(loss.item())
                print(f"train loss: {loss.item()}, iter time: {iter_time}")
                wandb.log({
                    "train_loss": loss.item(),
                })

            if epoch%1==0 or epoch==num_epochs-1:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    # torch.save(model.state_dict(), os.path.join(directory, f"model_epoch{epoch}.pt"))
                    torch.save(checkpoint, os.path.join(directory, f"checkpoint_epoch{epoch}.pt"))
            print('Forward time {}, backward time {}'.format(forward_time, backward_time))

            model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(device)
                    y = y.to(device)
                    
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    
                    test_err += computeErr(pred)

                    test_loss_list.append(loss.item())

            train_loss = np.mean(train_loss_list)
            test_loss = np.mean(test_loss_list)
            
            train_err = train_err/len(train_dataset)
            test_err = test_err/len(test_dataset)

            wandb.log({
                "avg_train_loss": train_loss,
                "avg_test_loss": test_loss,
                "train_error": train_err,
                "test_error": test_err,
                "total_forward_time": forward_time,
                "total_backward_time": backward_time,
            })
            
            avg_train_err.append(train_err)
            avg_test_err.append(test_err)
            
            avg_train_loss.append(train_loss)
            avg_test_loss.append(test_loss)
            print("Epoch: {}, Train Loss: {}, Test Loss: {}, train error: {}, test error: {}".format(epoch, train_loss, test_loss, train_err, test_err))

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Error/train', train_err, epoch)
            writer.add_scalar('Error/test', test_err, epoch)

            file.write('{},{},{},{},{},{},{}\n'.format(epoch, train_loss, test_loss, forward_time, backward_time, train_err, test_err))
            file.flush()

        writer.flush()

    
    




 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ffocp_eq', help='ffocp_eq, lpgd, qpth, cvxpylayer')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--resume_epoch', type=int, help="epoch number of the model you want to resume training")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n', type=int, default=2, help='n^2 is the board side length')

    parser.add_argument('--alpha', type=float, default=100, help='alpha')
    parser.add_argument('--dual_cutoff', type=float, default=1e-3, help='dual cutoff')
    parser.add_argument('--slack_tol', type=float, default=1e-8, help='slack tolerance')
    
    args = parser.parse_args()
    
    experiment_dir = '../sudoku_results_{}/{}/'.format(args.batch_size, args.method)
    os.makedirs(experiment_dir, exist_ok=True)
    central_logger = create_logger(logging_root=experiment_dir, log_name="central_failures.log")
    
    n = args.n
    failure_id = '{}_n{}_lr{}_seed{}'.format(args.method, n, args.lr, args.seed)

    wandb.login(key="9459f0100021f1abd3867bedcda1b47716e21a34")
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not os.path.exists(f"wandb/{args.method}"):
        os.makedirs(f"wandb/{args.method}")
    wandb.init(project=f"bilevel_layer_sudoku_{args.method}", name=f"sudoku_{time_str}", config=vars(args), dir=f"wandb/{args.method}")
    
    try:
        train_test_loop(args, experiment_dir, n=n)
    except Exception as e:
        central_logger.exception(f"{failure_id}: An error occurred: {e}")
        print(f"{failure_id}: An error occurred: {e}")
        raise e
    