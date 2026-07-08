# import cvxtorch
# from cvxtorch import TorchExpression
# import cvxtorch, sys
# print("11111 cvxtorch file:", getattr(cvxtorch, "__file__", "NO FILE"))
# print("sys.path[0]:", sys.path[0])
import numpy as np
import torch
import time
import sys
import os
import argparse
import copy

from models import OptModel
from data import genData

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='bilevel layer method')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate') #0.00001
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--ydim', type=int, default=1000, help='dimension of y')
    parser.add_argument('--backward_eps', type=float, default=1e-5, help='backward tolerance') #0.00001
    
    parser.add_argument('--alpha', type=float, default=100, help='alpha')
    parser.add_argument('--dual_cutoff', type=float, default=1e-3, help='dual cutoff')
    parser.add_argument('--slack_tol', type=float, default=1e-8, help='slack tolerance')

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--learn_constraint', type=int, default=0, help='whether to learn constraint')
    parser.add_argument('--suffix', type=str, default="", help='suffix to the result directory')
    
    
    args = parser.parse_args()

    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    learning_rate = args.lr
    print("learning rate: ", learning_rate)
    alpha = args.alpha
    dual_cutoff = args.dual_cutoff
    slack_tol = args.slack_tol
    backward_eps = args.backward_eps

    # Set random seed for reproducibility
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    input_dim   = 640
    ydim  = args.ydim
    n = ydim
    num_samples = 2000 #2048
    batch_size = args.batch_size

    train_loader, test_loader = genData(device, input_dim, ydim, num_samples, batch_size)
    # print(len(train_loader))
    # assert(len(train_loader)*batch_size == 1600)

    model = OptModel(input_dim, ydim, layer_type=method, constraint_learnable=(args.learn_constraint==1), batch_size=batch_size, device=device, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, backward_eps=backward_eps, is_QP=True).to(device)
    cvxpylayer = OptModel(input_dim, ydim, layer_type="cvxpylayer", constraint_learnable=(args.learn_constraint==1), batch_size=batch_size, device=device, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, backward_eps=backward_eps, is_QP=True).to(device)
    cvxpylayer.predictor = copy.deepcopy(model.predictor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writer = SummaryWriter()

    deltas = [torch.zeros_like(parameter) for parameter in model.parameters()]
    gradients = [torch.zeros_like(parameter) for parameter in model.parameters()]

    if method == "qpth" and device != "cpu":
        directory = '../synthetic_results_{}{}/{}_gpu/'.format(args.batch_size, args.suffix, method)
    else:
        directory = '../synthetic_results_{}{}/{}/'.format(args.batch_size, args.suffix, method)

    if method == "ffocp_eq":
        filename = '{}_ydim{}_lr{}_seed{}_backwardTol{}.csv'.format(method, ydim, learning_rate, seed, backward_eps)
    else:
        filename = '{}_ydim{}_lr{}_seed{}.csv'.format(method, ydim, learning_rate, seed)
    # if os.path.exists(directory + filename):
    #     os.remove(directory + filename)

    # if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)
        
    step_experiment_dir = '../synthetic_results_compare_grad_{}{}/{}_steps/'.format(args.batch_size, args.suffix, method)
    # if os.path.exists(step_experiment_dir + filename):
    #     os.remove(step_experiment_dir + filename)
    # if not os.path.exists(step_experiment_dir):
    os.makedirs(step_experiment_dir, exist_ok=True)
    with open(step_experiment_dir + filename, 'w') as step_file:
        step_file.write('epoch, iter, train_df_loss, iter_forward_time, iter_backward_time, forward_solve_time, backward_solve_time, forward_setup_time, backward_setup_time, cosine_sim_val, l2_dist_val, l1_dist_val, l_inf_dist_val, grad_norm_val\n')
        step_file.flush()
        
        
    # if method=="ffocp_eq":
    #     timing_directory = '../synthetic_results_{}{}/{}_debug_timing/'.format(args.batch_size, args.suffix, method)
    #     timing_filename = '{}_ydim{}_lr{}_seed{}.csv'.format(method, ydim, learning_rate, seed)
    #     if os.path.exists(timing_directory + timing_filename):
    #         os.remove(timing_directory + timing_filename)

    #     if not os.path.exists(timing_directory):
    #         os.makedirs(timing_directory)
            
    #     with open(timing_directory + timing_filename, 'w') as timing_file:
    #         timing_file.write('epoch, forward_time, backward_time, forward_opt_time, backward_opt_time, forward_num_iters, backward_num_iters, forward_setup_time, backward_setup_time, forward_solve_time, backward_solve_time, pre_autograd_time, autograd_time\n')
    #         timing_file.flush()

    
    
    ts_weight = 0
    norm_weight = 0
    
    experiment_start_time = 0
    
    with open(directory + filename, 'w') as file:
        file.write('epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, forward_time, backward_time\n')
        file.flush()
        
        for epoch in range(num_epochs):
            print(f"##### epoch {epoch}: ")
            
            train_ts_loss_list, test_ts_loss_list = [], []
            train_df_loss_list, test_df_loss_list = [], []
            forward_time = 0
            backward_time = 0
            
            forward_optimization_time = 0
            backward_optimization_time = 0
            pre_autograd_time = 0
            autograd_time = 0
            
            forward_solve_time = 0
            backward_solve_time = 0
            forward_setup_time = 0
            backward_setup_time = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                if i%10==0:
                    time_elapsed = experiment_start_time - time.time()
                    print(f"\t\t train example: {i}/{len(train_loader)}, time elpased:{time_elapsed/60} minutes")
                
                iter_start_time = time.time()
                
                start_time = time.time()
                cvxpylayer.predictor = copy.deepcopy(model.predictor)

                z, y_pred = model(x) # (opt solution, predicted q)

                z_cvx, y_pred_cvx = cvxpylayer(x)

                ts_loss = loss_fn(y_pred, y)
                ts_loss_cvx = loss_fn(y_pred_cvx, y)
                
                df_loss = torch.mean(y * z)
                loss = df_loss + ts_loss * ts_weight + torch.norm(z) * norm_weight

                df_loss_cvx = torch.mean(y * z_cvx)
                loss_cvx = df_loss_cvx + ts_loss_cvx * ts_weight + torch.norm(z_cvx) * norm_weight
                
                forward_time_ = time.time() - start_time
                
                start_time = time.time()
                
                with torch.no_grad():
                    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                    grads_cvx = torch.autograd.grad(loss_cvx, cvxpylayer.parameters(), create_graph=True)

                    grads_flat = torch.cat([g.flatten() for g in grads])
                    grads_cvx_flat = torch.cat([g.flatten() for g in grads_cvx])
                    
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        grads_flat.unsqueeze(0), 
                        grads_cvx_flat.unsqueeze(0), 
                        dim=1
                    )
                    l2_distance = torch.norm(grads_flat - grads_cvx_flat, p=2)
                    l1_distance = torch.norm(grads_flat - grads_cvx_flat, p=1)
                    l_inf_distance = torch.norm(grads_flat - grads_cvx_flat, p=float('inf'))
                    grad_norm = torch.norm(grads_flat, p=2)

                    cosine_sim_val = cosine_similarity.item()
                    l2_dist_val = l2_distance.item()
                    l1_dist_val = l1_distance.item()
                    l_inf_dist_val = l_inf_distance.item()
                    grad_norm_val = grad_norm.item()

                    # print(f"cosine similarity: {cosine_sim_val}")
                    # print(f"l2 distance: {l2_distance}")
                    # print(f"l1 distance: {l1_distance}")
                    # print(f"l_inf distance: {l_inf_distance}")
                
                loss.backward()
                backward_time_ = time.time() - start_time
                
                iter_time = time.time() - iter_start_time
                
                do_record = True #not(epoch==0 and i==0)
                if do_record:
                    forward_time += forward_time_
                    backward_time += backward_time_
                
                    if method=="ffocp_eq":
                        forward_solve_time_ = model.optlayer.forward_solve_time
                        backward_solve_time_ = model.optlayer.backward_solve_time
                        forward_setup_time_ = model.optlayer.forward_setup_time
                        backward_setup_time_ = model.optlayer.backward_setup_time
                    elif method=="lpgd" or method=="cvxpylayer":
                        forward_solve_time_ = model.optlayer.info['solve_time']
                        backward_solve_time_ = model.optlayer.info['dDT_time']
                        forward_setup_time_ = model.optlayer.info['canon_time']
                        backward_setup_time_ = model.optlayer.info['dcanon_time']
                    else:
                        forward_solve_time_ = forward_time_
                        backward_solve_time_ = backward_time_
                        forward_setup_time_ = 0
                        backward_setup_time_ = 0
                        
                        
                    forward_solve_time += forward_solve_time_
                    backward_solve_time += backward_solve_time_
                    forward_setup_time += forward_setup_time_
                    backward_setup_time += backward_setup_time_
                
                
                    with open(step_experiment_dir + filename, 'a') as step_file:
                        step_file.write(f'{epoch},{i},{df_loss.item()},{forward_time_},{backward_time_},{forward_solve_time_},{backward_solve_time_},{forward_setup_time_},{backward_setup_time_},{cosine_sim_val},{l2_dist_val},{l1_dist_val},{l_inf_dist_val},{grad_norm_val}\n')
                        step_file.flush()
                        
                    train_ts_loss_list.append(ts_loss.item())
                    train_df_loss_list.append(df_loss.item())
                    


                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                #if epoch > 0:
                optimizer.step()

                optimizer.zero_grad()

                
                
                print(f"train loss: {loss.item()}, forward solve time: {forward_solve_time_}, backward solve time: {backward_solve_time_}, forward setup time: {forward_setup_time_}, backward setup time: {backward_setup_time_}, backward time: {backward_time_}, iter time: {iter_time}")
                
                # if method=="ffocp_eq":
                #     print(f"forward_opt_time: {forward_optimization_time_}, backward_opt_time: {backward_optimization_time_}")
                #     print(f"forward_num_iters: {forward_num_iters}, backward_num_iters: {backward_num_iters}")

            print('Forward time {}, backward time {}'.format(forward_time, backward_time))

            # model.eval()
            # with torch.no_grad():
            #     for i, (x, y) in enumerate(test_loader):
            #         # print("batch size : ", x.shape[0])
            #         z, y_pred = model(x) #opt solution, predicted q
            #         ts_loss = loss_fn(y_pred, y)
            #         df_loss = torch.mean(y * z)
                    
            #         optimizer.zero_grad()

            #         test_ts_loss_list.append(ts_loss.item())
            #         test_df_loss_list.append(df_loss.item())
            # model.eval()
            # for i, (x, y) in enumerate(test_loader):
            #     # print("batch size : ", x.shape[0])
            #     z, y_pred = model(x) #opt solution, predicted q
            #     ts_loss = loss_fn(y_pred, y)
            #     df_loss = torch.mean(y * z)
                
            #     optimizer.zero_grad()

            #     test_ts_loss_list.append(ts_loss.item())
            #     test_df_loss_list.append(df_loss.item())
            
            test_ts_loss_list.append(0)
            test_df_loss_list.append(0)


            train_ts_loss = np.mean(train_ts_loss_list)
            train_df_loss = np.mean(train_df_loss_list)
            test_ts_loss = np.mean(test_ts_loss_list)
            test_df_loss = np.mean(test_df_loss_list)
            print("Epoch {}, Train TS Loss {}, Test TS Loss {}, Train DF Loss {}, Test DF Loss {}".format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss))

            writer.add_scalar('Loss/TS/train', train_ts_loss, epoch)
            writer.add_scalar('Loss/TS/test', test_ts_loss, epoch)
            writer.add_scalar('Loss/DF/train', train_df_loss, epoch)
            writer.add_scalar('Loss/DF/test', test_df_loss, epoch)

            file.write('{},{},{},{},{},{},{}\n'.format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, forward_time, backward_time))
            file.flush()
            
            # if method=="ffocp_eq":
            #     with open(timing_directory + timing_filename, 'a') as timing_file:
            #         timing_file.write('{},{},{},{},{},{},{}\n'.format(epoch, forward_time, backward_time, forward_optimization_time, backward_optimization_time, pre_autograd_time, autograd_time))
            #         timing_file.flush()  
            
            writer.flush()
            
        file.close()