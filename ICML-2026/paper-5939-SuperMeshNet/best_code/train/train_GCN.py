import torch
import torch.nn.functional as F
import numpy as np
from models.GCN import GCN_shared
from models.Common import Decoder_F, Decoder_G
from tqdm import tqdm
from utils.knn_interpolate import knn_interpolate
import os
import copy
from torch.cuda.amp import autocast, GradScaler

def train_GCN_comp(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, hidden_size=64, output_size=1, depth=4,  residual=True, ib_n=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    ema_decay = 0.999
    stage1_epochs = 10  # supervised-only pre-training
    ramp_epochs = 5     # linear ramp of complementary loss
    for i in range(num_exp):
        model_shared=GCN_shared(depth, y_input_size, pos_input_size, hidden_size, ib_n).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        model_G= Decoder_G(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters())+list(model_G.parameters()), lr=1e-3)

        # EMA shadow parameters
        ema_shared = copy.deepcopy(model_shared.state_dict())
        ema_F = copy.deepcopy(model_F.state_dict())
        ema_G = copy.deepcopy(model_G.state_dict())

        loss_list=[]
        test_loss_list=[]

        for epoch in tqdm(range(5000)):
            # Ramp complementary loss: 0.0 in stage1, linearly→1.0 during ramp, 1.0 after
            if epoch < stage1_epochs:
                lambda_comp = 0.0
            else:
                lambda_comp = min(1.0, (epoch - stage1_epochs) / ramp_epochs)

            mean_loss=[]
            for _ in (range(len(train_data2)+len(train_data1))):
                with autocast():
                    idx3=np.random.choice(list(range(len(train_data2))),1)[0]
                    idx12=np.random.choice(list(range(len(train_data1))),2,replace=False)
                    idx1=idx12[0]
                    idx2=idx12[1]

                    l_pos1, l_y1, l_e1, h_pos1, h_e1, y1=train_data1[idx1]
                    l_pos2, l_y2, l_e2, h_pos2, h_e2, y2=train_data1[idx2]
                    l_pos3, l_y3, l_e3, h_pos3, h_e3=train_data2[idx3]
                    optimizer.zero_grad()

                    emb1=model_shared(l_pos1, l_y1, l_e1,  h_pos1, h_e1)
                    emb2=model_shared(l_pos2, l_y2, l_e2,  h_pos2, h_e2)
                    emb3=model_shared(l_pos3, l_y3, l_e3,  h_pos3, h_e3)

                    out1=model_F(emb1,l_y1, l_pos1, h_pos1)
                    out2=model_F(emb2, l_y2, l_pos2, h_pos2)
                    out3=model_F(emb3, l_y3, l_pos3, h_pos3)

                    out12=model_G(emb1, l_y1, l_pos1, h_pos1, emb2, l_y2, l_pos2, h_pos2)
                    out23=model_G(emb2, l_y2, l_pos2, h_pos2, emb3, l_y3, l_pos3, h_pos3)
                    out31=model_G(emb3, l_y3, l_pos3, h_pos3, emb1, l_y1, l_pos1, h_pos1)

                    # Supervised loss (always active)
                    loss_sup = F.mse_loss(out1,y1)+F.mse_loss(out2,y2)

                    if lambda_comp > 0:
                        # Complementary loss (ramped in)
                        loss_F_comp = F.mse_loss(out3, (out31+knn_interpolate(y1,h_pos1,h_pos3)).detach())+F.mse_loss(out3, knn_interpolate(y2-out23, h_pos2,h_pos3).detach())
                        loss_G = F.mse_loss(out12,y1-knn_interpolate(y2,h_pos2,h_pos1))+F.mse_loss(out31, out3.detach()-knn_interpolate(y1,h_pos1,h_pos3))+F.mse_loss(out23,y2-knn_interpolate(out3,h_pos3,h_pos2).detach())
                        loss = loss_sup + lambda_comp * (loss_F_comp + loss_G)
                    else:
                        loss = loss_sup

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_shared.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(model_F.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(model_G.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    # EMA update
                    with torch.no_grad():
                        for ema_p, p in zip(ema_shared.values(), model_shared.state_dict().values()):
                            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                        for ema_p, p in zip(ema_F.values(), model_F.state_dict().values()):
                            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                        for ema_p, p in zip(ema_G.values(), model_G.state_dict().values()):
                            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

                    mean_loss.append(loss.item())
            mean_loss=np.mean(mean_loss)

            # Test loss with EMA weights
            mean_test_loss=[]
            with torch.no_grad():
                orig_shared = copy.deepcopy(model_shared.state_dict())
                orig_F = copy.deepcopy(model_F.state_dict())
                orig_G = copy.deepcopy(model_G.state_dict())
                model_shared.load_state_dict(ema_shared)
                model_F.load_state_dict(ema_F)
                model_G.load_state_dict(ema_G)

                for l_pos, l_y, l_e, h_pos, h_e,  y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = F.mse_loss(out,y)
                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)

                model_shared.load_state_dict(orig_shared)
                model_F.load_state_dict(orig_F)
                model_G.load_state_dict(orig_G)

            if epoch%5==0 and verbose:
                stage = "S1" if lambda_comp == 0 else ("R"+str(int(lambda_comp*100))) if lambda_comp < 1.0 else "S2"
                print("epoch {} {} train loss {} test loss {}".format(epoch, stage, mean_loss, mean_test_loss))

            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)

            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break

        np.save(dir+"/GCN_comp_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0], i), test_loss_list)
    return None



def train_GCN_sup(device, train_data, test_data, dir, num_exp=5, y_input_size=1, pos_input_size=2, hidden_size=64, output_size=1, depth=4, residual=True,  ib_n=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    ema_decay = 0.999
    for i in range(num_exp):
        model_shared=GCN_shared(depth, y_input_size, pos_input_size, hidden_size, ib_n).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters()), lr=1e-3)

        # EMA shadow parameters
        ema_shared = copy.deepcopy(model_shared.state_dict())
        ema_F = copy.deepcopy(model_F.state_dict())

        loss_list=[]
        test_loss_list=[]

        for epoch in tqdm(range(5000)):
            mean_loss=[]
            for _ in (range(len(train_data))):
                with autocast():
                    idx=np.random.choice(list(range(len(train_data))),1)[0]
                    l_pos, l_y, l_e, h_pos, h_e, y=train_data[idx]

                    optimizer.zero_grad()
                    emb=model_shared(l_pos, l_y, l_e,  h_pos, h_e)
                    out=model_F(emb, l_y, l_pos, h_pos)
                    loss = F.mse_loss(out,y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_shared.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(model_F.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    # EMA update
                    with torch.no_grad():
                        for ema_p, p in zip(ema_shared.values(), model_shared.state_dict().values()):
                            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                        for ema_p, p in zip(ema_F.values(), model_F.state_dict().values()):
                            ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

                    mean_loss.append(loss.item())

                mean_loss.append(loss.item())
            mean_loss=np.mean(mean_loss)

            # Test loss with EMA weights
            mean_test_loss=[]
            with torch.no_grad():
                orig_shared = copy.deepcopy(model_shared.state_dict())
                orig_F = copy.deepcopy(model_F.state_dict())
                model_shared.load_state_dict(ema_shared)
                model_F.load_state_dict(ema_F)

                for l_pos, l_y, l_e, h_pos, h_e, y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = F.mse_loss(out,y)
                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)

                model_shared.load_state_dict(orig_shared)
                model_F.load_state_dict(orig_F)

            if epoch%5==0 and verbose:
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))

            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)

            if epoch> 15:
                if -(np.mean(test_loss_list[-15:])-np.mean(test_loss_list[-16:-1]))/np.mean(test_loss_list[-16:-1])<0.01:
                    break

        np.save(dir+"/GCN_sup_{}_{}_{}_{}.npy".format(len(train_data), len(train_data), str(ib_n)[0], i), test_loss_list)
    return None
