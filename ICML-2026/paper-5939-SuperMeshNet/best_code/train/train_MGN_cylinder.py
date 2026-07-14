import torch
import torch.nn.functional as F
import numpy as np
from models.MGN import MGN_shared
from models.Common import Decoder_F, Decoder_G 
from tqdm import tqdm
#from torch_geometric.nn.unpool import knn_interpolate
from utils.knn_interpolate import knn_interpolate
import os
from torch.cuda.amp import autocast, GradScaler



def weighted_mse(pred, target, wu=99, wp=1):
    # Assumes first 3 channels are velocity (u), last channel is pressure (p).
    if pred.size(-1) < 4 or target.size(-1) < 4:
        return F.mse_loss(pred, target)

    loss_u = F.mse_loss(pred[..., :2], target[..., :2])
    loss_p = F.mse_loss(pred[..., 3:4], target[..., 3:4])
    return wu * loss_u + wp * loss_p

def train_MGN_comp(device, train_data1, train_data2, test_data, dir, num_exp=5, y_input_size=4, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=4, depth=3,  residual=True, ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    scaler = GradScaler()
    for i in range(num_exp):

        model_shared=MGN_shared(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        model_G= Decoder_G(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters())+list(model_G.parameters()), lr=1e-3)
        
        loss_list=[]
        test_loss_list=[]

       
    
        for epoch in tqdm(range(5000)):
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
                        
                    loss_F = weighted_mse(out1,y1)+weighted_mse(out2,y2)+weighted_mse(out3, (out31+knn_interpolate(y1,h_pos1,h_pos3)).detach())+weighted_mse(out3, knn_interpolate(y2-out23, h_pos2,h_pos3).detach())              
                    loss_G = weighted_mse(out12,y1-knn_interpolate(y2,h_pos2,h_pos1))+weighted_mse(out31, out3.detach()-knn_interpolate(y1,h_pos1,h_pos3))+weighted_mse(out23,y2-knn_interpolate(out3,h_pos3,h_pos2).detach())
                    loss=loss_F+loss_G
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                            
                    mean_loss.append(loss.item())

            mean_loss=np.mean(mean_loss)        
            mean_test_loss=[]

            
            with torch.no_grad():
                
                for l_pos, l_y, l_e, h_pos, h_e,  y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = weighted_mse(out,y)
                        mean_test_loss.append(loss.item())

                   
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 100:
                break
        
        np.save(dir+"/MGN_comp_{}_{}_{}_{}.npy".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
        torch.save(model_F.cpu().state_dict(), dir+"/F_MGN_comp_{}_{}_{}_{}.pt".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i))
        torch.save(model_shared.cpu().state_dict(), dir+"/shared_MGN_comp_{}_{}_{}_{}.pt".format(len(train_data1), len(train_data2)+len(train_data1), str(ib_n)[0]+str(ib_e)[0], i))
    return None





def train_MGN_sup(device, train_data, test_data, dir, num_exp=5, y_input_size=4, pos_input_size=2, edge_input_size=4, hidden_size=30, output_size=4, depth=3, residual=True,  ib_n=True, ib_e=True, verbose=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    scaler = GradScaler()
    
    for i in range(num_exp):
        model_shared=MGN_shared(depth, y_input_size, pos_input_size, edge_input_size, hidden_size, ib_n, ib_e).to(device)
        model_F= Decoder_F(hidden_size, output_size, residual).to(device)
        optimizer=torch.optim.Adam(list(model_shared.parameters())+list(model_F.parameters()), lr=1e-3)
        
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
                    loss = weighted_mse(out,y)                    

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    mean_loss.append(loss.item())
            mean_loss=np.mean(mean_loss)
                    
            mean_test_loss=[]
            with torch.no_grad():
                for l_pos, l_y, l_e, h_pos, h_e, y in test_data:
                    with autocast():
                        emb=model_shared(l_pos, l_y, l_e, h_pos, h_e)
                        out = model_F(emb, l_y, l_pos, h_pos)
                        loss = weighted_mse(out,y)
                        mean_test_loss.append(loss.item())
                mean_test_loss=np.mean(mean_test_loss)
            
            if epoch%5==0 and verbose:    
                print("epoch {} train loss {} test loss {}".format(epoch, mean_loss, mean_test_loss))
            
            loss_list.append(mean_loss)
            test_loss_list.append(mean_test_loss)
            
            if epoch> 100:
                break
        np.save(dir+"/MGN_sup_{}_{}_{}_{}.npy".format(len(train_data), len(train_data), str(ib_n)[0]+str(ib_e)[0], i), test_loss_list)
        torch.save(model_F.cpu().state_dict(), dir+"/F_MGN_sup_{}_{}_{}_{}.pt".format(len(train_data), len(train_data), str(ib_n)[0]+str(ib_e)[0], i))
        torch.save(model_shared.cpu().state_dict(), dir+"/shared_MGN_sup_{}_{}_{}_{}.pt".format(len(train_data), len(train_data), str(ib_n)[0]+str(ib_e)[0], i))
    return None




