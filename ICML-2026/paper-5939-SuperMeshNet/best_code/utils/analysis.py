import numpy as np

def RMSE(result_dir, model, learning, N_paired, N_total, ib, exp_list=list(range(5))):
    out=[]
    
    for exp_num in exp_list:
        loss_curve=np.load(result_dir+"/"+model+"_"+learning+"_"+str(N_paired)+"_"+str(N_total)+"_"+ib+"_"+str(exp_num)+".npy")
        out.append(np.mean(np.sqrt(loss_curve[-6:-1])))
    return np.mean(out), np.std(out)

