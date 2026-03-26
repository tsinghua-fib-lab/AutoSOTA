import cuml
import numpy as np
import cudf
from cuml.svm import LinearSVC,SVC
import torch
import dgl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
from sklearn.cluster import SpectralClustering


warnings.filterwarnings("ignore")




def train(dataset,norm,c):

    a = dgl.data.TUDataset(dataset)
    x = torch.load(f'cache/{dataset}.pt')
    if norm ==1:
        x = torch.nn.functional.normalize(x, p=2, dim=1)

    x = x.numpy()
    scaler = StandardScaler()
    if norm ==2:
        x = scaler.fit_transform(x)
    y = a.graph_labels.numpy()

    skf = StratifiedKFold(n_splits=10, shuffle=True,random_state = 0)

    val_acc_list = []
    for train_idx, test_idx in skf.split(x, y):
        svc = SVC(kernel='rbf')
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        val_acc_list.append(accuracy)

    print("Dataset:{} norm:{} C:{} Accuracy:{}, STD:{}".format(dataset,norm,c,np.mean(val_acc_list),np.std(val_acc_list)))

folder_path = "./cache"

pt_files = [file for file in os.listdir(folder_path) if file.endswith("count.pt")]
for path in pt_files:
    dataset = path.replace('_len_count.pt','')
    for i in [1,2]:
        for j in (0.2,0.5,1,2,3):
            train(dataset,i,j)