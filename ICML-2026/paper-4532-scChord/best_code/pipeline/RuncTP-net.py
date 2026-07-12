# Run with scvi environment
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.optim as optim
import sys
from scipy import stats
import scanpy as sc
from utils import compute_PCC, compute_CMD, compute_RMSE
interactive = True

# load datasets
train_id = "24"
test_id = "23"
data_path = "../dataset/"
train_data_path = data_path + "dataset" + train_id + "_adata.h5ad" # path to training data
test_data_path = data_path + "dataset" + test_id + "_adata.h5ad" # path to test data
output_path = "../Results/" # path to results

train_data = sc.read(train_data_path)
test_data = sc.read(test_data_path)

genes_inter = train_data.var_names.intersection(test_data.var_names)
train_data = train_data[:,genes_inter].copy()
test_data = test_data[:,genes_inter].copy()

X_file = train_data.to_df().T
y_file = train_data.obsm['protein_expression'].T

header_list = ['']
loc = output_path+'ctpnet_model/'+test_id+'from'+train_id+'_'

repi = 0
n_batches = 32
protein_list = None 
gene_list = None
X_final = None
y_final = None
max_epochs=4

x_batch2 = test_data.to_df().T
y_list_savePath = output_path +test_id+"_from_"+train_id+"_cTPnet1.csv"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(14505, 1000)
        self.fc1 = nn.Linear(X.shape[1], 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.ModuleDict({})
        for p in protein_list:
            self.fc3[p] = nn.Linear(256, 64)
        self.fc4 = nn.ModuleDict({})
        for p in protein_list:
            self.fc4[p] = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = {}
        for p in protein_list:
            outputs[p] = self.fc4[p](F.relu(self.fc3[p](x)))
        return outputs

X = X_file
y = y_file
if interactive:
    X=X.transpose().sample(frac=0.1,random_state=4905).transpose()
	# Dealing with X's dimensionality
gene = set(X.index.tolist())
if gene_list is None:
    gene_list=gene
else:
    gene_list=set(gene_list).intersection(gene)
    
gene_list=list(gene_list)
gene_list.sort()
X=X.loc[gene_list,]

if not X_final is None:
    X_final=X_final.loc[gene_list,]
	
    # Dealing with Y's dimensionality
protein = set(y.index)
if protein_list is None:
    protein_list = protein
else:
    protein_list=set(protein_list).intersection(protein)
protein_list = list(protein_list)   
	# print(protein)
	# y.index=protein
y = y[X.columns]
y = y.loc[protein_list,]
	# Add header to cell
X.columns = list(map(lambda x: header_list[0]+'-'+x, X.columns.tolist()))
y.columns = list(map(lambda x: header_list[0]+'-'+x, y.columns.tolist()))
    
X_final = X
y_final = y   

# Normalize y
shared_proteins = y_final.apply(lambda x: not x.isna().any(),axis=1)
y = y_final.apply(lambda x: np.log((x+1.0)/stats.gmean(x[shared_proteins]+1.0)), axis=0)
del(y_final)
# Use direct UMI count is hard to transfer across experiment
# Let's try normalize the data with seurat like method
X=X_final.apply(lambda x: np.log((x*10000.0/sum(x))+1.0), axis=0)
del(X_final)
# random cell order
X=X.T
X=X.sample(frac=1,random_state=4905)
# separate test data from train data
if interactive:
	X_test=X.sample(frac=0.15,random_state=4905)# Need change after test
else:
	X_test=X.sample(frac=0.15,random_state=4905)# Need change after test

y_test=y[X_test.index]

X=X.drop(X_test.index)
y=y.drop(columns=y_test.columns)
y=y[X.index]


# covert to tenor
X=torch.tensor(X.values)
X=X.type(torch.FloatTensor)
y=torch.tensor(y.values)
y=y.type(torch.FloatTensor)
y=torch.t(y)

X_test=torch.tensor(X_test.values)
X_test=X_test.type(torch.FloatTensor)
y_test=torch.tensor(y_test.values)
y_test=y_test.type(torch.FloatTensor)
y_test=torch.t(y_test)


net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001,amsgrad=True, weight_decay=0.001)
# optimizer = optim.Adagrad(net.parameters(), lr=lr_vec[repi], lr_decay=0.001)

train_loss=pd.DataFrame(np.zeros(shape=(len(protein_list),max_epochs)),index=protein_list)
test_loss=pd.DataFrame(np.zeros(shape=(len(protein_list),max_epochs)),index=protein_list)

# Init early stop
patience=30
best_score=None
Dy=len(protein_list)
estop_counter=pd.Series(np.zeros(Dy),index=protein_list)
early_stop=pd.Series([False]*Dy,index=protein_list)

for epoch in range(max_epochs):
	if all(early_stop):
		break
	running_loss=pd.Series(np.zeros(Dy),index=protein_list)
	for i in range(int(y.shape[0]/n_batches)):
		# Local batches and labels
		local_X, local_y = X[i*n_batches:min((i+1)*n_batches,X.shape[0]-1),], y[i*n_batches:min((i+1)*n_batches,y.shape[0]-1),]
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs_dict = net(local_X)
		loss=None
		loss_count=0.0
		for p in protein_list:
			notNaN=(local_y[:,protein_list.index(p):(protein_list.index(p)+1)]==local_y[:,protein_list.index(p):(protein_list.index(p)+1)])
			loss_p=criterion(outputs_dict[p][notNaN],local_y[:,protein_list.index(p):(protein_list.index(p)+1)][notNaN])
			if not torch.isnan(loss_p):
				loss_count+=1.0
				running_loss[p]+=loss_p.item()
				if loss is None:
					loss=loss_p
				else:
					loss=loss+loss_p
		loss.backward()
		optimizer.step()
		if(i==(int(y.shape[0]/n_batches)-1)):
			train_loss.iloc[:,epoch]=(running_loss / 150)
		if i % 150 == 149:    # print every mini-batches
			# print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_loss / 150)))
			running_loss=pd.Series(np.zeros(Dy),index=protein_list)
			sys.stdout.flush()
	test_outputs = net(X_test)
	test_outputs = [test_outputs[p] for p in protein_list]
	test_outputs=torch.transpose(torch.stack(test_outputs),0,1).view(X_test.shape[0],-1)
	test_loss_i=pd.Series([criterion(test_outputs[:,pi][y_test[:,pi]==y_test[:,pi]], y_test[:,pi][y_test[:,pi]==y_test[:,pi]]).item() for pi in range(Dy)],index=protein_list)
	test_loss.iloc[:,epoch]=test_loss_i
	# if epoch % 10 == 9:
	# 	f,ax=plt.subplots(figsize=(6,6))
	# 	ax.scatter(y_test.detach().numpy(),test_outputs.detach().numpy())
	# 	ax.plot([-2,5],[-2,5],ls='--',c='.3')
	# 	# ax.text(3,-2,'correlation: '+str(np.corrcoef(test_outputs.detach().numpy().flatten(),y_test.detach().numpy().flatten())[1,0]))
	# 	df = pd.DataFrame({"y_pred":test_outputs.detach().numpy().flatten(),'y_truth':y_test.detach().numpy().flatten()})
	# 	ax.text(3,-2,'correlation: '+str(round(df.corr().values[1,0],4)))
	# 	fig = ax.get_figure()
	# 	fig.savefig(loc+'figure_rep'+str(repi)+'_ep'+str(epoch)+'.pdf')
	# 	sys.stdout.flush()
	# 	plt.close(fig)
	if epoch % 5 == 4:
		torch.save(net.state_dict(), loc+'model_rep'+str(repi)+'_ep'+str(epoch))
	# Implement early stopping
	if best_score is None:
		best_score=test_loss_i
	else:
		for p in protein_list:
			if test_loss_i[p]>(best_score[p]-0.001) and (not early_stop[p]):
				estop_counter[p]+=1
				if estop_counter[p]>=patience:
					early_stop[p]=True
			else:
				best_score[p]=test_loss_i[p]
				estop_counter[p]=0
		# print(estop_counter)

print('Finished Training')


torch.save(net.state_dict(), loc+'model_rep'+str(repi)+'_ep'+str(epoch))

train_loss.index=['train_'+p for p in protein_list]
test_loss.index=['test_'+p for p in protein_list]
log=pd.concat([train_loss,test_loss])

log.to_csv(loc+'log_rep'+str(repi)+'.csv')

X_pred_raw = x_batch2
# X_pred_raw = pd.read_csv(X_list_batch2, index_col=0)
X_pred = X_pred_raw.loc[gene_list,]
X_pred = X_pred.apply(lambda x: np.log((x*10000.0/sum(x))+1.0), axis=0).T
X_pred = torch.tensor(X_pred.values)
X_pred = X_pred.type(torch.FloatTensor)
test_outputs_ = net(X_pred)
def fun(x):
    test_outputs_[x] = test_outputs_[x].detach().numpy().flatten()
ing = [fun(i) for i in test_outputs_]
test_outputs_ = pd.DataFrame.from_dict(test_outputs_, orient='index')
test_outputs_.columns = X_pred_raw.columns
test_outputs_.to_csv(y_list_savePath)

# Compute evaluation metrics
print('Computing evaluation metrics...')

# Load ground truth data
true_data = test_data.obsm['protein_expression'].copy()
true_data.columns = true_data.columns.str.replace("-", "_")
true_data.columns = true_data.columns.str.replace(".", "_")

# Process prediction data: test_outputs_ has proteins as rows, cells as columns, need to transpose
pred_data = test_outputs_.T.copy()
# Process prediction data column names (protein names)
pred_data.columns = pred_data.columns.str.replace("-", "_")
pred_data.columns = pred_data.columns.str.replace(".", "_")

# Process training data protein column names
train_proteins = train_data.obsm['protein_expression'].columns.copy()
train_proteins = train_proteins.str.replace("-", "_")
train_proteins = train_proteins.str.replace(".", "_")

# Find intersection (using processed column names)
proteins_inter = train_proteins.intersection(true_data.columns)
pred_data = pred_data[proteins_inter]
true_data = true_data[proteins_inter]

# Ensure index consistency (cell names)
common_index = pred_data.index.intersection(true_data.index)
pred_data = pred_data.loc[common_index]
true_data = true_data.loc[common_index]

# For cTP-net, ground truth data needs CLR transformation for PCC calculation
clr_true = true_data.apply(lambda x: np.log((x + 1.0) / stats.gmean(x + 1.0)), axis=0)

# Compute PCC (per protein)
pcc_protein = compute_PCC(pred_data, clr_true, by='protein')
pcc_protein_df = pd.DataFrame([pcc_protein], index=['cTp_net'], columns=proteins_inter)

# Compute PCC (per cell)
pcc_cell = compute_PCC(pred_data, clr_true, by='cell')
pcc_cell_df = pd.DataFrame([pcc_cell], index=['cTp_net'], columns=common_index)

# Compute CMD (cell-cell)
cmd_cell = compute_CMD(pred_data, true_data, by='cell')
cmd_cell_df = pd.DataFrame([cmd_cell], index=['cTp_net'], columns=['cmd_values'])

# Compute CMD (protein-protein)
cmd_protein = compute_CMD(pred_data, true_data, by='protein')
cmd_protein_df = pd.DataFrame([cmd_protein], index=['cTp_net'], columns=['cmd_values'])

# Compute RMSE
rmse_value = compute_RMSE(pred_data, true_data)
rmse_df = pd.DataFrame([rmse_value], index=['cTp_net'], columns=['rmse_values'])

# Save metric results
pcc_protein_path = output_path + test_id + "_from_" + train_id + "_pcc_protein.csv"
pcc_cell_path = output_path + test_id + "_from_" + train_id + "_pcc_cell.csv"
cmd_cell_path = output_path + test_id + "_from_" + train_id + "_cmd_cell.csv"
cmd_protein_path = output_path + test_id + "_from_" + train_id + "_cmd_protein.csv"
rmse_path = output_path + test_id + "_from_" + train_id + "_rmse.csv"

pcc_protein_df.to_csv(pcc_protein_path)
pcc_cell_df.to_csv(pcc_cell_path)
cmd_cell_df.to_csv(cmd_cell_path)
cmd_protein_df.to_csv(cmd_protein_path)
rmse_df.to_csv(rmse_path)

print('Metrics saved:')
print(f'  PCC (protein): mean = {np.mean(pcc_protein):.4f}')
print(f'  PCC (cell): mean = {np.mean(pcc_cell):.4f}')
print(f'  CMD (cell): {cmd_cell:.4f}')
print(f'  CMD (protein): {cmd_protein:.4f}')
print(f'  RMSE: {rmse_value:.4f}')
