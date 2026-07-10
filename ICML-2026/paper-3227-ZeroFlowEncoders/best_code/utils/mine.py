# %%
# mutual information estimator

import torch 
import torch.nn as nn
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a standard MLP model with 2d + 1 * 500 * 500 * 500 * d
class MLP(nn.Module):
    def __init__(self, dx, dy):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dx + dy, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# mine loss function
def loss(x, y, model):
    # joint 
    fxy = model(torch.cat([x, y], dim=1))
    
    # marginal, concatenate shuffled x and y
    xshuffle = x[torch.randperm(x.shape[0]), :]
    yshuffle = y[torch.randperm(y.shape[0]), :]
    fxyshuffle = model(torch.cat([xshuffle, yshuffle], dim=1))
    
    # MINE loss
    loss = - (fxy.mean() - torch.log(torch.exp(fxyshuffle).mean()))
    
    return loss

def train_test_split(x, y):
    idx = torch.randperm(x.shape[0])
    x = x[idx, :]
    y = y[idx, :]
    
    # split x and y into train and test, 80% train and 20% test
    xtrain = x[:int(0.8*x.shape[0]), :]
    ytrain = y[:int(0.8*y.shape[0]), :]
    xtest = x[int(0.8*x.shape[0]):, :]
    ytest = y[int(0.8*y.shape[0]):, :]
    
    return xtrain, ytrain, xtest, ytest

# the main method
def MINE(x0,y0):
    x, y, xtest, ytest = train_test_split(x0, y0)
    
    model = MLP(x.shape[1], y.shape[1]).to(device)
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.002)
    tensorset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(tensorset, batch_size=100, shuffle=True)
    
    from tqdm import tqdm
    
    for epoch in tqdm(range(100)):
        for iter, (xi, yi) in enumerate(trainloader):
            xi = xi.to(device)
            yi = yi.to(device)
            
            optimizer.zero_grad()
            lossi = loss(xi, yi, model)
            lossi.backward()
            optimizer.step()
            
            # if iter % 100 == 0:
            #     #print loss two decimal places
            #     print(f'Epoch {epoch}, loss = {lossi.item():.2f}')
            
    
    return -loss(xtest.to(device), ytest.to(device), model).item(), model
                
# main function
if __name__ == '__main__':
    torch.manual_seed(1234)
    
    # test the code 
    d = 10
    n = 5000

    x = torch.randn(n, d)
    y1 = torch.randn(n, d) 
    print(MINE(x, y1))

    y2 = x + torch.randn(n, d)
    print(MINE(x, y2))

    y3 = x + torch.randn(n, d) * .25
    print(MINE(x, y3))

# %%