import torch
from models import MLP

def genData(device, input_dim=64, output_dim=10, num_samples=2048, batch_size=32):
    # Generate some data
    y = torch.rand(num_samples, output_dim).to(device)
    model = MLP(input_dim=output_dim, output_dim=input_dim).to(device)

    x = model(y).detach() 

    # Add some noise
    x = x + torch.rand(num_samples, input_dim).to(device) * 0.1

    dataset = list(zip(x,y))
    train_split = 0.8
    training_size = int(num_samples * train_split)
    
    train_loader = torch.utils.data.DataLoader(dataset[:training_size], batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(dataset[training_size:], batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader
