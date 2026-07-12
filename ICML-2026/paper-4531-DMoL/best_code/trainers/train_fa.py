from .train_backprop import train_backprop

def train_fa(model, train_loader, optimizer, criterion, device):
    # FA的训练循环与Backprop完全相同
    return train_backprop(model, train_loader, optimizer, criterion, device)
