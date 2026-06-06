import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as scio

# Import Opacus for Differential Privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

def Regularized_loss(model, n, y_pred, y_true, p=4, lam=0.01):
    y_true = torch.argmax(y_true, dim=1)
    classification_loss = F.multi_margin_loss(y_pred, y_true) # hinge loss
    # classification_loss = -torch.mean(y_true * torch.log_softmax(y_pred, dim=1)) # logits
    
    # Fix for accessing weights in GradSampleModule
    if hasattr(model, "_module"):
        weight = model._module.weight
    else:
        weight = model.weight
        
    RG_loss = 1/n * torch.norm(weight.unsqueeze(1) - weight.unsqueeze(0), p=2, dim=2).pow(p).sum()
    loss = classification_loss + lam*RG_loss
    return loss


def R_MLR(para):
    path = f'./dataset/{para.data}.mat'
    X = scio.loadmat(path)['X']
    y = scio.loadmat(path)['Y'].squeeze()
    print(X.shape, y.shape)

    n, d = X.shape[0], X.shape[1]
    num_class = len(np.unique(y))

    if para.If_scale == True:
        scaler = StandardScaler()#MinMaxScaler()#StandardScaler()
        X = scaler.fit_transform(X)

    y = y-1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = \
        train_test_split(X_tensor, y_tensor, test_size=para.test_size, random_state=para.state)

    # Convert to one-hot encoding
    y_train_one_hot = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=num_class).float()
    y_test_one_hot = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=num_class).float()

    # Create DataLoader for batch processing (required for Opacus)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Define the model
    model = torch.nn.Linear(d, num_class)
    
    # Make sure the model is compatible with Opacus
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    
    # Create optimizer based on selection
    if para.optimizer == 'adam' or para.optimizer == 'adagrad':
        if para.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
        elif para.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
        print(f"Using Adam optimizer with lr={para.lr}")

        if para.loss == 'svm':
            loss_fn = Regularized_loss
        elif para.loss == 'CE':
            loss_fn = torch.nn.CrossEntropyLoss()
        
        # Create privacy engine
        privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to the optimizer
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=para.num_epoch,
            target_epsilon=para.eps,
            target_delta=para.delta,
            max_grad_norm=1,
        )
        print(f"Using sigma={optimizer.noise_multiplier} for privacy")
        
        # Training loop for Adam
        for epoch in range(para.num_epoch):
            total_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                y_batch_one_hot = F.one_hot(y_batch, num_classes=num_class).float()
                
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(model, n, y_pred, y_batch_one_hot, para.p, para.lam)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{para.num_epoch}], Loss: {avg_loss:.4f}")
                
                with torch.no_grad():
                    y_pred = model(X_test)
                    correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test_one_hot, dim=1)).sum().item()
                    test_acc = correct / len(X_test)
                    print(f"Test Accuracy: {test_acc:.4f}")
                    
                    # Print current privacy spent
                    epsilon = privacy_engine.get_epsilon(delta=para.delta)
                    print(f"Current ε: {epsilon:.2f} (target: {para.eps})")
                
                if epoch == para.num_epoch - 1:
                    final_epsilon = privacy_engine.get_epsilon(delta=para.delta)
                    print(f"Final Privacy Guarantee: (ε = {final_epsilon:.2f}, δ = {para.delta})")
        
    elif para.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=para.lr)
        print(f"Using LBFGS optimizer with lr={para.lr}")
        
        # Training loop for LBFGS
        for epoch in range(para.num_epoch):
            # Define closure function for LBFGS
            def closure():
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = Regularized_loss(model, n, y_pred, y_train_one_hot, para.p, para.lam)
                loss.backward()
                return loss
            
            # Perform optimization step with closure
            loss = optimizer.step(closure)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{para.num_epoch}], Loss: {loss.item():.4f}")
                
                with torch.no_grad():
                    y_pred = model(X_test)
                    correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test_one_hot, dim=1)).sum().item()
                    test_acc = correct / len(X_test)
                    print(f"Test Accuracy: {test_acc:.4f}")
                    
                # No privacy guarantees for LBFGS
                print("LBFGS optimizer doesn't support differential privacy")
    
    else:  # SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)
        print(f"Using SGD optimizer with lr={para.lr}")
        
        # Create privacy engine
        privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to the optimizer
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=para.num_epoch,
            target_epsilon=para.eps,
            target_delta=para.delta,
            max_grad_norm=1,
        )

        print(f"Using sigma={optimizer.noise_multiplier} for privacy")
        
        # Training loop for SGD
        for epoch in range(para.num_epoch):
            total_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                y_batch_one_hot = F.one_hot(y_batch, num_classes=num_class).float()
                
                # Forward pass
                y_pred = model(X_batch)
                loss = Regularized_loss(model, n, y_pred, y_batch_one_hot, para.p, para.lam)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{para.num_epoch}], Loss: {avg_loss:.4f}")
                
                with torch.no_grad():
                    y_pred = model(X_test)
                    correct = (torch.argmax(y_pred, dim=1) == torch.argmax(y_test_one_hot, dim=1)).sum().item()
                    test_acc = correct / len(X_test)
                    print(f"Test Accuracy: {test_acc:.4f}")
                    
                    # Print current privacy spent
                    epsilon = privacy_engine.get_epsilon(delta=para.delta)
                    print(f"Current ε: {epsilon:.2f} (target: {para.eps})")
                
                if epoch == para.num_epoch - 1:
                    final_epsilon = privacy_engine.get_epsilon(delta=para.delta)
                    print(f"Final Privacy Guarantee: (ε = {final_epsilon:.2f}, δ = {para.delta})")

    print(f"Training complete!")

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    parser = argparse.ArgumentParser(
        description='train with differential privacy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='vehicle_uni')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--lr_scale', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--state', type=int, default=42)
    parser.add_argument('--If_scale', default=True)
    parser.add_argument('--eps', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adam', 'lbfgs', 'adagrad'], 
                        help='Optimizer to use')
    parser.add_argument('--loss', type=str, default='svm', 
                        choices=['svm','CE'], 
                        help='Optimizer to use')    
    para = parser.parse_args()
    para.test_size = 0.2

    if para.data == 'HHAR':  # n=10229,d=561,c=6
        para.lr = 0.02
        para.lam = 0.0005
    elif para.data == 'Cornell':  # n=827,d=4134,c=7
        para.lr = 0.1
        para.lam = 0.005
        para.If_scale = False
    elif para.data == 'USPS':  # n=9298,d=256,c=10
        para.lr = 0.01
        para.lam = 0.001
    elif para.data == 'ISOLET':  # n=1560,d=617,c=26
        para.lr = 0.001
        para.lam = 0.001
        para.If_scale = False
    elif para.data == 'Dermatology':  # n=366,d=34,c=6
        para.lr = 0.01
        para.lam = 0.1
        para.If_scale = False
    elif para.data == 'Vehicle':  # n=946,d=18,c=4
        para.lr = 0.05
        para.lam = 0.0001
    elif para.data == 'Glass':  # n=214,d=9,c=6
        para.lr = 0.01
        para.lam = 0.0001

    para.lr *= para.lr_scale
    if para.optimizer=="sgd":
        para.lr *= 100
    
    R_MLR(para)

