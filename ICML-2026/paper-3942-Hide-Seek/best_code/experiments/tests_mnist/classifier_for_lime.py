import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=2):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Flatten the input if it's not already
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def baseline_forward(self, x): #looks like I can remove all the baseline_forward methods.
        """Forward pass for prediction"""
        return self.forward(x)

class MNIST3vs8Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        # ✅ CrossEntropyLoss expects integer class indices (LongTensor)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # ✅ Ensure input is float
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            val_acc = self.evaluate(val_loader)
            train_loss = running_loss / len(train_loader)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return train_losses, val_accuracies
    
    def predict_proba(self, x):
        """For LIME or probability-based evaluation — no labels involved here"""
        self.model.eval()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        else:
            assert isinstance(x, np.ndarray)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        x = torch.from_numpy(x).float()
        
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.model.baseline_forward(x)
            probs = F.softmax(logits, dim=1)  # ✅ probabilities for LIME
            
        return probs.cpu().numpy()   # Always return numpy floats
    
    def evaluate(self, data_loader):
        """Evaluate accuracy using integer labels"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def baseline_forward(self, x):
        return self.model.baseline_forward(x)

def prepare_data(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Prepare data loaders for training and validation
    
    Args:
        X_train, X_val: numpy arrays of shape (n_samples, 28, 28) or (n_samples, 784)
        y_train, y_val: numpy arrays with labels 
                        (can be class indices [0,1] or one-hot [[1,0],[0,1]])
    """
    # Flatten the images if they're not already flattened
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if len(X_val.shape) > 2:
        X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Convert to float32 for consistency
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    
    # ✅ Handle one-hot labels by converting to class indices
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val = np.argmax(y_val, axis=1)
    
    # Ensure labels are integers
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def run_model_lime_nn_classifier(X_train, y_train, X_test, y_test, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    assert X_train is not None
    # Prepare data loaders
    train_loader, test_loader = prepare_data(X_train, y_train, X_test, y_test)
    
    # Create model
    model = MNISTClassifier()
    trainer = MNIST3vs8Trainer(model, device)
    
    print("Starting training...")
    train_losses, val_accuracies = trainer.train(train_loader, test_loader, num_epochs=epochs)
    
    # Final evaluation
    final_accuracy = trainer.evaluate(test_loader)
    print(f"Final test accuracy: {final_accuracy:.4f}")
    
    # Example prediction using your required interface
    sample_data = X_test
    probs = trainer.predict_proba(sample_data)
    predicted_classes = np.argmax(probs, axis=1)
    
    # print((predicted_classes == y_test[:,1]).mean())

    print("\nSample predictions:")
    print(f"Predicted probabilities: {probs[:5]}")
    print(f"Predicted classes: {predicted_classes[:5]}")
    print(f"True classes: {y_test[:5]}")
    return trainer, device