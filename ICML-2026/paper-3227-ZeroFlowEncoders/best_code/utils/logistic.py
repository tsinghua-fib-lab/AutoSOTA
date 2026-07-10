import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def acc_classifier(encoder, trainx, trainy, testx, testy, batch_size=512, epochs=100):
    """
    Trains a linear classifier on the encoder's frozen features and evaluates accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval() # Freeze encoder (batchnorm stats, dropout, etc.)

    # --- 1. FEATURE EXTRACTION HELPER ---
    # We extract features in batches to avoid running out of VRAM
    def get_features(images):
        features_list = []
        # Create a temporary loader for efficient batching
        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for (batch_imgs,) in loader:
                batch_imgs = batch_imgs.to(device)
                # Get the flat latent vector
                batch_feats = encoder(batch_imgs) 
                features_list.append(batch_feats.cpu()) # Store on CPU to save GPU memory
        
        return torch.cat(features_list, dim=0)

    print("Extracting features...")
    # Extract features ONCE. 
    # This transforms (N, 3, 64, 64) -> (N, Latent_Dim)
    train_feats = get_features(trainx)
    test_feats = get_features(testx)
    
    # Infer dimensions automatically
    input_dim = train_feats.shape[1]
    num_classes = len(torch.unique(trainy)) # Assumes labels are 0, 1, 2...
    
    print(f"Feature Dim: {input_dim} | Classes: {num_classes}")

    # --- 2. TRAIN LOGISTIC REGRESSION ---
    # The "Classifier" is just a single Linear Layer
    classifier = nn.Linear(input_dim, num_classes).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Move features to GPU for the fast linear training
    # (Since features are small vectors, they usually fit in VRAM easily now)
    train_feats_gpu = train_feats.to(device)
    trainy_gpu = trainy.to(device)
    
    # Create a loader for the features (not the images)
    feat_dataset = TensorDataset(train_feats_gpu, trainy_gpu)
    feat_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True)

    print("Training Linear Probe...")
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_feats, batch_labels in feat_loader:
            optimizer.zero_grad()
            logits = classifier(batch_feats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(feat_loader):.4f}")

    # --- 3. EVALUATION ---
    classifier.eval()
    test_feats_gpu = test_feats.to(device)
    testy_gpu = testy.to(device)

    with torch.no_grad():
        logits = classifier(test_feats_gpu)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == testy_gpu).sum().item()
        accuracy = correct / len(testy_gpu)

    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

from torch.utils.data import TensorDataset, DataLoader

def fine_tune(encoder, xtrain, ytrain, xtest, ytest, batch_size=128, epochs=50):
    """
    Performs End-to-End Fine-Tuning on the encoder using the provided data.
    
    Args:
        encoder (nn.Module): The pre-trained MAE backbone (e.g., ResNet18).
        xtrain (Tensor): Training images (N, 3, 64, 64).
        ytrain (Tensor): Training labels (N).
        xtest  (Tensor): Testing images (N, 3, 64, 64).
        ytest  (Tensor): Testing labels (N).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. SETUP DATA LOADERS
    # We use DataLoaders to handle batching, shuffling, and GPU transfer efficiently
    train_ds = TensorDataset(xtrain, ytrain)
    test_ds = TensorDataset(xtest, ytest)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. SETUP MODEL HEAD
    # Determine the feature dimension by passing a dummy input
    encoder = encoder.to(device)
    dummy_input = torch.zeros(1, 3, 64, 64).to(device)
    with torch.no_grad():
        # Assumes encoder returns a flat vector (N, Dim)
        feature_dim = encoder(dummy_input).shape[1]
    
    num_classes = len(torch.unique(ytrain))
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 3. SETUP OPTIMIZER (The Key to Fine-Tuning)
    # We optimize BOTH the Encoder (Backbone) and the Classifier
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-4},   # Lower LR for pre-trained backbone
        {'params': classifier.parameters(), 'lr': 1e-3} # Higher LR for new linear head
    ])
    
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Fine-Tuning on {len(xtrain)} images...")
    print(f"Backbone LR: 1e-4 | Head LR: 1e-3 | Batch Size: {batch_size}")

    # 4. TRAINING LOOP
    for epoch in range(epochs):
        encoder.train()    # UNFREEZE backbone
        classifier.train() # UNFREEZE head
        
        total_loss = 0
        
        for batch_imgs, batch_labels in train_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            features = encoder(batch_imgs)
            logits = classifier(features)
            
            loss = criterion(logits, batch_labels)
            
            # Backward Pass (Updates gradients in both Encoder and Classifier)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 5. EVALUATION LOOP
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            
            features = encoder(batch_imgs)
            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total
    print(f"Final Fine-Tuning Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy