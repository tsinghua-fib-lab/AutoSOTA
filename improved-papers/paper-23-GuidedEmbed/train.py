import torch
from torch.utils.data import DataLoader, random_split
from model import GSTransformModel, ExampleDataset, contrastive_loss
from datetime import datetime
from data import load_embeddings
import json
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

def get_label_mapping(labels):
    # Dynamically generate valid label list
    valid_labels = sorted(set(
        label.strip() 
        for label in labels
        if label.strip()  # Exclude empty labels
    ))
    
    logger.info(f"Discovered categories: {valid_labels}")
    logger.info(f"Number of categories: {len(valid_labels)}")
    
    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(valid_labels)}
    return label_mapping

def load_train_data(json_path):
    """Prepare training data from classification_result.json"""
    # Read classification results
    with open(json_path, 'r', encoding='utf-8') as f:
        classification_result = json.load(f)

    # Extract sample IDs and labels
    selected_indices = list(classification_result.keys())
    train_indices = [int(i) for i in selected_indices]
    selected_labels = list(classification_result.values())

    label_mapping = get_label_mapping(selected_labels)
    
    # Get embedding vectors
    train_embeddings = load_embeddings(indices=train_indices)
    train_labels = [label_mapping[label] for label in selected_labels]
    
    # Output sample count for each category
    logger.info("\nCategory distribution:")
    for label, idx in label_mapping.items():
        count = sum(1 for l in train_labels if l == idx)
        logger.info(f"{label}: {count} records")
    
    return train_embeddings, train_labels, label_mapping, train_indices

def train_with_validation(model, train_dataloader, val_dataloader, 
                         num_epochs=500, learning_rate=0.0001, 
                         lambda1=1.0, lambda2=1.0, margin=10.0, 
                         patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_triplet = 0
        total_train_recon = 0
        
        for batch_x, batch_labels in train_dataloader:
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            z, x_reconstructed = model(batch_x)
            
            loss_contrastive = contrastive_loss(
                z, batch_labels, 
                margin=margin
            )
            loss_reconstruction = torch.nn.MSELoss()(x_reconstructed, batch_x)
            loss = lambda1 * loss_contrastive + lambda2 * loss_reconstruction

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_triplet += loss_contrastive.item()
            total_train_recon += loss_reconstruction.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_triplet = total_train_triplet / len(train_dataloader)
        avg_train_recon = total_train_recon / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_labels in val_dataloader:
                batch_x = batch_x.to(device)
                batch_labels = batch_labels.to(device)
                
                z, x_reconstructed = model(batch_x)
                loss_contrastive = contrastive_loss(z, batch_labels, margin=margin)
                loss_reconstruction = torch.nn.MSELoss()(x_reconstructed, batch_x)
                loss = lambda1 * loss_contrastive + lambda2 * loss_reconstruction
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Print information every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f} "
                  f"(Triplet: {avg_train_triplet:.4f}, Recon: {avg_train_recon:.4f}), "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
    
    return train_losses, val_losses


def main():
    # Prepare training data
    logger.info("Preparing training data...")
    train_embeddings, train_labels, label_mapping, train_indices = \
        load_train_data(json_path='./cache/classifications_xxx.json')
    
    # Create dataset and split into training and validation
    dataset = ExampleDataset(train_embeddings, train_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model
    input_dim = train_embeddings.shape[1]
    hidden_dim = 512
    embedding_dim = 128
    model = GSTransformModel(input_dim, embedding_dim).to(device)
    
    # Train model
    logger.info("Starting model training...")
    train_losses, val_losses = train_with_validation(
        model, train_dataloader, val_dataloader,
        num_epochs=500,
        learning_rate=0.0001,
        lambda1=1.0,
        lambda2=1.0,
        margin=10.0,
        patience=10
    )
    
    # Save model and training information
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'label_mapping': label_mapping
    }, './cache/model.pth')
    logger.info(f"Model saved to: ./cache/model.pth")

if __name__ == "__main__":
    main()
