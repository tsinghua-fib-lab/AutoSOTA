import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

def calculate_prototypes_from_labels(embeddings,
                                     labels,
                                     max_label=None):
  """Calculates prototypes from labels.

  This function calculates prototypes (mean direction) from embedding
  features for each label. This function is also used as the m-step in
  k-means clustering.

  Args:
    embeddings: A 2-D or 4-D float tensor with feature embedding in the
      last dimension (embedding_dim).
    labels: An N-D long label map for each embedding pixel.
    max_label: The maximum value of the label map. Calculated on-the-fly
      if not specified.

  Returns:
    A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
  """
  embeddings = embeddings.view(-1, embeddings.shape[-1])

  if max_label is None:
    max_label = labels.max() + 1
  prototypes = torch.zeros((max_label, embeddings.shape[-1]),
                           dtype=embeddings.dtype,
                           device=embeddings.device)
  labels = labels.view(-1, 1).expand(-1, embeddings.shape[-1])
  prototypes = prototypes.scatter_add_(0, labels, embeddings)
  prototypes = F.normalize(prototypes, dim=1)

  return prototypes


def find_nearest_prototypes(embeddings, prototypes):
  """Finds the nearest prototype for each embedding pixel.

  This function calculates the index of nearest prototype for each
  embedding pixel. This function is also used as the e-step in k-means
  clustering.

  Args:
    embeddings: An N-D float tensor with embedding features in the last
      dimension (embedding_dim).
    prototypes: A 2-D float tensor with shape
      `[num_prototypes, embedding_dim]`.

  Returns:
    A 1-D long tensor with length `[num_pixels]` containing the index
    of the nearest prototype for each pixel.
  """
  embeddings = embeddings.view(-1, prototypes.shape[-1])
  similarities = torch.mm(embeddings, prototypes.t())

  return torch.argmax(similarities, 1)


def kmeans_with_initial_labels(embeddings,
                               initial_labels,
                               max_label=None,
                               iterations=10):
  """Performs the von-Mises Fisher k-means clustering with initial
  labels.

  Args:
    embeddings: A 2-D float tensor with shape
      `[num_pixels, embedding_dim]`.
    initial_labels: A 1-D long tensor with length [num_pixels].
      K-means clustering will start with this cluster labels if
      provided.
    max_label: An integer for the maximum of labels.
    iterations: Number of iterations for the k-means clustering.

  Returns:
    A 1-D long tensor of the cluster label for each pixel.
  """
  if max_label is None:
    max_label = initial_labels.max() + 1

  labels = initial_labels
  for iteration in range(iterations):
    print(f"K-means iteration {iteration+1}/{iterations}")
    # M-step of the vMF k-means clustering.
    prototypes = calculate_prototypes_from_labels(
        embeddings, labels, max_label)
    # E-step of the vMF k-means clustering.
    labels = find_nearest_prototypes(embeddings, prototypes)

  return labels, prototypes

def preprocess_cifar10_for_ppo(
    save_path='./cifar10_100_cluster.pt',
    num_clusters=100,
    subclusters_per_class=1
):
    """
    Preprocess CIFAR-10 data with correct format for RectifiedFlow.
    
    CRITICAL: Images must be stored in [-1, 1] range to match RectifiedFlow output.
    """
    
    print("=== Preprocessing CIFAR-10 with Correct Format ===")
    
    # Transform for RectifiedFlow: [-1, 1] range
    rectified_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # [0, 1]
        torchvision.transforms.Lambda(lambda x: x * 2 - 1)  # Convert to [-1, 1]
    ])
    
    # Transform for DINO: keep [0, 1] for proper ImageNet normalization
    dino_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # [0, 1]
    ])
    
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    rectified_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=True, transform=rectified_transform, download=True
    )
    dino_dataset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=True, transform=dino_transform, download=False
    )
    
    rectified_loader = Data.DataLoader(rectified_dataset, batch_size=128, shuffle=False,num_workers=8, pin_memory=True)
    dino_loader = Data.DataLoader(dino_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    # Load DINO model
    print("Loading DINO ResNet50 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    dino_model = dino_model.to(device).eval()
    
    # Collect data
    all_rectified_images = []  # Will be [-1, 1]
    all_dino_features = []
    all_class_labels = []
    
    # Collect RectifiedFlow format images
    print("Collecting RectifiedFlow format images...")
    for images, labels in tqdm(rectified_loader):
        all_rectified_images.append(images)
        all_class_labels.extend(labels.tolist())

    for images, _ in tqdm(dino_loader):
        images = images.to(device)
        with torch.no_grad():
            # Resize to 224x224
            images_224 = F.interpolate(images, size=(224, 224), 
                                      mode='bilinear', align_corners=False)
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            images_norm = (images_224 - mean) / std
            
            # Extract and normalize features
            features = dino_model(images_norm)
            features = F.normalize(features, dim=1)
            all_dino_features.append(features.cpu())
    
    # Concatenate all data
    all_rectified_images = torch.cat(all_rectified_images, dim=0)
    all_dino_features = torch.cat(all_dino_features, dim=0)

    all_class_labels = torch.tensor(all_class_labels)
    all_class_labels = kmeans_with_initial_labels(
        all_dino_features.cuda(),
        torch.linspace(0, num_clusters - 1, all_dino_features.shape[0]).long().cuda(),
        iterations=50
    ).flatten().cpu()
    print(torch.unique(all_class_labels).numel(), "unique classes found after initial clustering")
    # Verify data format
    print(f"\nData Format Verification:")
    print(f"  Images shape: {all_rectified_images.shape}")
    print(f"  Images range: [{all_rectified_images.min():.4f}, {all_rectified_images.max():.4f}]")
    print(f"  DINO features shape: {all_dino_features.shape}")
    
    # Perform clustering (keeping original logic)
    print("\nPerforming clustering for each class...")
    cluster_labels = all_class_labels.clone()
    cluster_to_class = {}

    # Build class to indices mapping
    class_to_indices = {}
    for class_id in range(num_clusters):
        class_indices = torch.where(all_class_labels == class_id)[0]
        class_to_indices[class_id] = class_indices.tolist()
    
    # Save corrected data
    save_data = {
        'images': all_rectified_images,       # [-1, 1] range, matches RectifiedFlow
        'dino_features': all_dino_features,   # L2-normalized features
        'class_labels': all_class_labels,
        'cluster_labels': cluster_labels,
        'class_to_indices': class_to_indices,
        'cluster_to_class': cluster_to_class,
        'num_classes': num_clusters,
        'subclusters_per_class': subclusters_per_class,
        'total_clusters': num_clusters * subclusters_per_class,
        'data_format': {
            'image_range': '[-1, 1]',
            'description': 'RectifiedFlow compatible format',
            'dino_features': 'L2 normalized, computed from [0,1] images'
        }
    }
    
    torch.save(save_data, save_path)
    print(f"\n Corrected data saved to: {save_path}")
    
    return save_data

def generate_visualization(data):
    """
    Generate clustering visualization results
    
    FIXED: Correctly handle [-1, 1] range images for visualization
    """
    
    # Clone images to avoid modifying original data
    images = data['images'].clone()
    
    # CRITICAL FIX: Images are in [-1, 1] range, NOT ImageNet normalized!
    # Convert from [-1, 1] to [0, 1] for visualization
    images = (images + 1) / 2  # [-1, 1] → [0, 1]
    images = torch.clamp(images, 0, 1)
    
    # Convert to visualization format [N, H, W, C]
    images = images.permute(0, 2, 3, 1)  # [50000, 32, 32, 3]
    images = (images * 255).byte()
    
    cluster_labels = data['cluster_labels']
    total_clusters = data['total_clusters']
    
    H, W = 32, 32
    num_img_per_cluster = 20
    canvas = np.zeros((H * total_clusters, W * num_img_per_cluster, 3), dtype=np.uint8)
    
    for cluster_id in range(total_clusters):
        idx = (cluster_labels == cluster_id).nonzero().flatten()
        if len(idx) == 0:
            continue
            
        if len(idx) < num_img_per_cluster:
            idx = idx[torch.randint(0, len(idx), (min(num_img_per_cluster, len(idx)),))]
        else:
            idx = idx[torch.randperm(len(idx))[:num_img_per_cluster]]
        
        selected_imgs = images[idx]
        
        # Arrange images
        for i, img in enumerate(selected_imgs):
            if i < num_img_per_cluster:
                y_start = cluster_id * H
                y_end = (cluster_id + 1) * H
                x_start = i * W
                x_end = (i + 1) * W
                canvas[y_start:y_end, x_start:x_end, :] = img.numpy()
    
    Image.fromarray(canvas, mode='RGB').save('cifar10_clustering_visualization_50.png')
    print("Visualization results saved to 'cifar10_clustering_visualization.png'")


def load_ppo_data(path='./cifar10_ppo_data.pt'):
    """Load preprocessed PPO data"""
    data = torch.load(path)
    print(f"Data loaded: {len(data['images'])} images, {data['total_clusters']} clusters")
    return data


def get_class_sample(data, target_class, device='cuda'):
    """Randomly sample a DINO feature from specified class"""
    class_indices = data['class_to_indices'][target_class]
    random_idx = np.random.choice(class_indices)
    
    target_feature = data['dino_features'][random_idx].to(device)
    target_image = data['images'][random_idx].to(device)
    
    return target_feature, target_image, random_idx


if __name__ == "__main__":
    print("Starting CIFAR-10 data preprocessing...")
    data = preprocess_cifar10_for_ppo()