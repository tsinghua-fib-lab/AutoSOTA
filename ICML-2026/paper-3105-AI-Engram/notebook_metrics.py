# Extracted from fig_resnet18_cifar10.ipynb cell 4
# ===== engram/metric/__init__.py  (verbatim from the engram package) =====
import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.inference_mode()
def compute_classwise_accuracy(model, dataloader, num_classes, device=None, batch_fn=None):
    device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval().to(device)
        
    class_correct = torch.zeros(num_classes, dtype=torch.float, device=device)
    class_total = torch.zeros(num_classes, dtype=torch.float, device=device)
    if batch_fn is None:
        batch_fn = lambda batch: (batch[0].to(device), batch[1].to(device))
       
    for batch in tqdm(dataloader):
        inputs, labels = batch_fn(batch)
       
        # Forward pass: handle dict or tensor inputs, extract logits
        if device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                else:
                    outputs = model(inputs)
        else:
            if isinstance(inputs, dict):
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
            else:
                outputs = model(inputs)
       
        _, predicted = torch.max(outputs, 1)
       
        correct = (predicted == labels).float()
        class_correct.scatter_add_(0, labels, correct)
        class_total.scatter_add_(0, labels, torch.ones_like(correct))

    classwise_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            classwise_accuracy[i] = float((class_correct[i] / class_total[i]).item())
   
    return classwise_accuracy

def compute_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine similarity matrix for a given tensor.
    
    Args:
        x: Input tensor of shape [num_classes, feature_dim] (e.g., [10, 500])
        
    Returns:
        A symmetric matrix of shape [num_classes, num_classes] 
        where entry (i, j) is the cosine similarity between row i and row j.
    """
    x_normalized = F.normalize(x, p=2, dim=1, eps=1e-8)
    sim_matrix = torch.mm(x_normalized, x_normalized.t())
    
    return sim_matrix