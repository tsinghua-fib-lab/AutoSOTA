import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from logdiff.score.sampling_compositional import And, Or_CI, Or_ME, Not, Expression
from logdiff.score.sampling_compositional_constant import NotConstant, AndConstant

import lpips
from hydra.utils import instantiate
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

class ConformityScorer:
    def __init__(self, classifier, feature_list, atom_class_to_name, preprocess=None):
        """
        Initializes the scorer.
        Args:
            classifier: The torch model used for prediction.
            feature_list: List of feature names corresponding to classifier outputs.
        """
        self.classifier = classifier
        self.feature_list = feature_list
        self.atom_class_to_name = atom_class_to_name
        self.preprocess = preprocess

    def evaluate(self, images, expr: Expression):
        """
        Args:
            images: Tensor of generated images.
            expr: The 'comp_sampling' expression object used for generation.
        Returns:
            accuracy (float): Percentage of images satisfying the logic.
            mask (bool tensor): Boolean mask of which images passed.
        """
        # Get predictions for ALL features, stored in a dictionary
        predictions_dict = self.get_predictions(images)
        
        # Recursively check logic against the predictions dictionary
        result_mask = self.__recursive_logic_check(expr, predictions_dict)
        
        accuracy = result_mask.float().mean().item()
        return accuracy, result_mask, predictions_dict

    def get_predictions(self, images):
        """
        Runs the classifier and returns a dictionary mapping feature names 
        (e.g., 'ObjectColor') to their predicted values (argmax tensor).
        """
        predictions = {}
        with torch.no_grad():
            if self.preprocess is not None:
                images = self.preprocess(images)
            outputs = self.classifier(images)
            
            if not isinstance(outputs, (tuple, list)):
                raise TypeError("Classifier output must be a list of logit tensor.")
            
            for feature_name, feature_logits in zip(self.feature_list, outputs):
                predictions[feature_name] = torch.argmax(feature_logits, dim=1).cpu()
            
            return predictions

    def __recursive_logic_check(self, node, predictions_dict):
        """
        Visits the logic nodes and applies Boolean logic by looking up the 
        correct prediction in the predictions_dict based on the node's type.
        """
        
        if type(node) in self.atom_class_to_name:
            feature_name = self.atom_class_to_name[type(node)]
            target_value = node.value
            
            predicted_values = predictions_dict.get(feature_name)
            
            if predicted_values is None:
                 raise KeyError(f"Feature '{feature_name}' not found in predictions dictionary. Check feature_ranges.")

            return predicted_values == target_value
            
        # --- Check for Compositional Logic Nodes ---
        if isinstance(node, (And, AndConstant)):
            return self.__recursive_logic_check(node.left, predictions_dict) & \
                   self.__recursive_logic_check(node.right, predictions_dict)

        if isinstance(node, (Or_CI, Or_ME)):
            return self.__recursive_logic_check(node.left, predictions_dict) | \
                   self.__recursive_logic_check(node.right, predictions_dict)

        if isinstance(node, (Not, NotConstant)):
            return ~self.__recursive_logic_check(node.expression, predictions_dict)

        raise ValueError(f"Unknown Expression Type: {type(node)}")
    

class LPIPSEvaluator:
    def __init__(self, device, cfg):
        # Initialize LPIPS model (AlexNet is standard for metrics)
        self.loss_fn = lpips.LPIPS(net='alex').eval().to(device)
        self.device = device
        self.ref_index = self.build_reference_index(cfg, device, max_samples_per_class=50)

    def preprocess(self, x):
        """
        Preprocesses images for LPIPS:
        1. Expand Grayscale -> RGB
        2. Resize small images -> 64x64 (Prevent pooling crash)
        3. Scale [0, 1] -> [-1, 1]
        """
        # 1. Handle Grayscale (B, 1, H, W) -> (B, 3, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 2. Resize if too small (Critical Fix for MNIST/CIFAR)
        # AlexNet requires at least ~64x64 to avoid vanishing feature maps
        if x.shape[2] < 64 or x.shape[3] < 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        # 3. Scale from [0, 1] to [-1, 1]
        x = x.clamp(0, 1) * 2.0 - 1.0
        return x

    @torch.no_grad()
    def compute(self, gen_imgs, pred_labels):
        """
        Computes LPIPS distance between generated and reference images.
        """
        ref_imgs, valid_mask = self.get_random_references(pred_labels, self.ref_index, self.device)
        gen = self.preprocess(gen_imgs[valid_mask].to(self.device))
        ref = self.preprocess(ref_imgs[valid_mask].to(self.device))
        
        # Returns (B, 1, 1, 1) -> squeeze to (B,)
        return self.loss_fn(gen, ref).flatten()

    # --- 2. Reference Data Management ---
    @staticmethod
    def build_reference_index(cfg, device, max_samples_per_class=50):
        """
        Builds a dictionary {label_string: Tensor_of_images} using BOTH Train and Val datasets.
        """
        datasets = []
        
        # Always load train
        if hasattr(cfg.dataset, "train_dataset"):
            datasets.append(instantiate(cfg.dataset.train_dataset))
            
        # Conditionally load val if it exists in config
        if hasattr(cfg.dataset, "val_dataset"):
            datasets.append(instantiate(cfg.dataset.val_dataset))
            
        if not datasets:
            raise ValueError("No datasets found in config (checked 'train_dataset' and 'val_dataset')")

        # 2. Combine them into one seamless source
        combined_dataset = ConcatDataset(datasets)
        
        loader = torch.utils.data.DataLoader(
            combined_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4
        )
        
        ref_index = defaultdict(list)
        
        max_batches = 1000 
        batches_processed = 0
        
        for batch in loader:
            imgs = batch["X"]
            labels = batch["label"] 
            
            for i in range(len(imgs)):
                current_labels = labels[i]
                
                # --- Robust Label Parsing ---
                active_attributes = []
                
                if isinstance(current_labels, torch.Tensor):
                    active_attributes = current_labels.tolist()
                else:
                    active_attributes = current_labels
                
                # --- Indexing ---
                label_id = "-".join([f"{l}" for l in active_attributes])
                
                if len(ref_index[label_id]) < max_samples_per_class:
                    ref_index[label_id].append(imgs[i].to(device))
            
            batches_processed += 1
            if batches_processed >= max_batches:
                break
                
        final_index = {}
        for attr, img_list in ref_index.items():
            if img_list:
                final_index[attr] = torch.stack(img_list)
                
        print(f"Reference index built. Covered {len(final_index)} unique attribute combinations.")
        return final_index

    @staticmethod
    def get_random_references(pred_attrs, ref_index, device):
        """
        Retrieves random real images corresponding to the generated labels.
        """
        ref_imgs = []
        valid_mask = []

        attr_names = list(pred_attrs.keys())
        # Get batch size from the first tensor found
        batch_size = len(pred_attrs[attr_names[0]])

        composite_keys = []

        batch_size = len(pred_attrs[attr_names[0]])
    
        for i in range(batch_size):
            # Extract the value for this sample (i) from every attribute tensor
            # We cast to .item() to get a standard Python int, then to str
            vals = [str(pred_attrs[name][i].item()) for name in attr_names]
            
            # Join them with a hyphen
            key = "-".join(vals)
            composite_keys.append(key)
        
        for i, key in enumerate(composite_keys):
            if key in ref_index.keys():
                # Pick one random real image of this class
                idx = torch.randint(len(ref_index[key]), (1,)).item()
                ref_imgs.append(ref_index[key][idx])
                valid_mask.append(True)
            else:
                # Fallback if class not in reference (e.g. rare class)
                ref_imgs.append(torch.zeros_like(ref_index[list(ref_index.keys())[0]][0]))
                valid_mask.append(False)
        return torch.stack(ref_imgs).to(device), torch.tensor(valid_mask, device=device)
    

def shannon_entropy_from_counts(counts: torch.Tensor):
    probs = counts / counts.sum()

    probs = probs[probs > 0]
    return -(probs * torch.log2(probs)).sum().item()

def classifier_confidence(images, classifier):
    with torch.no_grad():
        logits_list = classifier(images) 
        
        head_confidences = []
        for logits in logits_list:
            probs = logits.softmax(dim=1)
            head_confidences.append(probs.max(dim=1).values)
            
        all_confidences = torch.stack(head_confidences)
    return all_confidences.mean(dim=0) 
