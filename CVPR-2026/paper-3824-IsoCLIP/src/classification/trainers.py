import pickle
import sys
import os
from pathlib import Path
from functools import partial
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).absolute().parents[3].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager

from utils import load_clip
from encode_no_projection import get_encode_image_with_noproj, get_encode_text_with_noproj, get_projection_layers
from retrieval import apply_iso  


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


# custom implementation
@TRAINER_REGISTRY.register()
class ClipZeroshot(TrainerX):
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip_model.tokenizer(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        _, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        

# custom implementation
@TRAINER_REGISTRY.register()
class ClipNCM(TrainerX):
    def build_model(self):
        cfg = self.cfg
        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)
        clip_model.to(self.device)

        self.clip_model = clip_model
        self.image_protos = self.compute_image_protos()
  
        with torch.no_grad():
            features, labels = [], []
            test_loader_iter = iter(self.test_loader)

            for batch_idx in tqdm(range(len(self.test_loader)), desc="Computing features for test images"):
                try:
                    batch_x = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(self.test_loader)
                    batch_x = next(test_loader_iter)

                features.append(clip_model.encode_image(batch_x["img"].to(self.device)))
                labels.append(batch_x["label"].to(self.device))

            test_features = torch.cat(features, dim=0)
            test_labels = torch.cat(labels, dim=0)
        
        self.test_data = [test_features, test_labels]
 

    def model_inference(self, image_feats):
 
        gallery_features = self.image_protos    
        query_features = image_feats  
 
        logit_scale = self.clip_model.logit_scale.exp()
        # Compute the similarity matrices
        gallery_features = F.normalize(gallery_features)
        query_features = F.normalize(query_features)

        logits = logit_scale * query_features @ gallery_features.t()
        return logits


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        clip_model, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        print(f"Evaluate on the *{split}* set")
 
        batch = self.test_data[0]
        label = self.test_data[1]
 
        output = self.model_inference(batch)
        self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        # first result is the accuracy
        return list(results.values())[0]
                    

    def compute_image_protos(self):
        "Compute and store image prototypes for each class"
        with torch.no_grad():
            class_sums = None 
            class_counts = torch.zeros(self.num_classes, device=self.device)

            train_loader_x_iter = iter(self.train_loader_x)

            for batch_idx in tqdm(range(len(self.train_loader_x)), desc="Computing Class-wise image Prototypes"):
                try:
                    batch_x = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(self.train_loader_x)
                    batch_x = next(train_loader_x_iter)

                # Extract features (assuming model returns raw feature embeddings)
                features = self.clip_model.encode_image(batch_x["img"].to(self.device))  # Shape: (batch_size, feature_dim)
                labels = batch_x["label"].to(self.device)
                
                if class_sums is None:
                    class_sums = torch.zeros(self.num_classes, features.shape[1], device=self.device)  

                # Accumulate sum of features for each class
                for i in range(labels.shape[0]):
                    class_sums[labels[i]] += features[i]
                    class_counts[labels[i]] += 1

            # Compute class prototypes (mean feature per class)
            self.image_protos = class_sums / class_counts.unsqueeze(1)
            print("Class-wise image prototypes:", self.image_protos.shape)

        return self.image_protos

 
 

# custom implementation
@TRAINER_REGISTRY.register()
class IsoNCM(TrainerX):
    def build_model(self):
        cfg = self.cfg
        clip_model, _, _ = load_clip(cfg.MODEL.BACKBONE.NAME,
                                     cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                     cfg.MODEL.BACKBONE.USE_OPEN_CLIP, self.device)
        clip_model.to(self.device)

        encode_image_noproj = get_encode_image_with_noproj(clip_model)  # encode image returns features before the projection
        clip_model.encode_image = partial(encode_image_noproj, clip_model)

        self.clip_model = clip_model
        self.image_protos = self.compute_image_protos()
        print("Running IsoNCM with K-Top: {}, K-Bottom: {}".format(cfg.iso_ktop, cfg.iso_kbottom))
        with torch.no_grad():
            features, labels = [], []
            test_loader_iter = iter(self.test_loader)

            for batch_idx in tqdm(range(len(self.test_loader)), desc="Computing features for test images"):
                try:
                    batch_x = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(self.test_loader)
                    batch_x = next(test_loader_iter)
                    
                features.append(clip_model.encode_image(batch_x["img"].to(self.device)))
                labels.append(batch_x["label"].to(self.device))

            test_features = torch.cat(features, dim=0)
            test_labels = torch.cat(labels, dim=0)
        
        self.test_data = [test_features, test_labels]
 

    def model_inference(self, image_feats):
 
        gallery_features = self.image_protos    
        query_features = image_feats  
        
        W_image, W_text = get_projection_layers(self.clip_model, self.cfg.MODEL.BACKBONE.NAME)
        W_image = W_image.T
        W_text = W_text.T
        
        _, W_image_iso = apply_iso(W_text, W_image, self.cfg.iso_ktop, self.cfg.iso_kbottom)
        
        query_features = query_features @ W_image_iso
        gallery_features = gallery_features @ W_image_iso 

        
        logit_scale = self.clip_model.logit_scale.exp()
        # Compute the similarity matrices
        gallery_features = F.normalize(gallery_features)
        query_features = F.normalize(query_features)

        logits = logit_scale * query_features @ gallery_features.t()
        
        return logits


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        clip_model, _, clip_preprocess = load_clip(self.cfg.MODEL.BACKBONE.NAME,
                                          self.cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED,
                                          self.cfg.MODEL.BACKBONE.USE_OPEN_CLIP, 'cpu')
        dm = DataManager(self.cfg, custom_tfm_train=clip_preprocess, custom_tfm_test=clip_preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        print(f"Evaluate on the *{split}* set")
 
        batch = self.test_data[0]
        label = self.test_data[1]
 
        output = self.model_inference(batch)
        self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        # first result is the accuracy
        return list(results.values())[0]
                    

    def compute_image_protos(self):
        "Compute and store image prototypes for each class"
        with torch.no_grad():
            class_sums = None 
            class_counts = torch.zeros(self.num_classes, device=self.device)

            train_loader_x_iter = iter(self.train_loader_x)

            for batch_idx in tqdm(range(len(self.train_loader_x)), desc="Computing Class-wise image Prototypes"):
                try:
                    batch_x = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(self.train_loader_x)
                    batch_x = next(train_loader_x_iter)

                # Extract features (assuming model returns raw feature embeddings)
                features = self.clip_model.encode_image(batch_x["img"].to(self.device))  # Shape: (batch_size, feature_dim)
                labels = batch_x["label"].to(self.device)
                
                if class_sums is None:
                    class_sums = torch.zeros(self.num_classes, features.shape[1], device=self.device)  

                # Accumulate sum of features for each class
                for i in range(labels.shape[0]):
                    class_sums[labels[i]] += features[i]
                    class_counts[labels[i]] += 1

            # Compute class prototypes (mean feature per class)
            self.image_protos = class_sums / class_counts.unsqueeze(1)
            print("Class-wise image prototypes:", self.image_protos.shape)

        return self.image_protos

 
 