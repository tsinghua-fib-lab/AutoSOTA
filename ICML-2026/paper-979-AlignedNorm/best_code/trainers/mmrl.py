import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from tqdm import tqdm
import copy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def load_clip_to_cpu(cfg, model_name="CLIP"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"model": model_name,
                      "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
                      "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS}
    model = clip.build_model_MMRL(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):

        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        eot_index = tokenized_prompts.argmax(dim=-1)
        combined = [x, compound_rep_tokens_text, 0, eot_index]  # third argument is the counter which denotes depth of representation tokens
        outputs, _ = self.transformer(combined)

        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection
 
        
        return x


class TextEncoder_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer(x)

        x = outputs
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  
        return x


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device

    text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME
    template = CUSTOM_TEMPLATES[dataset]

    with torch.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  #(n_tokens)
            tokens = tokens.to(device)
            tokenized_prompts.append(tokens) 
        tokenized_prompts = torch.cat(tokenized_prompts) # (n_classes, n_tokens)  

        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype) # (n_classes, n_tokens, embed_dim)
        outputs = text_encoder(embeddings.cuda(), tokenized_prompts.cuda()) 

        text_embeddings = outputs

    text_encoder = text_encoder.to(device)
    return text_embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_rep_tokens = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRL.REP_DIM

        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)  # max=12
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        dataset = cfg.DATASET.NAME

        template = CUSTOM_TEMPLATES[dataset]
        
        tokenized_prompts = []
        for text in classnames:
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  # (n_tokens)
            tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.cat(tokenized_prompts)  # (n_classes, n_tokens)

        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype) # (n_classes, n_tokens, embed_dim)

        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)

        single_layer_r2v = nn.Linear(rep_dim, visual_dim)
        single_layer_r2t = nn.Linear(rep_dim, text_dim)

        self.compound_rep_tokens_r2vproj = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)
       

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
 
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens)                        
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))      

        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRL.ALPHA
        self.classnames = classnames
        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(clip_model.dtype)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None


    def forward(self, image):
        
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
        else:
            if self.text_features_for_inference is None:
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference = self.representation_learner()
                self.text_features_for_inference = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, self.compound_rep_tokens_text_for_inference)

            compound_rep_tokens_text, compound_rep_tokens_visual = self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference
            text_features = self.text_features_for_inference

        image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])
    
        alpha = self.alpha
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100. * image_features @ text_features.t()
        logits_rep = 100. * image_features_rep @ text_features.t()
        logits_fusion = alpha * logits + (1 - alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features


class MMRL_Loss(_Loss):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super(MMRL_Loss, self).__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha 

    def forward(self, logits, logits_rep,
                image_features, text_features, 
                image_features_clip, text_features_clip, 
                label):
    
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)

        cossim_reg_img = 1 - torch.mean(F.cosine_similarity(image_features, image_features_clip, dim=1))
        cossim_reg_text = 1 - torch.mean(F.cosine_similarity(text_features, text_features_clip, dim=1))

        return self.alpha * xe_loss1 + (1-self.alpha) * xe_loss2 +  + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text



@TRAINER_REGISTRY.register()
class MMRL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMRL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMRL.PREC == "fp32" or cfg.TRAINER.MMRL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype

        with torch.no_grad():
            self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip)
            self.text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
        self.image_encoder_clip = clip_model_zero_shot.visual  

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)


        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["representation_learner", "image_encoder.proj_rep"]

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
            param.requires_grad_(update)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        

        self.image_encoder_clip.to(self.device)    
    
        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        alpha = cfg.TRAINER.MMRL.ALPHA
        self.criterion = MMRL_Loss(reg_weight=reg_weight, alpha=alpha)

        # NOTE: only give representation_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalRepresentationLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMRL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.image_encoder_clip = nn.DataParallel(self.image_encoder_clip)
        

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MMRL.PREC
        if prec == "amp":
            with autocast():
                with torch.no_grad():
                    image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                    image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)
                          
                logits, logits_rep, logits_fusion, image_features, text_features = model(image)
                text_features = text_features[0:self.num_classes] #Crop the returned text_features for multi-GPU compatibility

                loss = self.criterion(logits, logits_rep, 
                                      image_features, text_features, 
                                      image_features_clip, self.text_features_clip, 
                                      label)
            
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            with torch.no_grad():
                image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

            logits, logits_rep, logits_fusion, image_features, text_features = model(image)
            text_features = text_features[0:self.num_classes] #Crop the returned text_features for multi-GPU compatibility
            
            loss = self.criterion(logits, logits_rep, 
                                    image_features, text_features, 
                                    image_features_clip, self.text_features_clip, 
                                    label)

            optim.zero_grad()
            loss.backward()
            optim.step()


        output = logits_fusion
        loss_summary = {"loss": loss.item(),
                        'acc': compute_accuracy(output, label)[0].item()
                        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits, _, logits_fusion, _, _, = self.model(input)

            if task == "B2N":
                output = logits_fusion if sub_cls == "base" else logits_fusion
            elif task == "FS":
                output = logits_fusion
            elif task == "CD":
                output = logits_fusion if dataset == "ImageNet" else logits
            else:
                raise ValueError("The TASK must be either B2N, CD, or FS.")

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        # model_file = 'model-best.pth.tar'

        # if epoch is not None:
        #     model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            #model_path = osp.join(directory, name, model_file)
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path_prefix)
                )
            for file in os.listdir(model_path_prefix):
                if "model-best.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
 
            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )            


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)