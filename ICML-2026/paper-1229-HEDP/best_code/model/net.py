import torch
import torch.nn as nn
import copy

from .prompt import PrefixOne,tensor_prompt

from .prompt_learner import load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames


class PromptNet(nn.Module):

    def __init__(self, args):
        super(PromptNet, self).__init__()
        
        self.clip_model = load_clip_to_cpu()

        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.args=args
        self.class_num = 1
        if args["dataset"] == "cddb":
            self.classifier_pool = PromptLearner(args["textual_prompt_length"], list(cddb_classnames.values()), self.clip_model,args["class_token_position"])
                
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.classifier_pool = PromptLearner(args["textual_prompt_length"], list(domainnet_classnames.values()), self.clip_model,args["class_token_position"])
                
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.classifier_pool = PromptLearner(args["textual_prompt_length"], list(core50_classnames.values()), self.clip_model,args["class_token_position"])
            self.class_num = 50
        else:#
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))
        
        self.fix_textual_prompt_weights=[]
        self.e_p_length=args["textual_prompt_length"]
        self.csc=args["CSC"]
        self.emb_d=self.clip_model.ln_final.weight.shape[0]
        self.textual_old_prompt = tensor_prompt(self.e_p_length,self.emb_d)
        self.textual_prompt=copy.deepcopy(self.textual_old_prompt)
        
        self.visual_old_prompt = PrefixOne(args["embd_dim"],args["visual_prompt_length"],args["visual_prompt_layers"]) 
        self.visual_prompt=copy.deepcopy(self.visual_old_prompt)
        
        self.fix_visual_prompt_weights=[]
       
        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
    
    #已经训练好的prompt和ctx得到logits
    def getOldLogits(self, image,taskId):
        if len(self.fix_textual_prompt_weights)<=taskId or len(self.fix_visual_prompt_weights)<=taskId:
            return None
        visual_prompt=self.fix_visual_prompt_weights[taskId]   
        logits = []
        #拼接image与textual_prompt.weight
        image_features = self.image_encoder(image.type(self.dtype), prefix_prompt=visual_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
        ctx=self.fix_textual_prompt_weights[taskId].to(image_features.device)
        prompts = self.classifier_pool
        #选出当前任务的text 矩阵，从而反向优化，使得text矩阵更适应于本任务语境（上下文）
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(ctx), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)#2,512
        logit_scale = self.logit_scale.exp()
        # @矩阵乘法
        logits.append(logit_scale * image_features @ text_features.t())
        return torch.cat(logits, dim=1) 
        

    def forward(self, image):
        logits = []
        #拼接image与textual_prompt.weight
        image_features = self.image_encoder(image.type(self.dtype), prefix_prompt=self.visual_prompt)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #128,512
        if self.csc:
            ctx=torch.stack([i.weight for i in self.textual_prompt], 0).type(self.dtype)
        else:
            ctx=self.textual_prompt.weight.type(self.dtype)
        prompts = self.classifier_pool
        #选出当前任务的text 矩阵，从而反向优化，使得text矩阵更适应于本任务语境（上下文）
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(ctx), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)#2,512
        logit_scale = self.logit_scale.exp()
        # @矩阵乘法
        logits.append(logit_scale * image_features @ text_features.t())
        return torch.cat(logits, dim=1)
            


    def update_fc(self):
        self.numtask +=1

        ctx=self.textual_prompt.weight.type(self.dtype)
        self.textual_prompt=copy.deepcopy(self.textual_old_prompt)
        self.fix_textual_prompt_weights.append(ctx.detach().clone())
        
        old_prompt=copy.deepcopy(self.visual_prompt)
        self.fix_visual_prompt_weights.append(old_prompt)
        self.visual_prompt=copy.deepcopy(self.visual_old_prompt)
        

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
