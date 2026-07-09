
# Code adapted from https://github.com/YBZh/DMN/ with associated paper 
# Zhang, Y., Zhu, W., Tang, H., Ma, Z., Zhou, K., & Zhang, L. (2024). Dual memory networks: A versatile adaptation approach for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 28718-28728).
# https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dual_Memory_Networks_A_Versatile_Adaptation_Approach_for_Vision-Language_Models_CVPR_2024_paper.html


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
from argparse import Namespace
import torch.nn as nn
from clip import load, tokenize
import torch

class DMNClipWrapper(nn.Module):
    def __init__(self, clip_model, transform, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        learned_cls=False, memory_size=10, text_prompt_type='custom'):
        super(DMNClipWrapper, self).__init__()
        self.clip = clip_model
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.first_flag = True
        self.memory_size = memory_size
        self.return_local_feat = False
        if text_prompt_type != 'custom':
            raise RuntimeError('Only custom prompts are supported.')
        self.text_prompt_type = text_prompt_type

        self.logit_scale = self.clip.logit_scale.data
        self.text_feat = None
        self.few_shot_mem = False


    def reset_classnames(self, dataset):
        self.n_cls = len(dataset.classnames)  ## 200
        self.classnames = [name.replace("_", " ") for name in dataset.classnames]
        self.text_prompt = dataset.template

        self.first_flag = True

    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        text_feat = []
        text_label = []
        count = 0
        for name in self.classnames:
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type =='tip_cupl':
                text_prompts += self.cupl_prompts[name]
            texts = tokenize(text_prompts).cuda()  # tokenize
            class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
            class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()
            text_feat.append(class_embedding_mean) ### 1024
            one_hot_target = torch.zeros(self.n_cls).to(class_embedding_mean.device)
            one_hot_target[count] = 1
            text_label.append(one_hot_target)  ## 1 * d, turn it to one hot labels.
            count = count + 1
        self.text_feat = torch.stack(text_feat, dim=0).cuda() ## N*1024
        self.text_label = torch.stack(text_label, dim=0).cuda()  ## N*N

        self.text_feat_full = self.text_feat ## not used.
        self.fixed_global_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_global_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C

        self.fixed_global_label = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label = self.text_label.clone().unsqueeze(1)
        self.fixed_global_label_vanilla = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label_vanilla = self.text_label.clone().unsqueeze(1)

        if self.first_flag:  ## initlize
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device) 
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag = False

        return self.text_feat, self.text_feat_full

    
    def DMN_encode_image(self, x):
        if len(x.shape) == 3:
            x = x[None,...]
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  ## torch.Size([128, 197, 768])

        x = self.clip.visual.ln_post(x) ## 128*197*768

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x
    
    
    def get_image_features(self, image):
        image_features = self.DMN_encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_local = image_features[:,1:,:]  ## B*L*C
        image_features_global = image_features[:, 0, :] ## B*C
        self.image_features_local = None #image_features_local
        self.image_features_global = image_features_global

        return self.image_features_global, self.image_features_local

    def forward(self, input):
        pass

def select_confident_samples(prob, top):
    # ipdb.set_trace()
    # print('prob in select_confident_sample: ',prob)
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    # print(batch_entropy)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] ## pick the min entropy
    # print(idx)
    idx_confused = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * top):] ## pick the max entropy
    return prob[idx], idx, prob[idx_confused], idx_confused

## the main component.
class DMNDualMem(nn.Module):
    def __init__(self, args=None, beta=5.5, feat_dim=1024, class_num=1000, mapping='bias'):
        super(DMNDualMem, self).__init__()
        self.args =  args
        self.indice = args.indice  ## indice of important channels.
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        if args.shared_param:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_bias_key = self.global_bias
            self.global_bias_value = self.global_bias

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.text_affine = self.global_ffn_affine
            self.text_bias = self.global_ffn_bias
        else:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_bias_key = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_value = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.text_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = args.mapping ### bias | affine | all


    def update_memory_bank(self, model):
        # updating 
        mean_prob = self.init_pred[0]  
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print('pseudo label in update_memory_bank: ', pseudo_label)
        text_features = model.text_feat[pseudo_label]  ## 512
        selected_image_features_global = model.image_features_global[:1]
        current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
        if model.image_feature_count[pseudo_label] == model.memory_size:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                pass  ## the entropy of current test image is very large.
            else:
                _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                to_replace_indice = indice[-1]  ## with max entropy, ascending.
                model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
        else:
            model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global
            model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
            model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy
            model.image_feature_count[pseudo_label] += 1


    
    def get_image_pred(self, model, return_full=False, return_logit=False):
        ## prediction with dynamic memory.
        img_feat = model.image_features_global[:1]  # 1*1024
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | entropy_weighted | similarity_weighted
        ### similarity_weighted achieves the best results.
        memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla), dim=1)  ## 200*11*1024

        if image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized features according to similarity.
            img_feat_mappling = img_feat
            memorized_image_feat_K = memorized_image_feat
            memorized_image_feat_V = memorized_image_feat
            with torch.no_grad():
                if self.args.position == 'query':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                elif self.args.position == 'key':
                    memorized_image_feat_K = memorized_image_feat  + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
                elif self.args.position == 'value':
                    memorized_image_feat_V = memorized_image_feat  + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
                elif self.args.position == 'qkv' or self.args.position == 'all':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
                else:
                    pass
                memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
                ## some memorized_image_feat slots are empty before mapping, reseting them to empty.
                memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
                memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
                memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
                img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

            similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(-1) ## 200*11  idealy [-1,1], practically [0.1, 0.2]  
            similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            ### weighting memoried features with similarity weights. 
            adaptive_image_feat = (memorized_image_feat_V * similarity_matrix.unsqueeze(-1)).sum(1)
            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            if self.args.position == 'output' or self.args.position == 'all':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024

            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()

            logits = logit_scale * adaptive_image_feat.squeeze() @ img_feat.T  ## used feat is not update.
            logits = logits.squeeze()
            return logits.softmax(dim=0).squeeze()
        else:
            raise NotImplementedError

    def get_image_pred_fewshot_global(self, model, return_full=False, return_logit=False):
        ## prediction with static memory.
        if return_full:
            img_feat = model.image_features_global  # 1*1024
        else:
            img_feat = model.image_features_global[:1, :]  # 1*1024
        num_class = model.image_feature_memory.shape[0]
        memorized_image_feat = model.fixed_global_feat  ## 200*11*1024, few shot samples and text features.
        img_feat_mappling = img_feat
        memorized_image_feat_K = memorized_image_feat
        memorized_image_feat_V = memorized_image_feat

        if self.args.position == 'query':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
        elif self.args.position == 'key':
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'value':
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'qkv' or self.args.position == 'all':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024

        memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
        memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
        img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)
        ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
        ##  200*11*200；
        similarity_matrix = memorized_image_feat_K @ img_feat_mappling.T ## class*shot*Batch
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
        adaptive_image_feat = memorized_image_feat_V.transpose(1,2) @ similarity_matrix ## class * D * batch, 102*1024*204
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        adaptive_image_feat = adaptive_image_feat.transpose(0,2).transpose(1,2) ## 204*102*1024
        if self.args.position == 'output' or self.args.position == 'all':
            adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024

        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
        logits = logit_scale * adaptive_image_feat[..., self.args.indice] @ img_feat[..., self.args.indice].unsqueeze(-1) ## memoried features are not updated.
        if return_logit:
            return logits[:,:,0]
        else:
            return logits[:,:,0].softmax(dim=1)

    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        if self.args.position == 'output' or self.args.position == 'all':
            text_feat = model.text_feat + self.text_bias
        else:
            text_feat = model.text_feat
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        img_text_logit = logit_scale * model.image_features_global @ text_feat.t() ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)
        

def get_cfg_DMN():
    dmn_args = Namespace()
    dmn_args.indice = 0
    dmn_args.shared_param = None
    dmn_args.mapping = 'bias'
    dmn_args.position = 'all'
    dmn_args.n_shot = 0 #zero shot
    dmn_args.n_augments = 32
    dmn_args.memory_size = 50
    dmn_args.selection_p = 0.1
    beta = 10

    return dmn_args, beta
