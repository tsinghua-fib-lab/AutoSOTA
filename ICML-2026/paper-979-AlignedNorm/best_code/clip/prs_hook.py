import numpy as np
import torch
from pathlib import Path
import os, json
import einops
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns


def _list_init(length: int):
    return [[] for _ in range(length)]

class PRSLogger(object):
    def __init__(self, model, device, layer_info, cfg, textual_len=None):
        self.device = device
        self.model_name = cfg.TRAINER.NAME
        self.dataset_name = cfg.DATASET.NAME
        self.seed = cfg.SEED
        self.logs_root = os.path.join(cfg.OUTPUT_DIR, "prs_logs")
        self.npz_root = os.path.join(cfg.OUTPUT_DIR, "prs_npz")
        os.makedirs(self.logs_root, exist_ok=True)
        self.visual_layers, self.textual_layers = layer_info
        self.layers_length = len(cfg.TRAINER.ALIGNEDNORM.REP_LAYERS) + 1
        self.first_layer = cfg.TRAINER.ALIGNEDNORM.REP_LAYERS[0]
        self.repr_token_len = cfg.TRAINER.ALIGNEDNORM.N_REP_TOKENS
        self.eot_pos = textual_len + self.repr_token_len
        self.visual_cls_token_after_attn = _list_init(self.visual_layers)
        self.visual_prompt_token_after_attn = _list_init(self.visual_layers)
        self.visual_content_token_after_attn = _list_init(self.visual_layers)
        self.visual_cls_token_post = _list_init(self.visual_layers)
        self.visual_prompt_token_post = _list_init(self.visual_layers)
        self.visual_content_token_post = _list_init(self.visual_layers)
        self.textual_eot_token_after_attn = _list_init(self.textual_layers)
        self.textual_content_token_after_attn = _list_init(self.textual_layers)
        self.textual_eot_token_post = _list_init(self.textual_layers)
        self.textual_content_token_post = _list_init(self.textual_layers)
        self.visual_attention = _list_init(self.visual_layers)
        
        self.visual_proj_post = []
        self.visual_proj_pre = []
        self.visual_proj_rep_post = []
        self.visual_proj_rep_pre = []
        self.textual_post_ln = []
        self.textual_post_proj = []
        
        self.sub_cls = cfg.DATASET.SUBSAMPLE_CLASSES
        self.cur_num = 0
        self.len_limit = 1e9 if self.sub_cls == "base" else 4096
        self.img = []
        self.img0 = []
        self.img_path = []
        self.name_to_idx = None
        self.model = model

    @torch.no_grad()
    def compute_visual_attentions_matrix(self, ret, layer):
        assert len(ret.shape) == 3, "Verify that you catch the attention weights correctly" # [b, n, n]
        ret_tmp = ret.detach()
        self.visual_attention[layer - 1].append(ret_tmp.cpu().numpy())  # [b, n]
        return ret

    @torch.no_grad()
    def compute_visual_sequence_after_attn(self, ret, layer):
        assert len(ret.shape) == 3, "Verify that you catch the attention weights correctly" # [b, n, d]
        ret_tmp = ret.detach().clone().permute(1, 0, 2)
        if self.cur_num <= self.len_limit:
            if layer >= self.first_layer:
                self.visual_cls_token_after_attn[layer - 1].append(ret_tmp[:, 0, :].norm(dim=-1).cpu().numpy())  # [b]
                self.visual_prompt_token_after_attn[layer - 1].append(ret_tmp[:, 1:1+self.repr_token_len, :].norm(dim=-1).cpu().numpy())  # [b, repr_n]
                self.visual_content_token_after_attn[layer - 1].append(ret_tmp[:, 1+self.repr_token_len:, :].norm(dim=-1).cpu().numpy())  # [b, n - repr_n]
            else:
                b, _, _ = ret_tmp.shape
                self.visual_cls_token_after_attn[layer - 1].append(ret_tmp[:, 0, :].norm(dim=-1).cpu().numpy())  # [b]
                self.visual_prompt_token_after_attn[layer - 1].append(np.zeros((b, self.repr_token_len), dtype=ret_tmp.cpu().numpy().dtype))  # [b, repr_n]
                self.visual_content_token_after_attn[layer - 1].append(ret_tmp[:, 1:, :].norm(dim=-1).cpu().numpy())  # [b, n - 1]
        return ret
    
    @torch.no_grad()
    def compute_visual_sequence_post(self, ret, layer):
        assert len(ret.shape) == 3, "Verify that you catch the attention weights correctly" # [b, n, d]
        ret_tmp = ret.detach().clone().permute(1, 0, 2)
        if self.cur_num <= self.len_limit:
            if layer >= self.first_layer:
                self.visual_cls_token_post[layer - 1].append(ret_tmp[:, 0, :].norm(dim=-1).cpu().numpy())  # [b]
                self.visual_prompt_token_post[layer - 1].append(ret_tmp[:, 1:1+self.repr_token_len, :].norm(dim=-1).cpu().numpy())  # [b, repr_n]
                self.visual_content_token_post[layer - 1].append(ret_tmp[:, 1+self.repr_token_len:, :].norm(dim=-1).cpu().numpy())  # [b, n - repr_n]
            else:
                b, _, _ = ret_tmp.shape
                self.visual_cls_token_post[layer - 1].append(ret_tmp[:, 0, :].norm(dim=-1).cpu().numpy())  # [b]
                self.visual_prompt_token_post[layer - 1].append(np.zeros((b, self.repr_token_len), dtype=ret_tmp.cpu().numpy().dtype))  # [b, repr_n]
                self.visual_content_token_post[layer - 1].append(ret_tmp[:, 1:, :].norm(dim=-1).cpu().numpy())  # [b, n - 1]
        return ret

    @torch.no_grad()
    def compute_textual_sequence_after_attn(self, ret, layer):
        assert len(ret.shape) == 3, "Verify that you catch the attention weights correctly" # [n, l, d]
        ret_tmp = ret.detach().clone().permute(1, 0, 2)
        if self.cur_num <= self.len_limit:
            self.textual_eot_token_after_attn[layer - 1].append(ret_tmp[torch.arange(ret_tmp.shape[0]), self.eot_pos].norm(dim=-1).cpu().numpy())  # [n]
            self.textual_content_token_after_attn[layer - 1].append(ret_tmp[:, :, :].norm(dim=-1).cpu().numpy())  # [n, l]
        return ret
    
    @torch.no_grad()
    def compute_textual_sequence_post(self, ret, layer):
        assert len(ret.shape) == 3, "Verify that you catch the attention weights correctly" # [n, l, d]
        ret_tmp = ret.detach().clone().permute(1, 0, 2)
        if self.cur_num <= self.len_limit:
            self.textual_eot_token_post[layer - 1].append(ret_tmp[torch.arange(ret_tmp.shape[0]), self.eot_pos].norm(dim=-1).cpu().numpy())  # [n]
            self.textual_content_token_post[layer - 1].append(ret_tmp[:, :, :].norm(dim=-1).cpu().numpy())  # [n, l]
        return ret
    
    def finalize(self, epoch):
        pass
                    
    def show_attn(self, data, load_path, idx=None, n=18, is_prompted=False):
        if not is_prompted: return
        total_tokens = data.shape[0]
        fixed_n = 6
        fixed_indices = list(range(1, min(fixed_n, total_tokens)))
        
        if total_tokens > fixed_n:
            n_to_select = n - len(fixed_indices)
            remaining_indices = np.arange(fixed_n, total_tokens)
            weights = data[:, remaining_indices].sum(axis=0)
            sorted_args = np.argsort(weights)
            k = min(n_to_select, len(remaining_indices))
            top_k_indices = remaining_indices[sorted_args][-k:]
            selected_indices = np.sort(np.concatenate([fixed_indices, top_k_indices]))
        else:
            selected_indices = np.arange(total_tokens)

        # 4. 提取子矩阵
        sub_matrix = data[np.ix_(selected_indices, selected_indices)]
        eps = 1e-8
        processed_data = np.log(sub_matrix + eps)
        
        npy_filename = f"attn_data.npy"
        npy_path = os.path.join(load_path, npy_filename)
        np.save(npy_path, processed_data)
        
        processed_data = processed_data - np.median(processed_data) # 零点对齐
        
        # 5. 可视化与保存
        sns.set_theme(style="white")
        plt.figure(figsize=(8, 8))
        
        ax = sns.heatmap(
            processed_data,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar=True,
        )
        ax.axis('off')
        
        # 自动处理路径
        save_path = os.path.join(load_path, f"attn.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
                    
    def bar_mark(self, ax, data, color, single=False):
        if single:
            ax.axhline(data, color=color, ls='--', lw=1)
        else:
            for iter in range(data.shape[0]):
                ax.axhline(data[iter], color=color, ls='--', lw=1)
                
    def set_bar_color(self, ax, pos):
        ax.patches[pos].set_facecolor('red')
        # ax.patches[pos].set_edgecolor('red')
    
    def show_bar_labels(self, ax, data, label, title):
        ax.set_title(title)
        ax.bar(np.arange(data.shape[0]), data, tick_label=label)
        
    def show_bar(self, ax, data, title):
        ax.set_title(title)
        ax.bar(np.arange(data.shape[0]), data)
    
    def show_img(self, ax, img, title):
        ax.imshow(img, aspect='auto')
        ax.set_title(title)
        ax.axis('off')
    
    def show_jet(self, ax, jet_map, title):
        ax.imshow(jet_map, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.axis('off')
    
    def add(self, img0, img, img_path):
        self.cur_num += img.shape[0]
        if self.cur_num <= self.len_limit:
            self.img.append(img.detach().permute(0, 2, 3, 1).numpy())
            self.img0.append(img0.detach().permute(0, 2, 3, 1).numpy())
        self.img_path.append(img_path)
    
    def test(self):
        self.len_limit = 50
        
    def compute_visual_proj_rep_post(self, ret):
        ret_tmp = ret.detach().clone() # [b, d]
        self.visual_proj_rep_post.append(ret_tmp.norm(dim=-1, keepdim=True).cpu().numpy())
        return ret
    
    def compute_visual_proj_rep_pre(self, ret):
        ret_tmp = ret.detach().clone() # [b, d]
        self.visual_proj_rep_pre.append(ret_tmp.norm(dim=-1, keepdim=True).cpu().numpy())
        return ret
    
    def compute_visual_proj_post(self, ret):
        ret_tmp = ret.detach().clone() # [b, d]
        self.visual_proj_post.append(ret_tmp.norm(dim=-1, keepdim=True).cpu().numpy())
        return ret
    
    def compute_visual_proj_pre(self, ret):
        ret_tmp = ret.detach().clone() # [b, d]
        self.visual_proj_pre.append(ret_tmp.norm(dim=-1, keepdim=True).cpu().numpy())
        return ret
    
    def compute_textual_post_ln(self, ret):
        ret_tmp = ret.detach().clone() # [n_cls, d]
        self.textual_post_ln.append(ret_tmp.norm(dim=-1).cpu().numpy())
        return ret
    
    def compute_textual_post_proj(self, ret):
        ret_tmp = ret.detach().clone() # [n_cls, d]
        self.textual_post_proj.append(ret_tmp.norm(dim=-1).cpu().numpy())
        return ret

    def reinit(self):
        self.visual_cls_token_after_attn = _list_init(self.visual_layers)
        self.visual_prompt_token_after_attn = _list_init(self.visual_layers)
        self.visual_content_token_after_attn = _list_init(self.visual_layers)
        self.visual_cls_token_post = _list_init(self.visual_layers)
        self.visual_prompt_token_post = _list_init(self.visual_layers)
        self.visual_content_token_post = _list_init(self.visual_layers)
        self.textual_eot_token_after_attn = _list_init(self.textual_layers)
        self.textual_content_token_after_attn = _list_init(self.textual_layers)
        self.textual_eot_token_post = _list_init(self.textual_layers)
        self.textual_content_token_post = _list_init(self.textual_layers)
        self.visual_proj_post = []
        self.visual_proj_pre = []
        self.visual_proj_rep_post = []
        self.visual_proj_rep_pre = []
        self.textual_post_ln = []
        self.textual_post_proj = []
        self.cur_num = 0
        self.img = []
        self.img0 = []
        self.img_path = []
        torch.cuda.empty_cache()


def hook_prs_logger(model, device, layer_info=None, cfg=None, textual_len=None):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, device, layer_info, cfg, textual_len)
    model.hook_manager.register(
        "visual.transformer.resblocks.*.after_attn", prs.compute_visual_sequence_after_attn
    )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.post", prs.compute_visual_sequence_post
    )
    model.hook_manager.register(
        "textual.resblocks.*.after_attn", prs.compute_textual_sequence_after_attn
    )
    model.hook_manager.register(
        "textual.resblocks.*.post", prs.compute_textual_sequence_post
    )
    model.hook_manager.register(
        "visual.proj_rep.post", prs.compute_visual_proj_rep_post
    )
    model.hook_manager.register(
        "visual.proj_rep.pre", prs.compute_visual_proj_rep_pre
    )
    model.hook_manager.register(
        "visual.proj.post", prs.compute_visual_proj_post
    )
    model.hook_manager.register(
        "visual.proj.pre", prs.compute_visual_proj_pre
    )
    model.hook_manager.register(
        "textual_post.post_ln", prs.compute_textual_post_ln
    )
    model.hook_manager.register(
        "textual_post.post_proj", prs.compute_textual_post_proj
    )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.attn_weight", prs.compute_visual_attentions_matrix
    )
    return prs
