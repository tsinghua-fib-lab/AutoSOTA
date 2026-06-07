import torch
from mmengine.model import BaseModel
from torch import nn
import math
import copy

from mmdet.registry import MODELS
from mmdet.models.layers import SinePositionalEncoding

from mmengine.model import caffe2_xavier_init, normal_init, xavier_init
from mmengine.model import ModuleList
from mmdet.models.layers import DeformableDetrTransformerDecoderLayer, DetrTransformerDecoderLayer
from torch import Tensor
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention

from mmdet.models.layers import MLP, coordinate_to_encoding, inverse_sigmoid

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=True,
                 scale=None,
                 use_coordinate_to_encoding=False):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.use_coordinate_to_encoding = use_coordinate_to_encoding

    def forward(self, boxes, image_shape_list):
        """
        use_coordinate_to_encoding represents the new PE method modified on Aug 8th; both boxes_normalized and embeddings are changed
        """
        if self.use_coordinate_to_encoding:
            eps = 1e-6
            boxes_normalized = boxes / (image_shape_list + eps)
            embeddings = coordinate_to_encoding(boxes_normalized, num_feats=self.num_pos_feats)
        else:
            if self.normalize:
                eps = 1e-6
                boxes_normalized = boxes / (image_shape_list + eps) * self.scale
            else:
                boxes_normalized = boxes
            half_dim = self.num_pos_feats // 2
            embeddings = math.log(self.temperature) / (half_dim - 1)
            embeddings = torch.exp(
                torch.arange(half_dim, device=boxes.device) * -embeddings)
            # embeddings = boxes_normalized[:, :, None] * embeddings[None, :]   # for boxes.shape: torch.Size([K, 4])
            embeddings = boxes_normalized[:, :, :, None] * embeddings[None, :]  # for boxes.shape: torch.Size([bs, K, 4])
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings, boxes_normalized


@MODELS.register_module()
class VisualPromptBertModel(BaseModel):

    def __init__(self,
                 max_tokens=256,
                 pad_to_max=False,
                 use_global_box=False,
                 prompt_across_batch=False,
                 prompt_across_batch_v2=False,
                 aggregate_NonSelf_prompts=False,
                 use_coordinate_to_encoding=False,
                 hidden_dim_D=256,
                 query_dim=256,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.hidden_dim_D = hidden_dim_D
        self.query_dim = query_dim
        self.use_global_box = use_global_box
        self.prompt_across_batch = prompt_across_batch
        self.prompt_across_batch_v2 = prompt_across_batch_v2
        self.aggregate_NonSelf_prompts = aggregate_NonSelf_prompts

        # positional encoding
        self.PositionEmbedding = SinusoidalPositionEmbeddings(hidden_dim_D, normalize=True, use_coordinate_to_encoding=use_coordinate_to_encoding)
        self.content_embedding = nn.Embedding(1, query_dim)
        if self.use_global_box:
            self.universal_content_embedding = nn.Embedding(1, query_dim)
        self.use_coordinate_to_encoding = use_coordinate_to_encoding
    
    def init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.content_embedding.weight)
        if self.use_global_box:
            nn.init.xavier_uniform_(self.universal_content_embedding.weight)
    
    def get_VisualAttentionMask_and_PositionIds(self, prompts_labels_padded, return_visual_prompt_idx=False):
        """
        Refer to generate_masks_with_special_tokens_and_transfer_map in bert.py.
        Obtain visual_attention_mask and position_ids.
        """
        bs, max_len = prompts_labels_padded.shape[0], prompts_labels_padded.shape[1]
        idxs, pre_label = [], None
        for i, prompts_label in enumerate(prompts_labels_padded):
            for j, label in enumerate(prompts_label):
                if pre_label is not None and label != pre_label:
                    idxs.append([i, j-1])
                if j == max_len - 1 and label != -1:
                    idxs.append([i, j])
                pre_label = label
            pre_label = None
        # NOTE. Previously used torch.int8, which caused values >128 to become negative and trigger errors later; changed to torch.int64
        idxs = torch.tensor(idxs, device=prompts_labels_padded[0].device, dtype=torch.int64)
        # NOTE. Should use torch.eye, not torch.zeros, otherwise self_attention output will contain NaN values
        visual_attention_mask = (
            # torch.zeros(*(max_shape[0], max_shape[0]),
            torch.eye(max_len,
                    device=prompts_labels_padded[0].device).bool().unsqueeze(0).repeat(bs, 1, 1))
        position_ids = torch.zeros((bs, max_len), device=prompts_labels_padded[0].device, dtype=torch.uint8)
        # NOTE. Generate visual_attention_mask and position_ids from idxs
        previous_row, previous_col = 0, -1
        for i in range(idxs.shape[0]):
            row, col = idxs[i].cpu().tolist()
            if row != previous_row:
                previous_col = -1
            visual_attention_mask[row, previous_col + 1:col + 1,
                        previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=prompts_labels_padded[0].device)
            previous_col = col
            previous_row = row
        # NOTE. visual_prompt_idx is used to pick memory_visual_final from memory_visual in the encoder
        # NOTE. visual_prompt_idx changed from List[Tensor] to Tensor, padded with -1. 2025/08/13
        if return_visual_prompt_idx:
            visual_prompt_idx_list = []
            for i in range(bs):
                visual_prompt_idx_list.append(idxs[:, 1][idxs[:, 0] == i])
            pad_Value = -1
            max_prompt_idx_len = max(t.size(0) for t in visual_prompt_idx_list)
            visual_prompt_idx = torch.full((len(visual_prompt_idx_list), max_prompt_idx_len), pad_Value, dtype=visual_prompt_idx_list[0].dtype, device=visual_prompt_idx_list[0].device)
            for i, t in enumerate(visual_prompt_idx_list):
                visual_prompt_idx[i, :t.size(0)] = t
            return visual_attention_mask, position_ids, visual_prompt_idx
        return visual_attention_mask, position_ids

    def forward(self, visual_prompts, visual_prompts_labels, batch_data_samples, **kwargs) -> dict:
        """Forward function."""

        # NOTE Collect img_shape of each image, for global box and coordinate normalization.
        # In pet_dino_layers.py, normalized coordinates (relative to img_shape) will be converted to (relative to batch_input_shape).
        bs = len(visual_prompts)
        image_shape_list = []
        for data_sample in batch_data_samples:
            img_h, img_w = data_sample.prompt_img_shape
            image_shape_list.append([img_w, img_h, img_w, img_h])
        image_shape_list = torch.tensor(image_shape_list, device=visual_prompts[0].device).unsqueeze(1)

        if True:
            assert self.use_global_box
            # NOTE Special label definitions
            pad_Lable = -1

            # NOTE Insert global_box into visual_prompts
            new_visual_prompts = []
            new_visual_prompts_labels = []
            learnable_embeddings = []
            for visual_prompt, visual_prompts_label, image_shape in zip(visual_prompts, visual_prompts_labels, image_shape_list):
                # get the global_box to insert
                global_box = torch.tensor([[0, 0, 1, 1]], device=visual_prompt.device)
                global_box = global_box * image_shape
                # find change_indices
                change_indices = (visual_prompts_label[1:] != visual_prompts_label[:-1]).nonzero().squeeze() + 1
                change_indices = change_indices.tolist()
                # if change_indices is a single element, convert to list
                if not isinstance(change_indices, list):
                    change_indices = [change_indices]
                # insert global_box at change_indices in visual_prompt, and pair learnable_embedding accordingly
                visual_prompt_inserted = []
                visual_prompt_label_inserted = []
                learnable_embedding_list = []
                prev_index = 0
                for index in change_indices:
                    visual_prompt_inserted.append(visual_prompt[prev_index:index])
                    visual_prompt_inserted.append(global_box)
                    visual_prompt_label_inserted.append(visual_prompts_label[prev_index:index])
                    visual_prompt_label_inserted.append(visual_prompts_label[prev_index:prev_index+1])
                    learnable_embedding_list.append(self.content_embedding.weight.repeat(index-prev_index, 1))
                    learnable_embedding_list.append(self.universal_content_embedding.weight)
                    prev_index = index
                # append one more global_box at the end of visual_prompt, and pair learnable_embedding accordingly
                visual_prompt_inserted.append(visual_prompt[prev_index:])
                visual_prompt_inserted.append(global_box)
                visual_prompt_label_inserted.append(visual_prompts_label[prev_index:])
                visual_prompt_label_inserted.append(visual_prompts_label[prev_index:prev_index+1])
                learnable_embedding_list.append(self.content_embedding.weight.repeat(visual_prompt[prev_index:].shape[0], 1))
                learnable_embedding_list.append(self.universal_content_embedding.weight)
                # concatenate results into a single tensor
                new_visual_prompt = torch.cat(visual_prompt_inserted, dim=0)
                new_visual_prompt_label = torch.cat(visual_prompt_label_inserted, dim=0)
                learnable_embedding = torch.cat(learnable_embedding_list, dim=0)
                new_visual_prompts.append(new_visual_prompt)
                new_visual_prompts_labels.append(new_visual_prompt_label)
                learnable_embeddings.append(learnable_embedding)

            # NOTE When prompt bbox counts differ across images in a batch, pad to the max length with global_box for parallel computation
            max_len = max([new_visual_prompt.shape[0] for new_visual_prompt in new_visual_prompts]) if not self.pad_to_max else self.max_tokens
            prompt_bboxes_padded = torch.zeros(bs, *(max_len, 4), device=new_visual_prompts[0].device)                              # torch.Size([bs, K, 4])
            visual_prompts_labels_padded = torch.zeros(bs, max_len, device=new_visual_prompts_labels[0].device, dtype=torch.int64)          # torch.Size([bs, K])
            visual_token_mask = torch.ones(bs, max_len, device=new_visual_prompts_labels[0].device, dtype=torch.uint8) 
            learnable_embeddings_padded = torch.zeros(bs, *(max_len, self.query_dim), device=learnable_embeddings[0].device)
            for i, new_visual_prompt in enumerate(new_visual_prompts):
                # 1. for box
                global_box = torch.tensor([[0, 0, 1, 1]], device=new_visual_prompt.device)
                global_box = global_box * image_shape_list[i]
                prompt_bboxes_padded[i, :new_visual_prompt.shape[0], :] = new_visual_prompt
                need_pad_num = max_len - new_visual_prompt.shape[0]
                prompt_bboxes_padded[i, new_visual_prompt.shape[0]:, :] = global_box.repeat(need_pad_num, 1)
                # 2. for label
                visual_prompts_labels_padded[i, :new_visual_prompt.shape[0]] = new_visual_prompts_labels[i]
                visual_prompts_labels_padded[i, new_visual_prompt.shape[0]:] = pad_Lable
                # 3. for token_mask
                visual_token_mask[i, new_visual_prompt.shape[0]:] = 0
                # 4. for learnable_embedding
                learnable_embeddings_padded[i, :new_visual_prompt.shape[0], :] = learnable_embeddings[i]
                learnable_embeddings_padded[i, new_visual_prompt.shape[0]:, :] = self.universal_content_embedding.weight.repeat(need_pad_num, 1)
            
            # NOTE Convert prompt_bboxes_padded from [x1, y1, x2, y2] to [cx, cy, w, h], refer to gen_encoder_output_proposals in deformable_detr.py
            prompt_bboxes_x = (prompt_bboxes_padded[..., 0] + prompt_bboxes_padded[..., 2]) / 2
            prompt_bboxes_y = (prompt_bboxes_padded[..., 1] + prompt_bboxes_padded[..., 3]) / 2
            prompt_bboxes_w = (prompt_bboxes_padded[..., 2] - prompt_bboxes_padded[..., 0])
            prompt_bboxes_h = (prompt_bboxes_padded[..., 3] - prompt_bboxes_padded[..., 1])
            prompt_bboxes_padded = torch.stack((prompt_bboxes_x, prompt_bboxes_y, prompt_bboxes_w, prompt_bboxes_h), dim=-1)

            # NOTE Compute embedded and bbox_normalized
            bbox_embed, bbox_normalized = self.PositionEmbedding(prompt_bboxes_padded, image_shape_list)
            if not self.use_coordinate_to_encoding:
                bbox_embed = bbox_embed.view(bs, prompt_bboxes_padded.shape[1], -1)

            # NOTE The paper's description of this logic is incorrect; bug_fixed=True is the corrected logic
            # NOTE Content embedding and position embedding will not be concatenated but added during attention.
            bug_fixed = True
            if bug_fixed:
                query = Q = learnable_embeddings_padded
            # else:
            #     Q = torch.cat((learnable_embeddings_padded, bbox_embed), dim=-1)
            #     query = self.query_linear(Q) 

            # NOTE Generate visual_attention_mask from visual_prompts_labels_padded for self_attention in encoder; position_ids generated as well
            visual_attention_mask, position_ids, visual_prompt_idx = self.get_VisualAttentionMask_and_PositionIds(visual_prompts_labels_padded, return_visual_prompt_idx=True)

            # NOTE Generate unique_visual_token_mask and unique_visual_prompts_labels_padded
            unique_visual_prompts_labels = []
            for visual_prompts_label in visual_prompts_labels:
                unique_visual_prompts_labels.append(torch.unique(visual_prompts_label))
            max_len_unique = max([unique_visual_prompts_label.shape[0] for unique_visual_prompts_label in unique_visual_prompts_labels]) if not self.pad_to_max else self.max_tokens
            unique_visual_prompts_labels_padded = torch.zeros(bs, max_len_unique, device=unique_visual_prompts_labels[0].device, dtype=torch.int64)
            unique_visual_token_mask = torch.ones(bs, max_len_unique, device=unique_visual_prompts_labels[0].device, dtype=torch.uint8)
            for i, unique_visual_prompts_label in enumerate(unique_visual_prompts_labels):
                # 1. for label
                unique_visual_prompts_labels_padded[i, :unique_visual_prompts_label.shape[0]] = unique_visual_prompts_label
                unique_visual_prompts_labels_padded[i, unique_visual_prompts_label.shape[0]:] = pad_Lable
                # 2. for token_mask
                unique_visual_token_mask[i, unique_visual_prompts_label.shape[0]:] = 0
            
            # NOTE Generate batch_visual_attention_mask from unique_visual_prompts_labels_padded for computing visual_positive_maps
            unique_visual_attention_mask, _ = self.get_VisualAttentionMask_and_PositionIds(unique_visual_prompts_labels_padded)

            # NOTE. Assemble into dict and return
            visual_dict_features = dict()
            visual_dict_features['embedded'] = query                                # corresponds to visual_prompts_labels_padded
            visual_dict_features['normalized_coordinate'] = bbox_normalized         # same as above
            visual_dict_features['masks'] = visual_attention_mask                   # same as above, used for self_attention in encoder
            visual_dict_features['position_ids'] = position_ids                     # same as above
            visual_dict_features['visual_token_mask'] = visual_token_mask.bool()    # same as above
            # -----
            visual_dict_features['visual_prompt_idx'] = visual_prompt_idx           # used to pick global box features after self attention
            # -----
            visual_dict_features['unique_visual_prompts_labels'] = unique_visual_prompts_labels_padded   # used to compute visual_positive_maps, put into data_samples.gt_instances
            visual_dict_features['unique_masks'] = unique_visual_attention_mask                                 # used to compute visual_positive_maps, put into data_samples.gt_instances
            visual_dict_features['unique_visual_token_mask'] = unique_visual_token_mask.bool()                  # put into data_samples.gt_instances as visual_token_mask

            # NOTE. check if all samples from the same dataset and if all samples have reverse_label_remap
            label_map_tags = [
                data_samples.metainfo.get('label_map_tag', None) for data_samples in batch_data_samples
            ]
            all_samples_from_the_same_dataset = len(set(label_map_tags)) == 1
            have_reverse_label_remap = [
                data_samples.metainfo.get('reverse_label_remap_dict', None) is not None for data_samples in batch_data_samples
            ]
            all_samples_have_reverse_label_remap = False not in have_reverse_label_remap

            # NOTE. Only training goes through the code below, since all_samples_have_reverse_label_remap is True only during training.
            if self.prompt_across_batch_v2 and all_samples_from_the_same_dataset and all_samples_have_reverse_label_remap:
                # NOTE. prompt_across_batch_v2 requires all three conditions above
                valid_visual_token_mask_list = []
                valid_visual_prompts_labels_list = []
                valid_visual_prompts_batch_idxs_list = []   # used to place current image's prompts first
                for i in range(bs):
                    valid_index = visual_dict_features['unique_visual_token_mask'][i]
                    valid_visual_token_mask_list.append(visual_dict_features['unique_visual_token_mask'][i][valid_index])
                    valid_visual_prompts_labels_list.append(visual_dict_features['unique_visual_prompts_labels'][i][valid_index])
                    valid_visual_prompts_batch_idxs_list.append(torch.full_like(visual_dict_features['unique_visual_prompts_labels'][i][valid_index], i))
                # NOTE. batch_sorted_indices requires that within the same category, current image's prompts come first; differs for each image in batch (needed for cross-batch prompt aggregation)
                # ==== original code ====
                # batch_visual_prompts_labels = torch.cat(valid_visual_prompts_labels_list, dim=0)
                # _, sorted_indices = torch.sort(batch_visual_prompts_labels, stable=True)
                # batch_sorted_indices = sorted_indices.unsqueeze(dim=0).repeat(bs, 1)
                # ==== modified code ====
                batch_visual_prompts_labels = torch.cat(valid_visual_prompts_labels_list, dim=0)
                valid_visual_prompts_batch_idxs = torch.cat(valid_visual_prompts_batch_idxs_list, dim=0)
                sorted_indices_list = []
                for i in range(bs):
                    # Create temp sort key: mark 0 when batch_idx equals current i, otherwise 1; keeps current image's elements first within same category
                    temp_sort_key = torch.where(
                        valid_visual_prompts_batch_idxs == i, 0, 1).to(batch_visual_prompts_labels.dtype)
                    # Composite key: primary sort by label, secondary sort by whether current batch_idx
                    composite_key = batch_visual_prompts_labels * 10 + temp_sort_key
                    # Stable sort to preserve original order for elements with same label and batch_idx
                    _, sorted_indices = torch.sort(composite_key, stable=True)
                    sorted_indices_list.append(sorted_indices)
                batch_sorted_indices = torch.stack(sorted_indices_list, dim=0)
                # NOTE. Sort batch_visual_prompts_labels by batch_sorted_indices
                batch_visual_prompts_labels_list = []
                batch_visual_token_mask_list = []
                batch_unaggregated_visual_prompts_labels_list = [] # NOTE. used when self.aggregate_NonSelf_prompts==True
                batch_unaggregated_visual_prompts_batch_idxs_list = [] # NOTE. used when self.aggregate_NonSelf_prompts==True
                for i in range(bs):
                    sorted_indice = batch_sorted_indices[i]
                    sorted_batch_visual_prompts_labels = batch_visual_prompts_labels[sorted_indice]
                    sorted_batch_visual_prompts_batch_idxs = valid_visual_prompts_batch_idxs[sorted_indice]
                    # NOTE. Optional, used to aggregate visual prompts with same category from other images
                    if self.aggregate_NonSelf_prompts:
                        mask = torch.ones_like(sorted_batch_visual_prompts_labels, dtype=torch.bool)
                        not_cur_batch_indices = (sorted_batch_visual_prompts_batch_idxs != i).nonzero().squeeze(1).view(-1)
                        # NOTE. Compute nonself_mask
                        nonself_visual_prompts_labels = sorted_batch_visual_prompts_labels[not_cur_batch_indices]
                        unique_values = torch.unique(nonself_visual_prompts_labels)
                        # Create mask
                        nonself_mask = torch.ones_like(nonself_visual_prompts_labels, dtype=torch.bool)
                        # Process each unique value
                        for val in unique_values:
                            # Find all indices of this value
                            indices = (nonself_visual_prompts_labels == val).nonzero().squeeze(1).view(-1)
                            # If more than 1, keep only one, corresponding to the aggregated prompt from non-self images of same category
                            if len(indices) > 1:
                                nonself_mask[indices[1:]] = False
                        # NOTE. Compute mask
                        mask[not_cur_batch_indices] = nonself_mask
                        # Apply mask to sorted_batch_visual_prompts_labels
                        batch_visual_prompts_labels_list.append(sorted_batch_visual_prompts_labels[mask])
                        # Apply mask to batch_valid_visual_token_mask
                        batch_valid_visual_token_mask = torch.cat(valid_visual_token_mask_list, dim=0)
                        batch_visual_token_mask_list.append(batch_valid_visual_token_mask[mask])
                        # Record unaggregated labels and batch_idxs, used to aggregate memory visual
                        batch_unaggregated_visual_prompts_labels_list.append(sorted_batch_visual_prompts_labels)
                        batch_unaggregated_visual_prompts_batch_idxs_list.append(sorted_batch_visual_prompts_batch_idxs)
                    else:
                        batch_visual_prompts_labels_list.append(sorted_batch_visual_prompts_labels)
                        batch_valid_visual_token_mask = torch.cat(valid_visual_token_mask_list, dim=0)
                        batch_visual_token_mask_list.append(batch_valid_visual_token_mask)
                
                # NOTE. Due to aggregate_NonSelf_prompts, visual prompt lengths may differ within a batch;
                # Affects batch_visual_prompts_labels_list and batch_visual_token_mask_list, which need padding;
                max_len = max([sorted_batch_visual_prompts_labels.shape[0] for sorted_batch_visual_prompts_labels in batch_visual_prompts_labels_list])
                batch_visual_prompts_labels_padded = torch.zeros(bs, max_len, device=batch_visual_prompts_labels_list[0].device, dtype=torch.int64)
                batch_visual_token_mask_padded = torch.zeros(bs, max_len, device=batch_visual_token_mask_list[0].device, dtype=torch.bool)
                for i, sorted_batch_visual_prompts_labels in enumerate(batch_visual_prompts_labels_list):
                    # 1. for label
                    batch_visual_prompts_labels_padded[i, :sorted_batch_visual_prompts_labels.shape[0]] = batch_visual_prompts_labels_list[i]
                    batch_visual_prompts_labels_padded[i, sorted_batch_visual_prompts_labels.shape[0]:] = pad_Lable
                    # 2. for visual_token_mask
                    batch_visual_token_mask_padded[i, :sorted_batch_visual_prompts_labels.shape[0]] = batch_visual_token_mask_list[i]
                    batch_visual_token_mask_padded[i, sorted_batch_visual_prompts_labels.shape[0]:] = False

                # NOTE. Generate batch_visual_attention_mask for computing visual_positive_maps
                batch_visual_attention_mask, _ = self.get_VisualAttentionMask_and_PositionIds(batch_visual_prompts_labels_padded)
                
                # NOTE. Put into visual_dict
                visual_dict_features['batch_valid_indices'] = unique_visual_token_mask.bool()               # used to get valid memory_visual, same value as visual_dict_features['unique_visual_token_mask']
                visual_dict_features['batch_sorted_indices'] = batch_sorted_indices                         # used for re-sorting after prompt_across_batch_v2
                visual_dict_features['batch_visual_prompts_labels'] = batch_visual_prompts_labels_padded    # used to compute visual_positive_maps, put into data_samples.gt_instances (aggregated, padded)
                visual_dict_features['batch_masks'] = batch_visual_attention_mask                           # used to compute visual_positive_maps, put into data_samples.gt_instances (aggregated, padded)
                visual_dict_features['batch_visual_token_mask'] = batch_visual_token_mask_padded            # put into data_samples.gt_instances as visual_token_mask (aggregated, padded)
                if self.aggregate_NonSelf_prompts:
                    # NOTE. Record pre-aggregation batch_unaggregated_visual_prompts_labels_ for aggregating memory visual
                    batch_unaggregated_visual_prompts_labels_ = torch.stack(batch_unaggregated_visual_prompts_labels_list, dim=0)
                    visual_dict_features['batch_unaggregated_visual_prompts_labels'] = batch_unaggregated_visual_prompts_labels_
                    batch_unaggregated_visual_prompts_batch_idxs = torch.stack(batch_unaggregated_visual_prompts_batch_idxs_list, dim=0)
                    visual_dict_features['batch_unaggregated_visual_prompts_batch_idxs'] = batch_unaggregated_visual_prompts_batch_idxs
                
                # NOTE. Unify variable names for downstream use, prefixed with final
                visual_dict_features['final_visual_prompts_labels'] = copy.deepcopy(visual_dict_features['batch_visual_prompts_labels'])    # used to compute visual_positive_maps, put into data_samples.gt_instances
                visual_dict_features['final_masks'] = copy.deepcopy(visual_dict_features['batch_masks'])                                    # used to compute visual_positive_maps, put into data_samples.gt_instances
                visual_dict_features['final_visual_token_mask'] = copy.deepcopy(visual_dict_features['batch_visual_token_mask'])            # put into data_samples.gt_instances as visual_token_mask

            else:
                # NOTE. Unify variable names for downstream use, prefixed with final
                visual_dict_features['final_visual_prompts_labels'] = copy.deepcopy(visual_dict_features['unique_visual_prompts_labels'])   # used to compute visual_positive_maps, put into data_samples.gt_instances
                visual_dict_features['final_masks'] = copy.deepcopy(visual_dict_features['unique_masks'])                                   # used to compute visual_positive_maps, put into data_samples.gt_instances
                visual_dict_features['final_visual_token_mask'] = copy.deepcopy(visual_dict_features['unique_visual_token_mask'])           # put into data_samples.gt_instances as visual_token_mask
        return visual_dict_features