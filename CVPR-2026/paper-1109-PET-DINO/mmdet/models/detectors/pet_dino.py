# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import random
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from ..layers.transformer.pet_dino_layers import (PetDINOTransformerEncoder, PetDINOTransformerDecoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)

from mmdet.structures.bbox import bbox_project
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from mmengine.logging import print_log
from .base import ForwardResults
from ..language_models.memory_bank import MemoryBank


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class PetDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 visual_prompt_model,
                 visual_prompt_encoder,
                 memory_bank={},
                 *args,
                 use_autocast=False,
                 visual_prompt_enhancer={},
                 encoder_first_then_visual_prompt_encoder=False,    # default to False
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self.visual_prompt_model_cfg = visual_prompt_model
        self.visual_prompt_encoder_cfg = visual_prompt_encoder
        self.visual_prompt_enhancer_cfg = visual_prompt_enhancer
        self.memory_bank_cfg = memory_bank
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.prompt_type = 'Text'
        assert self.prompt_type in ['Text', 'Visual']
        self.encoder_first_then_visual_prompt_encoder = encoder_first_then_visual_prompt_encoder
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = PetDINOTransformerEncoder(**self.encoder)
        self.decoder = PetDINOTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

        # visual prompt model
        self.visual_prompt_model = MODELS.build(self.visual_prompt_model_cfg)
        self.use_global_box = self.visual_prompt_model_cfg.get('use_global_box', False)
        self.prompt_across_batch = self.visual_prompt_model_cfg.get('prompt_across_batch', False)
        self.prompt_across_batch_v2 = self.visual_prompt_model_cfg.get('prompt_across_batch_v2', False)
        self.use_coordinate_to_encoding = self.visual_prompt_model_cfg.get('use_coordinate_to_encoding', False)
        self.aggregate_NonSelf_prompts = self.visual_prompt_model_cfg.get('aggregate_NonSelf_prompts', False)
        # visual feat map
        self.visual_feat_map = nn.Linear(self.embed_dims, self.embed_dims, bias=True)
        # visual prompt encoder
        self.visual_prompt_encoder = MODELS.build(self.visual_prompt_encoder_cfg)
        # visual prompt enhancer
        if self.visual_prompt_enhancer_cfg:
            self.visual_prompt_enhancer = MODELS.build(self.visual_prompt_enhancer_cfg)
        # memory bank
        self.memory_banks = {}
        for k, v in self.memory_bank_cfg.items():
            self.memory_banks[k] = MemoryBank(**v)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
        # visual prompt model init
        nn.init.constant_(self.visual_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.visual_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ):
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        visual_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        if True:
            # run forward_encoder first
            encoder_outputs_dict = self.forward_encoder(
                **encoder_inputs_dict, text_dict=text_dict)
            
            if 'memory_visual' in visual_dict:
                # NOTE. Use precomputed memory_visual for inference (predict and eval only).
                assert self.use_global_box
                assert 'memory_visual' in visual_dict
                assert 'final_visual_token_mask' in visual_dict
            else:
                # NOTE. Apply pre_transformer to prompt_image.
                if 'prompt_visual_feats' in visual_dict:
                    prompt_visual_feats = visual_dict.pop('prompt_visual_feats')
                    if prompt_visual_feats:
                        # for predict(visual_prompt==Visual), compute encoder_inputs_dict_for_prompt
                        batch_data_samples_tmp = copy.deepcopy(batch_data_samples)
                        for data_samples_tmp in batch_data_samples_tmp:
                            new_metainfo = dict()
                            new_metainfo['img_path'] = data_samples_tmp.prompt_img_path
                            new_metainfo['ori_shape'] = data_samples_tmp.prompt_ori_shape
                            new_metainfo['img_shape'] = data_samples_tmp.prompt_img_shape
                            new_metainfo['pad_shape'] = data_samples_tmp.prompt_pad_shape
                            new_metainfo['batch_input_shape'] = data_samples_tmp.prompt_batch_input_shape
                            data_samples_tmp.set_metainfo(new_metainfo)
                        encoder_inputs_dict_for_prompt, _ = self.pre_transformer(
                            prompt_visual_feats, batch_data_samples_tmp)
                    else:
                        # for train(visual_prompt==Visual) and eval(visual_prompt==Visual)
                        encoder_inputs_dict_for_prompt = {}
                else:
                    # for predict(prompt_type==Text), visual_dict is {}.
                    pass

                # NOTE. forward_visual_prompt_encoder, used to obtain memory_visual.
                if visual_dict:
                    if encoder_inputs_dict_for_prompt:
                        if self.encoder_first_then_visual_prompt_encoder:
                            # NOTE. replace encoder_inputs_dict_for_prompt['feat'] with encoder_outputs_dict_for_prompt['memory']
                            encoder_outputs_dict_for_prompt = self.forward_encoder(
                                **encoder_inputs_dict_for_prompt, text_dict=text_dict)
                            encoder_inputs_dict_for_prompt['feat'] = encoder_outputs_dict_for_prompt['memory']
                        # for predict(visual_prompt==Visual)
                        memory_visual = self.forward_visual_prompt_encoder(**encoder_inputs_dict_for_prompt, visual_dict=visual_dict)
                    else:
                        if self.encoder_first_then_visual_prompt_encoder:
                            # NOTE. replace encoder_inputs_dict['feat'] with encoder_outputs_dict['memory']
                            encoder_inputs_dict['feat'] = encoder_outputs_dict['memory']
                        # for train(visual_prompt==Visual) and eval(visual_prompt==Visual)
                        memory_visual = self.forward_visual_prompt_encoder(**encoder_inputs_dict, visual_dict=visual_dict)
                    visual_dict['memory_visual'] = memory_visual
                else:
                    # for predict(prompt_type==Text), visual_dict is {}, memory_visual will return None.
                    visual_dict['memory_visual'] = None
                    visual_dict['final_visual_token_mask'] = None
            
            # NOTE. Contrastive alignment loss (placeholder).
            # Some parameters are unused in loss computation; this avoids
            # unused parameter errors in multi-GPU training.
            if self.training:
                contrastive_alignment_dict = dict(
                    ca_memory_text=encoder_outputs_dict['memory_text'],
                    ca_text_token_mask=encoder_outputs_dict['text_token_mask'],
                    ca_memory_visual=visual_dict['memory_visual'],
                    ca_visual_token_mask=visual_dict['final_visual_token_mask'],
                )
            else:
                contrastive_alignment_dict = dict()
            
            # NOTE. If prompt_type == 'Visual', replace memory_text with memory_visual for downstream.
            if self.prompt_type == 'Visual':
                encoder_outputs_dict['memory_text'] = visual_dict['memory_visual']
                encoder_outputs_dict['text_token_mask'] = visual_dict['final_visual_token_mask']

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict, contrastive_alignment_dict
    
    def forward_visual_prompt_encoder(self, feat: Tensor, feat_mask: Tensor,
                                      feat_pos: Tensor, spatial_shapes: Tensor,
                                      level_start_index: Tensor, valid_ratios: Tensor,
                                      visual_dict: Dict) -> Tensor:
        memory_visual = self.visual_prompt_encoder(
            # for prompt image feat
            prompt_query=feat,
            prompt_query_pos=feat_pos,
            prompt_key_padding_mask=feat_mask,
            prompt_spatial_shapes=spatial_shapes,
            prompt_level_start_index=level_start_index,
            prompt_valid_ratios=valid_ratios,
            # for visual encoder
            memory_visual=visual_dict['embedded'] if visual_dict else None,
            visual_attention_mask=~visual_dict['visual_token_mask'] if visual_dict else None,
            visual_position_ids=visual_dict['position_ids'] if visual_dict else None,       # unused
            visual_self_attention_masks=visual_dict['masks'] if visual_dict else None,
            normalized_coordinate=visual_dict['normalized_coordinate'] if visual_dict else None,
            visual_prompt_idx=visual_dict['visual_prompt_idx'] if visual_dict and 'visual_prompt_idx' in visual_dict else None,
            visual_prompts_batch_idx=None,
            # use_coordinate_to_encoding in visual prompt
            use_coordinate_to_encoding=self.use_coordinate_to_encoding)

        # NOTE. Training-only logic below (not used for inference).
        if self.training:
            batch_size = memory_visual.shape[0]
            # NOTE. for prompt_across_batch_v2 training process.
            if 'batch_valid_indices' in visual_dict and 'batch_sorted_indices' in visual_dict:
                assert self.use_global_box
                assert self.prompt_across_batch_v2
                valid_memory_visual_list = []
                for i in range(batch_size):
                    valid_index = visual_dict['batch_valid_indices'][i]
                    valid_memory_visual_list.append(memory_visual[i][valid_index])
                batch_memory_visual = torch.cat(valid_memory_visual_list, dim=0).unsqueeze(dim=0).repeat(batch_size, 1, 1)
                # NOTE. Sort batch_memory_visual by batch_sorted_indices
                sorted_batch_memory_visual_list = []
                for i in range(batch_size):
                    sorted_indice = visual_dict['batch_sorted_indices'][i]
                    sorted_batch_memory_visual_list.append(batch_memory_visual[i][sorted_indice])
                batch_memory_visual = torch.stack(sorted_batch_memory_visual_list, dim=0)

                # NOTE. for aggregate_NonSelf_prompts training process (inference goes through mmdet/apis/det_inferencer.py).
                if self.training and self.aggregate_NonSelf_prompts:
                    assert 'batch_unaggregated_visual_prompts_labels' in visual_dict
                    assert 'batch_unaggregated_visual_prompts_batch_idxs' in visual_dict
                    
                    # NOTE. Mean pooling of non-self visual prompts
                    batch_memory_visual_list = []
                    for i in range(batch_size):
                        mask = torch.ones_like(visual_dict['batch_unaggregated_visual_prompts_labels'][i], dtype=torch.bool)
                        not_cur_batch_indices = (visual_dict['batch_unaggregated_visual_prompts_batch_idxs'][i] != i).nonzero().squeeze(1).view(-1)
                        nonself_visual_prompts_labels = visual_dict['batch_unaggregated_visual_prompts_labels'][i][not_cur_batch_indices]
                        unique_values = torch.unique(nonself_visual_prompts_labels)
                        for val in unique_values:
                            indices = (nonself_visual_prompts_labels == val).nonzero().squeeze(1).view(-1)
                            # If more than one, aggregate into a single prompt via mean pooling
                            if len(indices) > 1:
                                mask[not_cur_batch_indices[indices[1:]]] = False        # NOTE. Double indexing like mask[not_cur_batch_indices][indices[1:]] = False would not work
                                aggregated_NonSelf_memory_visual = batch_memory_visual[i][not_cur_batch_indices[indices]].mean(dim=0, keepdim=True)
                                batch_memory_visual[i][not_cur_batch_indices[indices[0]]] = aggregated_NonSelf_memory_visual
                        batch_memory_visual_list.append(batch_memory_visual[i][mask])
                
                elif self.training and not self.aggregate_NonSelf_prompts:
                    batch_memory_visual_list = []
                    for i in range(batch_size):
                        batch_memory_visual_list.append(batch_memory_visual[i])
            else:
                batch_memory_visual_list = []
                for i in range(batch_size):
                    batch_memory_visual_list.append(memory_visual[i][visual_dict['unique_visual_token_mask'][i]])
            
            if hasattr(self, 'visual_prompt_enhancer'):
                batch_sampled_features_list = []
                batch_sampled_count = []
                batch_sampled_count_memory_valid = [[], [], [], []]
                for i in range(batch_size):
                    label_map_tag = visual_dict['label_map_tag_list'][i]
                    assert label_map_tag in self.memory_banks
                    memory_sampled_visual_prompt_labels = visual_dict['memory_sampled_visual_prompt_labels_list'][i]
                    if len(memory_sampled_visual_prompt_labels) == 0:
                        # Create a random tensor to pass through the enhancer module
                        batch_sampled_count.append(1)
                        batch_sampled_count_memory_valid[i].append(16)
                        class_features = torch.rand(16, 256)
                        batch_sampled_features_list.append(class_features)
                    else:
                        batch_sampled_count.append(len(memory_sampled_visual_prompt_labels))
                        for label in memory_sampled_visual_prompt_labels:
                            class_features = self.memory_banks[label_map_tag].get_class_features(class_id=int(label))
                            # class_features = class_features.requires_grad_(True)  
                            batch_sampled_features_list.append(class_features)
                            batch_sampled_count_memory_valid[i].append(class_features.shape[0])
                # Initialize a random tensor of shape (N, 16, 256)
                batch_sampled_features = torch.rand(len(batch_sampled_features_list), 16, 256, device=batch_memory_visual.device)
                batch_visual_attention_mask = torch.eye(16, device=batch_memory_visual.device).bool().unsqueeze(0).repeat(len(batch_sampled_features_list), 1, 1)
                # Fill valid data (overwriting random parts)
                for i, tensor in enumerate(batch_sampled_features_list):
                    M = tensor.shape[0]
                    batch_sampled_features[i, :M, :] = tensor  # Only fill valid portion, rest remains random
                    batch_visual_attention_mask[i,:M,:M] = True
                # Pass through visual_prompt_enhancer
                batch_sampled_features = batch_sampled_features.requires_grad_(True)        # Force enable gradients
                enhanced_batch_sampled_features = self.visual_prompt_enhancer(
                    memory_visual=batch_sampled_features,
                    visual_self_attention_masks=batch_visual_attention_mask)
                # Split enhanced features back per sample
                enhanced_batch_sampled_features_list = torch.split(enhanced_batch_sampled_features, batch_sampled_count, dim=0)
                for i in range(batch_size):
                    # NOTE. Mean pooling of enhanced features
                    sampled_visual_prompt_memory_list = []
                    memory_sampled_visual_prompt_labels = visual_dict['memory_sampled_visual_prompt_labels_list'][i]
                    for j in range(len(enhanced_batch_sampled_features_list[i])):
                        valid_num = batch_sampled_count_memory_valid[i][j]
                        sampled_visual_prompt_memory_list.append(enhanced_batch_sampled_features_list[i][j][:valid_num].mean(dim=0))
                    if len(memory_sampled_visual_prompt_labels) == 0:
                        # sampled_visual_prompt_memory_list contains dummy random tensors here
                        sampled_visual_prompt_memory = torch.empty((0, 256), dtype=batch_memory_visual_list[0].dtype, device=batch_memory_visual_list[0].device)
                        sampled_visual_prompt_memory = sampled_visual_prompt_memory + torch.stack(sampled_visual_prompt_memory_list) * 0     # Establish explicit gradient dependency
                    else:                                           
                        sampled_visual_prompt_memory = torch.stack(sampled_visual_prompt_memory_list)
                    # NOTE. Update memory bank with features before mean pooling (aggregate_NonSelf_prompts).
                    visual_prompts_labels_valid = visual_dict['unique_visual_prompts_labels'][i][visual_dict['unique_visual_token_mask'][i]]
                    memory_visual_valid = memory_visual[i][visual_dict['unique_visual_token_mask'][i]]
                    self.memory_banks[label_map_tag].update(memory_visual_valid, visual_prompts_labels_valid, ignore_labels=[])
                    # NOTE. Concatenate sampled_visual_prompt_memory into batch_memory_visual_list[i] (no sorting needed)
                    batch_memory_visual_list[i] = torch.cat((batch_memory_visual_list[i], sampled_visual_prompt_memory), dim=0)
            else:
                # NOTE. for memory bank training process.
                for i in range(batch_size):
                    label_map_tag = visual_dict['label_map_tag_list'][i]
                    if label_map_tag in self.memory_banks:
                        # NOTE. Sample visual prompts from memory bank
                        memory_sampled_visual_prompt_labels = visual_dict['memory_sampled_visual_prompt_labels_list'][i]
                        sampled_visual_prompt_memory_list = []
                        for label in memory_sampled_visual_prompt_labels:
                            if self.memory_banks[label_map_tag].use_ema_feature:
                                sampled_visual_prompt_memory_list.append(self.memory_banks[label_map_tag].get_class_ema_feature(class_id=int(label)))
                            elif self.memory_banks[label_map_tag].use_mean_feature:
                                sampled_visual_prompt_memory_list.append(self.memory_banks[label_map_tag].get_class_mean_feature(class_id=int(label)))
                            else:
                                sampled_visual_prompt_memory_list.append(self.memory_banks[label_map_tag].get_class_any_feature(class_id=int(label)))
                        # NOTE. The sampled visual prompts from memory bank
                        if sampled_visual_prompt_memory_list:
                            sampled_visual_prompt_memory = torch.stack(sampled_visual_prompt_memory_list)
                        else:
                            sampled_visual_prompt_memory = torch.empty((0, 256), dtype=batch_memory_visual_list[0].dtype, device=batch_memory_visual_list[0].device)

                        # NOTE. Update memory bank with features before mean pooling (aggregate_NonSelf_prompts).
                        visual_prompts_labels_valid = visual_dict['unique_visual_prompts_labels'][i][visual_dict['unique_visual_token_mask'][i]]
                        memory_visual_valid = memory_visual[i][visual_dict['unique_visual_token_mask'][i]]
                        self.memory_banks[label_map_tag].update(memory_visual_valid, visual_prompts_labels_valid, ignore_labels=[])

                        # NOTE. Concatenate sampled_visual_prompt_memory into batch_memory_visual_list[i] (no sorting needed)
                        batch_memory_visual_list[i] = torch.cat((batch_memory_visual_list[i], sampled_visual_prompt_memory), dim=0)

            # NOTE. aggregate_NonSelf_prompts / memory bank may cause variable-length visual prompts within a batch, need padding
            max_len = max([memory_visual_item.shape[0] for memory_visual_item in batch_memory_visual_list])
            batch_memory_visual = torch.zeros(batch_size, *(max_len, batch_memory_visual_list[0].shape[-1]), device=batch_memory_visual_list[0].device)
            for i, memory_visual_item in enumerate(batch_memory_visual_list):
                need_pad_num = max_len - memory_visual_item.shape[0]
                batch_memory_visual[i, :memory_visual_item.shape[0]] = batch_memory_visual_list[i]
                # Compute mean and std of current features for random padding
                mean_feat = memory_visual_item.mean()
                std_feat = memory_visual_item.std()
                # Generate random padding values following the same distribution
                random_padding = torch.randn(need_pad_num, memory_visual_item.shape[1], device=memory_visual_item.device) * std_feat + mean_feat
                batch_memory_visual[i, memory_visual_item.shape[0]:] = random_padding

            return batch_memory_visual
            
        return memory_visual  

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Tuple[Dict]:
        text_token_mask = text_dict['text_token_mask'] if text_dict else None

        # NOTE. If prompt_type is Text, output has cross-attended with text;
        # if prompt_type is Visual, output only passed through self-attention.
        memory, memory_text = self.encoder(             # Feature Enhancer Layer
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text
            memory_text=text_dict['embedded'] if text_dict else None,
            text_attention_mask=~text_token_mask if text_dict else None,
            position_ids=text_dict['position_ids'] if text_dict else None,
            text_self_attention_masks=text_dict['masks'] if text_dict else None,
            # prompt_type
            prompt_type=self.prompt_type)

        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask
        )
        
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        # NOTE language-guided query selection module for query initialization
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[        # torch.Size([1, 20097, 256])
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask, self.prompt_type)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(                              # torch.Size([1, 900, 256])
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # NOTE ref OV-DINO. get E_so.
        topk_output_memory = torch.gather(                      # torch.Size([1, 900, 256])
            output_memory, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
            # ref OV-DINO.
            memory_so=topk_output_memory,  # torch.Size([1, 900, 256])
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        # kwargs: memory_text \ text_attention_mask
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            # prompt_type
            prompt_type=self.prompt_type,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # NOTE. text prompt
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        # NOTE. visual prompt
        assert 'prompt_bboxes' in batch_data_samples[0]
        visual_prompts = []
        visual_prompts_labels = []
        for i, data_samples in enumerate(batch_data_samples):
            if not data_samples.metainfo['prompt_bboxes_already_transformed']:
                # NOTE. Project prompt_bboxes from ori_shape to img_shape via homography_matrix.
                prompt_bboxes_in_ori_shape = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device).float()
                prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                prompt_bboxes_in_img_shape = bbox_project(
                        prompt_bboxes_in_ori_shape,
                        torch.from_numpy(data_samples.homography_matrix).to(
                            self.data_preprocessor.device), data_samples.prompt_img_shape)
                visual_prompts.append(prompt_bboxes_in_img_shape)
                visual_prompts_labels.append(prompt_bboxes_labels)
            else:
                prompt_bboxes = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device)
                prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                visual_prompts.append(prompt_bboxes)
                visual_prompts_labels.append(prompt_bboxes_labels)

        # check if all samples from the same dataset and if all samples have reverse_label_remap
        label_map_tag_list = [
            data_samples.metainfo.get('label_map_tag', None) for data_samples in batch_data_samples
        ]
        all_samples_from_the_same_dataset = len(set(label_map_tag_list)) == 1
        have_reverse_label_remap = [
            data_samples.metainfo.get('reverse_label_remap_dict', None) is not None for data_samples in batch_data_samples
        ]
        all_samples_have_reverse_label_remap = False not in have_reverse_label_remap

        # NOTE. label remap.
        if self.prompt_across_batch_v2 and all_samples_from_the_same_dataset and all_samples_have_reverse_label_remap:
            # NOTE. prompt_across_batch_v2 requires all three conditions above.
            for i, data_samples in enumerate(batch_data_samples):
                reverse_label_remap_dict = data_samples.metainfo['reverse_label_remap_dict']
                visual_prompts_labels[i] = torch.tensor(list(map(lambda x: reverse_label_remap_dict[x], visual_prompts_labels[i].cpu().numpy())))
                visual_prompts_labels[i] = visual_prompts_labels[i].to(data_samples.gt_instances.labels.device)
                # NOTE. visual_prompts_labels must be sorted in ascending order, with visual_prompts matching accordingly.
                sorted_idx = torch.argsort(visual_prompts_labels[i])
                visual_prompts[i] = visual_prompts[i][sorted_idx]
                visual_prompts_labels[i] = visual_prompts_labels[i][sorted_idx]
        else:
            # NOTE. Use memory bank if a matching label_map_tag is found.
            for i, data_samples in enumerate(batch_data_samples):
                label_map_tag = data_samples.metainfo.get('label_map_tag', None)
                if label_map_tag in self.memory_banks:
                    reverse_label_remap_dict = data_samples.metainfo['reverse_label_remap_dict']
                    visual_prompts_labels[i] = torch.tensor(list(map(lambda x: reverse_label_remap_dict[x], visual_prompts_labels[i].cpu().numpy())))
                    visual_prompts_labels[i] = visual_prompts_labels[i].to(data_samples.gt_instances.labels.device)
                    # NOTE. visual_prompts_labels must be sorted in ascending order, with visual_prompts matching accordingly.
                    sorted_idx = torch.argsort(visual_prompts_labels[i])
                    visual_prompts[i] = visual_prompts[i][sorted_idx]
                    visual_prompts_labels[i] = visual_prompts_labels[i][sorted_idx]

        # run visual_prompt_model
        visual_dict = self.visual_prompt_model(visual_prompts, visual_prompts_labels, batch_data_samples)
        if self.visual_feat_map is not None:
            visual_dict['embedded'] = self.visual_feat_map(visual_dict['embedded'])
        
        # NOTE. For Train(prompt_type==Visual), prompt_visual_feats is None.
        visual_dict['prompt_visual_feats'] = None

        # NOTE. set label_map_tag for memory bank
        visual_dict['label_map_tag_list'] = label_map_tag_list
        visual_dict['all_samples_from_the_same_dataset'] = all_samples_from_the_same_dataset
        visual_dict['all_samples_have_reverse_label_remap'] = all_samples_have_reverse_label_remap
        
        # NOTE. Add memory bank features to visual prompts; manage labels in advance.
        if True:
            assert self.use_global_box
            ## args ##
            pad_Lable = -1
            ##########
            visual_dict['memory_sampled_visual_prompt_labels_list'] = []
            visual_prompts_labels_with_memory_list = []
            # NOTE. Look up memory bank if a matching label_map_tag is found.
            for i, data_samples in enumerate(batch_data_samples):
                prompt_bboxes_labels = visual_dict['final_visual_prompts_labels'][i]
                visual_token_mask = visual_dict['final_visual_token_mask'][i]
                label_map_tag = data_samples.metainfo.get('label_map_tag', None)
                if label_map_tag in self.memory_banks:
                    valid_classes = self.memory_banks[label_map_tag].get_FeatureNum_valid_classes(min_FeatureNum=self.memory_banks[label_map_tag].min_FeatureNum)
                    if hasattr(self.memory_banks[label_map_tag], 'distinguish_pos_and_neg') and self.memory_banks[label_map_tag].distinguish_pos_and_neg:
                        pos_candidate_list, neg_candidate_list = [], []
                        for label in valid_classes:
                            if label in prompt_bboxes_labels:
                                pos_candidate_list.append(label)
                            else:
                                neg_candidate_list.append(label)
                        num_sample_neg_visual_prompt = min(self.memory_banks[label_map_tag].max_num_sample_neg_visual_prompt, len(neg_candidate_list))
                        memory_sampled_visual_prompt_labels = torch.tensor(pos_candidate_list + random.sample(neg_candidate_list, num_sample_neg_visual_prompt), device=prompt_bboxes_labels.device)
                    else:
                        if self.memory_banks[label_map_tag].only_sample_neg:
                            # Strategy 1: For each image in the batch, only sample negative prompts.
                            candidate_list = []
                            for label in valid_classes:
                                if label not in prompt_bboxes_labels:
                                    candidate_list.append(label)
                        else:
                            # Strategy 2: For each image in the batch, sample both negative and positive prompts.
                            candidate_list = valid_classes
                        # Randomly sample from memory bank
                        num_sample_visual_prompt = min(self.memory_banks[label_map_tag].max_num_sample_visual_prompt, len(candidate_list))
                        memory_sampled_visual_prompt_labels = torch.tensor(random.sample(candidate_list, num_sample_visual_prompt), device=prompt_bboxes_labels.device)
                    visual_dict['memory_sampled_visual_prompt_labels_list'].append(memory_sampled_visual_prompt_labels)
                    # Record the final visual_prompts_labels
                    visual_prompts_labels_with_memory = torch.cat((prompt_bboxes_labels[visual_token_mask], memory_sampled_visual_prompt_labels))
                    visual_prompts_labels_with_memory_list.append(visual_prompts_labels_with_memory)
                else:
                    # sampled_visual_prompt_memory = torch.empty((0, 256), dtype=batch_memory_visual_list[0].dtype, device=batch_memory_visual_list[0].device)
                    visual_dict['memory_sampled_visual_prompt_labels_list'].append(torch.tensor([], device=prompt_bboxes_labels.device))
                    visual_prompts_labels_with_memory_list.append(prompt_bboxes_labels[visual_token_mask])

            # NOTE. After adding memory bank features, visual prompt lengths may vary within a batch; need padding.
            max_len = max([visual_prompts_labels_with_memory.shape[0] for visual_prompts_labels_with_memory in visual_prompts_labels_with_memory_list])
            visual_prompts_labels_with_memory_padded = torch.zeros(len(batch_data_samples), max_len, device=visual_prompts_labels_with_memory_list[0].device, dtype=torch.int64)
            visual_token_mask_with_memory_padded = torch.ones(len(batch_data_samples), max_len, device=visual_prompts_labels_with_memory_list[0].device, dtype=torch.bool)
            for i, visual_prompts_labels_with_memory in enumerate(visual_prompts_labels_with_memory_list):
                # 1. for label
                visual_prompts_labels_with_memory_padded[i, :visual_prompts_labels_with_memory.shape[0]] = visual_prompts_labels_with_memory_list[i]
                visual_prompts_labels_with_memory_padded[i, visual_prompts_labels_with_memory.shape[0]:] = pad_Lable
                # 2. for visual_token_mask
                visual_token_mask_with_memory_padded[i, visual_prompts_labels_with_memory.shape[0]:] = False
            # NOTE. Update final labels and masks. Note: final_masks is skipped here since it becomes
            # inaccurate after memory bank insertion, so visual_positive_map is generated without final_masks.
            visual_dict['final_visual_prompts_labels'] = visual_prompts_labels_with_memory_padded
            visual_dict['final_visual_token_mask'] = visual_token_mask_with_memory_padded

        # NOTE. Compute visual_positive_map
        if True:
            assert self.use_global_box
            visual_positive_maps = []
            for i, data_samples in enumerate(batch_data_samples):
                label_map_tag = data_samples.metainfo.get('label_map_tag', None)
                prompt_bboxes_labels = visual_dict['final_visual_prompts_labels'][i]
                if (self.prompt_across_batch_v2 and all_samples_from_the_same_dataset and all_samples_have_reverse_label_remap) or (label_map_tag in self.memory_banks):
                    # Convert text-remapped labels back to original labels
                    labels = copy.deepcopy(data_samples.gt_instances.labels)
                    reverse_label_remap_dict = data_samples.metainfo['reverse_label_remap_dict']
                    labels = torch.tensor(list(map(lambda x: reverse_label_remap_dict[x], labels.cpu().numpy())))
                    labels = labels.to(data_samples.gt_instances.labels.device)
                    data_samples.gt_instances.original_labels = labels
                else:
                    labels = data_samples.gt_instances.labels
                # NOTE. Equivalent to the commented-out code above, but does not use final_masks,
                # compatible with memory_bank feature insertion.
                visual_positive_map = torch.zeros(*(len(labels), 256), device=visual_dict['final_visual_prompts_labels'][i].device)
                for j, label in enumerate(labels):
                    idx = torch.where(prompt_bboxes_labels==label.cpu())[0]
                    visual_positive_map[j][idx] = 1.0
                visual_positive_maps.append(visual_positive_map)
        
        # NOTE. Store visual_positive_maps and visual_token_mask in data_samples.gt_instances.
        for i, data_samples in enumerate(batch_data_samples):
            visual_positive_map = visual_positive_maps[i].to(batch_inputs.device)
            visual_token_mask = visual_dict['final_visual_token_mask'][i]
            data_samples.gt_instances.visual_positive_maps = visual_positive_map
            data_samples.gt_instances.visual_token_mask = \
                visual_token_mask.unsqueeze(0).repeat(
                    len(visual_positive_map), 1)

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict, contrastive_alignment_dict = self.forward_transformer(visual_features, text_dict, visual_dict,
                                                    batch_data_samples)
        head_inputs_dict.update(contrastive_alignment_dict)
        head_inputs_dict['prompt_type'] = self.prompt_type

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses
    
    def get_visual_token_positive_map(self, visual_prompts_label):
        """Compute visual_token_positive_map and visual_labels_map from visual_prompts_label.
        If self.use_global_box, visual_prompts_label should be torch.unique(visual_prompts_label).
        """
        visual_token_positive_map = {}
        visual_labels_map = {}
        # NOTE. Compute visual_labels_map; visual_prompts_label must be sorted (ensured in both eval and predict).
        for idx, value in enumerate(torch.unique(visual_prompts_label).cpu().tolist()):
            visual_labels_map[value] = idx
        # NOTE. Map visual_prompts_label to sequential labels starting from 0.
        mapped_visual_prompts_label = torch.tensor([visual_labels_map[x.item()] for x in visual_prompts_label],
                                                    device=visual_prompts_label.device)
        # NOTE. Compute visual_token_positive_map; keys are original labels + 1.
        for idx, value in enumerate(mapped_visual_prompts_label.cpu().tolist()):
            # NOTE. value + 1 because: ref mmdet/models/dense_heads/atss_vlfusion_head.py Line44
            # and mmdet/models/dense_heads/pet_dino_head.py _predict_by_feat_single.
            # For COCO, visual_token_positive_map keys range from 1-80 (real labels),
            # while predicted labels range from 0-79.
            # The value indicates which embeddings correspond to the current label;
            # if multiple, average similarity is computed.
            if value + 1 not in visual_token_positive_map:
                visual_token_positive_map[value + 1] = [idx]
            else:
                visual_token_positive_map[value + 1].append(idx)
        return visual_token_positive_map, visual_labels_map

    def predict(self, batch_inputs, batch_data_samples, prompt_batch_inputs=None, rescale: bool = True):
        """Predict results.
        prompt_batch_inputs is None when prompt_type is Text, or when prompt_type is Visual but no prompt_image.
        """
        if 'prompt_type' not in batch_data_samples[0].metainfo:
            # NOTE. Predict(prompt_type=='Text') or Eval: prompt_type set via test_cfg, defaults to Text.
            self.prompt_type = self.test_cfg.get('prompt_type', 'Text')
        else:
            # NOTE. Predict(prompt_type=='Visual'), prompt_type set via parameters.
            self.prompt_type = batch_data_samples[0].metainfo['prompt_type']
            print_log(f"Predict : Set prompt type to {self.prompt_type}.")

        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        # NOTE. visual prompt
        if self.prompt_type == 'Visual':
            if 'memory_visual' in batch_data_samples[0]:
                # Use precomputed memory_visual to build visual_dict
                memory_visual_batch_list, memory_label_batch_list, memory_class_name_batch_list = [], [], []
                for data_samples in batch_data_samples:
                    memory_visual_batch_list.append(data_samples.metainfo['memory_visual'].to(batch_inputs.device))
                    memory_label_batch_list.append(data_samples.metainfo['memory_label'])
                    memory_class_name_batch_list.append(data_samples.metainfo['memory_class_name'])
                memory_visual = torch.stack(memory_visual_batch_list)
                visual_dict = {'memory_visual': memory_visual}
                # NOTE. Compute final_visual_token_mask
                final_visual_token_mask = torch.ones(memory_visual.shape[:2]).bool().to(batch_inputs.device)
                visual_dict['final_visual_token_mask'] = final_visual_token_mask
                # NOTE. Compute visual_token_positive_maps, visual_labels_map and visual_entities
                visual_token_positive_maps, visual_labels_maps = [], []
                for memory_label in memory_label_batch_list:
                    visual_token_positive_map, visual_labels_map = self.get_visual_token_positive_map(memory_label)
                    visual_token_positive_maps.append(visual_token_positive_map)
                    visual_labels_maps.append(visual_labels_map)
                visual_entities = memory_class_name_batch_list
                # NOTE. Fake class names (unused).
                # fake_visual_prompts_label = torch.arange(memory_visual.shape[1])
                # visual_token_positive_map, visual_labels_map = self.get_visual_token_positive_map(fake_visual_prompts_label)
                # visual_token_positive_maps = [visual_token_positive_map] * len(batch_inputs)
                # visual_labels_maps = [visual_labels_map] * len(batch_inputs)
                # visual_entities = [
                #     [f'class {item}' for item in torch.unique(fake_visual_prompts_label).cpu().tolist()]
                # ] * len(batch_inputs)
            else:
                # NOTE. Compute visual_dict
                assert 'prompt_bboxes' in batch_data_samples[0]
                visual_prompts = []
                visual_prompts_labels = []
                for data_samples in batch_data_samples:
                    if not data_samples.metainfo['prompt_bboxes_already_transformed']:
                        # NOTE. Project prompt_bboxes from ori_shape to img_shape via homography_matrix.
                        prompt_bboxes_in_ori_shape = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device).float()
                        prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                        prompt_bboxes_in_img_shape = bbox_project(
                                prompt_bboxes_in_ori_shape,
                                torch.from_numpy(data_samples.homography_matrix).to(
                                    self.data_preprocessor.device), data_samples.prompt_img_shape)
                        visual_prompts.append(prompt_bboxes_in_img_shape)
                        visual_prompts_labels.append(prompt_bboxes_labels)
                    else:
                        prompt_bboxes = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device)
                        prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                        visual_prompts.append(prompt_bboxes)
                        visual_prompts_labels.append(prompt_bboxes_labels)
                visual_dict = self.visual_prompt_model(visual_prompts, visual_prompts_labels, batch_data_samples)
                if self.visual_feat_map is not None:
                    visual_dict['embedded'] = self.visual_feat_map(visual_dict['embedded'])

                # NOTE. Extract prompt_visual_feats from prompt_batch_inputs.
                if prompt_batch_inputs is not None:
                    # NOTE. for Predict(prompt_type==Visual)
                    prompt_visual_feats = self.extract_feat(prompt_batch_inputs)
                    visual_dict['prompt_visual_feats'] = prompt_visual_feats
                else:
                    # NOTE. for Eval(prompt_type==Visual), prompt_visual_feats is None.
                    visual_dict['prompt_visual_feats'] = None
                
                # NOTE. Compute visual_token_positive_maps, visual_labels_map and visual_entities.
                # visual_token_positive_maps: stored in data_samples.gt_instances as token_positive_maps for prediction.
                # visual_labels_map: maps visual_prompts_label, used to revert pred_instances['labels'] back (visual prompt specific).
                # visual_entities: class names for display, only used in pet_dino.py.
                visual_token_positive_maps, visual_labels_maps = [], []
                for visual_prompts_label in visual_prompts_labels:
                    assert self.use_global_box
                    visual_prompts_label = torch.unique(visual_prompts_label)
                    visual_token_positive_map, visual_labels_map = self.get_visual_token_positive_map(visual_prompts_label)
                    visual_token_positive_maps.append(visual_token_positive_map)
                    visual_labels_maps.append(visual_labels_map)
                visual_entities = [
                    [f'class {item}' for item in visual_prompts_label.cpu().tolist()]
                    for visual_prompts_label in visual_prompts_labels
                ]
        else:
            visual_dict = {}

        if isinstance(text_prompts[0], list):   # len(text_prompts[0]) : ceil(1203/chunked_size)
            # chunked text prompts, only bs=1 is supported

            # NOTE. Convert visual_dict to chunked mode;
            # generate chunked visual_token_positive_maps (each class has one memory_visual during eval).
            if visual_dict:
                visual_dict_list = []
                accumulate_c = 0
                visual_token_positive_maps = [[]]
                for entity_lst in entities[0]:
                    c_size = len(entity_lst)
                    visual_dict_tmp = {}
                    visual_dict_tmp['memory_visual'] = visual_dict['memory_visual'][:, accumulate_c:accumulate_c+c_size, :]
                    visual_dict_tmp['final_visual_token_mask'] = visual_dict['final_visual_token_mask'][:, accumulate_c:accumulate_c+c_size]
                    visual_dict_list.append(visual_dict_tmp)
                    visual_token_positive_maps[0].append({key: [key - 1] for key in range(1, c_size + 1)})
                    accumulate_c += c_size
            else:
                visual_dict_list = [{},] * len(entities[0])
            
            assert len(batch_inputs) == 1
            count = 0
            results_list = []
            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once
                
                # Override token_positive_map for Visual prompt
                if self.prompt_type == 'Visual':
                    batch_data_samples[
                        0].token_positive_map = visual_token_positive_maps[0][b]

                head_inputs_dict, _ = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, visual_dict_list[b], batch_data_samples)
                head_inputs_dict['prompt_type'] = self.prompt_type
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # NOTE. Both inference and evaluation use this path.
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            # Override token_positive_map for visual prompt
            if self.prompt_type == 'Visual':
                for i, data_samples in enumerate(batch_data_samples):
                    data_samples.token_positive_map = visual_token_positive_maps[i]

            head_inputs_dict, _ = self.forward_transformer(
                visual_feats, text_dict, visual_dict, batch_data_samples)
            head_inputs_dict['prompt_type'] = self.prompt_type
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # Override entity for Visual prompt
        if self.prompt_type == 'Visual':
            entities = visual_entities

        for idx, (data_sample, pred_instances, entity, is_rec_task) in enumerate(zip(
                batch_data_samples, results_list, entities, is_rec_tasks)):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            # NOTE. Remap labels back for Visual prompt
            if self.prompt_type == 'Visual':
                visual_labels_map = visual_labels_maps[idx]
                visual_labels_map_revert = {value: key for key, value in visual_labels_map.items()}
                pred_instances['labels'] = torch.tensor([visual_labels_map_revert[x.item()] for x in pred_instances['labels']],
                                                        device=pred_instances['labels'].device)
            data_sample.pred_instances = pred_instances
        return batch_data_samples
    
    def test_step(self, data: Union[dict, tuple, list], prompt_img_data: Union[dict, tuple, list] = None):
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        if prompt_img_data:     # NOTE. for Predict(prompt_type==Visual)
            prompt_img_data = self.data_preprocessor(prompt_img_data, False)
            data = self.update_promptImgData_to_Data(data=data, prompt_img_data=prompt_img_data)
        else:
            # NOTE. for Predict(prompt_type==Text) and Eval
            pass
        return self._run_forward(data, mode='predict')  # type: ignore

    def update_promptImgData_to_Data(self, data: Union[dict, tuple, list],
                     prompt_img_data: Union[dict, tuple, list]) -> Union[dict, tuple, list]:
        """update_promptImgData_to_Data

        Args:
            prompt_img_data (dict or tuple or list): Data sampled from dataset.
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            dict or tuple or list: similar to data.
        """
        if isinstance(data, dict):
            data['prompt_inputs'] = prompt_img_data['inputs']
            assert len(data['data_samples']) == len(prompt_img_data['data_samples'])
            new_metainfo = dict()
            for i in range(len(data['data_samples'])):
                # type1: collected from PackDetInputs
                new_metainfo['prompt_img_path'] = prompt_img_data['data_samples'][i].prompt_img_path
                new_metainfo['prompt_ori_shape'] = prompt_img_data['data_samples'][i].prompt_ori_shape
                new_metainfo['prompt_img_shape'] = prompt_img_data['data_samples'][i].prompt_img_shape
                # type2: not collected from PackDetInputs
                new_metainfo['prompt_batch_input_shape'] = prompt_img_data['data_samples'][i].batch_input_shape
                new_metainfo['prompt_pad_shape'] = prompt_img_data['data_samples'][i].pad_shape
                # prompt-specific fields
                new_metainfo['prompt_type'] = prompt_img_data['data_samples'][i].prompt_type
                new_metainfo['prompt_bboxes_already_transformed'] = prompt_img_data['data_samples'][i].prompt_bboxes_already_transformed
                new_metainfo['homography_matrix'] = prompt_img_data['data_samples'][i].homography_matrix
                new_metainfo['prompt_bboxes'] = prompt_img_data['data_samples'][i].prompt_bboxes
                new_metainfo['prompt_bboxes_labels'] = prompt_img_data['data_samples'][i].prompt_bboxes_labels
                # memory_visual & memory_label & memory_class_name
                if 'memory_visual' in prompt_img_data['data_samples'][i]:
                    new_metainfo['memory_visual'] = prompt_img_data['data_samples'][i].memory_visual
                    new_metainfo['memory_label'] = prompt_img_data['data_samples'][i].memory_label
                    new_metainfo['memory_class_name'] = prompt_img_data['data_samples'][i].memory_class_name
                data['data_samples'][i].set_metainfo(new_metainfo)
        elif isinstance(data, (list, tuple)):
            raise Exception('not impl yet')
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return data
    
    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                prompt_inputs: torch.Tensor = None,         # May be non-None during predict; always None for eval and train.
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, prompt_inputs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_visual_embedding(self, prompt_img_data: Union[dict, tuple, list] = None) -> Tensor:
        # test_step
        prompt_img_data = self.data_preprocessor(prompt_img_data, False)
        if isinstance(prompt_img_data, dict):
            new_metainfo = dict()
            for i in range(len(prompt_img_data['data_samples'])):
                # type2: not collected from PackDetInputs
                new_metainfo['prompt_batch_input_shape'] = prompt_img_data['data_samples'][i].batch_input_shape
                new_metainfo['prompt_pad_shape'] = prompt_img_data['data_samples'][i].pad_shape
                prompt_img_data['data_samples'][i].set_metainfo(new_metainfo)
        elif isinstance(prompt_img_data, (list, tuple)):
            raise Exception('not impl yet')
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(prompt_img_data)}')
        prompt_batch_inputs = prompt_img_data['inputs']
        batch_data_samples = prompt_img_data['data_samples']

        # predict
        if 'prompt_type' not in batch_data_samples[0].metainfo:
            # NOTE. Eval: prompt_type set via test_cfg, defaults to Text.
            self.prompt_type = self.test_cfg.get('prompt_type', 'Text')
        else:
            # NOTE. Predict: prompt_type set via parameters.
            self.prompt_type = batch_data_samples[0].metainfo['prompt_type']
            print_log(f"Predict : Set prompt type to {self.prompt_type}.")
        
        # NOTE. visual prompt
        if self.prompt_type == 'Visual':
            # NOTE. Compute visual_dict
            assert 'prompt_bboxes' in batch_data_samples[0]
            visual_prompts = []
            visual_prompts_labels = []
            for data_samples in batch_data_samples:
                if not data_samples.metainfo['prompt_bboxes_already_transformed']:
                    # NOTE. Project prompt_bboxes from ori_shape to img_shape via homography_matrix.
                    prompt_bboxes_in_ori_shape = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device).float()
                    prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                    prompt_bboxes_in_img_shape = bbox_project(
                            prompt_bboxes_in_ori_shape,
                            torch.from_numpy(data_samples.homography_matrix).to(
                                self.data_preprocessor.device), data_samples.prompt_img_shape)
                    visual_prompts.append(prompt_bboxes_in_img_shape)
                    visual_prompts_labels.append(prompt_bboxes_labels)
                else:
                    prompt_bboxes = data_samples.metainfo['prompt_bboxes'].to(data_samples.gt_instances.bboxes.device)
                    prompt_bboxes_labels = data_samples.metainfo['prompt_bboxes_labels'].to(data_samples.gt_instances.labels.device)
                    visual_prompts.append(prompt_bboxes)
                    visual_prompts_labels.append(prompt_bboxes_labels)
            visual_dict = self.visual_prompt_model(visual_prompts, visual_prompts_labels, batch_data_samples)
            if self.visual_feat_map is not None:
                visual_dict['embedded'] = self.visual_feat_map(visual_dict['embedded'])

            # NOTE. Extract prompt_visual_feats from prompt_batch_inputs.
            if prompt_batch_inputs is not None:
                # NOTE. for Predict(prompt_type==Visual)
                prompt_visual_feats = self.extract_feat(prompt_batch_inputs)
                visual_dict['prompt_visual_feats'] = prompt_visual_feats
            else:
                # NOTE. for Eval(prompt_type==Visual), prompt_visual_feats is None.
                visual_dict['prompt_visual_feats'] = None

            # NOTE. Compute visual_token_positive_maps, visual_labels_map and visual_entities.
            # visual_token_positive_maps: stored in data_samples.gt_instances as token_positive_maps for prediction.
            # visual_labels_map: maps visual_prompts_label, used to revert pred_instances['labels'] back (visual prompt specific).
            # visual_entities: class names for display, only used in pet_dino.py.
            visual_token_positive_maps, visual_labels_maps = [], []
            for visual_prompts_label in visual_prompts_labels:
                if self.use_global_box:
                    visual_prompts_label = torch.unique(visual_prompts_label)
                visual_token_positive_map, visual_labels_map = self.get_visual_token_positive_map(visual_prompts_label)
                visual_token_positive_maps.append(visual_token_positive_map)
                visual_labels_maps.append(visual_labels_map)
            visual_entities = [
                [f'class {item}' for item in visual_prompts_label.cpu().tolist()]
                for visual_prompts_label in visual_prompts_labels
            ]
        else:
            visual_dict = {}
            raise Exception('INFO: check your prompt_type when extract visual embedding.')

        # Override token_positive_map for Visual prompt
        if self.prompt_type == 'Visual':
            for i, data_samples in enumerate(batch_data_samples):
                data_samples.token_positive_map = visual_token_positive_maps[i]

        # NOTE. Apply pre_transformer to prompt_image.
        prompt_visual_feats = visual_dict.pop('prompt_visual_feats')
        
        # NOTE. Since prompt_image and image are the same, we can directly run inference.
        encoder_inputs_dict_for_prompt, _ = self.pre_transformer(
                prompt_visual_feats, batch_data_samples)

        if self.encoder_first_then_visual_prompt_encoder:
            # run forward_encoder first
            encoder_outputs_dict = self.forward_encoder(
                **encoder_inputs_dict_for_prompt, text_dict={})
            # then run forward_visual_prompt_encoder
            # NOTE. replace encoder_inputs_dict['feat'] with encoder_outputs_dict['memory']
            encoder_inputs_dict_for_prompt['feat'] = encoder_outputs_dict['memory']
            # for train(visual_prompt==Visual) and eval(visual_prompt==Visual)
            memory_visual = self.forward_visual_prompt_encoder(**encoder_inputs_dict_for_prompt, visual_dict=visual_dict)
        else:
            # NOTE. forward_visual_prompt_encoder, used to obtain memory_visual.
            memory_visual = self.forward_visual_prompt_encoder(**encoder_inputs_dict_for_prompt, visual_dict=visual_dict)
        
        return memory_visual
