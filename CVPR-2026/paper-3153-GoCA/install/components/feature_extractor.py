import math
import json
import torch
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class FeatureStore:
    def __init__(self, to_store, resize_ratio, train_unet):
        if to_store:
            self.to_store = to_store
            self.accept_all = False
        else:
            self.to_store = {}
            self.accept_all = True
        self.feats = {}
        self.status = 'active'
        self.resize_ratio = resize_ratio
        self.train_unet = train_unet
        self.store_idx = None

    def pause(self):
        self.status = 'pause'

    def resume(self):
        self.status = 'active'

    def reset(self):
        self.feats = {}

    def store(self, feat, feat_id):
        if self.status == 'pause':
            return

        # perform filtering
        if (feat_id in self.to_store.keys() and self.to_store[feat_id]) or self.accept_all:
            # filter cross-k and cross-v
            if 'cross-k' in feat_id or 'cross-v' in feat_id:
                return

            # do resize for ffn features
            # if 'ffn' in feat_id:
            #     feat = F.interpolate(feat, size=feat.shape[-1]//4, mode='linear')

            # do reshape for ViT features
            if 'special' not in feat_id:
                if len(feat.shape) == 3:
                    size = int(math.sqrt(feat.shape[1]))
                    feat = rearrange(feat, 'b (h w) c -> b c h w', h=size)

            # do resize
            if 'special' not in feat_id:
                if self.resize_ratio > 1:
                    target_size = (feat.shape[2]//self.resize_ratio, feat.shape[3]//self.resize_ratio)
                    feat = F.adaptive_avg_pool2d(feat, target_size)

            # normalize
            if 'special' not in feat_id:
                feat = TF.normalize(feat, mean=0, std=1)

            # dtype and device
            feat = feat.to(torch.float16)#.to('cuda')

            # add to store
            if not self.train_unet:
                feat = feat.detach()
            if self.accept_all:
                feat = feat.cpu()

            if self.store_idx is None:
                self.feats[feat_id] = feat
            else:
                entry = self.feats[feat_id] if feat_id in self.feats else {'feat': {}, 'count': 0}
                current_idx = entry['count'] + 1
                if current_idx in self.store_idx:
                    entry['feat'][current_idx] = feat
                entry['count'] = current_idx
                self.feats[feat_id] = entry

    @property
    def stored_feats(self):
        return self.feats


class FeatureGatherer:
    def __init__(self, module_id, feature_store):
        self.module_id = module_id
        self.feature_store = feature_store

    def gather(self, feat, feat_id):
        self.feature_store.store(feat, '-'.join([self.module_id, feat_id]))


def prepare_feature_extractor(version, pipe, config, resize_ratio, train_unet):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    feature_store = FeatureStore(config, resize_ratio, train_unet)

    if version == 'flux':
        pipe.transformer.feature_gatherer = FeatureGatherer('vit', feature_store)
        for i, basic_block in enumerate(pipe.transformer.transformer_blocks):
            # basic block
            block_id = '-'.join(['vit', f'block{i}'])
            # print('\t', block_id)
            basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
            # attention
            block_id = '-'.join(['vit', f'block{i}'])
            # print('\t', block_id)
            basic_block.attn.feature_gatherer = FeatureGatherer(block_id, feature_store)
            # ffn
            block_id = '-'.join(['vit', f'block{i}', 'ffn'])
            # print('\t', block_id)
            basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)
        for basic_block in pipe.transformer.single_transformer_blocks:
            i += 1
            # basic block
            block_id = '-'.join(['vit', f'block{i}'])
            # print('\t', block_id)
            basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
            # attention
            block_id = '-'.join(['vit', f'block{i}'])
            # print('\t', block_id)
            basic_block.attn.feature_gatherer = FeatureGatherer(block_id, feature_store)
        return feature_store

    if not hasattr(pipe, 'transformer'):
        pipe.unet.feature_gatherer = FeatureGatherer('unet', feature_store)

        # print('down')
        stage_name = 'down'
        for level_name, level in enumerate(pipe.unet.down_blocks):
            # print('\t', type(level))
            for repeat_name, repeat in enumerate(range(len(level.resnets))):
                block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'res'])
                # print('\t\t', block_id)
                level.resnets[repeat].feature_gatherer = FeatureGatherer(block_id, feature_store)
                if hasattr(level, 'attentions') and len(level.attentions) > 0:
                    block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit'])
                    # print('\t\t', block_id)
                    level.attentions[repeat].feature_gatherer = FeatureGatherer(block_id, feature_store)
                    if hasattr(level.attentions[repeat], 'transformer_blocks'):
                        for i, basic_block in enumerate(level.attentions[repeat].transformer_blocks):
                            # basic block
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}'])
                            # print('\t\t\t', block_id)
                            basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # self-attention
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'self'])
                            # print('\t\t\t', block_id)
                            basic_block.attn1.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # cross-attention
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'cross'])
                            # print('\t\t\t', block_id)
                            basic_block.attn2.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # ffn
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'ffn'])
                            # print('\t\t\t', block_id)
                            basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)
                    else:
                        i = 0
                        basic_block = level.attentions[repeat]
                        # basic block
                        block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'cross'])
                        # print('\t\t\t', block_id)
                        basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
            if level.downsamplers:
                for downsampler in level.downsamplers:
                    block_id = '-'.join([stage_name, f'level{level_name}', 'downsampler'])
                    # print('\t\t', type(downsampler))
                    downsampler.feature_gatherer = FeatureGatherer(block_id, feature_store)

        # print('mid')
        stage_name = 'mid'
        level = pipe.unet.mid_block
        for repeat_name, repeat in enumerate(range(len(level.resnets))):
            block_id = '-'.join([stage_name, f'repeat{repeat_name}', 'res'])
            # print('\t\t', block_id)
            level.resnets[repeat].feature_gatherer = FeatureGatherer(block_id, feature_store)
        if hasattr(level, 'attentions') and len(level.attentions) > 0:
            block_id = '-'.join([stage_name, 'vit'])
            # print('\t\t', block_id)
            level.attentions[0].feature_gatherer = FeatureGatherer(block_id, feature_store)
            if hasattr(level.attentions[0], 'transformer_blocks'):
                for i, basic_block in enumerate(level.attentions[0].transformer_blocks):
                    # basic block
                    block_id = '-'.join([stage_name, 'vit', f'block{i}'])
                    # print('\t\t\t', block_id)
                    basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
                    # self-attention
                    block_id = '-'.join([stage_name, 'vit', f'block{i}', 'self'])
                    # print('\t\t\t', block_id)
                    basic_block.attn1.feature_gatherer = FeatureGatherer(block_id, feature_store)
                    # cross-attention
                    block_id = '-'.join([stage_name, 'vit', f'block{i}', 'cross'])
                    # print('\t\t\t', block_id)
                    basic_block.attn2.feature_gatherer = FeatureGatherer(block_id, feature_store)
                    # ffn
                    block_id = '-'.join([stage_name, 'vit', f'block{i}', 'ffn'])
                    # print('\t\t\t', block_id)
                    basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)
            else:
                i = 0
                basic_block = level.attentions[0]
                # basic block
                block_id = '-'.join([stage_name, 'vit', f'block{i}', 'cross'])
                # print('\t\t\t', block_id)
                basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
        
        # print('up')
        stage_name = 'up'
        for level_name, level in enumerate(pipe.unet.up_blocks):
            # print('\t', type(level))
            for repeat_name, repeat in enumerate(range(len(level.resnets))):
                block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'res'])
                # print('\t\t', block_id)
                level.resnets[repeat].feature_gatherer = FeatureGatherer(block_id, feature_store)
                if hasattr(level, 'attentions') and len(level.attentions) > 0:
                    block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit'])
                    # print('\t\t', block_id)
                    level.attentions[repeat].feature_gatherer = FeatureGatherer(block_id, feature_store)
                    if hasattr(level.attentions[repeat], 'transformer_blocks'):
                        for i, basic_block in enumerate(level.attentions[repeat].transformer_blocks):
                            # basic block
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}'])
                            # print('\t\t\t', block_id)
                            basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # self-attention
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'self'])
                            # print('\t\t\t', block_id)
                            basic_block.attn1.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # cross-attention
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'cross'])
                            # print('\t\t\t', block_id)
                            basic_block.attn2.feature_gatherer = FeatureGatherer(block_id, feature_store)
                            # ffn
                            block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'ffn'])
                            # print('\t\t\t', block_id)
                            basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)
                    else:
                        i = 0
                        basic_block = level.attentions[repeat]
                        # basic block
                        block_id = '-'.join([stage_name, f'level{level_name}', f'repeat{repeat_name}', 'vit', f'block{i}', 'cross'])
                        # print('\t\t\t', block_id)
                        basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
            if level.upsamplers:
                for upsampler in level.upsamplers:
                    # print('\t\t', type(upsampler))
                    block_id = '-'.join([stage_name, f'level{level_name}', 'upsampler'])
                    upsampler.feature_gatherer = FeatureGatherer(block_id, feature_store)
    else:
        if not hasattr(pipe.transformer, 'transformer_blocks'):
            for i, basic_block in enumerate(pipe.transformer.blocks):
                # basic block
                block_id = '-'.join(['vit', f'block{i}'])
                # print('\t', block_id)
                basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # self-attention
                block_id = '-'.join(['vit', f'block{i}', 'self'])
                # print('\t', block_id)
                basic_block.attn1.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # cross-attention
                block_id = '-'.join(['vit', f'block{i}', 'cross'])
                # print('\t', block_id)
                basic_block.attn2.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # ffn
                block_id = '-'.join(['vit', f'block{i}', 'ffn'])
                # print('\t', block_id)
                basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)
        else:
            for i, basic_block in enumerate(pipe.transformer.transformer_blocks):
                # basic block
                block_id = '-'.join(['vit', f'block{i}'])
                # print('\t', block_id)
                basic_block.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # self-attention
                block_id = '-'.join(['vit', f'block{i}', 'self'])
                # print('\t', block_id)
                basic_block.attn1.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # cross-attention
                block_id = '-'.join(['vit', f'block{i}', 'cross'])
                # print('\t', block_id)
                basic_block.attn2.feature_gatherer = FeatureGatherer(block_id, feature_store)
                # ffn
                block_id = '-'.join(['vit', f'block{i}', 'ffn'])
                # print('\t', block_id)
                basic_block.ff.feature_gatherer = FeatureGatherer(block_id, feature_store)

    return feature_store
