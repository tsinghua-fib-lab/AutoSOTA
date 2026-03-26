import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
from modeling.clip.model import QuickGELU
import torch


class MDReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(MDReID, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        if self.GLOBAL_LOCAL:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
            self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
            self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())

        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

            if self.cfg.MODEL.ADD_SHARE is True:
                self.classifier_s = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
                self.classifier_s.apply(weights_init_classifier)
                self.bottleneck_s = nn.BatchNorm1d(3 * self.feat_dim)
                self.bottleneck_s.bias.requires_grad_(False)
                self.bottleneck_s.apply(weights_init_kaiming)

            if cfg.MODEL.ADD_ELOSS is True:
                self.classifier_rs = nn.Linear(2 * self.feat_dim, self.num_classes, bias=False)
                self.classifier_rs.apply(weights_init_classifier)
                self.bottleneck_rs = nn.BatchNorm1d(2 * self.feat_dim)
                self.bottleneck_rs.bias.requires_grad_(False)
                self.bottleneck_rs.apply(weights_init_kaiming)
                self.classifier_ns = nn.Linear(2 * self.feat_dim, self.num_classes, bias=False)
                self.classifier_ns.apply(weights_init_classifier)
                self.bottleneck_ns = nn.BatchNorm1d(2 * self.feat_dim)
                self.bottleneck_ns.bias.requires_grad_(False)
                self.bottleneck_ns.apply(weights_init_kaiming)
                self.classifier_ts = nn.Linear(2 * self.feat_dim, self.num_classes, bias=False)
                self.classifier_ts.apply(weights_init_classifier)
                self.bottleneck_ts = nn.BatchNorm1d(2 * self.feat_dim)
                self.bottleneck_ts.bias.requires_grad_(False)


    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.to("cuda:"+self.cfg.MODEL.DEVICE_ID).eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            img_In = torch.cat((RGB, NI, TI), dim=0)
            B = img_In.shape[0]//3
            if cam_label.shape.numel() > 1:
                cam_label = torch.cat([cam_label, cam_label, cam_label], dim=0)
                view_label = torch.cat([view_label, view_label, view_label], dim=0)
            RNT_cash, RNT_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
            RGB_cash, NI_cash, TI_cash = RNT_cash[0:B,], RNT_cash[B:B*2,], RNT_cash[B*2:B*3,]
            RGB_global, NI_global, TI_global = RNT_global[0:B,], RNT_global[B:B*2,], RNT_global[B*2:B*3,]

            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
            if self.direct:
                if self.cfg.MODEL.ADD_SHARE is True:
                    ori = torch.cat(
                        [RGB_global, NI_global, TI_global, RGB_cash[:, 0, :], NI_cash[:, 0, :], TI_cash[:, 0, :]], dim=-1)
                    ori_global = self.bottleneck(ori[:, ori.shape[1]//2:])
                    ori_score = self.classifier(ori_global)
                else:
                    ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                    ori_global = self.bottleneck(ori)
                    ori_score = self.classifier(ori_global)
            else:
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))
                if self.cfg.MODEL.ADD_SHARE is True:
                    ori = torch.cat(
                        [RGB_global, NI_global, TI_global, RGB_cash[:, 0, :], NI_cash[:, 0, :], TI_cash[:, 0, :]], dim=-1)
                    share_ori_global = ori[:, ori.shape[1] // 2:]
                    share_ori_score = self.classifier_s(self.bottleneck_s(share_ori_global))
                if self.cfg.MODEL.ADD_ELOSS is True:
                    RGB_Share = torch.cat([RGB_global, RGB_cash[:, 0, :]],dim=-1)
                    RS_ori_score = self.classifier_rs(self.bottleneck_rs(RGB_Share))
                    NI_Share = torch.cat([NI_global, NI_cash[:, 0, :]],dim=-1)
                    NS_ori_score = self.classifier_rs(self.bottleneck_rs(NI_Share))
                    TI_Share = torch.cat([TI_global, TI_cash[:, 0, :]],dim=-1)
                    TS_ori_score = self.classifier_rs(self.bottleneck_rs(TI_Share))

            if self.direct:
                return ori_score, ori
            else:
                if self.cfg.MODEL.ADD_SHARE is True:
                    if self.cfg.MODEL.ADD_ELOSS is False:
                        return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, share_ori_score, share_ori_global
                    else:
                        return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, share_ori_score, share_ori_global, RS_ori_score, RGB_Share, NS_ori_score, NI_Share, TS_ori_score, TI_Share
                else:
                    return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global
        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            if self.cfg.MODEL.ADD_SHARE is True:
                if 'cam_label' in x:
                    cam_label = x['cam_label']
                if self.miss_type == 'r':
                    if cam_label.shape.numel() > 1:
                        cam_label = torch.cat([cam_label, cam_label], dim=0)
                        view_label = torch.cat([view_label, view_label], dim=0)
                    img_In = torch.cat((NI, TI), dim=0)
                    NT_cash, NT_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
                    B = img_In.shape[0] // 2
                    NI_cash, TI_cash = NT_cash[0:B, ], NT_cash[B:B * 2, ]
                    NI_global, TI_global = NT_global[0:B, ], NT_global[B:B * 2, ]
                    if self.GLOBAL_LOCAL:
                        NI_local = self.pool(NI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        TI_local = self.pool(TI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                        TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
                    ori = torch.cat(
                        [NI_global, TI_global, NI_cash[:, 0, :], TI_cash[:, 0, :]], dim=-1)
                elif self.miss_type == 'n':
                    if cam_label.shape.numel() > 1:
                        cam_label = torch.cat([cam_label, cam_label], dim=0)
                        view_label = torch.cat([view_label, view_label], dim=0)
                    img_In = torch.cat((RGB, TI), dim=0)
                    RT_cash, RT_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
                    B = img_In.shape[0] // 2
                    RGB_cash, TI_cash = RT_cash[0:B, ], RT_cash[B:B * 2, ]
                    RGB_global, TI_global = RT_global[0:B, ], RT_global[B:B * 2, ]
                    if self.GLOBAL_LOCAL:
                        RGB_local = self.pool(RGB_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        TI_local = self.pool(TI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                        TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
                    ori = torch.cat(
                        [RGB_global, TI_global, RGB_cash[:, 0, :], TI_cash[:, 0, :]], dim=-1)
                elif self.miss_type == 't':
                    if cam_label.shape.numel() > 1:
                        cam_label = torch.cat([cam_label, cam_label], dim=0)
                        view_label = torch.cat([view_label, view_label], dim=0)
                    img_In = torch.cat((RGB, NI), dim=0)
                    RN_cash, RN_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
                    B = img_In.shape[0] // 2
                    RGB_cash, NI_cash = RN_cash[0:B, ], RN_cash[B:B * 2, ]
                    RGB_global, NI_global = RN_global[0:B, ], RN_global[B:B * 2, ]
                    if self.GLOBAL_LOCAL:
                        RGB_local = self.pool(RGB_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        NI_local = self.pool(NI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                        NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                    ori = torch.cat(
                        [RGB_global, NI_global, RGB_cash[:, 0, :], NI_cash[:, 0, :]], dim=-1)
                elif self.miss_type == 'rn':
                    TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
                    if self.GLOBAL_LOCAL:
                        TI_local = self.pool(TI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
                    ori = torch.cat(
                        [TI_global, TI_cash[:, 0, :]], dim=-1)
                elif self.miss_type == 'rt':
                    NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
                    if self.GLOBAL_LOCAL:
                        NI_local = self.pool(NI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                    ori = torch.cat(
                        [NI_global, NI_cash[:, 0, :]], dim=-1)
                elif self.miss_type == 'nt':
                    RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                    if self.GLOBAL_LOCAL:
                        RGB_local = self.pool(RGB_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                    ori = torch.cat(
                        [RGB_global, RGB_cash[:, 0, :]], dim=-1)
                else:
                    if cam_label.shape.numel() > 1:
                        cam_label = torch.cat([cam_label, cam_label, cam_label], dim=0)
                        view_label = torch.cat([view_label, view_label, view_label], dim=0)
                    img_In = torch.cat((RGB, NI, TI), dim=0)
                    RNT_cash, RNT_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
                    B = img_In.shape[0] // 3
                    RGB_cash, NI_cash, TI_cash = RNT_cash[0:B, ], RNT_cash[B:B * 2, ], RNT_cash[B * 2:B * 3, ]
                    RGB_global, NI_global, TI_global = RNT_global[0:B, ], RNT_global[B:B * 2, ], RNT_global[
                                                                                                 B * 2:B * 3, ]
                    if self.GLOBAL_LOCAL:
                        RGB_local = self.pool(RGB_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        NI_local = self.pool(NI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        TI_local = self.pool(TI_cash[:, 1:, :].permute(0, 2, 1)).squeeze(-1)
                        RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                        NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                        TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
                    ori = torch.cat(
                        [RGB_global, NI_global, TI_global, RGB_cash[:, 0, :], NI_cash[:, 0, :], TI_cash[:, 0, :]],
                        dim=-1)
            else:
                if self.miss_type == 'r':
                    RGB = torch.zeros_like(RGB)
                elif self.miss_type == 'n':
                    NI = torch.zeros_like(NI)
                elif self.miss_type == 't':
                    TI = torch.zeros_like(TI)
                elif self.miss_type == 'rn':
                    RGB = torch.zeros_like(RGB)
                    NI = torch.zeros_like(NI)
                elif self.miss_type == 'rt':
                    RGB = torch.zeros_like(RGB)
                    TI = torch.zeros_like(TI)
                elif self.miss_type == 'nt':
                    NI = torch.zeros_like(NI)
                    TI = torch.zeros_like(TI)

                if 'cam_label' in x:
                    cam_label = x['cam_label']
                img_In = torch.cat((RGB, NI, TI), dim=0)
                if cam_label.shape.numel() > 1:
                    cam_label = torch.cat([cam_label, cam_label, cam_label], dim=0)
                    view_label = torch.cat([view_label, view_label, view_label], dim=0)
                B = img_In.shape[0] // 3
                RNT_cash, RNT_global = self.BACKBONE(img_In, cam_label=cam_label, view_label=view_label)
                if self.cfg.MODEL.ADD_SHARE is True:
                    RGB_cash, NI_cash, TI_cash = RNT_cash[0:B, 1:, ], RNT_cash[B:B*2, 1:, ], RNT_cash[B*2:B*3, 1:, ]
                else:
                    RGB_cash, NI_cash, TI_cash = RNT_cash[0:B, ], RNT_cash[B:B * 2, ], RNT_cash[B * 2:B * 3, ]
                RGB_global, NI_global, TI_global = RNT_global[0:B,], RNT_global[B:B*2,], RNT_global[B*2:B*3,]

                if self.GLOBAL_LOCAL:
                    RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                    NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                    TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                    RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                    NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                    TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            return ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = MDReID(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building MDReID===========')
    return model
